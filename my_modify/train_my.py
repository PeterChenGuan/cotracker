import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from my_modify.my_real_dataset import MyRealDataset
from cotracker.datasets.utils import collate_fn_train
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline


def get_args():
    parser = argparse.ArgumentParser()

    # 数据
    parser.add_argument("--data_root", type=str, required=True,
                        help="图片帧数据集根目录，下面每个子目录是一段序列")
    parser.add_argument("--crop_size", type=int, nargs=2, default=[384, 512],
                        help="训练裁剪尺寸 H W")
    parser.add_argument("--seq_len", type=int, default=24,
                        help="每段序列帧数")
    parser.add_argument("--traj_per_sample", type=int, default=256,
                        help="每个样本采样轨迹数")
    parser.add_argument("--limit_samples", type=int, default=10000,
                        help="最多用多少段序列")
    parser.add_argument("--num_workers", type=int, default=4)

    # 训练
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--save_freq", type=int, default=500,
                        help="每多少步保存一次 checkpoint")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")

    # 模型
    parser.add_argument("--model_type", type=str, default="offline",
                        choices=["offline", "online"],
                        help="offline=CoTrackerThreeOffline, online=CoTrackerThreeOnline")
    parser.add_argument("--student_ckpt", type=str, default=None,
                        help="student 模型初始化权重，不填则随机初始化")
    parser.add_argument("--teacher_ckpt", type=str, required=True,
                        help="teacher 模型权重路径（必须填）")

    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_frame_rate", action="store_true")
    parser.add_argument("--random_seq_len", action="store_true")
    parser.add_argument("--random_resize", action="store_true")

    return parser.parse_args()


def build_model(model_type, seq_len, ckpt_path=None):
    if model_type == "offline":
        model = CoTrackerThreeOffline(
            stride=4,
            corr_radius=3,
            window_len=seq_len,
            model_resolution=(384, 512),
            linear_layer_for_vis_conf=True,
        )
    else:
        model = CoTrackerThreeOnline(
            stride=4,
            corr_radius=3,
            window_len=seq_len,
            model_resolution=(384, 512),
            linear_layer_for_vis_conf=True,
        )

    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # 兼容不同保存格式
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"[build_model] loaded from {ckpt_path}")
    else:
        print(f"[build_model] random init")

    return model


def preprocess_video(video):
    # video: [B, T, 3, H, W], float32, 0~255
    # 归一化到 [-0.5, 0.5]
    return video / 255.0 - 0.5


def generate_pseudo_labels(teacher, video, traj_per_sample, device):
    """
    用 teacher 在 video 上生成伪标签轨迹和 visibility
    video: [B, T, 3, H, W], float32, 0~255
    返回:
        trajs: [B, T, N, 2]
        vis:   [B, T, N]
    """
    B, T, C, H, W = video.shape

    with torch.no_grad():
        # 在第一帧上均匀采样查询点
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(4, H - 4, int(traj_per_sample ** 0.5), device=device),
            torch.linspace(4, W - 4, int(traj_per_sample ** 0.5), device=device),
            indexing="ij",
        )
        # [N, 2] -> (x, y)
        queries_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        N = queries_xy.shape[0]

        # queries: [B, N, 3] -> (t, x, y), t=0 表示从第0帧开始跟踪
        t_col = torch.zeros(N, 1, device=device)
        queries = torch.cat([t_col, queries_xy], dim=-1)  # [N, 3]
        queries = queries.unsqueeze(0).expand(B, -1, -1)  # [B, N, 3]

        video_norm = preprocess_video(video)

        pred_trajs, pred_vis, pred_conf = teacher(
            video=video_norm,
            queries=queries,
            iters=6,
        )
        # pred_trajs: [B, T, N, 2]
        # pred_vis:   [B, T, N]

    return pred_trajs, pred_vis


def compute_loss(pred_trajs, pred_vis, gt_trajs, gt_vis):
    """
    pred_trajs: [B, T, N, 2]
    pred_vis:   [B, T, N]
    gt_trajs:   [B, T, N, 2]
    gt_vis:     [B, T, N]
    """
    # 轨迹 L1 loss，只在 teacher 认为可见的点上计算
    vis_mask = gt_vis.bool()  # [B, T, N]

    # 轨迹损失
    traj_diff = (pred_trajs - gt_trajs).abs()  # [B, T, N, 2]
    traj_loss = traj_diff[vis_mask].mean() if vis_mask.any() else torch.tensor(0.0)

    # visibility BCE loss
    vis_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_vis, gt_vis.float()
    )

    total_loss = traj_loss + vis_loss
    return total_loss, traj_loss, vis_loss


def save_checkpoint(model, optimizer, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"[save] checkpoint -> {path}")


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ── 数据集 ──────────────────────────────────────────────
    train_dataset = MyRealDataset(
        data_root=args.data_root,
        crop_size=tuple(args.crop_size),
        seq_len=args.seq_len,
        traj_per_sample=args.traj_per_sample,
        random_frame_rate=args.random_frame_rate,
        random_seq_len=args.random_seq_len,
        random_resize=args.random_resize,
        limit_samples=args.limit_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_train,
        drop_last=True,
        pin_memory=True,
    )

    # ── teacher 模型（frozen，不更新梯度）────────────────────
    teacher = build_model(args.model_type, args.seq_len, args.teacher_ckpt)
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("[teacher] loaded and frozen")

    # ── student 模型（需要训练）──────────────────────────────
    student = build_model(args.model_type, args.seq_len, args.student_ckpt)
    student = student.to(device)
    student.train()
    print("[student] ready")

    # ── optimizer ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.num_steps,
        pct_start=0.05,
    )

    # ── 训练循环 ─────────────────────────────────────────────
    step = 0
    data_iter = iter(train_loader)

    print(f"start training for {args.num_steps} steps")

    while step < args.num_steps:
        # 取一个 batch
        try:
            batch, gotit = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch, gotit = next(data_iter)

        # 跳过读取失败的 batch
        if not all(gotit):
            continue

        video = batch.video.to(device)   # [B, T, 3, H, W]
        B, T, C, H, W = video.shape

        # ── step1: teacher 生成伪标签 ─────────────────────────
        gt_trajs, gt_vis = generate_pseudo_labels(
            teacher, video, args.traj_per_sample, device
        )
        # gt_trajs: [B, T, N, 2]
        # gt_vis:   [B, T, N]

        N = gt_trajs.shape[2]

        # ── step2: 构造 student 的输入 queries ───────────────
        # 使用 teacher 预测的第 0 帧位置作为查询点
        queries_xy = gt_trajs[:, 0, :, :]   # [B, N, 2]  第0帧的 (x,y)
        t_col = torch.zeros(B, N, 1, device=device)
        queries = torch.cat([t_col, queries_xy], dim=-1)  # [B, N, 3]

        video_norm = preprocess_video(video)

        # ── step3: student 前向 ───────────────────────────────
        optimizer.zero_grad()

        pred_trajs, pred_vis, pred_conf = student(
            video=video_norm,
            queries=queries,
            iters=6,
        )

        # ── step4: 计算 loss ──────────────────────────────────
        loss, traj_loss, vis_loss = compute_loss(
            pred_trajs, pred_vis, gt_trajs, gt_vis
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        step += 1

        # ── 打印日志 ──────────────────────────────────────────
        if step % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:6d}/{args.num_steps} | "
                f"loss={loss.item():.4f} | "
                f"traj={traj_loss.item():.4f} | "
                f"vis={vis_loss.item():.4f} | "
                f"lr={lr_now:.6f}"
            )

        # ── 保存 checkpoint ───────────────────────────────────
        if step % args.save_freq == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"student_step{step:06d}.pth")
            save_checkpoint(student, optimizer, step, ckpt_path)

    # 训练结束保存最终模型
    final_path = os.path.join(args.ckpt_dir, "student_final.pth")
    save_checkpoint(student, optimizer, step, final_path)
    print("training done")


if __name__ == "__main__":
    main()