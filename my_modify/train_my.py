import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from my_modify.my_real_dataset import MyRealDataset
from cotracker.datasets.utils import collate_fn_train
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline


def get_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of image-sequence dataset. Each subdirectory is one sequence.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs=2,
        default=[384, 512],
        help="Training crop size: H W.",
    )
    parser.add_argument("--seq_len", type=int, default=24, help="Frames per clip.")
    parser.add_argument(
        "--traj_per_sample",
        type=int,
        default=256,
        help="Target number of pseudo-label trajectories per sample.",
    )
    parser.add_argument(
        "--limit_samples",
        type=int,
        default=10000,
        help="Maximum number of sequences to use.",
    )
    parser.add_argument("--num_workers", type=int, default=4)

    # Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--save_freq", type=int, default=500)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")

    # Model
    parser.add_argument(
        "--model_type",
        type=str,
        default="offline",
        choices=["offline", "online"],
        help="offline=CoTrackerThreeOffline, online=CoTrackerThreeOnline",
    )
    parser.add_argument(
        "--student_ckpt",
        type=str,
        default=None,
        help="Optional checkpoint used to initialize the student.",
    )
    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        required=True,
        help="Required checkpoint used by the frozen teacher.",
    )

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random_frame_rate", action="store_true")
    parser.add_argument("--random_seq_len", action="store_true")
    parser.add_argument("--random_resize", action="store_true")

    return parser.parse_args()


def build_model(model_type, seq_len, ckpt_path=None, crop_size=(384, 512)):
    model_kwargs = dict(
        stride=4,
        corr_radius=3,
        window_len=seq_len,
        model_resolution=tuple(crop_size),
        linear_layer_for_vis_conf=True,
    )
    if model_type == "offline":
        model = CoTrackerThreeOffline(**model_kwargs)
    else:
        model = CoTrackerThreeOnline(**model_kwargs)

    if ckpt_path is not None and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"[build_model] loaded from {ckpt_path}")
    else:
        print("[build_model] random init")

    return model


def preprocess_video(video):
    """Keep video in 0~255 range; CoTrackerThree normalizes internally."""
    return video.float()


def unpack_model_output(output):
    """Return (tracks, visibility, confidence, train_data) for CoTracker variants."""
    if not isinstance(output, (tuple, list)):
        raise TypeError(f"Expected model output tuple/list, got {type(output)!r}")
    if len(output) == 4:
        tracks, visibility, confidence, train_data = output
    elif len(output) == 3:
        tracks, visibility, train_data = output
        confidence = None
    else:
        raise ValueError(f"Unexpected model output length: {len(output)}")
    return tracks, visibility, confidence, train_data


def make_grid_queries(batch_size, traj_per_sample, height, width, device):
    grid_size = max(1, int(traj_per_sample ** 0.5))
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(4, height - 4, grid_size, device=device),
        torch.linspace(4, width - 4, grid_size, device=device),
        indexing="ij",
    )
    queries_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    t_col = torch.zeros(queries_xy.shape[0], 1, device=device)
    queries = torch.cat([t_col, queries_xy], dim=-1)
    return queries.unsqueeze(0).expand(batch_size, -1, -1).contiguous()


def generate_pseudo_labels(teacher, video, traj_per_sample, device):
    """
    Generate teacher pseudo labels.

    Args:
        teacher: frozen CoTracker model.
        video: [B, T, 3, H, W], float32, 0~255.
    Returns:
        trajs: [B, T, N, 2]
        vis:   [B, T, N], probability in [0, 1]
    """
    B, _, _, H, W = video.shape
    queries = make_grid_queries(B, traj_per_sample, H, W, device)
    video_for_model = preprocess_video(video)

    with torch.no_grad():
        pred_trajs, pred_vis, pred_conf, _ = unpack_model_output(
            teacher(video=video_for_model, queries=queries, iters=6)
        )
        if pred_conf is not None:
            pred_vis = pred_vis * pred_conf

    return pred_trajs.detach(), pred_vis.detach()


def compute_loss(pred_trajs, pred_vis, gt_trajs, gt_vis):
    """
    Args:
        pred_trajs: [B, T, N, 2]
        pred_vis:   [B, T, N], probability in [0, 1]
        gt_trajs:   [B, T, N, 2]
        gt_vis:     [B, T, N], probability in [0, 1]
    """
    gt_vis = gt_vis.float().clamp(0.0, 1.0)
    vis_mask = gt_vis > 0.5

    traj_diff = (pred_trajs - gt_trajs).abs()
    traj_loss = (
        traj_diff[vis_mask].mean()
        if vis_mask.any()
        else pred_trajs.new_tensor(0.0)
    )

    # CoTracker3 returns visibility probabilities, not logits.
    pred_vis = pred_vis.clamp(1e-6, 1.0 - 1e-6)
    vis_loss = torch.nn.functional.binary_cross_entropy(pred_vis, gt_vis)

    total_loss = traj_loss + vis_loss
    return total_loss, traj_loss, vis_loss


def save_checkpoint(model, optimizer, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    print(f"[save] checkpoint -> {path}")


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

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

    teacher = build_model(
        args.model_type,
        args.seq_len,
        args.teacher_ckpt,
        crop_size=tuple(args.crop_size),
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("[teacher] loaded and frozen")

    student = build_model(
        args.model_type,
        args.seq_len,
        args.student_ckpt,
        crop_size=tuple(args.crop_size),
    ).to(device)
    student.train()
    print("[student] ready")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=args.num_steps,
        pct_start=0.05,
    )

    step = 0
    data_iter = iter(train_loader)
    print(f"start training for {args.num_steps} steps")

    while step < args.num_steps:
        try:
            batch, gotit = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch, gotit = next(data_iter)

        if not all(gotit):
            continue

        video = batch.video.to(device)  # [B, T, 3, H, W], 0~255
        B, _, _, H, W = video.shape

        gt_trajs, gt_vis = generate_pseudo_labels(
            teacher, video, args.traj_per_sample, device
        )

        queries_xy = gt_trajs[:, 0]
        queries_t = torch.zeros(B, queries_xy.shape[1], 1, device=device)
        queries = torch.cat([queries_t, queries_xy], dim=-1)

        optimizer.zero_grad(set_to_none=True)
        pred_trajs, pred_vis, pred_conf, _ = unpack_model_output(
            student(video=preprocess_video(video), queries=queries, iters=6)
        )
        if pred_conf is not None:
            pred_vis = pred_vis * pred_conf

        loss, traj_loss, vis_loss = compute_loss(pred_trajs, pred_vis, gt_trajs, gt_vis)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        step += 1

        if step % 10 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"step {step:6d}/{args.num_steps} | "
                f"loss={loss.item():.4f} | "
                f"traj={traj_loss.item():.4f} | "
                f"vis={vis_loss.item():.4f} | "
                f"lr={lr_now:.6f}"
            )

        if step % args.save_freq == 0:
            ckpt_path = os.path.join(args.ckpt_dir, f"student_step{step:06d}.pth")
            save_checkpoint(student, optimizer, step, ckpt_path)

    final_path = os.path.join(args.ckpt_dir, "student_final.pth")
    save_checkpoint(student, optimizer, step, final_path)
    print("training done")


if __name__ == "__main__":
    main()
