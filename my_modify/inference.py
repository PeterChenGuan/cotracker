import torch
import cv2
import numpy as np
import glob
import os

from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline


# ── 配置 ──────────────────────────────────────────────────────
CKPT_PATH   = "./outputs/run1/student_final.pth"
IMAGE_DIR   = "./test_seq"        # 一段测试视频的图片帧目录
OUTPUT_VID  = "./result.mp4"      # 输出可视化视频
SEQ_LEN     = 24
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# 查询点：在第0帧上手动指定，格式 (x, y)
QUERY_POINTS = [
    (100, 150),
    (200, 250),
    (300, 180),
    (400, 300),
]


# ── 加载模型 ──────────────────────────────────────────────────
def load_model(ckpt_path):
    model = CoTrackerThreeOffline(
        stride=4,
        corr_radius=3,
        window_len=SEQ_LEN,
        model_resolution=(384, 512),
        linear_layer_for_vis_conf=True,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(DEVICE)
    print(f"[load_model] loaded from {ckpt_path}")
    return model


# ── 读取图片帧 ────────────────────────────────────────────────
def load_frames(image_dir, seq_len):
    paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.png"))
    )
    paths = paths[:seq_len]

    frames = []
    for p in paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)

    return frames   # list of [H, W, 3] uint8


# ── 预处理 ────────────────────────────────────────────────────
def preprocess(frames, crop_size=(384, 512)):
    H, W = crop_size
    processed = []
    for f in frames:
        f = cv2.resize(f, (W, H))
        processed.append(f)

    video = np.stack(processed, axis=0)               # [T, H, W, 3]
    video = torch.from_numpy(video).permute(0, 3, 1, 2).float()  # [T, 3, H, W]
    # Keep 0~255 values; CoTrackerThree normalizes internally.
    video = video.unsqueeze(0)                         # [1, T, 3, H, W]
    return video


# ── 推理 ──────────────────────────────────────────────────────
def run_inference(model, video, query_points):
    """
    video:         [1, T, 3, H, W]
    query_points:  list of (x, y)
    """
    N = len(query_points)
    device = next(model.parameters()).device

    # 构造 queries: [1, N, 3]  格式 (帧号t, x, y)
    queries = []
    for (x, y) in query_points:
        queries.append([0, float(x), float(y)])
    queries = torch.tensor(queries, dtype=torch.float32)  # [N, 3]
    queries = queries.unsqueeze(0).to(device)              # [1, N, 3]

    video = video.to(device)

    with torch.no_grad():
        output = model(
            video=video,
            queries=queries,
            iters=6,
        )
        if len(output) == 4:
            pred_trajs, pred_vis, pred_conf, _ = output
        elif len(output) == 3:
            pred_trajs, pred_vis, _ = output
            pred_conf = None
        else:
            raise ValueError(f"Unexpected model output length: {len(output)}")
        if pred_conf is not None:
            pred_vis = pred_vis * pred_conf

    # pred_trajs: [1, T, N, 2]
    # pred_vis:   [1, T, N]
    trajs = pred_trajs[0].cpu().numpy()   # [T, N, 2]
    vis   = pred_vis[0].cpu().numpy()     # [T, N]

    return trajs, vis


# ── 可视化 ────────────────────────────────────────────────────
COLORS = [
    (255,  0,  0),
    (  0,255,  0),
    (  0,  0,255),
    (255,255,  0),
    (255,  0,255),
    (  0,255,255),
    (255,128,  0),
    (128,  0,255),
]

def visualize(frames, trajs, vis, output_path, crop_size=(384, 512)):
    """
    frames: list of [H, W, 3] uint8 原始尺寸
    trajs:  [T, N, 2]
    vis:    [T, N]
    """
    H_crop, W_crop = crop_size
    T, N, _ = trajs.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 10, (W_crop, H_crop))

    # 存历史轨迹用于画尾迹
    history = [[] for _ in range(N)]

    for t in range(min(T, len(frames))):
        frame = cv2.resize(frames[t], (W_crop, H_crop))
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for n in range(N):
            x, y = trajs[t, n]
            v = vis[t, n]
            color = COLORS[n % len(COLORS)]

            # 只画可见的点
            if v > 0.5:
                cx, cy = int(round(x)), int(round(y))
                history[n].append((cx, cy))

                # 画尾迹
                for i in range(1, len(history[n])):
                    alpha = i / len(history[n])
                    c = tuple(int(c * alpha) for c in color)
                    cv2.line(
                        frame_bgr,
                        history[n][i-1],
                        history[n][i],
                        c, 1, cv2.LINE_AA
                    )

                # 画当前点
                cv2.circle(frame_bgr, (cx, cy), 4, color, -1)
                cv2.circle(frame_bgr, (cx, cy), 5, (255,255,255), 1)

                # 标序号
                cv2.putText(
                    frame_bgr, str(n),
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )

        out.write(frame_bgr)

    out.release()
    print(f"[visualize] saved -> {output_path}")


# ── 主流程 ────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. 加载模型
    model = load_model(CKPT_PATH)

    # 2. 读取测试帧
    frames = load_frames(IMAGE_DIR, SEQ_LEN)
    print(f"loaded {len(frames)} frames, size={frames[0].shape[:2]}")

    # 3. 预处理
    video = preprocess(frames, crop_size=(384, 512))
    print(f"video tensor: {video.shape}")

    # 4. 推理
    trajs, vis = run_inference(model, video, QUERY_POINTS)
    print(f"trajs: {trajs.shape}")   # [T, N, 2]
    print(f"vis:   {vis.shape}")     # [T, N]

    # 5. 打印每帧坐标
    print("\n── 每帧坐标 ──────────────────────")
    for t in range(trajs.shape[0]):
        for n in range(trajs.shape[1]):
            x, y = trajs[t, n]
            v = vis[t, n]
            print(f"  frame {t:3d} | point {n} | x={x:.1f} y={y:.1f} | visible={v:.2f}")

    # 6. 可视化保存视频
    visualize(frames, trajs, vis, OUTPUT_VID, crop_size=(384, 512))