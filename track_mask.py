import torch
import numpy as np
from PIL import Image
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path

DEFAULT_DEVICE =  "cpu"

# ── 1. 读取视频 ────────────────────────────────────────────────────────────────
video = read_video_from_path("./assets/apple.mp4")        # (T, H, W, 3)
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(DEFAULT_DEVICE)
# shape: (1, T, 3, H, W)
T = video.shape[1]

# ── 2. 读取 Mask，整理成 model 所需形状 (1, 1, H, W) ──────────────────────────
segm_mask_np = np.array(Image.open("./assets/apple_mask.png").convert("L"))
segm_mask_np = (segm_mask_np > 128).astype(np.uint8)

# model() 需要 (B, 1, H, W)
segm_mask_model = torch.from_numpy(segm_mask_np)[None, None].to(DEFAULT_DEVICE)

# ── 3. 加载模型 ────────────────────────────────────────────────────────────────
model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
model = model.to(DEFAULT_DEVICE)

# ── 4. 推理 ───────────────────────────────────────────────────────────────────
pred_tracks, pred_visibility = model(
    video,
    grid_size=50,
    grid_query_frame=0,
    segm_mask=segm_mask_model,      # (1, 1, H, W)  ← 传给 model 用于筛选采样点
    backward_tracking=True,
)
print(f"跟踪点数: {pred_tracks.shape[2]}")

# ── 5. 准备 visualize() 所需的 segm_mask：形状必须是 (B, T, H, W) ───────────
#   visualizer 内部会做 segm_mask[0, query_frame][y, x]
#   所以需要把 mask 在时间轴上 repeat 成 T 帧
segm_mask_vis = torch.from_numpy(segm_mask_np)[None, None]          # (1, 1, H, W)
segm_mask_vis = segm_mask_vis.repeat(1, T, 1, 1).to(DEFAULT_DEVICE) # (1, T, H, W)

# ── 6. 可视化 ─────────────────────────────────────────────────────────────────
vis = Visualizer(
    save_dir="./saved_videos",
    pad_value=120,
    linewidth=2,
    tracks_leave_trace=-1,
    mode="rainbow",
)
vis.visualize(
    video,
    pred_tracks,
    pred_visibility,
    segm_mask=segm_mask_vis,    # ✅ (1, T, H, W)  ← 修复点
    query_frame=0,
    filename="apple_tracks",
)