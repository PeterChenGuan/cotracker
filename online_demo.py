# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np

from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor


# DEFAULT_DEVICE = (
#     "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# )
DEFAULT_DEVICE = 'cpu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="./assets/apple.mp4",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        type=float,
        default=[0, 400, 350, 0, 400, 300, 0, 450, 350, 0, 450, 300],
        help=(
            "Query points as a flat list: frame x y frame x y ...\n"
            "Example: --queries 0 400 350 10 400 500 20 750 600 30 900 200"
        ),
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    # ── Build queries tensor  (N, 3)  →  [frame, x, y] ──────────────────────
    raw = args.queries
    if len(raw) % 3 != 0:
        raise ValueError("--queries must be a multiple of 3 values: frame x y ...")
    queries = torch.tensor(raw, dtype=torch.float32).reshape(-1, 3)  # (N, 3)
    queries = queries.to(DEFAULT_DEVICE)

    # ── Load model ────────────────────────────────────────────────────────────
    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)

    window_frames = []

    def _process_step(window_frames, is_first_step, queries):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries[None],   # (1, N, 3)
        )

    # ── Iterate over video frames, one window at a time ───────────────────────
    is_first_step = True
    for i, frame in enumerate(
        iio.imiter(args.video_path, plugin="FFMPEG")
    ):
        if i % model.step == 0 and i != 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                queries=queries,
            )
            is_first_step = False
        window_frames.append(frame)

    # ── Process the final (possibly partial) window ───────────────────────────
    pred_tracks, pred_visibility = _process_step(
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        queries=queries,
    )

    print("Tracks are computed")

    # ── Save visualisation ────────────────────────────────────────────────────
    seq_name = args.video_path.split("/")[-1]
    video = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
        0, 3, 1, 2
    )[None]  # (1, T, 3, H, W)

    # Use the earliest query frame as the visualisation anchor
    query_frame = int(queries[:, 0].min().item())

    vis = Visualizer(
        save_dir="./saved_videos",
        pad_value=120,
        linewidth=3,
        tracks_leave_trace=-1,
    )
    vis.visualize(
        video,
        pred_tracks,
        pred_visibility,
        query_frame=query_frame,
        filename="apple+3_queries",
    )