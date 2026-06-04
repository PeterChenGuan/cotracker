import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from cotracker.datasets.utils import CoTrackerData


class MyRealDataset(Dataset):
    def __init__(
        self,
        data_root,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        random_frame_rate=False,
        random_seq_len=False,
        random_resize=False,
        limit_samples=10000,
        image_exts=("jpg", "jpeg", "png", "bmp"),
    ):
        super().__init__()
        np.random.seed(0)
        torch.manual_seed(0)

        self.data_root = data_root
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.traj_per_sample = traj_per_sample
        self.random_frame_rate = random_frame_rate
        self.random_seq_len = random_seq_len
        self.random_resize = random_resize
        self.image_exts = image_exts

        self.sequences = self._collect_sequences(data_root)
        self.sequences = self.sequences[:limit_samples]

        print(f"[MyRealDataset] found {len(self.sequences)} sequences in {data_root}")
        if len(self.sequences) == 0:
            raise ValueError(f"No valid image sequences found in {data_root}")

    def _collect_sequences(self, data_root):
        sequences = []

        if not os.path.isdir(data_root):
            raise ValueError(f"data_root does not exist: {data_root}")

        seq_dirs = sorted(
            [
                os.path.join(data_root, d)
                for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d))
            ]
        )

        for seq_dir in seq_dirs:
            frame_paths = []
            for ext in self.image_exts:
                frame_paths.extend(glob.glob(os.path.join(seq_dir, f"*.{ext}")))
                frame_paths.extend(glob.glob(os.path.join(seq_dir, f"*.{ext.upper()}")))

            frame_paths = sorted(frame_paths)

            if len(frame_paths) >= 2:
                sequences.append(
                    {
                        "seq_name": os.path.basename(seq_dir),
                        "seq_dir": seq_dir,
                        "frame_paths": frame_paths,
                    }
                )

        return sequences

    def __len__(self):
        return len(self.sequences)

    def crop(self, video):
        # video: [T, C, H, W]
        T, C, H, W = video.shape
        crop_h, crop_w = self.crop_size

        y0 = 0 if crop_h >= H else np.random.randint(0, H - crop_h + 1)
        x0 = 0 if crop_w >= W else np.random.randint(0, W - crop_w + 1)

        video = video[:, :, y0:y0 + crop_h, x0:x0 + crop_w]
        return video

    def resize_video(self, video):
        # video: [T, C, H, W]
        T, C, H, W = video.shape
        crop_h, crop_w = self.crop_size
        out = []

        for i in range(T):
            frame = video[i].permute(1, 2, 0).cpu().numpy()  # HWC
            frame = cv2.resize(frame, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
            frame = torch.from_numpy(frame).permute(2, 0, 1)  # CHW
            out.append(frame)

        return torch.stack(out, dim=0)

    def _read_frame(self, frame_path):
        img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
        return img

    def _load_frames(self, frame_paths):
        frames = []
        for p in frame_paths:
            frame = self._read_frame(p)
            if frame is None:
                return None
            frames.append(frame)
        return torch.stack(frames, dim=0)  # [T, C, H, W]

    def _sample_clip_paths(self, all_frame_paths):
        L = len(all_frame_paths)

        if self.random_seq_len:
            seq_len = np.random.randint(max(8, self.seq_len // 2), self.seq_len + 1)
        else:
            seq_len = self.seq_len

        if seq_len < 8:
            return None

        paths = list(all_frame_paths)

        # 太短就镜像补长
        while len(paths) < seq_len:
            paths = paths + paths[::-1]

        frame_rate = 1
        if self.random_frame_rate:
            max_frame_rate = min(4, max(1, len(paths) // seq_len))
            if max_frame_rate > 1:
                frame_rate = np.random.randint(1, max_frame_rate + 1)

        needed = seq_len * frame_rate
        if needed < len(paths):
            start = np.random.randint(0, len(paths) - needed + 1)
        else:
            start = 0

        clip_paths = paths[start:start + needed:frame_rate]

        if len(clip_paths) < seq_len:
            while len(clip_paths) < seq_len:
                clip_paths = clip_paths + clip_paths[::-1]
            clip_paths = clip_paths[:seq_len]

        return clip_paths

    def _make_dummy_sample(self, seq_name="dummy"):
        return CoTrackerData(
            video=torch.zeros((self.seq_len, 3, self.crop_size[0], self.crop_size[1]), dtype=torch.float32),
            trajectory=torch.ones((self.seq_len, self.traj_per_sample, 2), dtype=torch.float32),
            visibility=torch.ones((self.seq_len, self.traj_per_sample), dtype=torch.float32),
            valid=torch.zeros((self.seq_len, self.traj_per_sample), dtype=torch.float32),
            seq_name=seq_name,
        )

    def __getitem__(self, index):
        seq_info = self.sequences[index]
        seq_name = seq_info["seq_name"]
        all_frame_paths = seq_info["frame_paths"]

        clip_paths = self._sample_clip_paths(all_frame_paths)
        if clip_paths is None:
            return self._make_dummy_sample(seq_name), False

        video = self._load_frames(clip_paths)
        if video is None:
            return self._make_dummy_sample(seq_name), False

        # video: [T, C, H, W], uint8-like values but tensor dtype is torch.uint8
        if self.random_resize and np.random.rand() < 0.5:
            video = self.resize_video(video)
        else:
            video = self.crop(video)
            _, _, H, W = video.shape
            if H != self.crop_size[0] or W != self.crop_size[1]:
                video = self.resize_video(video)

        video = video.float()  # 保持 0~255 数值范围，与原 RealDataset 风格一致
        T = video.shape[0]

        sample = CoTrackerData(
            video=video,
            trajectory=torch.ones((T, self.traj_per_sample, 2), dtype=torch.float32),
            visibility=torch.ones((T, self.traj_per_sample), dtype=torch.float32),
            valid=torch.ones((T, self.traj_per_sample), dtype=torch.float32),
            seq_name=seq_name,
        )
        return sample, True