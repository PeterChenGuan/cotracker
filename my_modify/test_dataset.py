from my_modify.my_real_dataset import MyRealDataset

ds = MyRealDataset(
    data_root="./my_data/my_tracking_frames",
    crop_size=(384, 512),
    seq_len=24,
    traj_per_sample=256,
)

sample, ok = ds[0]
print(ok)
print(sample.video.shape)       # [24, 3, 384, 512]
print(sample.trajectory.shape)  # [24, 256, 2]
print(sample.visibility.shape)  # [24, 256]
print(sample.valid.shape)       # [24, 256]
print(sample.seq_name)