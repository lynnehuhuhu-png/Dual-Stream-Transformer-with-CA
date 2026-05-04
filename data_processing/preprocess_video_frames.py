"""
预处理视频帧（一次性）
将所有视频的15帧提取并保存，避免训练时重复读取
"""

import os
import glob
import cv2
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def load_video_frames(video_path, max_frames=15):
    """加载视频的15帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 均匀采样
    if total_frames < max_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize + CenterCrop
            frame = cv2.resize(frame, (256, 256))
            h, w = frame.shape[:2]
            start_h, start_w = (h - 224) // 2, (w - 224) // 2
            frame = frame[start_h:start_h+224, start_w:start_w+224]
            frames.append(frame)

    cap.release()

    # 填充不足的帧
    while len(frames) < max_frames:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

    return np.array(frames[:max_frames])  # (15, 224, 224, 3)


def main():
    print("=" * 80)
    print("🚀 预处理视频帧（一次性）")
    print("=" * 80)

    VIDEO_DIR = "D:/paper/SARID-main/SARID/video-new"
    OUTPUT_FILE = "datatest1/video_frames_preprocessed.npy"

    video_files = sorted(glob.glob(os.path.join(VIDEO_DIR, '*.mp4')))

    print(f"视频数量: {len(video_files)}")
    print(f"输出文件: {OUTPUT_FILE}")
    print("-" * 80)

    all_frames = []
    failed_videos = []

    for video_path in tqdm(video_files, desc="提取帧"):
        try:
            frames = load_video_frames(video_path, max_frames=15)
            all_frames.append(frames)
        except Exception as e:
            print(f"\n❌ 失败: {os.path.basename(video_path)} - {str(e)}")
            failed_videos.append(os.path.basename(video_path))
            # 填充零数组
            all_frames.append(np.zeros((15, 224, 224, 3), dtype=np.uint8))

    # 保存
    all_frames = np.array(all_frames, dtype=np.uint8)  # (N, 15, 224, 224, 3)
    np.save(OUTPUT_FILE, all_frames)

    print(f"\n{'='*80}")
    print(f"✅ 预处理完成!")
    print(f"{'='*80}")
    print(f"帧数组形状: {all_frames.shape}")
    print(f"已保存到: {OUTPUT_FILE}")
    print(f"失败视频: {len(failed_videos)}个")
    print(f"{'='*80}")

    if failed_videos:
        with open("datatest1/failed_videos_preprocess.txt", "w") as f:
            f.write("\n".join(failed_videos))
        print("失败视频列表已保存: datatest1/failed_videos_preprocess.txt")


if __name__ == "__main__":
    os.makedirs("datatest1", exist_ok=True)
    main()
