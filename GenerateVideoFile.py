import os
import cv2
import numpy as np
from astropy.io import fits

def read_fits(file_path):
    with fits.open(file_path) as hdul:
        data = hdul[0].data.astype(np.float32)
        data -= data.min()
        if data.max() > 0:
            data /= data.max()
        return (data * 255).astype(np.uint8)

def fits_to_video(fits_folder, output_video, fps=10):
    fits_files = sorted([f for f in os.listdir(fits_folder) if f.endswith(".fits")])
    if not fits_files:
        print("未找到FITS文件。")
        return

    first_frame = read_fits(os.path.join(fits_folder, fits_files[0]))
    height, width = first_frame.shape
    video_writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*'avc1'),  # 或 'mp4v'，根据你的系统兼容性
        fps,
        (width, height),
        isColor=True
    )

    for fname in fits_files[1:100]:
        frame = read_fits(os.path.join(fits_folder, fname))
        frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video_writer.write(frame_color)
        print(f"写入帧：{fname}")

    video_writer.release()
    print(f"视频保存成功：{output_video}")

if __name__ == "__main__":
    fits_to_video("/Volumes/Others/观测数据/20200516_Ha_TiO/20200516/HA/Disk_Center/011736/B060",
                  "/Volumes/Others/观测数据/20200516_Ha_TiO/20200516/HA/Disk_Center/011736/B060/Unregistered_video.mp4", fps=10)
