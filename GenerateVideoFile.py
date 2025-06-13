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
        print(f"The FITS file was not found: {fits_files}")
        return

    first_frame = read_fits(os.path.join(fits_folder, fits_files[0]))
    height, width = first_frame.shape
    video_writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height), isColor=True)

    for fname in fits_files[1:]:
        frame = read_fits(os.path.join(fits_folder, fname))
        frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video_writer.write(frame_color)
        print(f"Write frame：{fname}")

    video_writer.release()
    print(f"The video was saved successfully ：{output_video}")

    # Modify file path
if __name__ == "__main__":
    fits_to_video("/Volumes/Others/B060",
                  "/Volumes/Others/B060/output_video.mp4", fps=10)
