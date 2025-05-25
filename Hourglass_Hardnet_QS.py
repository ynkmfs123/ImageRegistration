import cv2
import matplotlib.pyplot as plt
import numpy as np
from modules.registration import  REG as REG
import torch
from astropy.io import fits
import time
import os

from pathlib import Path


def convert_to_float32(img):
    # 对于大于float32数据类型的数据opencv运行时，偶尔会出现错误
    #将大于float32数据类型的图像转换为float32类型，其他类型保持不变
    # 如果是浮点类型
    if np.issubdtype(img.dtype, np.floating):
        # 如果是非 native-endian，先转为 native-endian
        if img.dtype.byteorder == '>' or (img.dtype.byteorder == '=' and np.little_endian is False):
            img = img.byteswap().newbyteorder()
        # 如果类型精度高于 float32，转换为 float32
        if img.dtype.itemsize > 4:
            img = img.astype(np.float32)
    return img

def AlignSeriesImage(fixed_data, moving_data):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reg = REG(dev = device)
    # 记录开始时间
    start_time = time.time()
    keypoints1, descriptors1 = reg.detectAndCompute(moving_data)
    keypoints2, descriptors2 = reg.detectAndCompute(fixed_data)

    matcher = cv2.BFMatcher(crossCheck = True)#创建一个Brute-Force匹配器，要求仅保留互相匹配的特征点对
    good_matches = matcher.match(descriptors1, descriptors2)#在两组特征描述符之间进行匹配，找到它们之间的最佳匹配点
    # good_matches = matches

    # 获取匹配点的坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    rigid_transform_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)


    # size = fixed_data.shape
    height, width = fixed_data.shape[:2]

    # moving_data = moving_data.astype(np.float32)

    aligned_image = cv2.warpAffine(moving_data, rigid_transform_matrix,(width, height),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


    print('totally registration time: ', time.time() - start_time, "s")

    return aligned_image

if __name__ == "__main__":

    # 设置文件夹路径
    input_folder = '/Volumes/Others/观测数据/20200516_Ha_TiO/20200516/HA/Disk_Center/011736/B060/'       # 源图像文件夹
    output_folder = os.path.join(input_folder, 'gre/') # 配准后图像输出文件夹

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取文件夹中所有fits文件
    fits_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.fits')])

    first_file_name = os.path.join(input_folder, fits_files[0])

    fixed_image = fits.open(first_file_name)
    fixed_data = fixed_image[0].data
    # fixed_data = fixed_data[150:2000,150:2300]
    fixed_head = fixed_image[0].header

    fixed_data = convert_to_float32(fixed_data)

    reference_file_name = Path(fits_files[0]).stem +'_gre' + Path(fits_files[0]).suffix

    # 将配准后的数据保存为FITS文件
    fixed_file_name = os.path.join(output_folder, reference_file_name)
    hdu = fits.PrimaryHDU(fixed_data, header=fixed_head)
    hdu.writeto(fixed_file_name, overwrite=True)

    print(f"Image saved to {reference_file_name}")

    for fits_file in fits_files[1:100]:
        # 读取fits图像
        input_path = os.path.join(input_folder, fits_file)
        with fits.open(input_path) as hdul:
            head = hdul[0].header
            moving_data = hdul[0].data  # 假设数据在第一个HDU中
            # moving_data = moving_data[150:2000,150:2300]
            moving_data = convert_to_float32(moving_data)

        aligned_data = AlignSeriesImage(fixed_data, moving_data)

        # diff_img1 = fixed_data - moving_data
        # diff_img2 = fixed_data - aligned_data
        #
        # fig, axes = plt.subplots(1, 2)  # 1行2列
        # axes[0].imshow(diff_img1, cmap = 'gray')
        # axes[0].axis('off')
        # axes[1].imshow(diff_img2, cmap = 'gray')
        # axes[1].axis('off')
        # plt.show()

        # 保存配准后的图像到输出文件夹
        output_file = Path(fits_file).stem +'_gre' + Path(fits_file).suffix
        output_path = os.path.join(output_folder, output_file)
        # 将配准后的数据保存为FITS文件
        hdu = fits.PrimaryHDU(aligned_data,header=head)
        hdu.writeto(output_path, overwrite=True)

        print(f"Image saved to {output_file}")

        fixed_data = aligned_data




