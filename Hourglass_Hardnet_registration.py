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
    '''
    When OpenCV processes data types larger than float32, errors may occasionally occur.
    To prevent this, convert images with data types exceeding float32 to float32,
    while leaving other data types unchanged.
    '''

    if np.issubdtype(img.dtype, np.floating):
        # If the data is non-native endian, convert it to native endian first
        if img.dtype.byteorder == '>' or (img.dtype.byteorder == '=' and np.little_endian is False):
            img = img.byteswap().newbyteorder()

        if img.dtype.itemsize > 4:
            img = img.astype(np.float32)
    return img

def AlignSeriesImage(fixed_data, moving_data):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reg = REG(dev = device)
    # record start time
    start_time = time.time()
    keypoints1, descriptors1 = reg.detectAndCompute(moving_data)
    keypoints2, descriptors2 = reg.detectAndCompute(fixed_data)

    matcher = cv2.BFMatcher(crossCheck = True)
    good_matches = matcher.match(descriptors1, descriptors2)


    # Retrieve the coordinates of matching points
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    rigid_transform_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

    height, width = fixed_data.shape[:2]

    aligned_image = cv2.warpAffine(moving_data, rigid_transform_matrix,(width, height),flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


    print('totally registration time: ', time.time() - start_time, "s")

    return aligned_image

if __name__ == "__main__":

    # set file path
    input_folder = '/Volumes/Others/观测数据/destr_libo/'
    output_folder = os.path.join(input_folder, 'gre/')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # obtain all 'fits' file
    fits_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.fts')])

    first_file_name = os.path.join(input_folder, fits_files[0])

    fixed_image = fits.open(first_file_name)
    fixed_data = fixed_image[0].data
    fixed_head = fixed_image[0].header

    fixed_data = convert_to_float32(fixed_data)

    reference_file_name = Path(fits_files[0]).stem +'_gre' + Path(fits_files[0]).suffix

    # save the reference image
    fixed_file_name = os.path.join(output_folder, reference_file_name)
    hdu = fits.PrimaryHDU(fixed_data, header=fixed_head)
    hdu.writeto(fixed_file_name, overwrite=True)

    print(f"Image saved to {reference_file_name}")

    for fits_file in fits_files[1:4]:
        input_path = os.path.join(input_folder, fits_file)
        with fits.open(input_path) as hdul:
            head = hdul[0].header
            moving_data = hdul[0].data
            moving_data = convert_to_float32(moving_data)

        aligned_data = AlignSeriesImage(fixed_data, moving_data)

        output_file = Path(fits_file).stem +'_gre' + Path(fits_file).suffix
        output_path = os.path.join(output_folder, output_file)

        hdu = fits.PrimaryHDU(aligned_data,header=head)
        hdu.writeto(output_path, overwrite=True)

        print(f"Image saved to {output_file}")

        fixed_data = aligned_data




