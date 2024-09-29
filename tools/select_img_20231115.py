import os
import cv2
import numpy as np
import pandas as pd

def calculate_mse(img1, img2):
    return np.sum((img1.astype("float") - img2.astype("float")) ** 2) / float(img1.shape[0] * img1.shape[1])

def mse_between_directories(directory1, directory2, output_excel_path):
    image_files1 = os.listdir(directory1)
    image_files2 = os.listdir(directory2)

    if len(image_files1) != len(image_files2):
        print("Error: The number of images in the two directories is not the same.")
        return

    mse_data = {'Filename': [], 'MSE': []}

    for file1, file2 in zip(image_files1, image_files2):
        
        img1 = cv2.imread(os.path.join(directory1, file1))
        img2 = cv2.imread(os.path.join(directory2, file2))

        if img1.shape != img2.shape:
            print(f"Error: Images {file1} and {file2} have different dimensions.")
            continue

        mse_value = calculate_mse(img1, img2)
        mse_data['Filename'].append(file1)
        mse_data['MSE'].append(mse_value)
        print("file:",file1, " MSE:",mse_value)

    df = pd.DataFrame(mse_data)
    df.to_excel(output_excel_path, index=False)
    print(f"MSE values saved to {output_excel_path}")

# Replace these paths with the actual paths to your image directories
directory_path1 = '/cluster/home3/wjs/ARC_2/work_dirs/20231108/arc_orcnn_r101fpn1x_ss_dota10_RxFFF_g16_0.3_2/vis_arc_noscore'
directory_path2 = '/cluster/home3/wjs/ARC_2/work_dirs/20231108/arc_orcnn_r101fpn1x_ss_dota10_RxFFF_g16_0.3_2/vis_noscore'

output_excel_path = '/cluster/home3/wjs/ARC_2/work_dirs/20231108/arc_orcnn_r101fpn1x_ss_dota10_RxFFF_g16_0.3_2/mse_results.xlsx'

mse_between_directories(directory_path1, directory_path2, output_excel_path)
