#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
from tqdm import tqdm
import os
import numpy as np
import shutil
from PIL import Image

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# Data directory
path_data = "data"
# Name of dataset
ds_name = "ultrasound"

#-----------------------------------------------------#
#                   Run Preparation                   #
#-----------------------------------------------------#
# Create solution directory
path_ds = os.path.join(path_data, ds_name)
if not os.path.exists(path_ds + ".prepared"):
    os.mkdir(path_ds + ".prepared")

# Iterate over each sample
for class_dir in ["benign", "malignant", "normal"]:
    file_list = os.listdir(os.path.join(path_ds, class_dir))
    for file in tqdm(file_list):
        if "mask" in file : continue
        index = file.split(".")[0]

        print(file, index)

        # Create sample dir in solution
        path_sample = os.path.join(path_ds + ".prepared", index)
        if not os.path.exists(path_sample) : os.mkdir(path_sample)
        # Copy image
        img_pil = Image.open(os.path.join(path_ds, class_dir, file))
        img_pil.save(os.path.join(path_sample, "imaging.png"))
        # result list
        seg_res = []
        # Load mask
        seg_raw = Image.open(os.path.join(path_ds, class_dir, index + "_mask.png"))
        # Prepare mask
        seg_pil = seg_raw.convert("LA")
        seg = np.array(seg_pil)
        seg_data = seg[:,:,0] / 255
        # Append to result list
        seg_res.append(seg_data)
        # Check for additional masks
        i = 1
        while True:
            next_file = index + "_mask_" + str(i) + ".png"
            if os.path.exists(os.path.join(path_ds, class_dir, next_file)):
                # Load mask
                seg_raw = Image.open(os.path.join(path_ds, class_dir, next_file))
                # Prepare mask
                seg_pil = seg_raw.convert("LA")
                seg = np.array(seg_pil)
                seg_data = seg[:,:,0] / 255
                # Append to result list
                seg_res.append(seg_data)

                i += 1
            else : break
        # Merge masks together
        mask = np.stack(seg_res, axis=0)
        mask = np.sum(mask, axis=0)
        # Store mask
        pillow_seg = Image.fromarray(mask.astype(np.uint8))
        pillow_seg.save(os.path.join(path_sample, "segmentation.png"))
