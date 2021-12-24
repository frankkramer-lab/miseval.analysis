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
ds_name = "braintumor"

#-----------------------------------------------------#
#                   Run Preparation                   #
#-----------------------------------------------------#
# Create solution directory
path_ds = os.path.join(path_data, ds_name)
if not os.path.exists(path_ds + ".prepared"):
    os.mkdir(path_ds + ".prepared")

# Identify directories
path_dir_img = os.path.join(path_ds, "images")
path_dir_seg = os.path.join(path_ds, "masks")

# Iterate over each sample
for index in tqdm(os.listdir(path_dir_img)):
    # Create sample dir in solution
    path_sample = os.path.join(path_ds + ".prepared", index[:-4])
    if not os.path.exists(path_sample) : os.mkdir(path_sample)
    # Copy image
    shutil.copy(os.path.join(path_dir_img, index),
                os.path.join(path_sample, "imaging.png"))
    # Access segmentation mask
    seg_raw = Image.open(os.path.join(path_dir_seg, index))
    # Prepare mask
    seg_pil = seg_raw.convert("LA")
    seg = np.array(seg_pil)
    seg_data = seg[:,:,0] / 255
    # Store segmentation on disk
    pillow_seg = Image.fromarray(seg_data.astype(np.uint8))
    pillow_seg.save(os.path.join(path_sample, "segmentation.png"))
