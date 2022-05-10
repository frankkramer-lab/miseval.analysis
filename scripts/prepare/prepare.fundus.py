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
ds_name = "drive"

#-----------------------------------------------------#
#                   Run Preparation                   #
#-----------------------------------------------------#
# Create solution directory
path_ds = os.path.join(path_data, ds_name)
if not os.path.exists(path_ds + ".prepared"):
    os.mkdir(path_ds + ".prepared")

# Iterate over each sample
for file in tqdm(os.listdir(os.path.join(path_ds, "training", "images"))):
    index = file[:2]

    # Create sample dir in solution
    path_sample = os.path.join(path_ds + ".prepared", index)
    if not os.path.exists(path_sample) : os.mkdir(path_sample)
    # Copy image
    img_pil = Image.open(os.path.join(path_ds, "training", "images", file))
    img_pil.save(os.path.join(path_sample, "imaging.png"))
    # Load mask
    seg_raw = Image.open(os.path.join(path_ds, "training", "1st_manual", index + "_manual1.gif"))
    # Prepare mask
    seg_pil = seg_raw.convert("LA")
    seg = np.array(seg_pil)
    seg_data = seg[:,:,0] / 255
    # Store mask
    pillow_seg = Image.fromarray(seg_data.astype(np.uint8))
    pillow_seg.save(os.path.join(path_sample, "segmentation.png"))
