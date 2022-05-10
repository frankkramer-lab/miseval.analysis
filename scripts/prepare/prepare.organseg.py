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

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# Data directory
path_data = "data"
# Name of dataset
ds_name = "organseg"

#-----------------------------------------------------#
#                   Run Preparation                   #
#-----------------------------------------------------#
# Create solution directory
path_ds = os.path.join(path_data, ds_name)
if not os.path.exists(path_ds + ".prepared"):
    os.mkdir(path_ds + ".prepared")

# Iterate over each sample
for file in tqdm(os.listdir(path_ds)):
    if not file.startswith("volume") : continue
    index = file.split(".")[0].split("-")[1]
    # Create sample dir in solution
    path_sample = os.path.join(path_ds + ".prepared", "sample_" + index)
    if not os.path.exists(path_sample) : os.mkdir(path_sample)
    # Copy image
    shutil.copy(os.path.join(path_ds, file),
                os.path.join(path_sample, "imaging.nii.gz"))
    shutil.copy(os.path.join(path_ds, "labels-" + index + ".nii.gz"),
                os.path.join(path_sample, "segmentation.nii.gz"))
