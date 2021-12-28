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
from miscnn import Data_IO
from miscnn.data_loading.interfaces import NIFTI_interface
from miscnn.processing.subfunctions import Resampling, Normalization
from PIL import Image

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# Data directory
path_data = "data"
# Name of dataset
ds_name = "covid"

#-----------------------------------------------------#
#                   Run Preparation                   #
#-----------------------------------------------------#
# Create solution directory
path_ds = os.path.join(path_data, ds_name)
if not os.path.exists(path_ds + ".prepared"):
    os.mkdir(path_ds + ".prepared")

# Intialize MIScnn Data IO
interface = NIFTI_interface(channels=1, classes=4)
data_io = Data_IO(interface, path_ds, delete_batchDir=True)

# Iterate over each sample
for index in tqdm(os.listdir(path_ds)):
    # Sample loading
    sample = data_io.sample_loader(index, load_seg=True)
    # Resample and normalize ct scan
    sf_resample = Resampling((1.58, 1.58, 2.70))
    sf_resample.preprocessing(sample, training=True)
    sf_normalize = Normalization(mode="grayscale")
    sf_normalize.preprocessing(sample, training=True)
    # Iterate over slices
    for slice in range(0, sample.img_data.shape[2]):
        index_slice = index + "." + str(slice)
        # Obtain slice
        img_slice = np.take(sample.img_data, slice, axis=2)
        seg_slice = np.take(sample.seg_data, slice, axis=2)
        # Combine lung left and lung right annotation
        seg_slice[seg_slice == 2] = 1
        seg_slice[seg_slice == 3] = 2
        # Store slice on disk
        path_slice = os.path.join(path_ds + ".prepared", index_slice)
        if not os.path.exists(path_slice) : os.mkdir(path_slice)
        pillow_img = Image.fromarray(img_slice[:,:,0].astype(np.uint8))
        pillow_img.save(os.path.join(path_slice, "imaging.png"))
        pillow_seg = Image.fromarray(seg_slice[:,:,0].astype(np.uint8))
        pillow_seg.save(os.path.join(path_slice, "segmentation.png"))
