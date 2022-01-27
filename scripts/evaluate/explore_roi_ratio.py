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
from miscnn.data_loading.interfaces import Image_interface
from miscnn import Data_IO
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# Data directory
path_data = "data"

# Datasets
# datasets = ["braintumor.prepared", "covid.prepared", "histopathology.prepared"]
datasets = ["braintumor.prepared", "histopathology.prepared"]

# Evaluation directory
path_evaluation = "evaluation"

#-----------------------------------------------------#
#                   Data Exploration                  #
#-----------------------------------------------------#
# Iterate over each dataset
for ds in datasets:
    # Initialize Data IO Interface for NIfTI data
    interface = Image_interface(img_type="grayscale", classes=2)

    # Create Data IO object to load and write samples in the file structure
    path_ds = os.path.join(path_data, ds)
    data_io = Data_IO(interface, path_ds, delete_batchDir=True)

    # Access all available samples in our file structure
    sample_list = data_io.get_indiceslist()
    sample_list.sort()

    # Now let's load each sample and obtain collect diverse information from them
    sample_data = {}
    for index in tqdm(sample_list):
        # Sample loading
        sample = data_io.sample_loader(index, load_seg=True)
        # Identify and store class distribution
        unique_data, unique_counts = np.unique(sample.seg_data, return_counts=True)
        class_freq = unique_counts / np.sum(unique_counts)
        class_freq = np.around(class_freq, decimals=6)
        sample_data[index] = tuple(class_freq)

    # Transform collected data into a pandas dataframe
    df = pd.DataFrame.from_dict(sample_data, orient="index",
                                columns=["class_0", "class_1"])

    # Print out the dataframe to console
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

    # Store to csv
    path_out = os.path.join(path_evaluation, ds.split(".")[0], "class_ratios.csv")
    df.index.name = "index"
    df.to_csv(path_out, index=True)
