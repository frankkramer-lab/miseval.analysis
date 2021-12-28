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
import os
from sklearn.model_selection import train_test_split
from miscnn.data_loading.interfaces import Image_interface
from miscnn import Data_IO, Preprocessor, Data_Augmentation, Neural_Network
from miscnn.processing.subfunctions import Normalization, Resize, Padding
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.neural_network.metrics import tversky_crossentropy, dice_soft, focal_tversky_loss
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, \
                                       CSVLogger, ModelCheckpoint

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

# Data directory
path_data = "data/braintumor.prepared"

# model directory
path_models = "models/braintumor"
if not os.path.exists("models") : os.mkdir("models")
if not os.path.exists(path_models) : os.mkdir(path_models)

# prediction directory
path_results = "results/braintumor"
if not os.path.exists("results") : os.mkdir("results")
if not os.path.exists(path_results) : os.mkdir(path_results)

#-----------------------------------------------------#
#                       Sampling                      #
#-----------------------------------------------------#
# Initialize Data IO Interface for data
## We are using 2 classes due to [background, cancer]
interface = Image_interface(img_type="grayscale", classes=2)

# Create Data IO object to load and write samples in the file structure
data_io = Data_IO(interface, input_path=path_data, delete_batchDir=False)

# Access all available samples in our file structure
sample_list = data_io.get_indiceslist()
sample_list.sort()

# Split dataset in train/test
train, test = train_test_split(sample_list, train_size=0.8, test_size=0.2, shuffle=True,
                               random_state=0)

#-----------------------------------------------------#
#               Setup of MIScnn Pipeline              #
#-----------------------------------------------------#
# Create and configure the Data Augmentation class
data_aug = Data_Augmentation(cycles=1, scaling=True, rotations=True,
                             elastic_deform=True, mirror=True,
                             brightness=True, contrast=True, gamma=True,
                             gaussian_noise=True)

# Create a resize Subfunction to resolution 512x512
sf_resize = Resize((512, 512))
# Create a pixel value normalization Subfunction for z-score scaling
sf_zscore = Normalization(mode="z-score")

# Assemble Subfunction classes into a list
sf = [sf_resize, sf_zscore]

# Create and configure the Preprocessor class
pp = Preprocessor(data_io, data_aug=data_aug, batch_size=24, subfunctions=sf,
                  prepare_subfunctions=True, prepare_batches=False,
                  analysis="fullimage")

# Initialize the Architecture
unet_standard = Architecture(depth=4, activation="softmax",
                             batch_normalization=True)

# Create the Neural Network model
model = Neural_Network(preprocessor=pp, architecture=unet_standard,
                       loss=focal_tversky_loss(),
                       metrics=[dice_soft, tversky_crossentropy],
                       learning_rate=0.001, batch_queue_size=3, workers=3)

# Define Callbacks
cb_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=8,
                          verbose=1, mode='min', min_delta=0.0001, cooldown=1,
                          min_lr=0.00001)
cb_es = EarlyStopping(monitor="loss", patience=30)
cb_cl = CSVLogger(os.path.join(path_models, "logs.csv"), separator=',',
                  append=True)
cb_mc = ModelCheckpoint(os.path.join(path_models, "model.{epoch}.hdf5"),
                        monitor="loss", verbose=1,
                        save_best_only=False, mode="min")
callback_list = [cb_lr, cb_es, cb_cl, cb_mc]

#-----------------------------------------------------#
#                Run Training Pipeline                #
#-----------------------------------------------------#
# Run Training
model.train(train, epochs=1000, iterations=75,
            callbacks=[cb_lr, cb_es, cb_cl, cb_mc])

# Dump latest model to disk
model.dump(os.path.join(path_models, "model.latest.hdf5"))

#-----------------------------------------------------#
#                Run Inference Pipeline               #
#-----------------------------------------------------#
# Compute predictions for first model
model.preprocessor.data_io = Data_IO(interface, input_path=path_data,
                                     delete_batchDir=False,
                                     output_path=os.path.join(path_results,
                                                              "first"))
model.load(os.path.join(path_models, "model.1.hdf5"))
model.predict(test, return_output=False)

# Compute predictions for final model
model.preprocessor.data_io = Data_IO(interface, input_path=path_data,
                                     delete_batchDir=False,
                                     output_path=os.path.join(path_results,
                                                              "final"))
model.load(os.path.join(path_models, "model.latest.hdf5"))
model.predict(test, return_output=False)
