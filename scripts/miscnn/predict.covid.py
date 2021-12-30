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

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

# Data directory
path_data = "data/covid.prepared"

# model directory
path_models = "models/covid"
if not os.path.exists("models") : os.mkdir("models")
if not os.path.exists(path_models) : os.mkdir(path_models)

# prediction directory
path_results = "results/covid"
if not os.path.exists("results") : os.mkdir("results")
if not os.path.exists(path_results) : os.mkdir(path_results)

#-----------------------------------------------------#
#             Custom Image I/O Interface              #
#-----------------------------------------------------#
""" Data I/O Interface for JPEG, PNG and other common 2D image files.
    Images are read by calling the imread function from the Pillow module.

Methods:
    __init__                Object creation function
    initialize:             Prepare the data set and create indices list
    load_image:             Load an image and associated data
    load_segmentation:      Load a segmentation
    load_prediction:        Load a prediction
    save_prediction:        Save a prediction to disk

Args:
    classes (int):          Number of classes of the segmentation
    img_type (string):      Type of imaging. Options: "grayscale", "rgb"
    img_format (string):    Imaging format: Popular formats: "png", "tif", "jpg"
    pattern (regex):        Pattern to filter samples
"""
class IIO_interface(Abstract_IO):
    #---------------------------------------------#
    #                   __init__                  #
    #---------------------------------------------#
    def __init__(self, classes=2, img_type="grayscale", img_format="png",
                 pattern=None):
        self.classes = classes
        self.img_type = img_type
        self.img_format = img_format
        self.three_dim = False
        self.pattern = pattern
        if img_type == "grayscale" : self.channels = 1
        elif img_type == "rgb" : self.channels = 3

    #---------------------------------------------#
    #                  initialize                 #
    #---------------------------------------------#
    def initialize(self, input_path):
        # Resolve location where imaging data set should be located
        if not os.path.exists(input_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(input_path))
            )
        # Cache data directory
        self.data_directory = input_path
        # Identify samples
        sample_list = os.listdir(input_path)
        # IF pattern provided: Remove every file which does not match
        if self.pattern != None and isinstance(self.pattern, str):
            for i in reversed(range(0, len(sample_list))):
                if not re.fullmatch(self.pattern, sample_list[i]):
                    del sample_list[i]
        # Return sample list
        return sample_list

    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    def load_image(self, index):
        # Make sure that the image file exists in the data set directory
        img_path = os.path.join(self.data_directory, index)
        if not os.path.exists(img_path):
            raise ValueError(
                "Sample could not be found \"{}\"".format(img_path)
            )
        # Load image from file
        img_raw = Image.open(os.path.join(img_path, "imaging" + "." + \
                                          self.img_format))
        # Convert image to rgb or grayscale if needed
        if self.img_type == "grayscale" and len(img_raw.getbands()) > 1:
            img_pil = img_raw.convert("LA")
        elif self.img_type == "rgb" and img_raw.mode != "RGB":
            img_pil = img_raw.convert("RGB")
        else:
            img_pil = img_raw
        # Convert Pillow image to numpy matrix
        img = np.array(img_pil)
        # Keep only intensity for grayscale images if needed
        if self.img_type == "grayscale" and len(img.shape) > 2:
            img = img[:, :, 0]
        # Return image
        return img, {"type": "image"}

    #---------------------------------------------#
    #              load_segmentation              #
    #---------------------------------------------#
    def load_segmentation(self, index):
        # Make sure that the segmentation file exists in the data set directory
        seg_path = os.path.join(self.data_directory, index)
        if not os.path.exists(seg_path):
            raise ValueError(
                "Segmentation could not be found \"{}\"".format(seg_path)
            )
        # Load segmentation from file
        seg_raw = Image.open(os.path.join(seg_path, "segmentation" + "." + \
                                          self.img_format))
        # Convert segmentation from Pillow image to numpy matrix
        seg_pil = seg_raw.convert("LA")
        seg = np.array(seg_pil)
        # Keep only intensity and remove maximum intensitiy range
        seg_data = seg[:,:,0]
        # Return segmentation
        return seg_data

    #---------------------------------------------#
    #               load_prediction               #
    #---------------------------------------------#
    def load_prediction(self, index, output_path):
        pass

    #---------------------------------------------#
    #               save_prediction               #
    #---------------------------------------------#
    def save_prediction(self, sample, output_path):
        # Resolve location where data should be written
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(output_path)
            )
        # Store numpy array
        pred_file = os.path.join(output_path, str(sample.index) + ".npy")
        numpy.save(pred_file, sample.pred_data.astype(np.uint8),
                   allow_pickle=True)

    #---------------------------------------------#
    #           check_file_termination            #
    #---------------------------------------------#
    @staticmethod
    def check_file_termination(termination):
        return termination in [".bmp", ".jpg", ".png", ".jpeg", ".gif"]

#-----------------------------------------------------#
#                       Sampling                      #
#-----------------------------------------------------#
# Initialize Data IO Interface for data
## We are using 3 classes due to [background, lung, covid-19]
interface = IIO_interface(img_type="grayscale", classes=3)

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

#-----------------------------------------------------#
#                Run Inference Pipeline               #
#-----------------------------------------------------#
# Compute predictions for start model
model.preprocessor.data_io = Data_IO(interface, input_path=path_data,
                                     delete_batchDir=False,
                                     output_path=os.path.join(path_results,
                                                              "start"))
model.load(os.path.join(path_models, "model.start.hdf5"))
model.predict(test, activation_output=True)

# Compute predictions for first model
model.preprocessor.data_io = Data_IO(interface, input_path=path_data,
                                     delete_batchDir=False,
                                     output_path=os.path.join(path_results,
                                                              "first"))
model.load(os.path.join(path_models, "model.1.hdf5"))
model.predict(test, activation_output=True)

# Compute predictions for 10 model
model.preprocessor.data_io = Data_IO(interface, input_path=path_data,
                                     delete_batchDir=False,
                                     output_path=os.path.join(path_results,
                                                              "10"))
model.load(os.path.join(path_models, "model.10.hdf5"))
model.predict(test, activation_output=True)

# Compute predictions for 75 model
model.preprocessor.data_io = Data_IO(interface, input_path=path_data,
                                     delete_batchDir=False,
                                     output_path=os.path.join(path_results,
                                                              "75"))
model.load(os.path.join(path_models, "model.75.hdf5"))
model.predict(test, activation_output=True)


# Compute predictions for final model
model.preprocessor.data_io = Data_IO(interface, input_path=path_data,
                                     delete_batchDir=False,
                                     output_path=os.path.join(path_results,
                                                              "final"))
model.load(os.path.join(path_models, "model.latest.hdf5"))
model.predict(test, activation_output=True)
