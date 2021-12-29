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
import numpy as np
from PIL import Image
import cv2
import pandas as pd

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# Desired image resolution for patches
img_size = 512
gap = 8
border = 8

# Data directory
path_data = "data/braintumor.prepared"

# Prediction directory
path_results = "results/braintumor"
pred_modes = ["start", "first", "final"]
path_preds = [os.path.join(path_results, x) for x in pred_modes]

# Evaluation directory
path_evaluation = "evaluation/braintumor"
if not os.path.exists("evaluation") : os.mkdir("evaluation")
if not os.path.exists(path_evaluation) : os.mkdir(path_evaluation)

#-----------------------------------------------------#
#                 Metric Computations                 #
#-----------------------------------------------------#
def calc_DSC(truth, pred, classes):
    dice_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            # Calculate Dice
            dice = 2*np.logical_and(pd, gt).sum() / (pd.sum() + gt.sum())
            dice_scores.append(dice)
        except ZeroDivisionError:
            dice_scores.append(0.0)
    # Return computed Dice Similarity Coefficients
    return dice_scores

def calc_IoU(truth, pred, classes):
    iou_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            # Calculate iou
            iou = np.logical_and(pd, gt).sum() / (pd.sum() + gt.sum() - np.logical_and(pd, gt).sum())
            iou_scores.append(iou)
        except ZeroDivisionError:
            iou_scores.append(0.0)
    # Return computed IoU
    return iou_scores

def calc_Sensitivity(truth, pred, classes):
    sens_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            # Calculate sensitivity
            sens = np.logical_and(pd, gt).sum() / gt.sum()
            sens_scores.append(sens)
        except ZeroDivisionError:
            sens_scores.append(0.0)
    # Return computed sensitivity scores
    return sens_scores

def calc_Specificity(truth, pred, classes):
    spec_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            not_gt = np.logical_not(np.equal(truth, i))
            not_pd = np.logical_not(np.equal(pred, i))
            # Calculate specificity
            spec = np.logical_and(not_pd, not_gt).sum() / (not_gt).sum()
            spec_scores.append(spec)
        except ZeroDivisionError:
            spec_scores.append(0.0)
    # Return computed specificity scores
    return spec_scores

def calc_Accuracy(truth, pred, classes):
    acc_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            not_gt = np.logical_not(np.equal(truth, i))
            not_pd = np.logical_not(np.equal(pred, i))
            # Calculate accuracy
            acc = (np.logical_and(pd, gt).sum() + \
                   np.logical_and(not_pd, not_gt).sum()) /  gt.size
            acc_scores.append(acc)
        except ZeroDivisionError:
            acc_scores.append(0.0)
    # Return computed accuracy scores
    return acc_scores

def calc_Precision(truth, pred, classes):
    prec_scores = []
    # Iterate over each class
    for i in range(classes):
        try:
            gt = np.equal(truth, i)
            pd = np.equal(pred, i)
            # Calculate precision
            prec = np.logical_and(pd, gt).sum() / pd.sum()
            prec_scores.append(prec)
        except ZeroDivisionError:
            prec_scores.append(0.0)
    # Return computed precision scores
    return prec_scores

#-----------------------------------------------------#
#                    Visualization                    #
#-----------------------------------------------------#
def overlay_segmentation(img_rgb, seg):
    # Initialize segmentation in RGB
    shp = seg.shape
    seg_rgb = np.zeros((shp[0], shp[1], 3), dtype=np.int)
    # Set class to appropriate color
    seg_rgb[np.equal(seg, 1)] = [255, 0,   0]
    # Get binary array for places where an ROI lives
    segbin = np.greater(seg, 0)
    repeated_segbin = np.stack((segbin, segbin, segbin), axis=-1)
    # Weighted sum where there's a value to overlay
    alpha = 0.3
    img_overlayed = np.where(
        repeated_segbin,
        np.round(alpha*seg_rgb+(1-alpha)*img_rgb).astype(np.uint8),
        np.round(img_rgb).astype(np.uint8)
    )
    # Return final image with segmentation overlay
    return img_overlayed

def rect_with_rounded_corners(img, r, t, c):
    """
    Source: https://stackoverflow.com/questions/60382952/how-to-add-a-round-border-around-an-image/60392932#60392932
    Credits: https://stackoverflow.com/users/7355741/fmw42
    Slightly modified by me (Dominik Müller, Augsburg, Germany)

    :param image: image as NumPy array
    :param r: radius of rounded corners
    :param t: thickness of border
    :param c: color of border
    :return: new image as NumPy array with rounded corners
    """
    hh, ww = img.shape[0:2]
    # create white image of size of input
    white = np.full_like(img, (255,255,255))
    # add black border of thickness
    border = cv2.copyMakeBorder(white, t, t, t, t,
                                borderType=cv2.BORDER_CONSTANT,
                                value=(0,0,0))
    # blur image by rounding amount as sigma
    blur = cv2.GaussianBlur(border, (0,0), r, r)
    # threshold blurred image
    thresh1 = cv2.threshold(blur, 128, 255, cv2.THRESH_BINARY)[1]
    # create thesh2 by eroding thresh1 by 2*t
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*t,2*t))
    thresh2 = cv2.morphologyEx(thresh1, cv2.MORPH_ERODE, kernel, iterations=1)
    # subtract white background with thresholded image to make a border mask
    mask = white - thresh2[t:hh+t, t:ww+t]
    # create colored image the same size as input
    color = np.full_like(img, c)
    # combine input and color with mask
    result = cv2.bitwise_and(color, mask) + cv2.bitwise_and(img, 255-mask)
    # Return image with borders
    return result

def visualize_segmentation(img, seg):
    # Squeeze image files to remove channel axis
    img = np.squeeze(img, axis=-1)
    # Convert volume to RGB
    img_rgb = np.stack([img, img, img], axis=-1)
    # Color image with segmentation if present
    if not seg is None:
        img_rgb = overlay_segmentation(img_rgb, seg)
    # Resize image to desired resolution
    img_rgb = cv2.resize(img_rgb, (img_size, img_size),
                         interpolation=cv2.INTER_LINEAR)
    # Add border to image
    img_rgb = rect_with_rounded_corners(img_rgb, border, border,
                                        (255, 255, 255))
    # Rotate braintumor image via 90° clockwise
    img_rgb = cv2.rotate(img_rgb, rotateCode=0)
    # Return visualized sample as NumPy matrix
    return img_rgb

def create_visualization(img, seg_list, index, vis_path):
    # Visualize each segmentation
    img_list = [visualize_segmentation(img, seg) for seg in seg_list]
    # Add white gap between each image
    for i in reversed(range(len(img_list))):
        if i == 0 : continue
        else : img_list.insert(i, np.full((img_size, gap, 3), 255))
    # Stack images together
    img_stacked = np.hstack((img_list))
    # Convert NumPy image matrix to Pillow
    PIL_image = Image.fromarray(img_stacked.astype('uint8'), 'RGB')
    # Set up the output path
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)
    file_name = str(index) + ".png"
    out_path = os.path.join(vis_path, file_name)
    # Save the animation (gif)
    PIL_image.save(out_path)

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
train, test = train_test_split(sample_list, train_size=0.8, test_size=0.2,
                               shuffle=True, random_state=0)

#-----------------------------------------------------#
#                      Evaluation                     #
#-----------------------------------------------------#
# Initialize some stuff
dt = []

# Iterate over each sample
for index in test:
    # Access sample
    sample = data_io.sample_loader(index, load_seg=True)
    img = sample.img_data
    # Load ground truth & predictions
    gt = np.squeeze(sample.seg_data, axis=-1)
    pd_start = data_io.interface.load_prediction(index, path_preds[0])
    pd_first = data_io.interface.load_prediction(index, path_preds[1])
    pd_final = data_io.interface.load_prediction(index, path_preds[2])
    # Create artificial no & full annotation predictions
    ap_no = np.full(gt.shape, 0)
    ap_full = np.full(gt.shape, 1)
    ap_rand = np.random.choice([0,1], (gt.shape))
    # Pack segmentations to a list together
    seg_list = [gt, ap_no, ap_full, ap_rand, pd_start, pd_first, pd_final]
    seg_names = ["gt", "ap_no", "ap_full", "ap_rand", "pd_start", "pd_first", "pd_final"]
    # Compute various scores
    for i, seg in enumerate(seg_list[1:]):
        dt.append([index, seg_names[i+1], "DSC", calc_DSC(gt, seg, 2)[1]])
        dt.append([index, seg_names[i+1], "IoU", calc_IoU(gt, seg, 2)[1]])
        dt.append([index, seg_names[i+1], "ACC", calc_Accuracy(gt, seg, 2)[1]])
        dt.append([index, seg_names[i+1], "SPEC", calc_Specificity(gt, seg, 2)[1]])
        dt.append([index, seg_names[i+1], "SENS", calc_Sensitivity(gt, seg, 2)[1]])
        dt.append([index, seg_names[i+1], "PREC", calc_Precision(gt, seg, 2)[1]])
    # Compute visualization
    create_visualization(img, seg_list, index, path_evaluation)
    break

dt = pd.DataFrame(dt)
print(dt)


# REWORK EVERYTHING WITH ACTIVATION OUTPUTS instead of argmax mask values from predictions
# In order to implement ROC curves

# what to implement
# - scores per sample in large csv
# - large row of 6 images: Ground Truth, No Annotation, Full-Image Annotation, Untrained Model, Bad Model, Good Model


    # # Combine images
    # top = np.hstack((inner_cache[0][2], np.full((ss,gap,3), 255),
    #                  inner_cache[1][2]))
    # down = np.hstack((inner_cache[2][2], np.full((ss,gap,3), 255),
    #                   inner_cache[3][2]))
    # final = np.vstack((np.full((30,ss*2+gap,3), 255), top,
    #                    np.full((gap,ss*2+gap,3), 255), down))
    # final = final.astype(np.uint8)
    #
    # # Add text labels
    # img = Image.fromarray(final)
    # draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf", 18)
    # # Add classifications
    # draw.text((2, 32), inner_cache[0][1], (173,255,47), font=font)
    # draw.text((2+ss+gap, 32), inner_cache[1][1], (173,255,47), font=font)
    # draw.text((2, 32+ss+gap), inner_cache[2][1], (173,255,47), font=font)
    # draw.text((2+ss+gap, 32+ss+gap), inner_cache[3][1], (173,255,47), font=font)
    # # Add dataset header on top
    # draw.text((((ss*2+gap)/2)-20, 2), ds.upper(), (0,0,0), font=font)
    #
    # # Convert back to NumPy and store
    # mat = np.array(img)
    # outer_cache.append(mat)
