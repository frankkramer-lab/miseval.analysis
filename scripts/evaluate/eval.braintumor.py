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
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from plotnine import *
from scipy import ndimage
from hausdorff import hausdorff_distance as simple_hausdorff_distance
# Experimental
import warnings
warnings.filterwarnings("ignore")

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
# Desired image resolution for patches
img_size = 512
gap = 8
border = 8

# ROC curves
rounding_precision = 8

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
                   np.logical_and(not_pd, not_gt).sum()) / gt.size
            acc_scores.append(acc)
        except ZeroDivisionError:
            acc_scores.append(0.0)
    # Return computed accuracy scores
    return acc_scores

def calc_Precision(truth, pred, classes):
    prec_scores = []
    # Iterate over each class
    for i in range(classes):
        gt = np.equal(truth, i)
        pd = np.equal(pred, i)
        # Calculate precision
        if pd.sum() == 0.0 : prec_scores.append(0.0)
        else:
            prec = np.logical_and(pd, gt).sum() / pd.sum()
            prec_scores.append(prec)
    # Return computed precision scores
    return prec_scores

def calc_AUC(truth, pred, classes):
    auc_scores = []
    # Iterate over each class
    for i in range(classes):
        prob = np.round(pred[:,:,i], rounding_precision)
        gt = np.equal(truth, i).astype(int)
        auc = roc_auc_score(gt.flatten(), prob.flatten())
        auc_scores.append(auc)
    # Return computed AUC scores
    return auc_scores

def calc_Kappa(truth, pred):
    # Compute confusion matrix
    gt = np.equal(truth, 1)
    pd = np.equal(pred, 1)
    not_gt = np.logical_not(np.equal(truth, 1))
    not_pd = np.logical_not(np.equal(pred, 1))
    tp = np.logical_and(pd, gt).sum()
    tn = np.logical_and(not_pd, not_gt).sum()
    fp = np.logical_and(pd, not_gt).sum()
    fn = np.logical_and(not_pd, gt).sum()
    # Compute kappa
    fa = tp + tn
    fc = ((tn+fn)*(tn+fp) + (fp+tp)*(fn+tp)) / gt.size
    kappa = (fa-fc) / (gt.size-fc)
    # Return computed kappa score
    return kappa

def border_map(binary_img,neigh):
    """
    Creates the border for a 3D image
    """
    binary_map = np.asarray(binary_img, dtype=np.uint8)
    neigh = neigh
    west = ndimage.shift(binary_map, [-1, 0], order=0)
    east = ndimage.shift(binary_map, [1, 0], order=0)
    north = ndimage.shift(binary_map, [0, 1], order=0)
    south = ndimage.shift(binary_map, [0, -1], order=0)
    cumulative = west + east + north + south
    border = ((cumulative < 4) * binary_map) == 1
    return border

def border_distance(ref,seg):
    """
    This functions determines the map of distance from the borders of the
    segmentation and the reference and the border maps themselves
    """
    neigh=8
    border_ref = border_map(ref,neigh)
    border_seg = border_map(seg,neigh)
    oppose_ref = 1 - ref
    oppose_seg = 1 - seg
    # euclidean distance transform
    distance_ref = ndimage.distance_transform_edt(oppose_ref)
    distance_seg = ndimage.distance_transform_edt(oppose_seg)
    distance_border_seg = border_ref * distance_seg
    distance_border_ref = border_seg * distance_ref
    return distance_border_ref, distance_border_seg#, border_ref, border_seg

# Source: https://link.springer.com/chapter/10.1007/978-3-030-11726-9_4
# Implementation: https://github.com/Issam28/Brain-tumor-segmentation
def calc_AveragedHausdorff(ref, seg):
    """
    This functions calculates the average symmetric distance and the
    hausdorff distance between a segmentation and a reference image
    :return: hausdorff distance and average symmetric distance
    """
    ref_border_dist, seg_border_dist = border_distance(ref,seg)
    hausdorff_distance = np.max(
        [np.max(ref_border_dist), np.max(seg_border_dist)])
    return hausdorff_distance

# Source: https://ieeexplore.ieee.org/document/7053955
# Implementation: https://github.com/mavillan/py-hausdorff
def calc_SimpleHausdorff(truth, pred):
    return simple_hausdorff_distance(truth, pred, distance="euclidean")

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
    if not os.path.exists(vis_path) : os.mkdir(vis_path)
    file_name = str(index) + ".png"
    out_path = os.path.join(vis_path, file_name)
    # Save the animation (gif)
    PIL_image.save(out_path)

#-----------------------------------------------------#
#                    Plot ROC Curve                   #
#-----------------------------------------------------#
def create_roc(gt, seg_list_activation, seg_names, index, path_roc):
    # Set up the roc directory if required
    if not os.path.exists(path_roc) : os.mkdir(path_roc)
    # Compute ROC and plot it for each predicted segmentation
    list_df = []
    for i, seg in enumerate(seg_list_activation):
        # Compute FPR & TPR on flattened matrices
        prob = np.round(seg[:,:,1], rounding_precision)
        fpr, tpr, _ = roc_curve(gt.flatten(), prob.flatten())
        # Create dataframe out of it
        df_seg = pd.DataFrame(data=[fpr, tpr], index=["fpr", "tpr"])
        df_seg = df_seg.transpose()
        df_seg = df_seg.apply(pd.Series.explode)
        # Append segmentation label
        df_seg["seg"] = seg_names[i]
        list_df.append(df_seg)
    # Concat dataframes together
    df_roc = pd.concat(list_df, axis=0, ignore_index=True)
    # Plot roc results
    fig = (ggplot(df_roc, aes("fpr", "tpr", color="seg"))
               + geom_line(size=2)
               + geom_abline(intercept=0, slope=1, color="black",
                             linetype="dashed")
               + ggtitle("Receiver Operating Characteristic")
               + xlab("False Positive Rate")
               + ylab("True Positive Rate")
               + scale_x_continuous(limits=[0, 1],
                                    breaks=np.arange(0.0, 1.1, 0.1))
               + scale_y_continuous(limits=[0, 1],
                                    breaks=np.arange(0.0, 1.1, 0.1))
               + scale_color_discrete(name="Segmentation:")
               + theme_bw(base_size=28))
    # Store figure to disk
    fig.save(filename=str(index) + ".png", path=path_roc, width=12, height=10,
             dpi=200, limitsize=False)

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
for index in tqdm(test):
    # Access sample
    sample = data_io.sample_loader(index, load_seg=True)
    img = sample.img_data
    # Load ground truth & predictions
    gt = np.squeeze(sample.seg_data, axis=-1)
    # pd_start = np.load(os.path.join(path_preds[0], index + ".npy"))
    # pd_start = np.squeeze(pd_start, axis=-1)                            # Dirty fix. Unnecessary with newest MIScnn verison
    pd_first = np.load(os.path.join(path_preds[1], index + ".npy"))
    pd_first = np.squeeze(pd_first, axis=-1)                            # Dirty fix. Unnecessary with newest MIScnn verison
    pd_final = np.load(os.path.join(path_preds[2], index + ".npy"))
    pd_final = np.squeeze(pd_final, axis=-1)                            # Dirty fix. Unnecessary with newest MIScnn verison
    # Create artificial no & full annotation predictions
    ap_no = np.full(gt.shape+(2,), [1.0, 0.0])
    ap_full = np.full(gt.shape+(2,), [0.0, 1.0])
    rand_tmp = np.array([[0.0, 1.0], [1.0, 0.0]])
    ap_rand = rand_tmp[np.random.choice(rand_tmp.shape[0], gt.shape,
                                        replace=True)]
    # Pack segmentations to a list together
    seg_list_activation = [ap_no, ap_full, ap_rand, pd_first, pd_final]
    seg_list_argmax = [np.argmax(x, axis=-1) for x in seg_list_activation]
    seg_names = ["No Annotation", "Full Annotation", "Random Annotation",
                 "Untrained Model", "Trained Model"]

    # Compute various scores
    for i, seg in enumerate(seg_list_argmax):
        dt.append([index, seg_names[i], "DSC", calc_DSC(gt, seg, 2)[1]])
        dt.append([index, seg_names[i], "IoU", calc_IoU(gt, seg, 2)[1]])
        dt.append([index, seg_names[i], "ACC", calc_Accuracy(gt, seg, 2)[1]])
        dt.append([index, seg_names[i], "SPEC", calc_Specificity(gt, seg, 2)[1]])
        dt.append([index, seg_names[i], "SENS", calc_Sensitivity(gt, seg, 2)[1]])
        # dt.append([index, seg_names[i], "PREC", calc_Precision(gt, seg, 2)[1]])
        dt.append([index, seg_names[i], "KAP", calc_Kappa(gt, seg)])
        # dt.append([index, seg_names[i], "SHD", calc_SimpleHausdorff(gt, seg)])
        dt.append([index, seg_names[i], "AHD", calc_AveragedHausdorff(gt, seg)])
    for i, seg in enumerate(seg_list_activation):
        dt.append([index, seg_names[i], "AUC", calc_AUC(gt, seg, 2)[1]])

    # Compute visualization
    path_viz = os.path.join(path_evaluation, "visualizations")
    create_visualization(img, [gt] + seg_list_argmax, index, path_viz)
    # Plot ROC curve
    path_roc = os.path.join(path_evaluation, "roc")
    create_roc(gt, seg_list_activation, seg_names, index, path_roc)

# Store scores as csv in evaluation directory
dt = pd.DataFrame(dt, columns=["index", "pred", "metric", "score"])
out_path = os.path.join(path_evaluation, "scores.csv")
dt.to_csv(out_path, sep=",", header=True, index=False)

# Plot performance for each metric
for metric in np.unique(dt["metric"]):
    fig = (ggplot(dt.loc[dt["metric"]==metric], aes("pred", "score", fill="pred"))
                  + geom_boxplot(show_legend=False)
                  + ggtitle("Performance via " + metric + " on dataset: Braintumor")
                  + xlab("Metric")
                  + ylab("Score")
                  + coord_flip()
                 # + scale_y_continuous(limits=[0, 1], breaks=np.arange(0.0,1.1,0.1))
                  + scale_fill_discrete(name="Classification")
                  + theme_bw(base_size=28))
    # Store figure to disk
    fig.save(filename="performance." + metric + ".png", path=path_evaluation,
             width=24, height=10, dpi=180)
