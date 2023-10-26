import numpy as np
from PIL import Image
import glob
import pandas as pd
from collections import OrderedDict
import torch 
from scipy.ndimage import label
from sklearn.metrics import jaccard_score

# variables
threshold = (0.1, 0.3, 0.5, 0.7, 0.9) # list of float values between 0-1 for where to threshold values when converting to binary
truthDir = "./truth/" # string directory of where ground truth mask png files are
predDir = "./prediction/" # string directory of where predicted mask npy files are
outDir = "./metrics/" # string output directory to save csv files with iou/dice scores
option = "new_dice" # option "new_dice" or "iou_dice" to use either function
iou_threshold = 0.5 # for new-dice, float value 0-1 to count an iou for a true positive or not

'''
takes single mask array of floats (0-1) [height, width]
puts out single binary mask (0 or 1) [height, width]
depending on threshold value
used with unet++ masks
'''
def thresholdmasks(mask, threshold):
    n, m = mask.shape
    newmask = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            if mask[i, j] >= threshold: # set mask to binary 0 or 1 depending on threshold value
                newmask[i, j] = 1
            else:
                newmask[i, j] = 0
                
    return newmask

'''
takes multiple mask arrays of floats (0-1) [instnum, height, width]
puts out single binary mask (0 or 1) [height, width]
depending on threshold value
used with maskrcnn masks
'''
def addmasks(mask, threshold):
    instNum, n, m = mask.shape
    tempmask = np.zeros((instNum, n, m))
    newmask = np.zeros((n, m))
    
    for i in range(instNum):
        for j in range(n):
            for k in range(m):
                if mask[i, j, k] >= threshold: # set mask to binary 0 or 1 depending on threshold value
                    tempmask[i, j, k] = 1
                else:
                    tempmask[i, j, k] = 0   

        newmask = np.add(newmask, tempmask[i]) # add all masks together to get single mask (0-instnum)
        
    for l in range(n):
        for m in range(m):
            if newmask[l, m] >= 1: # set mask to 1 if there is >= 1 instances
                newmask[l, m] = 1
    
    return newmask

'''
takes a binary image mask (0 or 255) [channels, height, width]
converts it to a binary array (0 or 1) [height, width]
'''
def convertimage(image):
    imgarray = np.asarray(image)
    n, m, _ = imgarray.shape
    newmask = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            if imgarray[i, j, 1] >= 250:
                newmask[i, j] = 1
                
    return newmask

'''
The dice coefficient is calculated individually for each image in the batch using the calculate_metrics function
then the mean dice coefficient is computed by averaging the dice coefficients over the entire batch.
'''
def new_dice(true_mask, pred_mask, iou_threshold, threshold): 
    smooth = 1e-6

    true_mask = true_mask > threshold
    pred_mask = pred_mask > threshold

    true_labels, _ = label(true_mask)
    pred_labels, _ = label(pred_mask)

    true_positives = 0
    false_negatives = 0
    visited = set()

    for j in range(1, np.max(true_labels) + 1):
        true_region = true_labels == j
        match_found = False

        for k in range(1, np.max(pred_labels) + 1):
            if k in visited:
                continue

            pred_region = pred_labels == k

            true_region_flat = true_region.flatten()
            pred_region_flat = pred_region.flatten()

            iou = jaccard_score(true_region_flat, pred_region_flat)

            if iou > iou_threshold:
                true_positives += 1
                match_found = True
                visited.add(k)
                break

        if not match_found:
            false_negatives += 1

    dice_coefficient = (true_positives + smooth) / (true_positives + false_negatives + smooth)

    return dice_coefficient

def iou_dice(predlist, truthlist, threshold, outDir):
    smooth = 1e-6

    for j in range(len(threshold)):
        log = OrderedDict([('iou', []), ('dice', [])])

        for i in range(len(predlist)):
            truthmask = convertimage(Image.open(truthlist[i]))
            
            #predmask = addmasks(np.load(predlist[i]), threshold[j]) # for maskrcnn masks
            predmask = thresholdmasks(np.load(predlist[i]), threshold[j]) # for unet++ masks
            
            # convert binary arrays to boolean arrays to use logical operators
            boolpredmask = np.array(predmask, dtype=bool)
            booltruthmask = np.array(truthmask, dtype=bool)
        
            intersection = boolpredmask * booltruthmask # logical AND
            union = boolpredmask + booltruthmask # logical OR 
            iou = (np.count_nonzero(intersection) + smooth)/(np.count_nonzero(union) + smooth)
            dice = (2 * np.count_nonzero(intersection) + smooth)/(np.count_nonzero(boolpredmask) + np.count_nonzero(booltruthmask) + smooth)

            log['iou'].append(iou)
            log['dice'].append(dice)
            pd.DataFrame(log).to_csv(outDir+"test_threshold_"+str(threshold[j])+".csv")
            
        print("threshold ", j, " completed.")
        
    print("done.")

def main():

    predlist = [f for f in glob.glob(predDir+"**.npy", recursive=False)]
    truthlist = [f for f in glob.glob(truthDir+"**.png", recursive=False)]
    predlist.sort()
    truthlist.sort()

    if option == "new_dice":
        for j in range(len(iou_threshold)):
            log = OrderedDict([('new dice', [])])
            
            for i in range(len(truthlist)):
                truthmask = convertimage(Image.open(truthlist[i]))
                predmask = addmasks(np.load(predlist[i]), 0.5)
                newdice = new_dice(torch.tensor(truthmask), torch.tensor(predmask), iou_threshold[j])
                
                log['new dice'].append(newdice)
                
                pd.DataFrame(log).to_csv(outDir+"new_dice_"+str(iou_threshold[j])+".csv")
                
            print("threshold ", j, " completed.")
            
        print("done.")
    elif option == "iou_dice":
        iou_dice(predlist, truthlist, threshold, outDir)
    else:
        print("invalid option. Change to \"new_dice\" or \"iou_dice\".")

if __name__ == "__main__":
    main()
