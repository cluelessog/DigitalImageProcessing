
# coding: utf-8

# In[2628]:


import numpy as np 
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
import sys
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
from skimage import color
import matplotlib.pyplot as plt 
import os 
import glob
import warnings
warnings.filterwarnings('ignore')


# In[2629]:


detect_path = "dataset/detect"
winSize = (64, 128)
winStride = (7,7)
downscale = 1.25


# In[2630]:


#faster nms
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
        np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


# In[2631]:


def sliding_window(image, window_size, step_size):
    '''Returns a patch of the input 'image' of size 
       equal to 'window_size(64x128)'. Increments in
       x and y direction by 'step_size'.The function returns a tuple - (x, y, im_window)
    '''
    for y in range(0, image.shape[0]-128, step_size[1]):
        for x in range(0, image.shape[1]-64, step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])


# In[2632]:


def detector(image):
    im = cv2.imread(image)
    im = imutils.resize(im, width = min(400, im.shape[1]))
    clf = joblib.load("person_final_hard.pkl")

    #List to store the detections
    detections = []
    #The current scale of the image 
    scale = 0
    num = 0
    count = 0
    #looping through images in a gaussian pyramid
    for im_scaled in pyramid_gaussian(im, downscale = downscale):
        #The list will store detections at the current scale
        #if the image dimension in gaussian pyramid goes below 64x128, break
        if im_scaled.shape[0] < winSize[1] or im_scaled.shape[1] < winSize[0]:
            break
        #at each scale in gaussian pyramid, image patch of 64x128 is extracted using sliding window
        for (x, y, im_window) in sliding_window(im_scaled, winSize, winStride):
            if im_window.shape[0] != winSize[1] or im_window.shape[1] != winSize[0]:
                continue
            im_window = color.rgb2gray(im_window)
            fd = hog(im_window,orientations=9,pixels_per_cell=(6,6),cells_per_block=(2,2),block_norm="L1",transform_sqrt=True)
            fd = fd.reshape(1, -1)
            pred = clf.predict(fd)
            num = num + 1
            #if it is a human
            if pred == 1:
                count = count + 1
                #and the probability is greater than 50 percent, count it as detection
                if clf.decision_function(fd) > 0.8:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), clf.decision_function(fd), 
                    int(winSize[0] * (downscale**scale)),
                    int(winSize[1] * (downscale**scale))))
            
                 

        sys.stdout.write("\r" + "Progress " + str((count/num)*100))   
        sys.stdout.flush()
        scale += 1

    clone = im.copy()
    #drawing original boxes
    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print ("sc: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression_fast(rects, overlapThresh = 0.30)
    print ("shape, ", pick.shape)
    #final bounding boxes
    for(xA, yA, xB, yB) in pick:
        cv2.rectangle(im, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    plt.figure(1)
    plt.title("Detections")
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show()


# In[2633]:


filenames = glob.iglob(os.path.join(detect_path, '*'))

for file in filenames:
    detector(file)


# In[ ]:




