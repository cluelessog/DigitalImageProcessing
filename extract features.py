
# coding: utf-8

# In[10]:


import numpy as np 
import cv2
from sklearn import svm
from sklearn.svm import SVC
import os 
import glob
import matplotlib.pyplot as plt
import sys
from skimage.feature import hog
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')


# In[11]:


# path to training data
train_path = "dataset/train"
hard_neg = "dataset/hardneg"


# In[12]:


def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0)
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window
    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in range(0, image.shape[0]-128, step_size[1]):
        for x in range(0, image.shape[1]-64, step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


# In[32]:


def hard_negative_mine(hard_neg, winSize, winStride):
    hard_negatives = []
    hard_negative_labels = []
    num_images = len(os.listdir(hard_neg))
    count = 0
    num = 0
    for x in range (1,num_images + 1):
        #filename, file_extension = os.path.splitext(neg_img_dir + imgfile)
        #filename = os.path.basename(filename)
        file = hard_neg + "/" + str(x) + ".jpg"
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for (x, y, im_window) in sliding_window(gray, winSize, winStride):
            fd = hog(im_window,orientations=9,pixels_per_cell=(6,6),cells_per_block=(2,2),block_norm="L1",transform_sqrt=True)
            if (clf.predict([fd]) == 1):
                hard_negatives.append(fd)
                hard_negative_labels.append(0)
                #joblib.dump(features, "features/neg_mined/" + str(filename) + str(imgcount) + ".feat")
                count = count + 1

        num = num + 1

        #print "Images Done: " + str(num)
        sys.stdout.write("\r" + "Images Done: " + str((num/218.0)*100) + "\tHard negatives: " + str(count))
        sys.stdout.flush()

        #print "Hard Negatives: " + str(count)
        #if (num == 10):
    #        break

    return hard_negatives, hard_negative_labels


# In[14]:


# get the training labels
train_labels = os.listdir(train_path)
fds = []
labels = []


# In[15]:


get_ipython().run_cell_magic('time', '', '# loop over the training data sub-folders\nfor training_name in train_labels:\n    # join the training data path and each species training folder\n    dir = os.path.join(train_path, training_name)\n\n    # get the current training label\n    current_label = training_name\n    images_per_class = len(os.listdir(dir))\n    \n    if(current_label == \'neg\'):\n        print ("Processing Negative images")\n        for x in range(1,images_per_class+1):\n            file = dir + "/" + str(x) + ".jpg"\n            img = cv2.imread(file)\n            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n            fd = hog(gray,orientations=9,pixels_per_cell=(6,6),cells_per_block=(2,2),block_norm="L1",transform_sqrt=True)\n            fds.append(fd)\n            labels.append(0)\n        print ("[STATUS] processed folder: {}".format(current_label))\n    elif(current_label == \'pos\'):\n        print ("Processing Positive images")\n        for x in range(1,images_per_class+1):\n            file = dir + "/" + str(x) + ".jpg"\n            img = cv2.imread(file)\n            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n            fd = hog(gray,orientations=9,pixels_per_cell=(6,6),cells_per_block=(2,2),block_norm="L1",transform_sqrt=True)\n            fds.append(fd)\n            labels.append(1)\n        print ("[STATUS] processed folder: {}".format(current_label))\n            \n\nprint ("[STATUS] completed Positive and Negative Feature Extraction...")')


# In[33]:


print (np.array(fds).shape,len(labels))
# Randomize data
#np.random.shuffle(fds)
#np.random.shuffle(labels)
print ("Images Read and Shuffled")
print ("Training Started")
# Initializing classifiers

clf = svm.LinearSVC(C=0.01)
clf.fit(fds,labels)

print ("Trained")

joblib.dump(clf, 'person.pkl')


# In[34]:


print ("Hard Negative Mining")
winStride = (8, 8)
winSize = (64, 128)
hard_negatives, hard_negative_labels = hard_negative_mine(hard_neg, winSize, winStride)


# In[35]:


sys.stdout.write("\n")
print(np.array(hard_negatives).shape,len(hard_negative_labels))
fds_final = np.concatenate((fds, hard_negatives))
labels_final = np.concatenate((labels, hard_negative_labels))

print ("Final Samples: " + str(len(fds_final)))
print ("Retraining the classifier with final data")

clf.fit(fds_final, labels_final)

print ("Trained and Dumping")

joblib.dump(clf,'person_final_hard.pkl')


# In[ ]:




