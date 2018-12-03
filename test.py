
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
import cv2
import sys
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import os
from sklearn.externals import joblib
from skimage.feature import hog


# In[2]:


clf = joblib.load('person_final_hard.pkl')
test_path = "testdataset"
testData = []
testLabels = []


# In[3]:


# get the test labels
train_labels = os.listdir(test_path)


# In[4]:


get_ipython().run_cell_magic('time', '', '# loop over the training data sub-folders\nfor training_name in train_labels:\n    # join the training data path and each species training folder\n    dir = os.path.join(test_path, training_name)\n\n    # get the current training label\n    current_label = training_name\n    images_per_class = len(os.listdir(dir))\n    num = 0\n    count = 0\n    if(current_label == \'neg\'):\n        print ("Processing Negative images")\n        for x in range(1,images_per_class+1):\n            file = dir + "/" + str(x) + ".jpg"\n            img = cv2.imread(file)\n            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n            fd = hog(gray,orientations=9,pixels_per_cell=(6,6),cells_per_block=(2,2),block_norm="L1",transform_sqrt=True)\n            testData.append(fd)\n            testLabels.append(0)\n            count = count + 1\n            num = num + 1\n            sys.stdout.write("\\r" + "Images Done: " + str((num/4553.0)*100) + "\\t negatives: " + str(count))\n            sys.stdout.flush()\n        print ("\\n[STATUS] processed folder: {}".format(current_label))\n    elif(current_label == \'pos\'):\n        print ("Processing Positive images")\n        for x in range(1,images_per_class+1):\n            file = dir + "/" + str(x) + ".jpg"\n            img = cv2.imread(file)\n            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n            fd = hog(gray,orientations=9,pixels_per_cell=(6,6),cells_per_block=(2,2),block_norm="L1",transform_sqrt=True)\n            testData.append(fd)\n            testLabels.append(1)\n            count = count + 1\n            num = num + 1\n            sys.stdout.write("\\r" + "Images Done: " + str((num/958.0)*100) + "\\t positives: " + str(count))\n            sys.stdout.flush()\n        print ("\\n[STATUS] processed folder: {}".format(current_label))\n            \n\nprint ("[STATUS] completed Positive and Negative Feature Extraction...")')


# In[5]:


prediction_result = clf.predict(testData)
score = accuracy_score(np.asarray(testLabels), prediction_result)


# In[6]:


decision = clf.decision_function(testData)
average_precision = average_precision_score(testLabels, decision)
print('\nAverage precision-recall score: {0:0.2f}'.format(
      average_precision))
print("\nClassification Report\n")
print(classification_report(testLabels, prediction_result))
print("Accuracy:" +str(score)+'\n')
print("Confusion Matrix")
cmx = confusion_matrix(testLabels, prediction_result, labels=[0,1])
df = pd.DataFrame(cmx, columns=[0,1], index=[0,1])
df.columns.name = 'prediction'
df.index.name = 'label'
df


# In[7]:


fpr_svm, tpr_svm, thresholds_svm = roc_curve(testLabels, decision)
fpr_, tpr_, thresholds_ = roc_curve(testLabels, decision)
auc_svm = auc(fpr_svm, tpr_svm)
auc_ = auc(fpr_, tpr_)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
#plt.plot(fpr_svm, tpr_svm, label='class 0 (area = {:.3f})'.format(auc_svm))
plt.plot(fpr_, tpr_, label='HoG (area = {:.3f})'.format(auc_))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('ROC.png')
plt.show()


# In[8]:


precision, recall, _ = precision_recall_curve(testLabels, decision)
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))
plt.show()

