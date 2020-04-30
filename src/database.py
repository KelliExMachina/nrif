# Standard Libraries
import pandas as pd
import numpy as np
import os
import sys

# Image Libraries
# from scikit-image
# from scikit-learn
import cv2
from cv2 import imread, imshow, waitKey, destroyAllWindows, rectangle, CascadeClassifier

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics, confusion_matrix

# NN Libraries
# import tensorflow as tf
# import mtcnn as mtcnn

# Database
import postgres

# Visualizations
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

sys.dont_write_bytecode = True  # Don't generate .pyc files / __pycache__ directories

print('*****************************************************************************************')
print('*****************************************************************************************')
print('***************************       CREATE DARABASE      **********************************')

-- Database: kennedy;

-- DROP DATABASE kennedy;

CREATE DATABASE kennedy
   WITH
   OWNER = kennedy
   ENCODING = 'UTF8'
   LC_COLLATE = 'en_US.UTF-8'
   LC_CTYPE = 'en_US.UTF-8'
    TABLESPACE = pg_default
   CONNECTION LIMIT = -1;

/* DROP TABLE */
DROP TABLE "images";

/* CREATE TABLE */
CREATE TABLE "images"
(
 "id" bigint DEFAULT 0 NOT NULL,
 "dir_path" text DEFAULT blah NOT NULL,
 "date" date DEFAULT 1/1/1970 NOT NULL,
 "img_orig" bytea DEFAULT NaN NOT NULL,
 "face" boolean DEFAULT False NOT NULL,
 "img_proc" bytea DEFAULT NaN,
 "face_class" integer DEFAULT 0 NOT NULL
)
;

COMMENT ON COLUMN "images"."id" IS E'This number will come from a counter programatically.';
COMMENT ON COLUMN "images"."dir_path" IS E'directory path to original image with the \'/\' changed to \'_\'.  This is so that you can trace back to the imported data for files with the same names in multiple directories.';
COMMENT ON COLUMN "images"."date" IS E'date image was imported.';
COMMENT ON COLUMN "images"."face_class" IS E'Classes based off _# in filename.  0,1,2,3,4,5. 0=no_class, 1=commecicomm,2=smile,3=shut,4=shocked,5=sunglasses.';
COMMENT ON COLUMN "images"."img_proc" IS E'Processed images!';
ALTER TABLE "images" ADD CONSTRAINT "dir_path" PRIMARY KEY("id");

print('*****************************************************************************************')
print('*****************************************************************************************')
print('**********************       PROCESS IMAGES FOR STORAGE      *****************************')

# Process Images Directory for Feeding into the Model
#! tar -zxvf TD_RGB_E_Set1.zip
#! tar -zxvf TD_RGB_E_Set2.zip
#! tar -zxvf TD_RGB_E_Set31.zip
#! tar -zxvf TD_RGB_E_Set41.zip


def build_image_list(path):
    '''
    Search (path) and make a recursive listing of paths to each file.
    INPUTS: path = top of directory tree to begin recursion for images. ex) "../data"
    OUTPUTS: fname = A full path listing of all files in the directory tree.  astype.list()
    '''
    fname = []
    for root, d_names, f_names in os.walk(path):
        f_names = [f for f in f_names if not f[0] == '.']  # skip hidden files: .DSstore
        d_names = [d for d in d_names if not d[0] == '.']  # skip hidden folders: .git
        for f in f_names:
            fname.append(os.path.join(root, f))
    return fname


def process_images(fname):
    i = 0
    for file in fname:
        global classifier
        global file_name
        global file_path
        file_name = os.path.basename(file)
        file_path = file
        #print('{}...{}...{}'.format(i, file_name, file_path))
        # insertBLOB(i, file_name, file_path) # call to import images into SQLite
        # load the photograph
        img = cv2.imread(file_path, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        i += 1
        perform_face_detection(img_rgb)
    return


print('*****************************************************************************************')
print('*****************************************************************************************')
print('**********************       DETECT FACES AND DRAW BOXES      ***************************')


def perform_face_detection(img_rgb):
    global classifier
    global file_name
    bounding_boxes = classifier.detectMultiScale(img_rgb, scaleFactor=3.0, minNeighbors=10, minSize=(150, 150))
    box_list = []
    for box in bounding_boxes:
        box_list.append(box)
    if len(box_list) < 1:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        cv2.imwrite('../sort/no_face/' + file_name + '.jpg', img_rgb)
    else:
        draw_boxes(img_rgb, box_list)
    return


def draw_boxes(img_rgb, box_list):
    for box in box_list:
        x, y, width, height = box
        x2, y2 = x + width, y + height
        cv2.rectangle(img_rgb, (x, y), (x2, y2), (0, 255, 0), 3)
    #show_image(img_rgb)
    return


print('*****************************************************************************************')
print('*****************************************************************************************')
print('**************************       STORE MODIFIED IMAGE      ******************************')


def show_image(img_rgb):
    global file_name
#     plt.imshow(img)
#     plt.xticks([]), plt.yticks([])
#     plt.show()
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    cv2.imwrite('../sort/face/' + file_name + '.jpg', img_rgb)
    return


print('*****************************************************************************************')
print('*****************************************************************************************')
print('*****************************       CONFUSION MATRIX      ********************************')

cm = confusion_matrix(y_test, y_pred_class)
tn = cm[0, 0]
fn = cm[0, 1]
fp = cm[1, 0]
tp = cm[1, 1]

accurracy = (tp + tn) / (tn + tp + fn + fp)

precision = tp / (tp + fp)

recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)
print('My model metrics were: Accurracy: {}, Precision: {}, Recall: {}, and F1: {}'.format(accurracy, precision, recall, f1_score))


plt.matshow(cm)
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


print('*****************************************************************************************')
print('*****************************************************************************************')
print('***********************************       MAIN      **************************************')


process_images(image_list)


print('*****************************************************************************************')
print('*****************************************************************************************')
print('***********************************       MAIN      **************************************')
print('*****************************************************************************************')
print('*****************************************************************************************')
print('***********************************       MAIN      **************************************')
print('*****************************************************************************************')
print('*****************************************************************************************')
print('***********************************       MAIN      **************************************')
print('*****************************************************************************************')
print('*****************************************************************************************')
print('***********************************       MAIN      **************************************')
print('*****************************************************************************************')
print('*****************************************************************************************')
print('***********************************       MAIN      **************************************')
print('*****************************************************************************************')
print('*****************************************************************************************')
print('***********************************       MAIN      **************************************')
print('*****************************************************************************************')
print('*****************************************************************************************')
print('***********************************       MAIN      **************************************')
print('*****************************************************************************************')
print('*****************************************************************************************')
print('***********************************       MAIN      **************************************')
print('*****************************************************************************************')
print('*****************************************************************************************')
print('***********************************       MAIN      **************************************')
print('*****************************************************************************************')
print('*****************************************************************************************')
print('***********************************       MAIN      **************************************')


ile_name = ''  # image name
file_path = ''  # path to file
classifier = CascadeClassifier('../src/haarcascade_frontalface_default.xml')  # load the pre-trained model


# load the model
print('**************************************************************************Load the model')
# Get the haarcascade .xml file and save it in the src directory
#! wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml


bvlc_alexnetprototxt_ = 'https://github.com/opencv/opencv_extra/testdata/dnn/bvlc_alexnet.prototxt'
bvlc_alexnetcaffemodel_ = 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel'
imagepath = '/Users/kennedy/Documents/GitHub/fiprojects/mod_4_look_at_me/data/test/img_33.jpg'

net = cv2.dnn.readNetFromCaffe(bvlc_alexnetprototxt_, bvlc_alexnetcaffemodel_)

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread('imagepath')
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network for detections and predictions
print('********************************************************************Detecting objects...')
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]
    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > args["confidence"]:
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)

# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
