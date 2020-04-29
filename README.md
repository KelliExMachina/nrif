<p align="center"><img src="img/YBCheeri-yo.jpg" title="Cheeri-yo"></a></p>  



# NRIF
## No, really.  I’m fine. 

Emotion detection application utilizing OpenCV and keras/Tensorflow.  http://www.notsurewhereitgoes.io  


## Context

The recent Covid-19 pandemic has forced many technology persons to work from home.  As humans, we are often conditioned to say things in a positive manner even when things are not positive at all.  This application was designed to allow human resource departments to monitor the emotionsl health of company resouces by reading facial expressions from a web camera and predicting the subjects emotions.  

## Table of Contents

- [Context](#Context)
- [Content](#Content)
- [Data Preparation](#Data-Preparation)
- [Analysis](#Analysis)
- [Model Evaluation](#Model-Evaluation)
- [Illustrations](#Illustrations)
- [Deployment](#Deployment)
- [Data Sources](#Data-Sources)
- [Next steps](#Next-steps)
- [Citations](#Citations)

## Content

I have over 200 GB of data in a dataset that was collected by Tufts University, and referenced on Kaggle.  Tufts Face Database is the latest, most comprehensive, large-scale (over 10,000 images, 74 females + 38 males, from more than 15 countries with an age range between 4 to 70 years old) face dataset that contains 6 image modalities: visible, near-infrared, thermal, computerized sketch, a recorded video, and 3D images.  I will be specifically using the TDRGBE and TDIRE image datasets for testing.

TDRGBE: The images were captured using a NIKON D3100 camera. Each participant was asked to pose with a neutral expression, a smile, eyes closed, exaggerated shocked expression, and finally wearing sunglasses.

TDIRE: Images were captured using a FLIR Vue Pro camera with participants posing as in the TDRGBE dataset.  

## Data-Preparation

There are basically two stages for the project: 1) face detection and 2) emotional detection.  I will train a cascade classifier from OpenCV and use the Viola & Jones method for the face detection.  

For the second step, I will match the face image (using thecFisherfaces method) to an image in a PostgreSQL database, built from the Tufts Face Database.  For the closest matching image, I will use the Softmax activation function to assign a probability for the particular emotion.

Specifically:  

Load the image dataset into a PostgreSQL database  
Programatically 'label' the images  
Split into train and test sets  
Load each train image and:  

1. Resize the image  
2. Convert to grayscale  
3. Detect face boundary with cascade classifier  
4. Save the face boundary (ROI) to be passed to the second function.  
5. Call the FisherFaces classifier on the train roi image.  
6. Use the resulting trained model to predict on the test data.  

In order to protect the data, I will implement EncryptedPickle to protect the model and data while in use.  The database content and metadata are encrypted with the SQLCipher extension.

Images are converted to grayscale during the verification process.

## Modeling

This will be a supervised learning model as the data in my dataset is tagged.  I will choose a statistical approach to modelling and therefore fit a PCA model initially.  I will add a Feed Forward Neural Network, to increase model accuracy. Then I will test a pre-trained Convolutional Neural Network (CNN) to increase the model efficiency.  Using:keras, tensorflow, how many hidden nodes, how many layers? Using 'Softmax' to output probabilities as I'm using a multi-class variable.  I'm also using 'Dropout' to prevent the model from overusing a particular path.  And finally, implemented 'Early Stopping' to prevent overtraining the model.

## Analysis

Enter analysis here.

## Model-Evaluation

I implemented CUDA and optimized the modelling and network training to run in GPU as opposed to CPU.  That alone decreased modelling and training time by a factor of 12.

- <type> Classifier
- Accuracy: %
- Recall: %
- The most significant features include: .  

I split the data into train and test portions using scikit-learn's train_test_split.  After the model was trained on 70% of the data, I generated a confusion matrix to illustrat the accuracy of the model.

It is worthwhile to note, that while the Cascade Classifier correctly found all of the faces in the RGB dataset, the same classifier failed to detect any faces in the infrared dataset.


## Illustrations 

![Place Holder](img/lucy.jpeg)  

[![Interactive Neural Network][5]][6]  

[5]: img/NN_example.png  
[6]: http://playground.tensorflow.org   

## Deployment  

The model is deployed as a Flask app that users can upload a previously taken photo or by activating a web cam to send their current picture.  The app will then uses the model trained here to predict the emotion of the subject.  


## Data-Sources  

[![Tufts Face Database][1]][2]  

[1]: img/tufts_university.png 
[2]: http://tdface.ece.tufts.edu "Tufts Face Database: Request permission to use this dataset!!!"  

## Next-steps  
 
- Extend the application to webcams.  
- Extend the application to video feeds.  
- Add tracking for video.
- As the application is used by others, the data can be saved, tagged, and then used as future training data to better the model.

## Citations

Any publication using this database must reference to this

- Website: http://tdface.ece.tufts.edu/, (Check note)

- Paper: Panetta, Karen, Qianwen Wan, Sos Agaian, Srijith Rajeev, Shreyas Kamath, Rahul Rajendran, Shishir Rao et al. "A comprehensive database for benchmarking imaging systems." IEEE Transactions on Pattern Analysis and Machine Intelligence (2018).

For any enquires regarding the Tufts Face Database, contact: panettavisonsensinglab@gmail.com
