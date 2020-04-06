<p align="center"><img src="img/YBCheeri-yo.jpg" title="Cheeri-yo" width="600" height="400" /></a></p>  



# cheeri-yo  

Emotion detection application utilizing OpenCV and a whole lot of research.  http://www.notsurewhereitgoes.io  


## Context

I often participate in many startup businesses i.e. restaurants, events, and such.  I’ve always wondered how to tell when someone is truly enjoying themselves, or when they are just trying to be nice.  This is always a concern for a business when selling/presenting something new for the first time, whether a new dish at a restaurant, a new lecture style at school, or even when viewing a real estate property.  

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

I have over 200 GB of data in two datasets: first is the industry standard Cohn-Kanade (CK and CK+) database.  I will use this database to train my model.  Once the model is trained, I will test it against the second dataset.

The second dataset was collected by Tufts University, and referenced on Kaggle.  Tufts Face Database is the latest, most comprehensive, large-scale (over 10,000 images, 74 females + 38 males, from more than 15 countries with an age range between 4 to 70 years old) face dataset that contains 6 image modalities: visible, near-infrared, thermal, computerized sketch, a recorded video, and 3D images.  I will be specifically using the TDRGBE and TDIRE image datasets for testing.

TDRGBE: The images were captured using a NIKON D3100 camera. Each participant was asked to pose with a neutral expression, a smile, eyes closed, exaggerated shocked expression, and finally wearing sunglasses.

TDIRE: Images were captured using a FLIR Vue Pro camera with participants posing as in the TDRGBE dataset.  

## Data-Preparation

There are basically two stages for my project: 1) face detection and 2) emotional detection.  I will train a cascade classifier from OpenCV and use the Viola & Jones method for the face detection.  For the second step, I will match the face image (using Eigenfaces or Fisherfaces method) to an image in a SQLite database, built from the Tufts Face Database.  For the closest matching image, I will assign a probability for the particular emotion.

In order to protect the data, I will implement EncryptedPickle to protect the model and data while in use.  The database content and metadata will be encrypted with the SQLCipher extension.

## Modeling

This will be a supervised learning model as the data in my dataset is tagged.  I will choose a statistical approach to modelling and therefore fit a PCA model initially.  I will add a Feed Forward Neural Network, to increase model accuracy. Then I will test a Convolutional Neural Network to increase the model efficiency.  

## Analysis


## Model-Evaluation

- <type> Classifier
- Accuracy: %
- Recall: %
- The most significant features include: .  

I will initially split the data from the Cohn-Kanade dataset into train and test portions.  After the model is trained, I will generate classification reports to test the accuracy while tuning the model.  After the model is trained, I will refit the data to the full Cohn-Kanade dataset and test against the unseen data in the Tufts Face Database.  If possible, I’d like to test against both the TDRGBE and the matching infrared dataset TDIRE.


## Illustrations 

![Pumps vs Population](img/pump_status_sns.jpg)  
![Pumps vs Population](img/pump_pop.jpg)  
![Installers](img/repairs_installers.jpg)     
![Well Locations](img/pump_locations.jpg)  
![Dropped needs repair](img/pump_locations_noyellow.jpg)  
![Model Image](img/model_mod.png)  
![Confusion Matrix](img/confusion.jpg)  

## Deployment  

The model will be deployed as a Flask app that users can upload a previously taken photo or by activating a web cam to send their picture.  The app will then use the model to predict the emotion of the person.  As the model is used by others, their data can be saved, tagged, and then used as future training data.




## Data-Sources

[![Cohn-Kanade (CK and CK+) database][1]][2]

[1]: img/pitt_edu_logo.jpg   
[2]: https://www.pitt.edu/ "Contact: Yaohan Ding YAD30@pitt.edu"  

[![Tufts Face Database][3]][4]

[3]: img/tufts_university.png 
[4]: http://tdface.ece.tufts.edu "You must request permission to use this dataset!!!"  


## Next-steps  
 

## Citations

Any publication using this database must reference to this

- Website: http://tdface.ece.tufts.edu/, (Check note)

- Paper: Panetta, Karen, Qianwen Wan, Sos Agaian, Srijith Rajeev, Shreyas Kamath, Rahul Rajendran, Shishir Rao et al. "A comprehensive database for benchmarking imaging systems." IEEE Transactions on Pattern Analysis and Machine Intelligence (2018).

For any enquires regarding the Tufts Face Database, contact: panettavisonsensinglab@gmail.com

- Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial
expression analysis. Proceedings of the Fourth IEEE International Conference
on Automatic Face and Gesture Recognition (FG'00), Grenoble, France, 46-53.
- Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I.
(2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression
dataset for action unit and emotion-specified expression. Proceedings of the
Third International Workshop on CVPR for Human Communicative Behavior
Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.

- Request access, contact: Yaohan Ding YAD30@pitt.edu.
