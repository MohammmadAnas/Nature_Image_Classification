# Nature_Image_Classification
HI this is an Image Classification Model.

## In this project we are trying to build a image classification model to classify Images of Nature.

## Data

Data is available in the data folder. The train.csv file contains the training set.

Dataset reference: https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction

Note: In order to download data from kaggle, you need to create an account on kaggle.com and download the kaggle.json file from your account. Then, you need to put this file in the same folder as the place where you are training the model.

there are six classes of Nature Images:

1. Buildings
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

## Dataset preparation

I trained multiple model for the project. The dataset is subdivided into catogeries for training within each categories for training, validation and test.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

## PLEASE NOTE: The model's tflite and h5 form and the dataset can be found under the releases which is on the right side of the screen just below the about heading.

------------------------------------------------------------------------------------------------------------------------------------------------------------

#### Run with Docker

You can run the project with Docker. To do so, you need to have Docker installed on your machine. Then, you need to build the image with the following command:

```bash
sudo docker build -t intel-model .
```
Now run the docker image using the following command:

```bash
sudo docker run -it --rm -p 8080:8080 intel-model:latest
```



As the model is also availablea on AWS Elastic Beanstalk you can test it using test.py. All you need to do is to run the file.

If you want to use some other image then change the url to your desired image.
