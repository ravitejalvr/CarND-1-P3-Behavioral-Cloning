# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[img_gid]: ./images/driving.gif "Driving GIF"
[img_model]: ./images/model.png "Model Visualization"
[img_map]: ./images/map01.jpg "Map Track 1"
[img_video_track1]: ./images/video-track_01.gif "Video Track 1"
[img_video_track2]: ./images/video-track_02.gif "Video Track 2"
[img_steering]: ./images/steering_angle_distribution.png "steering angle distribution"
[img_carcam_LCR]: ./images/img_carcam_LCR.png "img_carcam_LCR"
[img_data_augment]: ./images/data_augment.png "img_data_augment"

![img_gid]

## [Rubric Points](https://review.udacity.com/#!/rubrics/432/view)
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model used in this project is based on the model by Nvidia discussed in "[End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)".

![img_model]

| Layer (type) | Output Shape | Param # | 
|---|---|---|
|cropping2d_1 (Cropping2D)  | (None, 80, 320, 3) | 0  |       
|lambda_1 (Lambda)          |  (None, 80, 320, 3)  | 0      |   
|conv2d_1 (Conv2D)          |  (None, 38, 158, 24) | 1824   |   
|conv2d_2 (Conv2D)          |  (None, 17, 77, 36)  | 21636  |   
|conv2d_3 (Conv2D)          |  (None, 7, 37, 48)   | 43248  |   
|conv2d_4 (Conv2D)          |  (None, 5, 35, 64)   | 27712  |   
|conv2d_5 (Conv2D)          |  (None, 3, 33, 64)   | 36928  |   
|flatten_1 (Flatten)        |  (None, 6336)        | 0      |   
|dense_1 (Dense)            |  (None, 100)         | 633700 |   
|dense_2 (Dense)            |  (None, 50)          | 5050   |   
|dense_3 (Dense)            |  (None, 10)          | 510    |   
|dense_4 (Dense)            |  (None, 1)           | 11     |   


Total params: 770,619.0

Trainable params: 770,619.0

Non-trainable params: 0.0

When comparing the used model to the one by Nvidia it is clear that the Nvidia-model has more parameters (7,629,687.0). The number of parameters was reduced by skipping the first Dense-Layer ```model.add(Dense(1164, activation='relu'))```.


#### 2. Attempts to reduce overfitting in the model

In an attempt to reduce overfitting L2 regularization to the model (dense layers) to improve validation loss.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For training only track 1 has been used. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

After reading the paper by Nvidia about [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) my first step was to use the architecture presented in the paper.

Before starting to work with the data I had a look how the steering angles are distributed in my data.

![img_steering]

As expected, a lot of very light turns (+/-0.02) and nearly no sharp turns (>+/-0.3) the +/-1 values are from the sharp turns when recovering.


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified by dropping one dense layer and adding L2 regularization to the model.

I implemented augmentation as suggested by the paper and @mohankarthik.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (marked red in the image below) to improve the driving behavior in these cases. 

![img_map]



At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
As only track 1 has been used to collect data, the interesting part was to know how good the model has generalized driving in the center so I also tested it on track 2. It crashed a few times (when going up a slope) but overall a good result.

[![img_video_track1]](https://drive.google.com/file/d/0B7W453GPQ0J3eHIzMkhUcEJLRHc/view)

[![img_video_track2]](https://drive.google.com/file/d/0B7W453GPQ0J3QVkwR2tUQTF1R2M/view)

#### 2. Final Model Architecture

The final model architecture consisted of a cropping2d layer to cut off unneeded pixels and a lamda layer to normalize the input data (preprocessing). These layers are followed by five convolution layers. After the convolution layers the output is flattened and send through four dense layers.

| Layer (type) | Output Shape | Param # | 
|---|---|---|
|cropping2d_1 (Cropping2D)  | (None, 80, 320, 3) | 0  |       
|lambda_1 (Lambda)          |  (None, 80, 320, 3)  | 0      |   
|conv2d_1 (Conv2D)          |  (None, 38, 158, 24) | 1824   |   
|conv2d_2 (Conv2D)          |  (None, 17, 77, 36)  | 21636  |   
|conv2d_3 (Conv2D)          |  (None, 7, 37, 48)   | 43248  |   
|conv2d_4 (Conv2D)          |  (None, 5, 35, 64)   | 27712  |   
|conv2d_5 (Conv2D)          |  (None, 3, 33, 64)   | 36928  |   
|flatten_1 (Flatten)        |  (None, 6336)        | 0      |   
|dense_1 (Dense)            |  (None, 100)         | 633700 |   
|dense_2 (Dense)            |  (None, 50)          | 5050   |   
|dense_3 (Dense)            |  (None, 10)          | 510    |   
|dense_4 (Dense)            |  (None, 1)           | 11     |   

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded two laps on track one using center lane driving clockwise and two laps anticlockwise. I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to take sharp corners.

![img_carcam_LCR]

After the collection process (including the 3 cameras), I had ~39.000 number of data points. I then preprocessed this data by cutting off the top 60px and the bottom 20px.

To augment the data, I flipped images and angles randomly, added a change in brightness and moved the images a bit off center by random thinking that this would help the NN to create a more general model.

![img_data_augment]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
