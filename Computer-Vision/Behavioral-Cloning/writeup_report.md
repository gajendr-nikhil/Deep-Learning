#**Behavioral Cloning (Deep Learning)** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_image.jpg "Center Image"
[Recovery_left_1]: ./images/Recovery_left_1.jpg "Recovery Left 1"
[Recovery_left_2]: ./images/Recovery_left_2.jpg "Recovery Left 2"
[Recovery_left_3]: ./images/Recovery_left_3.jpg "Recovery Left 3"
[Recovery_left_4]: ./images/Recovery_left_4.jpg "Recovery Left 4"
[Recovery_right_1]: ./images/Recovery_right_1.jpg "Recovery Right 1"
[Recovery_right_2]: ./images/Recovery_right_2.jpg "Recovery Right 2"
[Recovery_right_3]: ./images/Recovery_right_3.jpg "Recovery Right 3"
[Recovery_right_4]: ./images/Recovery_right_4.jpg "Recovery Right 4"
[Recovery_left_3_flipped]: ./images/Recovery_left_3_flipped.jpg "Recovery Left 3 flipped"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 2x2 filter sizes and depths between 24 and 256 (model.py lines 94-102) 

The model includes RELU layers to introduce nonlinearity (code lines 94-111), the data is normalized in the model using a Keras lambda layer (code line 91), and top 70 pixels and bottom 25 pixels of data is cropped in the model using Keras Cropping2D layer (code line 92).  

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 103, 107, and 109). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 17-66). Data from all three cameras (left, right, and center) is used and each camera image has been horizontally flipped in order to augment more data. And, Python generators are used to generate data for training rather than storing the training data in memory.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 113). I also tuned the angle correction for left and right camera images to 0.18.
Left image angle = Center image angle + correction
Right image angle = Center image angle - correction

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. Each camera image has been horizontally flipped in order to augment more data.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a network architecture powerful than LeNet, train that network on the collected training data. Then, run the model in simulator to see how well performs to keep the car within the track. Based on the failures in the simulator fine tune the network parameters (for example: correction angle, dropout probability, Max/Average pooling), add higher level convolution layer to identify shapes even better, and finally modify the density of fully connected layers.

My first step was to use a convolution neural network model similar to the architecture published by Autonomous vehicle team in Nvidia, I thought this model might be appropriate because this is the network they use for training a real car to drive autonomously.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model was under performing in following 2 cases:
(1) In track 1, model failed to recognize thin yellow marking lanes on the track, and assumed that the pavement beyond the yellow lanes is the end of track, and it car started driving over the yellow lanes.
(2) In track 2, model failed to recognize darker shadows in some of the steep turns.
(3) In track 2, model also assumed that after a steep turn, track will be a straight line.
I think in the first cases model failed to recognize finer features in the image, so I reduced the convolution filter size to 3x3. In the second case, model failed to recognize higher level shapes such as trees, mountains, bridge side by the track, so I added 2 more layers of convolution on top of Nvidia's convolution layers. Third case is a genuine overfitting case. To combat the overfitting, I modified the model to have dropout layers between fully connected layers.

Then I increased the density of fully connected layers by doing trial and error, and chosing the one that gives least mse on validation set

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I thought this is because of lack of more generalized data. I collected more data from each track (Data collection techniques are explained below)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 89-112) consisted of a convolution neural network with the following layers and layer sizes

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 BGR image   							| 
| Convolution 3x3     	| (2,2) subsample, same padding, outputs 80x160x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 40x80x24 				|
| Convolution 3x3	    | (2,2) subsample, same padding, outputs 20x40x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 10x20x36 					|
| Convolution 3x3	    | (2,2) subsample, same padding, outputs 5x10x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x5x48 					|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 2x5x64 	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 2x5x128 	|
| RELU					|					
| Convolution 2x2	    | 1x1 stride, same padding, outputs 2x5x256 	|
| RELU					|					
| Flatten				| input 2x5x256,  outputs 2560 					|
| Fully connected		| Inputs = 2560 (2x5x256) Outputs = 500 			|
| Dropout				| With keep probability = 0.5					|
| Fully connected		| Inputs = 500, Outputs = 200 					|
| Dropout				| With keep probability = 0.5					|
| Fully connected		| Inputs = 200, Outputs = 20 					|
| Dropout				| With keep probability = 0.5					|
| Fully connected		| Inputs = 20, Outputs = 1 					|
| mse				| etc.        									|


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive within the track and middle of the road. These images show what a recovery looks like starting from left side to middle of the road :

![alt text][Recovery_left_1]
![alt text][Recovery_left_2]
![alt text][Recovery_left_3]
![alt text][Recovery_left_4]

These images show what a recovery looks like starting from right side to middle of the road :
![alt text][Recovery_right_1]
![alt text][Recovery_right_2]
![alt text][Recovery_right_3]
![alt text][Recovery_right_4]

Then I collected more imgaes by driving in opposite direction of where the simulator starts.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would generalize the model pretty will without overfitting to the any of the tracks that are being trained on For example, here is an image that has then been flipped:

![alt text][Recovery_left_3]
![alt text][Recovery_left_3_flipped]

After the collection process, I had 44001 number of data points. I then preprocessed this data by using all threee (center, left, right) images for each data point. Then I horizontally flipped each image as well leading to generate 6 times the data points which is 264006 images.


I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 9 as evidenced by doing lot of trail and error on the entire and subset of the data. I used an adam optimizer so that manually training the learning rate wasn't necessary.
