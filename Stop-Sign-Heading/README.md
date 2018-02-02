# lvl5 Stop Sign Heading Problem

## Challenge

Predict the heading (in degrees) of the stop sign (relative to the camera).

## Dataset

We've provided a total of 4830 images for training, testing and validation.  The image set spans the range of [-80,80] degrees of heading at increments of 1 degree, with 30 images sampled at each heading..  A sign has a relative heading of 0 degrees, when it is directly facing the camera.

Images are located in `images/sign_<HEADING_DEGREES>_XX.jpg`

## Network Architecture (Solution)
I used the deep learing approach to solve predict the heading of the stop sign.

The final model architecture consisted of a convolution neural network with the following layers and layer sizes


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 275x275x3 RGB image   							| 
| Convolution 3x3     	| (2,2) subsample, same padding, outputs 138x138x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 69x69x24 				|
| Convolution 3x3	    | (2,2) subsample, same padding, outputs 35x35x36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 18x18x36 					|
| Convolution 3x3	    | (2,2) subsample, same padding, outputs 9x9x48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48 					|
| Flatten				| input 5x5x48,  outputs 1200 					|
| Fully connected		| Inputs = 1200 (5x5x40) Outputs = 200 			|
| Dropout				| With keep probability = 0.1					|
| Fully connected		| Inputs = 200, Outputs = 20 					|
| Dropout				| With keep probability = 0.1					|
| Fully connected		| Inputs = 20, Outputs = 1 					|
| mse				| etc.        									|


My model consists of a convolution neural network with 3x3 filter with depths between 24 and 48, and Max pooling layers with 2x2 filters and stride. The model includes RELU layers to introduce nonlinearity the data is normalized in the model using a Keras lambda layer. The model contains dropout layers in order to reduce overfitting. The model was trained and validated on different data sets to ensure that the model was not overfitting. And, Python generators are used to generate data for training rather than storing the training data in memory.

The model used an adam optimizer with mean squared error begin the loss, so the learning rate was not tuned manually.

The overall strategy for deriving a model architecture was to start with a network architecture powerful than LeNet, train that network on the collected training data. Then, run the model on test set to see how well performs i.e., mean squared error on the test set. Based on the MSE on the test set, fine tune the network parameters (for example: dropout probability, Max/Average pooling), add/delete higher level convolution layer to identify shapes even better, and finally modify the density of fully connected layers.

In order to gauge how well the model was working, I split my images into a training, validation set, and test set. I randomly shuffled the given images and put 20% of them into test set. Out of the remaining images, 80% is used as training set and 20% is used as validation set.

I found that my first model was overfitting. In order to avoid that, I reduced the complexity of the network and added dropout layers between fully connected layers.

## Improvements
I tried to augment data by changing brightness, and rotating the images. That seems to not help. The best way to get MSE <= 1 is by generating more real-ish data i.e., in the given images, find the best homography matrix (using RANSAC) between images with heading 0 degrees and the rest, QR decompose the Homography matrix to get the rotation matrix, then finally use this rotation matrix to rotate all the images with 0 degree heading to any heading (-80, 80) we want. 
Now that we have more images, we can retrain the same network or add few more convolution and max pooling layers to reduce the MSE.