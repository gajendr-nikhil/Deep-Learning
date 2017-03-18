#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./writeup_images/bar_charts.png "Probablility distribution of Traffic Sign"
[image2]: ./writeup_images/augment_data.png "Augmented Images"
[image3]: ./writeup_images/0.png "Augmented Images"
[image4]: ./writeup_images/1.png "Augmented Images"
[image5]: ./writeup_images/2.png "Augmented Images"
[image6]: ./writeup_images/3.png "Augmented Images"
[image7]: ./writeup_images/4.png "Augmented Images"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of test set is 12630 images
* The shape of a traffic sign image is (32, 32, 3) unsigned integer numpy array
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth and fifth code cell of the IPython notebook.  

One image per class is displayed with title being 'class label' and 'count of samples of that class'. Fifth code cell displays probability distribution of each class label in training, validation, testing, and entire dataset respectively (Also attached below). As we can see from the below probablility distribution, each class have different distribution much like real world, where we see some traffic signs more than the others. That probablity distribution of classes in each of the datasets is unchanged so that we are inline with real world.


![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the sixth and seventh code cell of the IPython notebook.

I decided to rotate, increase/decrease intensity of, Zoom-in/Zoom-out of the images, and finally normalize the images because in the real world, camera recorded images will not always be clear and clean. Many of them will be rotated, with different pixel intensity depending on time of the day, closer or farther from camera. To mimic the real world behaviour, I decided to apply the above augmentation techniques.

I also normalized all the datasets to ensure that features used to train the network are of zero mean and unit variance. Because of that, optimizer can use the same learning rate for all the weights and bias vectors to achieve minimum loss.

Here is an example of a traffic sign image before and after augmenting with above mentioned techniques.

![alt text][image2]

As a last step, I normalized the image data because ...

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Data given to us has already been split into trianing, validation, and test set.
Data has been read in second code cell of the IPython notebook.

I used Keras on top of tensor flow to augment the data as described in the previous section because to make the model more robust to changes in real-world conditions such as viewing the sign from an angle, occlusions, brightness variation etc.,

I trained for 15 Epochs with each Epoch training on 34799 and validating on 4410 images. Therefore, my final training set had 521985 number of images. My validation set and test set had 66150 and 12630 number of images. Test set used only once at the end, post model training, to test the performance of the model

The seventh code cell of the IPython notebook contains the code for augmenting the data set.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eighth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 					|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 8x8x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x128 					|
| Convolution 1x1	    | 1x1 stride, same padding, outputs 4x4x256 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 2x2x256 					|
| Flatten				| input 2x2x256,  outputs 1024 					|
| Fully connected		| Inputs = 1024 (2x2x256) Outputs = 64 			|
| Dropout				| With keep probability = 0.5					|
| Fully connected		| Inputs = 64, Outputs = 43 					|
| Softmax				| etc.        									|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth and ninth cell of the ipython notebook. 

To train the model, I used softmax with binary cross entropy loss. I used RMSprop optimizer with default learning rate of 0.001. I used a batch size of 64 for 15 Epochs. These numbers were chosen after many trial and error iterations of using different values for batch size(= 16, 32, 64, 128, 256) and Epochs(=5, 10, 15, 25, 30, 40). This helped to increase the validation accuracy by atleast 1%. I also used dropout regularization technique with drop probability of 65%



####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the tenth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.45%
* validation set accuracy of 99.66%
* test set accuracy of 99.58%

If an iterative approach was chosen:
* The first architecture that I tried was LeNet architecture that had two 5x5 convolutional layers followed by three fully connected layers. It gave a validation accuracy of ~89%.
* Since the LeNet architecture was built to recognize only 10 classes and we have 43 classes, model was not able to capture all the required features with just two convolution layers. Obviously we need some changes with the architecture.
* I added two more convolution layers, which gets a total of 4 convolution layers. 3 of them have a filter of size 3x3, and the fourth convolution layer has a filter of size 1x1. I also increased the number of feature maps to 32, 64, 128, and 256 respectively for each convolution layer. Finally, reduced the number of feature maps of first fully connected layer to 64, and increased the number of feature maps of final fully connected layer to 43. This increased the validation accuracy to 96.76% after 25 Epochs, but it decreased after that when Training accuracy kept on increasing upto 99%. This indicated that the model is overfitting.
* Then I added a dropout layer followed by the first fully connected layer with drop probability of 65% (This was chosen after many trying different values for many iterations). This increased the validation accuracy to ~98.33%.
* Finally, I tried data augmentation techniques, on the training and validation data, such as rotation, intensity variation, zoom-in/zoom-out. This got me the best validation accuracy of 99.66%.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web after transforming them with random rotation angle and gaussian noise:

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6] 

![alt text][image7]

The first image might be difficult to classify because it's been rotated and gaussian noise been added.

The second image might be difficult to classify because it's been rotated as well as it has shaded areas on the right and bottom part of the image and gaussian noise been added.

The third image might be difficult to classify because it's been rotated, tilted, brightened, and gaussian noise is added.

The fourth image might be difficult to classify because it's been rotated, reduced brightness, and gaussian noise is added.

The fifth image might be difficult to classify because it's been rotated a little bit, and a lot of gaussian noise is added.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 12th, 13th, and 14th cell of the Ipython notebook.

Here are the results of the prediction:

| Image									| Prediction									| 
|:-------------------------------------:|:---------------------------------------------:| 
| 30 km/h								| 30 km/h   									| 
| Stop Sign    							| Stop Sign 									|
| No Entry								| No Entry										|
| Right of way at the next intersection | Right of way at the next intersection			|
| Go straight							| Go straight									|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 99.55%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the 30 km/h image, the top five soft max probabilities are (It is noticeable that a large probability is given to correct label, and the next 4 labels are closely related to each other)

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99886715			| 30km/h										| 
| 1.1138817e-19			| 20km/h 										|
| 3.2561287e-36			| 50km/h										|
| 0.0					| 60km/h						 				|
| 0.0					| 70km/h										|


For the Stop Sign image, the top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.98344165			| Stop Sign										| 
| 0.0					| 20km/h 										|
| 0.0					| 30km/h										| 
| 0.0					| 50km/h										|
| 0.0					| 60km/h						 				|

For the No Entry image, the top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0					| No entry										| 
| 0.0					| 20km/h 										|
| 0.0					| 30km/h										| 
| 0.0					| 50km/h										|
| 0.0					| 60km/h						 				|

For the Right of way at the next intersection image, the top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99991703			| Right-of-way at the next intersection			| 
| 2.0257838e-09			| Beware of ice/snow							|
| 7.55653e-21			| 60km/h						 				|
| 4.2894068e-33			| 80km/h						 				|
| 2.4159068e-33			| Pedestrians					 				|

For the Go straight image, the top five soft max probabilities are

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0					| Ahead only									| 
| 0.0					| 20km/h 										|
| 0.0					| 30km/h										| 
| 0.0					| 50km/h										|
| 0.0					| 60km/h						 				|
