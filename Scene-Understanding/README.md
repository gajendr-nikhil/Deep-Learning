# Semantic Segmentation
### Introduction
In this project, I label the pixels of a road in images using a Fully Convolutional Network (FCN). FCN has been trained on the [Kitti Road dataset](http://www.cvlibs.net/download.php?file=data_road.zip).

[//]: # (Image References)
[image1]: ./images/vgg16.png "VGG 16"
[image2]: ./images/fcn_8.png "FCN-8 architecture"
[image3]: ./images/um_000045.png "training sample data"
[image4]: ./images/um_lane_000045.png "training sample lable"
[image5]: ./images/umm_000077.png "Result 1"
[image6]: ./images/uu_000093.png "Result 2"

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Network Architecture
FCN-8 architecture built using VGG16 network. Two fully connected layers are replaced by 1x1 convolution to preserve spatial information. Network performance is improved using skip connections i.e., performing 1x1 convolutions on layer3 and layer4, and adding them elementwise to upsampled layer.


![alt text][image1]
![alt text][image2]

### Data augmentation
Training set only contained 290 images, I augmented the data by horizontally flipping images to avoid overfitting.
![alt text][image3]
![alt text][image4]

### Hyperparameters
I tried different values for the hyperparameters and got the best result for the following values.

| Hyperparameter         		|     value	        					| 
|:---------------------:|:---------------------------------------------:| 
|Epochs| 20 |
|Batch size| 1 |
|Learning rate| 1e-4 |
|Dropout keep probability| 0.80 |

### Test Result
As you can see from the below image, FCN is able to distinguish the road from other things in the image pretty accurately.

![alt text][image5]
![alt text][image6]