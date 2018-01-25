# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.


## Usage
```sh
jupyter notebook Traffic_Sign_Classifier.ipynb
```

### AWS EC2 GPU g2.x2large instance. Use udacity-carnd.

### Install docker-ce, nvidia-docker2
#### Pull a docker container with tensorflow gpu and python3

```sh
#sudo docker pull tensorflow/tensorflow:latest-gpu-py3
docker pull udacity/carnd-term1-starter-kit # for cpu
sudo docker build -t tf_py3_cv2 -f Dockerfile.gpu . #for gpu, build a docker image locally
```
#### Launch this workspace
```sh
#sudo nvidia-docker run -v `pwd`:/notebooks -it --rm -p 8888:8888  tensorflow/tensorflow:latest-gpu-py3
sudo nvidia-docker run -v `pwd`:/notebooks -it --rm -p 8888:8888  ttf_py3_cv2 #gpu
docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit # cpu
```

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image_train]: ./reports/train.png "Train"
[image_test]: ./reports/test.png "Test"
[image_valid]: ./reports/valid.png "Valid"
[new_data_speed]: ./new_signs/01_speed_30.jpg "New Speed"
[new_data_yield]: ./new_signs/13_yield.jpg "New Yield"
[new_data_priority]: ./new_signs/12_priority.jpg "New Priority"
[new_data_stop]: ./new_signs/14_stop.jpg "New Stop"
[new_data_no_ent]: ./new_signs/17_no_ent.jpg "New No Ent"
[new_data_keep_right]: ./new_signs/38_keep_right.jpg "New Keep Right"
[top7]: ./reports/top7.png "Priority Result"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/zhuangh/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410 .
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.



#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the data distributions, where x-axis is the the indices of labels and y-axis represents the size of samples for one category/label.

Train Set:
![Train Set][image_train]

Test Set:
![Test Set][image_test]

Validation Set
![Validation Set][image_valid]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color does not help the sign recogintion based on my experiment.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because zero-mean data will provide better conditioned distribution for numerical optimization during the training.

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x10 	|
| RELU					| 												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x10   |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x40 	|
| RELU					| 												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x40     |
| Fully connected		| 100x240      									|
| Dropout               | rate = 0.5                                    |
| Fully connected		| 240x84      									|
| Dropout               | rate = 0.5                                    |
| Softmax		        | 84x43      									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer discussed in the lecture. 
The batch size is 128.
The number of epochs is 51. 
The learning rate is 0.0008.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

0.7
Train Accuracy = 0.99833
Validation Accuracy = 0.94467
Test Accuracy = 0.92835

0.5
Train Accuracy = 0.99747
Validation Accuracy = 0.96054
Test Accuracy = 0.94125

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LeNet is choson since the lecture mentioned it has pretty good performance in this kind of task.

* What were some problems with the initial architecture?
The accuracy is not high enough, only around 89% for the test set.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I increased the filter depth to capture more pattern information from the inputs.

I added dropout for the fully-connected layers to avoid the overfitting.

* Which parameters were tuned? How were they adjusted and why?

Dropout rate. I set 0.9 then decreased to 0.7. 


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
I increase the filter depths for the convolution layers. 
The dropout is set to 0.7. It improved the accuracy of test set from 89% to 95%. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


![alt text][new_data_speed] ![alt text][new_data_yield] ![alt text][new_data_priority] ![alt text][new_data_stop] ![alt text][new_data_no_ent] ![alt text][new_data_keep_right] 


This is wrong prediction when we use unaugmented data set. 
![alt text][top7].

After adding the generated data. 


 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 


