# **Traffic Sign Recognition** 

---
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---

## Usage of [my code](https://github.com/zhuangh/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

```sh
jupyter notebook Traffic_Sign_Classifier.ipynb
```


#### Pull a docker container with tensorflow gpu and python3
CPU: use udacity-carnd. 
```sh
docker pull udacity/carnd-term1-starter-kit # for cpu
```

For the use of GPU, AWS EC2 GPU g2.x2large instance can be used. Install docker-ce, nvidia-docker2 on the instance. Plus, I need to build a docker image locally to support cv2, etc.

```sh
sudo docker build -t tf_py3_cv2 -f Dockerfile.gpu .
```

#### Launch this workspace
CPU only
```sh
docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit
```

GPU 
```sh
sudo nvidia-docker run -v `pwd`:/notebooks -it --rm -p 8888:8888  tf_py3_cv2 
```

---

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
[all_news]: ./reports/all_news.png
[top7_new]: ./reports/top7_new.png "Priority Result"
[new1]: ./reports/new1.png 
[new2]: ./reports/new2.png 
[new3]: ./reports/new3.png 
[new4]: ./reports/new4.png 
[new5]: ./reports/new5.png 
[new2_new]: ./reports/new2_new.png 
[rgb_set]: ./reports/rgb_set.png "RGB Images"
[gray_sets]: ./reports/gray_sets.png "Gray Images"
[norm_set]: ./reports/norm_set.png "Norm Images"
[scaled]: ./reports/scaled.png 
[scaled2]: ./reports/scaled2.png
[scaled3]: ./reports/scaled3.png
[scaled4]: ./reports/scaled4.png
[aug_set_dist]: ./reports/aug_set_dist.png
[perm]: ./reports/perm.png
[lenet]: ./reports/lenet.png
[confusion_matrix]: ./reports/confusion_matrix.png

### Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/zhuangh/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of the training set is 34799.
* The size of the validation set is 4410.
* The size of the test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.



#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the data distributions, where x-axis is the indices of labels and y-axis represents the size of samples for one category/label.

Train Set Distribution

![Train Set][image_train]

Test Set Distribution

![Test Set][image_test]

Validation Set Distribution

![Validation Set][image_valid]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the color does not help the sign recognition based on my experiment. Here the corresponding examples of traffic sign images before and after grayscaling.

The examples of RGB images are,

![alt text][rgb_set]

The corresponding grayscale images are shown as follows, 

![alt text][gray_sets]

As a last step, I normalized the image data because zero-mean data will provide better-conditioned distribution for numerical optimization during the training. The equation is 
```sh
gray_image_normalized = (gray_image - 128)/ 128
```
The normalized images are shown as follows

![alt text][norm_set]




I decided to generate additional data because I found a mispredicted example from the new signs I downloaded from the website. Due to the time constraint of this project, I used only the scaling and cropping method to generate extra data to help recognize this kind of images. Here the examples of original images and the augmented images:

![alt text][scaled] 

![alt text][scaled2]

![alt text][scaled3]

![alt text][scaled4]

Augmented train set has the sample distribution as

![alt text][aug_set_dist] 

Compared to the original distribution of train set.

![Train Set][image_train]

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
| Dropout               | keep prob = 0.5                                    |
| Fully connected		| 240x84      									|
| Dropout               | keep prob = 0.5                                    |
| Softmax		        | 84x43      									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer discussed in the lecture. 

The batch size is 128.

The number of epochs is 51. 

The learning rate is 0.0008.

The keep probability of dropout is 50.0%.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.98%.
* validation set accuracy of 96.12%.
* test set accuracy of 95.20%.
* new signs accuracy of 100.00% (80% without augmented train sets)

Here are the configuration and the accuracy performance I record the trials.

* Configuration and Performance Table

| Configuration			        |     Performance        					| 
|:---------------------:|:---------------------------------------------:|
| Dropout's keep prob=0.7 + Grayscale Data Sets | Train Accuracy = 0.99833, Validation Accuracy = 0.94467, Test Accuracy = 0.92835|
| Dropout's keep prob=0.5 + Grayscale Data Sets | Train Accuracy = 0.99747, Validation Accuracy = 0.96054, Test Accuracy = 0.94125|
| Dropout's keep prob=0.5 + Normalized Grayscale Data Sets | Train Accuracy = 0.99974, Validation Accuracy = 0.97483, Test Accuracy = 0.95408|
| Dropout's keep prob=0.5 + Normalized Grayscale Augmented Data Sets | Train Accuracy = 0.99976, Validation Accuracy = 0.96122, Test Accuracy = 0.95202|

The training performance figure is attached.

![alt text][perm]


#### Iterative approach was chosen

* What was the first architecture that was tried and why was it chosen?

Answer: I started with LeNet since the lecture mentioned it has pretty good performance in this kind of task.

![alt text][lenet]

* What were some problems with the initial architecture?

Answer: The accuracy is not high enough, only around 89% for the test set.

* How was the architecture adjusted and why was it adjusted? 

Answer: 

I increased the filter depth to capture more pattern information from the inputs. 

I added dropout for the fully-connected layers to avoid the overfitting.

* Which parameters were tuned? How were they adjusted and why?

Answer: I tuned the Dropout's keep probability. I set 0.7 then decreased it to 0.5. Check the Configuration and Performance Table table I added above. 
 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][all_news]

This is the wrong prediction without the augmented data set. 

![alt text|140%][new2]

After adding the generated data. 

![alt text][new2_new]

 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction with augmented data sets:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep          		| Stop sign   									| 
| Priority    			| Priority   									|
| Stop					| Stop						    				|
| Yield	      		    | Yield	    					 				|
| No Entry			    | No Entry     				        			|


Without augmented data, the model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep          		| Stop sign   									| 
| Priority    			|  Children crossing  							|
| Stop					| Stop						    				|
| Yield	      		    | Yield	    					 				|
| No Entry			    | No Entry     				        			|


This is the reason why I added scaled images as augmented data samples to help the deep neural network to get trained.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

Following figures should the top 5 softmax probabilities.

![alt text][new1]

![alt text][new2_new]

![alt text][new3]

![alt text][new4]

![alt text][new5]

The confidence of each prediction is pretty high.


## Summary

Based on the test set. 

The Precision of the model is 93.8%.

The Recall Score of the model 95.2%.

The confusion matrix of the model 

![alt text][confusion_matrix]

## Further Steps:

* Add more diversity samples to the train set. Due to the time constraint, I only added the data with different scaling factors. Actually, we can rotate the images, use different blur versions, and so on. 

* Balance the sample data distributions in the train set.

* Train the IJCNN'11 [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) mentioned.

* Visualization of the neural network's state. 


