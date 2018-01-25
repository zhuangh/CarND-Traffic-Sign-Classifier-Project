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
sudo nvidia-docker run -v `pwd`:/notebooks -it --rm -p 8888:8888  ttf_py3_cv2 
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

As a first step, I decided to convert the images to grayscale because the color does not help the sign recogintion based on my experiment. Here the corresponding examples of traffic sign images before and after grayscaling.
![alt text][rgb_set]

The grayscaled images are shown as follows, 

![alt text][gray_sets]

As a last step, I normalized the image data because zero-mean data will provide better conditioned distribution for numerical optimization during the training. The equation is 
```sh
gray_image_normalized = (gray_image - 128)/ 128
```
The normalized images are show as follows
![alt text][norm_set]




I decided to generate additional data because I found an mispredicted example from the new signs I downloaded from the website. Due to the time constraint of this project, I used only the scaling and cropping method to generate extra data to help recognize this kind of images. Here the examples of original images and the augmented images:
![alt text][scaled] 
![alt text][scaled2]
![alt text][scaled3]
![alt text][scaled4]

Augmented train set has the sample distribution as
![alt text][aug_set_dist]
label  0 (0, b'Speed limit (20km/h)')  sample count  180
label  1 (1, b'Speed limit (30km/h)')  sample count  1980
label  2 (2, b'Speed limit (50km/h)')  sample count  2010
label  3 (3, b'Speed limit (60km/h)')  sample count  1260
label  4 (4, b'Speed limit (70km/h)')  sample count  1770
label  5 (5, b'Speed limit (80km/h)')  sample count  1650
label  6 (6, b'End of speed limit (')  sample count  360
label  7 (7, b'Speed limit (100km/h')  sample count  1290
label  8 (8, b'Speed limit (120km/h')  sample count  1260
label  9 (9, b'No passing')  sample count  1320
label  10 (10, b'No passing for vehic')  sample count  1800
label  11 (11, b'Right-of-way at the ')  sample count  1170
label  12 (12, b'Priority road')  sample count  1890
label  13 (13, b'Yield')  sample count  1920
label  14 (14, b'Stop')  sample count  690
label  15 (15, b'No vehicles')  sample count  540
label  16 (16, b'Vehicles over 3.5 me')  sample count  360
label  17 (17, b'No entry')  sample count  990
label  18 (18, b'General caution')  sample count  1080
label  19 (19, b'Dangerous curve to t')  sample count  180
label  20 (20, b'Dangerous curve to t')  sample count  300
label  21 (21, b'Double curve')  sample count  270
label  22 (22, b'Bumpy road')  sample count  330
label  23 (23, b'Slippery road')  sample count  450
label  24 (24, b'Road narrows on the ')  sample count  240
label  25 (25, b'Road work')  sample count  1350
label  26 (26, b'Traffic signals')  sample count  540
label  27 (27, b'Pedestrians')  sample count  210
label  28 (28, b'Children crossing')  sample count  480
label  29 (29, b'Bicycles crossing')  sample count  240
label  30 (30, b'Beware of ice/snow')  sample count  390
label  31 (31, b'Wild animals crossin')  sample count  690
label  32 (32, b'End of all speed and')  sample count  210
label  33 (33, b'Turn right ahead')  sample count  599
label  34 (34, b'Turn left ahead')  sample count  360
label  35 (35, b'Ahead only')  sample count  1080
label  36 (36, b'Go straight or right')  sample count  330
label  37 (37, b'Go straight or left')  sample count  180
label  38 (38, b'Keep right')  sample count  1860
label  39 (39, b'Keep left')  sample count  270
label  40 (40, b'Roundabout mandatory')  sample count  300
label  41 (41, b'End of no passing')  sample count  210
label  42 (42, b'End of no passing by')  sample count  210
label  0 (0, b'Speed limit (20km/h)')  sample count  900
label  1 (1, b'Speed limit (30km/h)')  sample count  9900
label  2 (2, b'Speed limit (50km/h)')  sample count  10050
label  3 (3, b'Speed limit (60km/h)')  sample count  6300
label  4 (4, b'Speed limit (70km/h)')  sample count  8850
label  5 (5, b'Speed limit (80km/h)')  sample count  8250
label  6 (6, b'End of speed limit (')  sample count  1800
label  7 (7, b'Speed limit (100km/h')  sample count  6450
label  8 (8, b'Speed limit (120km/h')  sample count  6300
label  9 (9, b'No passing')  sample count  6600
label  10 (10, b'No passing for vehic')  sample count  9000
label  11 (11, b'Right-of-way at the ')  sample count  5850
label  12 (12, b'Priority road')  sample count  9450
label  13 (13, b'Yield')  sample count  9600
label  14 (14, b'Stop')  sample count  3450
label  15 (15, b'No vehicles')  sample count  2700
label  16 (16, b'Vehicles over 3.5 me')  sample count  1800
label  17 (17, b'No entry')  sample count  4950
label  18 (18, b'General caution')  sample count  5400
label  19 (19, b'Dangerous curve to t')  sample count  900
label  20 (20, b'Dangerous curve to t')  sample count  1500
label  21 (21, b'Double curve')  sample count  1350
label  22 (22, b'Bumpy road')  sample count  1650
label  23 (23, b'Slippery road')  sample count  2250
label  24 (24, b'Road narrows on the ')  sample count  1200
label  25 (25, b'Road work')  sample count  6750
label  26 (26, b'Traffic signals')  sample count  2700
label  27 (27, b'Pedestrians')  sample count  1050
label  28 (28, b'Children crossing')  sample count  2400
label  29 (29, b'Bicycles crossing')  sample count  1200
label  30 (30, b'Beware of ice/snow')  sample count  1950
label  31 (31, b'Wild animals crossin')  sample count  3450
label  32 (32, b'End of all speed and')  sample count  1050
label  33 (33, b'Turn right ahead')  sample count  2995
label  34 (34, b'Turn left ahead')  sample count  1800
label  35 (35, b'Ahead only')  sample count  5400
label  36 (36, b'Go straight or right')  sample count  1650
label  37 (37, b'Go straight or left')  sample count  900
label  38 (38, b'Keep right')  sample count  9300
label  39 (39, b'Keep left')  sample count  1350
label  40 (40, b'Roundabout mandatory')  sample count  1500
label  41 (41, b'End of no passing')  sample count  1050
label  42 (42, b'End of no passing by')  sample count  1050


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
* training set accuracy of 99.98%.
* validation set accuracy of 96.12%.
* test set accuracy of 95.20%.

Here are the configuration and the accuracy performance I record along the trials.

* Configuration and Performance Table

| Configuration			        |     Performance        					| 
|:---------------------:|:---------------------------------------------:|
| Dropout rate = 0.7 and Grayscale Data Sets | Train Accuracy = 0.99833, Validation Accuracy = 0.94467, Test Accuracy = 0.92835|
| Dropout rate = 0.5 and Grayscale Data Sets | Train Accuracy = 0.99747, Validation Accuracy = 0.96054, Test Accuracy = 0.94125|
| Dropout rate = 0.5 and Normalized Grayscale Data Sets | Train Accuracy = 0.99974, Validation Accuracy = 0.97483, Test Accuracy = 0.95408|
| Dropout rate = 0.5 and Normalized Grayscale Augemented Data Sets | Train Accuracy = 0.99976, Validation Accuracy = 0.96122, Test Accuracy = 0.95202|

The training performance figure is attached.
![alt text][perm]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
Answer: I started with LeNet since the lecture mentioned it has pretty good performance in this kind of task.
![alt text][lenet]

* What were some problems with the initial architecture?
Answer: The accuracy is not high enough, only around 89% for the test set.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Answer: I increased the filter depth to capture more pattern information from the inputs. I added dropout for the fully-connected layers to avoid the overfitting.
* Which parameters were tuned? How were they adjusted and why?
Answer: Dropout rate. I set 0.7 then decreased to 0.5. Check the Configuration and Performance Table table I added above. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Answer: In terms of achitecture, I added two dropout layers after fully-connected layers respectively. 



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:



![alt text][all_news]

This is wrong prediction when we use unaugmented data set. 
![alt text][new2].

After adding the generated data. 
![alt text][new2_new].

 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction with augumented data sets:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep          		| Stop sign   									| 
| Priority    			| Priority   									|
| Stop					| Stop						    				|
| Yield	      		    | Yield	    					 				|
| No Entry			    | No Entry     				        			|


Without augumented data, the model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 
| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep          		| Stop sign   									| 
| Priority    			|  Children crossing  									|
| Stop					| Stop						    				|
| Yield	      		    | Yield	    					 				|
| No Entry			    | No Entry     				        			|

This is the reason why I added scaled images as augumented data samples to help the deep neural network to get trained.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

Following figures should the top 5 softmax probablities.

![alt text][new1]

![alt text][new2]

![alt text][new3]

![alt text][new4]

![alt text][new5]
