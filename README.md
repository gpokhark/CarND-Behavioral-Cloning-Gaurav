# **Behavioral Cloning** 

## Writeup Gaurav

---
#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* New_Gaurav_Model.ipynb jupyter notebook for the code contained in model.py

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing. Please click this link for my simulation [video](./images_video/run1.mp4)
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

* ```view_imgCV2(image)``` - Created to view the image (```model.py``` line 43-46)
* ```pre_process(image)``` - Created to preprocess the image (apply gaussian blur and to crop the image). But after intial runs decided not to apply gaussian blur as the results were not good. Also decided to use keras ```Cropping2D``` to crop the images later as initial cropping in the ```pre_process(image)``` function gave error due to the input from ```drive.py``` was a complete full sized image (160x320) (```model.py``` line 51-56)
* ```vizualize_images(images, angles)``` - Created to view random 20 images in a subplot (```model.py``` line 83-90)
* ```plot_histogram(angles, n_bins)``` - Created to plot histogram (```model.py``` line 101-109)
* ```delete_values(image_paths, angles)``` - Created to delete data from the bins containing more than twice the average number of samples per bin. Implementation source [Jeremy Shanon](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project) (```model.py``` line 116-139)
* ```flip_images(image_paths, angles, flipped)``` - Created to augment the data having steering measurement more than 0.7 and less than -0.7 (```model.py``` line 151-168)
* ```image_generator(image_paths, angles, flipped, batch_size=32)``` - Created to read in images from the data folder and generate the data (```model.py``` line 178-205)
* ```X_train_paths, X_test_paths, y_train, y_test, flipped_train, flipped_test``` - Split the data into training and test data set for validation purpose using ```train_test_split(image_paths, angles, flipped, test_size=0.15)``` from ```sklearn.model_selection``` (```model.py``` line 173-176)
* nVidia model (```model.py``` line 213-228)



### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
I have implemented [nVidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

nVidia Model - 

![nVidia Model](./images_video/nVidia_Model.png)

My Model summary -

![My Model Summary](./images_video/nVidia_Model_Summary.JPG)


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I used the sample data provided by udacity. I utilized the images from left and right cameras as well by using a correction factor of 0.25 to the steering angle measurement.

Data Vizualization

![Sample Image](./images_video/data_1.JPG)

Plotting few random images with steering angles

![Random Images](./images_video/random_images.JPG)

I plotted histogram to have a look at the spread of the data.

![Histogram](./images_video/histogram_1.JPG)

As we can see we have more data for steering meansurement 0 and 0.25. It would be better for the model if we have more uniform data as shown by the black line. Hence to make it more uniform I deleted data from the bins containing more than average samples and kept the data having lesser than average. I implemented similar strategy as that of [Jeremy Shanon](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project).

New histogram distribution is -

![Histogram](./images_video/histogram_2.JPG)

The new distribution is better than the previos one which was more biased towards 0 and 0.25. Further I observed that there was very few data with measurements greater than + and - 0.7. Hence decided to fllp the images with measurments grater than 0.7 and less than -0.7. For that I created an array ```flipped``` to hold the tokens (0 - do not flip, 1- flip). Multiplied the measurements corresponding flipped images by -1.

New histogram distribution after data augmentation is -

![Histogram](./images_video/histogram_4.jpg)

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I implemented nVidia model as shown in the lecture notes nVidia model (```model.py``` line 213-228)
```
# Nvidia Architecture
# image = mpimg.imread(str(image_paths[0]))
row, col, ch = 90, 320, 3
model = Sequential()
```
Normalized the images 
```
model.add(Lambda(lambda x: (x/127.5) - 1, input_shape = (160, 320, 3))
```
Cropped the images to size 90x320x3 to eliminate the hood and the sky
```
model.add(Cropping2D(cropping = ((50,20), (0,0))))
```
24 layers of Convolution with 2x2 strides and 5x5 kernel size and relu activation function
```
model.add(Conv2D(24, (5,5), subsample = (2,2), activation = 'relu'))
```
36 layers of Convolution with 2x2 strides and 5x5 kernel size and relu activation function
```
model.add(Conv2D(36, (5,5), subsample = (2,2), activation = 'relu'))
```
48 layers of Convolution with 2x2 strides and 5x5 kernel size and relu activation function
```
model.add(Conv2D(48, (5,5), subsample = (2,2), activation = 'relu'))
```
64 layers of Convolution with 3x3 kernel size and relu activation function
```
model.add(Conv2D(64, (3,3), activation = 'relu')) 
```
64 layers of Convolution with 3x3 kernel size and relu activation function
```
model.add(Conv2D(64, (3,3), activation = 'relu'))
```
64 layers of Convolution with 3x3 kernel size and relu activation function
```
model.add(Conv2D(64, (3,3), activation = 'relu'))
```
Dropout layer with with 0.5 as the fraction of inputs to drop
```
model.add(Dropout(0.5))
```
Flatten layer
```
model.add(Flatten())
```
Layers to reduce it to a single value output as the steering value is a single point value
```
model.add(Dense(100,activation = 'relu'))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(10,activation = 'relu'))
model.add(Dense(1))
```

 I split my image and steering angle data into a training and validation set. 
 ```X_train_paths, X_test_paths, y_train, y_test, flipped_train, flipped_test``` - Split the data into training and test data set for validation purpose using ```train_test_split(image_paths, angles, flipped, test_size=0.15)``` from ```sklearn.model_selection``` (```model.py``` line 173-176)

The model contains dropout layers in order to reduce overfitting. 

The vehicle is able to drive autonomously around the track without leaving the road. 

[video](./images_video/run1.mp4)

I ran the model for 10 Epochs and saved results from each epoch. Plotted the mean square error v/s epochs and saw that mean square error for training and validation set remained almost constant after 5 epochs hence used the result after epoch 5 to run the model and it successfully completed the first track. Mean square error for validation set is more than the training set indicating that the model is not overfitting.

![Mean Square Error](./images_video/mean_square_error.JPG)

#### 2. Final Model Architecture

Has already been discussed above.

#### 3. Creation of the Training Set & Training Process

Has already been discussed above.