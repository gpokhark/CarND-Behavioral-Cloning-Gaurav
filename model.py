import csv
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import sklearn
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint


# PATH Variable
path =  './'
print(path)


# # Logging to file
import logging
# # DISABLE LOGGING
logging.disable(logging.CRITICAL)
# create logger with 'model_app' for debugging
logger = logging.getLogger('model_app')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(path+'/model_logging.txt', mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',datefmt='%m/%d/%Y %I:%M:%S %p')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info('============START============')


# for viewing the image
# cv2 reads in as BGR Format
# matplotlib reads in as RGB Format

def view_imgCV2(image):
    plt.imshow(image)
    plt.text(10, 20,'Width {0} Height {1}'.format(image.shape[1],image.shape[0]),backgroundcolor = [1,1,1])
    return 0

# image = mpimg.imread(path+'data/data/IMG/center_2016_12_01_13_30_48_287.jpg')
# view_imgCV2(image)

def pre_process(image):
#     Cropping the image bottom 20 and top 50 - New size 90 x 320
#     new_image = image[50:140,:,:]
#     Applying Gaussian Blur
#     new_image = cv2.GaussianBlur(image[50:140,:,:], (3,3), 0)
    return image

# view_imgCV2(pre_process(image))

path = path + 'data/data/'
image_paths = []
angles = []
# Steering angle correction
correction = 0.25
with open(path+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        line[0] = path + 'IMG/' + line[0].split('/')[-1]
        line[1] = path + 'IMG/' + line[1].split('/')[-1]
        line[2] = path + 'IMG/' + line[2].split('/')[-1]
        image_paths.append(line[0])
        image_paths.append(line[1])
        image_paths.append(line[2])
        
        angles.append(float(line[3]))
        angles.append(float(line[3])+correction)
        angles.append(float(line[3])-correction)

image_paths = np.array(image_paths)
angles = np.array(angles)

# function to view 20 images in sub-plot
def vizualize_images(images, angles):
    fig, axs = plt.subplots(5,4, figsize=(20, 10))
    axs = axs.ravel()
    for i in range(20):
        image = images[i]
        axs[i].imshow(image)
        axs[i].text(10, 20,'Angle {0:.2f}'.format(angles[i]),backgroundcolor = [1,1,1])
    return 0

# index = np.random.randint(image_paths.shape[0], size = 20)
# viz_images = []
# for i in index:
#     viz_images.append(pre_process(mpimg.imread(image_paths[i])))
# viz_images = np.asarray(viz_images)
# print(viz_images.shape)
# vizualize_images(viz_images, angles[index])

# Function to plot histgram
def plot_histogram(angles, n_bins):
    avg_samples_per_bin = len(angles)/n_bins
    plt.figure(1,figsize=(10,5))
    plt.hist(angles, bins=n_bins)
    plt.xlabel('Steering Angle')
    plt.plot((np.min(angles),(np.max(angles))),(avg_samples_per_bin, avg_samples_per_bin),'k-')
    plt.xlim((np.min(angles),(np.max(angles))))
    plt.show()
    return None

# # Original Data Histogram
# n_bins = 30
# plot_histogram(angles, n_bins)

# Function to delete extra data
def delete_values(image_paths, angles):
#     If number is below average do not delete anything
#     If number is above average delete all that is above the average
    hist, bins = np.histogram(angles, bins=n_bins)
    avg_samples_per_bin = len(angles)/n_bins
    keep_probs = []
    target = avg_samples_per_bin * .5
    for i in range(n_bins):
        if hist[i] < target:
            keep_probs.append(1.)
        else:
            keep_probs.append(1./(hist[i]/target))
            
    remove_list = []
    for i in range(len(angles)):
        for j in range(n_bins):
            if angles[i] > bins[j] and angles[i] <= bins[j+1]:
                # delete from X and y with probability 1 - keep_probs[j]
                if np.random.rand() > keep_probs[j]:
                    remove_list.append(i)
                    
    image_paths = np.delete(image_paths, remove_list, axis=0)
    angles = np.delete(angles, remove_list)
    return image_paths,angles

# # Delete data first time 
# image_paths,angles = delete_values(image_paths,angles)
# plot_histogram(angles, n_bins)

# Flipped -> 0 indicates do not flip the image 1 indicates flip the images
flipped = np.zeros(angles.shape,)
print(flipped.shape)

# Function to augment the data
# flipping images with absolute angles grater than 0.7
def flip_images(image_paths, angles, flipped):
    paths = []
    measurement = []
    flip = []
    for i in range(image_paths.shape[0]):
        paths.append(image_paths[i])
        measurement.append(angles[i])
#         0 indicates do not flip the image
        flip.append(0)
        
        if (abs(angles[i]) >= 0.7):
            paths.append(image_paths[i])
#             Multiply by -1 to flip the angles
            measurement.append(angles[i])
#             1 indicates to flip the image
            flip.append(1)

    return np.array(paths), np.array(measurement), np.array(flip)

# image_paths, angles, flipped = flip_images(image_paths, angles, flipped)
# plot_histogram(angles, n_bins)

X_train_paths, X_test_paths, y_train, y_test, flipped_train, flipped_test = train_test_split(image_paths, angles, flipped, test_size=0.15)
print('Original:', image_paths.shape,angles.shape, flipped.shape)
print('Train:', X_train_paths.shape, y_train.shape, flipped_train.shape)
print('Test:', X_test_paths.shape, y_test.shape, flipped_test.shape)

def image_generator(image_paths, angles, flipped, batch_size=32):
    num_samples = angles.shape[0]
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(image_paths, angles, flipped)
        for offset in range(0, num_samples, batch_size):
            batch_paths = image_paths[offset:offset+batch_size]
            batch_angles = angles[offset:offset+batch_size]
            batch_flip = flipped[offset:offset+batch_size]
            images = []
            measurements = []
            for i in range(len(batch_paths)):
                if (batch_flip[i] == 1):
                    image = cv2.flip(mpimg.imread(str(batch_paths[i])),1)
                    image = pre_process(image)
                    angle = float(- 1* batch_angles[i])
                else:
                    image = mpimg.imread(str(batch_paths[i]))
                    image = pre_process(image)
                    angle = float(batch_angles[i])
        
                images.append(image)
                measurements.append(angle)
            
        
    # trim image to only see section with road
            X = np.array(images)
            y = np.array(measurements)
            yield X, y

# compile and train the model using the generator function
train_generator = image_generator(X_train_paths, y_train, flipped_train, batch_size=32)
test_generator = image_generator(X_test_paths, y_test, flipped_test, batch_size=32)

# Nvidia Architecture
# image = mpimg.imread(str(image_paths[0]))
row, col, ch = 90, 320, 3
model = Sequential()
model.add(Lambda(lambda x: (x/127.5) - 1, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping = ((50,20), (0,0))))
model.add(Conv2D(24, (5,5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(36, (5,5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(48, (5,5), subsample = (2,2), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100,activation = 'relu'))
model.add(Dense(50,activation = 'relu'))
model.add(Dense(10,activation = 'relu'))
model.add(Dense(1))

model.summary()
print(path)

model.compile(loss = 'mse',optimizer = 'adam')
checkpoint = ModelCheckpoint('./20180925_10Epochs_Correct_Run/model{epoch:02d}.h5')
history_object = model.fit_generator(train_generator, steps_per_epoch= y_train.shape[0],
                                     validation_data=test_generator,
                                     validation_steps=y_test.shape[0], epochs = 10, verbose = 1, callbacks=[checkpoint])

# Model Save
model.save('./20180925_10Epochs_Correct_Run/model.h5')

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# CLOSING THE LOGGER FILE
logger.info('============CLOSE============')
fh.flush()
fh.close()
logger.removeHandler(fh)