# Behavioral Cloning for Self-Driving Cars

The goal of this project is to train a convolutional neural network to map the front-facing camera directly to the steering commands. With the training model, the system can drive a car autonomously around test tracks in the Udacity's driving simulator.

Here are two youtube videos that the self-driving cars ran under two anmiated roads from the simulation tool

| <a href="http://www.youtube.com/watch?feature=player_embedded&v=L6MeuvmfgOM" target="_blank"><img src="http://img.youtube.com/vi/L6MeuvmfgOM/0.jpg" alt="Road Track One" width="240" height="180" border="10" /></a> | <a href="http://www.youtube.com/watch?feature=player_embedded&v=MueOlce4iXw" target="_blank"><img src="http://img.youtube.com/vi/MueOlce4iXw/0.jpg" alt="Road Track One" width="240" height="180" border="10" /></a> |
|---|---|
|[Road Track one (easy one) - Youtube](https://youtu.be/L6MeuvmfgOM)|[Road Track two (hard one) - Youtube](https://youtu.be/MueOlce4iXw) |

## Simulator Download
* [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
* [macOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)

## Files and Usage
1. model.py
    * Load training data and training a model to predict steering angles.
    * `python model.py` to train the model with the output `model.h5` as the model file.
2. model.h5
    * Keras deep learning model file for "the easy road"
3. model.2.h5
    * Keras deep learning model file for "the hard road"    
4. drive.py
    * A server to take images and output the steer predictions based on a trained model
    * `python drive.py model.h5`
5. video.py
    * Generate an mp4 video from the previously saved images of an autonomous run.
    * `python video.py previous_run1`
6. video.mp4
    * A video recording of the vehicle driving autonomously in the first road

# Implementation Deep Dive

## Data Collection
The train data is collected by driving the car in the simulator. Each car equiped with three front-faceing cameras: left, center, and right positioned, to capture the images as below in real time. The car was driven in the center of the road as much as possible during the training process, although in the real world the car should stay in its own lane. For the simplicity of this project, it only aims for the center road driving.

<img src="images/left_1.jpg" width="200"><img src="images/center_1.jpg" width="200"><img src="images/right_1.jpg" width="200">

<img src="images/left_2.jpg" width="200"><img src="images/center_2.jpg" width="200"><img src="images/right_2.jpg" width="200">

## Network Architecture
I used Nvidia's architecture from their white paper [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf). It contains 9 layers, 5 convolutional layers and 4 fully connected layers. 

From bottom up, below is each layer's funcationalities

| Layer | Type | Description |
| :---: | :---: | --- |
| Input |  | 1) 160x320x3 images <br> 2) Normalize the pixel value to [-1, 1] range <br> 3) Chop the top and bottom portion of the images to remove noices |
| Layer 1 | Conv2D | 24 filters with 5x5 convolution window and 2x2 strides |
| Layer 2 | Conv2D | 36 filters with 5x5 convolution window and 2x2 strides |
| Layer 3 | Conv2D | 48 filters with 5x5 convolution window and 2x2 strides |
| Layer 4 | Conv2D | 64 filters with 3x3 convolution window and 1x1 strides |
| Layer 5 | Conv2D | 64 filters with 3x3 convolution window and 1x1 strides |
| Layer 6 | Fully connected | 1) 50% dropout <br> 2) Output 100 neurons |
| Layer 7 | Fully connected | 1) 50% dropout <br> 2) Output 50 neurons |
| Layer 8 | Fully connected | Output 10 neurons |
| Layer 9 | Fully connected | Output 1 neurons |

<img src="images/nvidia_cnn.png" alt="Architecture"><img src="images/detail_cnn.png" alt="Architecture" width="360">

The left deep neural network architecure used from [NVidia’s paper](https://arxiv.org/pdf/1604.07316.pdf). The right is the screen shot from Keras model.summary() call.

## Training Details

For each road in the simulator, I driven the cars for two cycles to collect the images, steering, throttle, brake, and speed information. 

### Data preprocessing

* Normalization: each pixel value is normalized to (-1, 1) range
* Noise removal: Chop the image to excludes the sky and/or the hood of the car

<img src="images/left_1_chopped.png" width="200"><img src="images/center_1_chopped.png" width="200"><img src="images/right_1_chopped.png" width="200">

* Train / validation data set split: 80% assigned to train and 20% to validation after shuffling the data

### Model hyperparameters

* Optimized on mean squared error as a standard unbiased estimate of error variance for this regression problem
* Adam optimizer is used, because it is [suggested as the default optimization method for deep learning applications](http://cs231n.github.io/neural-networks-3/)
* Activation function: ELU is used because it facilitates [a fast and accurate Deep Network Learning](https://arxiv.org/abs/1511.07289)

### Leverage left and right camera images

he left and right images are important to teach the network how to recover from a poor position. The center image represents the direction that the car go straight, and the left / right images records the direction if the car steer left or right on a certain angle. Since the distance and angle are not given for the left and right camera in the simulator, an empiricial approach is applied: adding an offset of 0.25 to the left images and substracting 0.25 from the right images. This is intend to steer the car back to the view of the center camera, if the center camera faces what the left / right cameras see.

### Data Augmentation through fliping images

Data augementation can be acheived through flipping images and taking the opposite sign of the steering measurement. This only only introduces more training data, but also alleviates the left turn bias because in the simulator the steering left dominates in the road condition

### Training data size

After the data augmentation, 

* Each Road: 38988 images
* Hard Road: 65250 images

### Prevent overfitting

Nvidia paper didn't offer any details how to alleviate overfitting. In this project, both dropout and L2 regulaization are used empirically. 50% dropout has applied to the first two fully connected layers, while the L2 regulaization has appiled to conv2D and fully connected layers.

### Model checkpoint

Due to the result of over-training, the model from the last epoch is not necessary the best one. We should pick the latest best model according to the monitored quantity, which in our case is the lowest loss on the validation set. Therefore, `keras.callbacks.ModelCheckpoint` is used to evaluate and capture the best model after every training epoch.

### Extra road driving for better training.

The first simulator road is smooth with few sharp turn, and so I only need to drive the car for a couple cycle to complete the training. The result works pretty well. However, the second simulator road is a lot harder, not only with sharp turns, but also multiple roads in the same images and thus making the model confused on which one is the right path.

<img src="images/multi_roads_1.jpg">
<img src="images/multi_roads_2.jpg">
<img src="images/multi_roads_3.jpg">

These roads seems to be close to each other on the images, but not connected. However, the training model might assume that they were connected and tried to drive over there, although the sharp turn signs were in sight. In order to make the model to handle these difficult situations, I had to drive these areas mulitple times to ensure the deep neural network to memorize the patterns.

## Evaluation and Discussion

### Can the model trained on road A directly drive successfully on road B?

If the road A and B's conditions are similar, it is totally possible the model can be directly applied to the other road. Here is the youtube video demostrating that the model that I trained in the easy road was able to drive car in an unseen mountain road successfully. (This road were found in another version of Udacity's driving simulator)

| <a href="http://www.youtube.com/watch?feature=player_embedded&v=Qb-Ik6BNcWE" target="_blank"><img src="http://img.youtube.com/vi/Qb-Ik6BNcWE/0.jpg" alt="Road Track One" width="240" height="180" border="10" /></a> |
|---|
|[Drive on an unseen road - Youtube](https://youtu.be/Qb-Ik6BNcWE)|

However, the model trained on the easy road cannot directly be used in the hard road, and visa versa. The road conditions (e.g. curves, cliff signs, river) are very different. Therefore, in this project, the model in each road is trained independently with their own road images and steering records. Ideally, I should consider transfer learning because both roads could share convonlutional neural network features on road lines and shapes.

|<img src="images/easy_track.png">|<img src="images/hard_track.png">|
|:---:|:---:|
|Easy Road|Hard Road|

