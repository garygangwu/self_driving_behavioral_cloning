# Behavioral Cloning for Self-Driving Cars

The goal of this project is to train a convolutional neural network to map the front-facing camera directly to the steering commands. With the training model, the system can drive a car autonomously around test tracks in the Udacity's driving simulator.

Here are two youtube videos that the self cars ran under two anmiated roads from the simulation tool

[Road Track one - Youtube](https://youtu.be/L6MeuvmfgOM)

[Road Track two (hard one) - Youtube](https://youtu.be/MueOlce4iXw)

## Files and Usage

# Solution

## Data Collection
The train data is collected by driving the car in the simulator in two loops

## Network Architecture
I used Nvidia's architecture from their white paper [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf). It contains 9 layers, 1 normalization layer, 5 convolutional layers, and 3 fully connected layers. 

<img src="images/nvidia_cnn.png" alt="Architecture"><img src="images/detail_cnn.png" alt="Architecture" width="360">
 


## Data Augmentation and Preprocessing

## Training Details

### Prevent overfitting

### Extra road training

## Evaluation


