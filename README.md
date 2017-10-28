# Behavioral Cloning for Self-Driving Cars

The goal of this project is to train a convolutional neural network to map the front-facing camera directly to the steering commands. With the training model, the system can drive a car autonomously around test tracks in the Udacity's driving simulator.

Here are two youtube videos that the self cars ran under two anmiated roads from the simulation tool

| <a href="http://www.youtube.com/watch?feature=player_embedded&v=L6MeuvmfgOM" target="_blank"><img src="http://img.youtube.com/vi/L6MeuvmfgOM/0.jpg" alt="Road Track One" width="240" height="180" border="10" /></a> | <a href="http://www.youtube.com/watch?feature=player_embedded&v=MueOlce4iXw" target="_blank"><img src="http://img.youtube.com/vi/MueOlce4iXw/0.jpg" alt="Road Track One" width="240" height="180" border="10" /></a> |
|---|---|
|[Road Track one - Youtube](https://youtu.be/L6MeuvmfgOM)|[Road Track two (hard one) - Youtube](https://youtu.be/MueOlce4iXw) |











## Files and Usage

# Solution

## Data Collection
The train data is collected by driving the car in the simulator in two loops


<img src="images/left_1.jpg" width="200"><img src="images/center_1.jpg" width="200"><img src="images/right_1.jpg" width="200">

<img src="images/left_2.jpg" width="200"><img src="images/center_2.jpg" width="200"><img src="images/right_2.jpg" width="200">

## Network Architecture
I used Nvidia's architecture from their white paper [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf). It contains 9 layers, 1 normalization layer, 5 convolutional layers, and 3 fully connected layers. 

<img src="images/nvidia_cnn.png" alt="Architecture"><img src="images/detail_cnn.png" alt="Architecture" width="360">
 


## Data Augmentation and Preprocessing

## Training Details

### Prevent overfitting

### Extra road training

## Evaluation


