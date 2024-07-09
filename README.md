# ARSLwithVMEA
Arabic Sign Language Video Classification using Video Masked Auto-Encoder

# Description
This repo is part of our machine learning work for our graduation project "Fluent Hands"
A mobile application helping people learn sign-language with AI reinfocement.

# Technologies
```
transformers
torch
torchvision
pytorchvideo
flask
```
to use this repo make sure to install all the dependecies in requirements.txt file
```
pip install -r requirements.txt
```
# Model Architecture
![model](images/videomae_architecture.jpeg?raw=true)
The model works using 2 attention methods:  
vanilla VIT (vision transformer) -> to understand each frame  
joint time-space attention ->to understand relations between frames  
but the special part about our model that makes it as robust and data effecient is masked auto encoding
you can read more about the model architecture in detail in the model's [paper](https://arxiv.org/abs/2203.12602)

# Deployment
the model was deployed using flask framework and for public API deployment lightning.ai was used  
the API was then called from a flutter application to be then incorperated into the app's ecosystem

#