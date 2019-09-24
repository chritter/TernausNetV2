# Modifications and Notes for STC

Christian Ritter

* HAN by Iglovikov 18b, implementation of https://github.com/ternaus/TernausNetV2

## Comments

* Compiles JiT cpp extnesions, need pip install ninja
* Interesting deconvoluation approach to avoid checkerboard features 
* Encoder
    * WideResNet 38, more channels while reducing layer number, 
    
    * 5 conv layers as encoder 
* Decoder
    * 5 decoder blocks 

* Tensorboard requires pip install tensorflow==1.4.0, does not work
* Modified loading of model to CPU via map_location=torch.device('cpu') for now..


## Input Data


## What needs to be done
