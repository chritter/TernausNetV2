# Modifications and Notes for STC

Christian Ritter

* U-Net architecture with WideRes-Net38,  by Iglovikov 18b, implementation of https://github.com/ternaus/TernausNetV2

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
* Maxpooling is performed in the WideResnet-38 modules before into into ResNet layers. Why does authors
preform again maxpooling on the input for these modules during forward pass?
* There is no information about data preparation to create the images in the code. Paper describes those though.
* Loss calculation and updating of the weights are not included in the code. Paper describes those


## Input Data

* as described in Iglovikov18b paper
* RGB : 3 bands
* MUL : 8 bands multispectral


## What needs to be done

* Implement loss calculation and learning based on description in paper