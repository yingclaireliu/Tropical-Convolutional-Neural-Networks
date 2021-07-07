# Tropical-Convolutional-Neural-Networks
TCNN-An Alternative Practice of Tropical Convolution to Traditional Convolutional Neural Networks

Paper：https://ui.adsabs.harvard.edu/abs/2021arXiv210302096F/abstract

## Requirement

python 3.6.5
pytorch 1.7.0


## Parameter Setting

Operation parameter reference the function "options_func()", and note that in "load_data()" needs to add the data set you need and its storage location.

运行参数参考options_func()，并注意在load_data()中需要添加你需要的数据集及其存放的位置。

## Running Example

python .\Traditional_Convolutional_Neural_Networks.py --net net0

## Code contains content

- 6 tropical convolution layers

 MinPlus-Sum_Conv Layer (MinP-S)

 MaxPlus-Sum-Conv Layer (MaxP-S)

 MinPlus-Max-Conv Layer (MinP-Max)

 MaxPlus-Min-Conv Layer (MaxP-Min)

 MinPlus-Min-Conv Layer (MinP-Min)

 MaxPlus-Max-Conv Layer (MaxP-Max)

- 6 network structures

See the paper for details.
