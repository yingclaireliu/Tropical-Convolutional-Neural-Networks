Tropical-Convolutional-Neural-Networks (TCNNs)
-----------------------------------------------------------------------------------------------------------------------------------------------
Based on the following paper on ICCDA2021

An Alternative Practice of Tropical Convolution to Traditional Convolutional Neural Networks
https://ui.adsabs.harvard.edu/abs/2021arXiv210302096F/abstract

Operating environment
-----------------------------------------------------------------------------------------------------------------------------------------------
python 3.6.5 + pytorch 1.7.0；


Set parameters:
-----------------------------------------------------------------------------------------------------------------------------------------------
You can refer to options_func to set your parameters，and you should add your own path of your dataset in the code.
Running case ： python .\Tropical_Convolutional_Neural_Networks.py --net net0

-----------------------------------------------------------------------------------------------------------------------------------------------

The code incudes

1.  6 types of tropical convolutional layers：

MinPlus-Sum_Conv Layer (MinP-S)

MaxPlus-Sum-Conv Layer (MaxP-S)

MinPlus-Max-Conv Layer (MinP-Max)

MaxPlus-Min-Conv Layer (MaxP-Min)

MinPlus-Min-Conv Layer (MinP-Min)

MaxPlus-Max-Conv Layer (MaxP-Max)

See the paper for details.

2、 Several TCNNs

![image](https://user-images.githubusercontent.com/86921131/124767651-94beed00-df6a-11eb-8748-63f7a62282ee.png)


Refer to the paper for more details.
