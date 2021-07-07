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
You can set your parameters in options_func

Options:
  -h, --help            show this help message and exit
  
  --net=NET             Choosing which net structure to use [1].
  
  --dataset=DATASET     Choosing  dataset [mnist ashion-mnist andomxt\CIFAR10\STL10\].
  
  --channels=CHANNELS   Input channels of train data.
  
  --dim=DIM             Dimension of train data. mnist-784/STL10-27468/CIFAR10-3072
                        
  --out=OUT             Dimension of out data.
  
  --k1=K1               The count of hidden nodes.
  
  --k2=K2               The count of hidden nodes.
  
  --k3=K3               The count of linner layers param.
  
  --kernel=KERNEL       Kernel size.
  
  --stride=STRIDE       stride.
  
  --lr=LR               The learning rate of training.
  
  --loss=LOSS           Loss Function.
  
  --epoch=EPOCH         Training epoch.
  
  --print_freq=PRINT_FREQ The frequency of printing.
  
  --save_loss_freq=SAVE_LOSS_FREQ [The frequency of saving loss.
  
  --bs=BS               batch_size
  
  --momentum=MOMENTUM   Momentum of optimizer
  
  --cuda=CUDA           use cuda or not
  
  --fitepoch=FITEPOCH   the min epoch after using early stopping
  
  --patience=PATIENCE   the patience with using early stopping
  
  --image_height=IMAGE_HEIGHT [length of image
  
  --image_width=IMAGE_WIDTH [width of image
  
and you should add your own path of your dataset in the code.


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

Refer to the paper for more details.

2、 Several TCNNs

![image](https://user-images.githubusercontent.com/86921131/124767651-94beed00-df6a-11eb-8748-63f7a62282ee.png)


Refer to the paper for more details.
