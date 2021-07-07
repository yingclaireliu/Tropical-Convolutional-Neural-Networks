# Tropical-Convolutional-Neural-Networks
TCNN-An Alternative Practice of Tropical Convolution to Traditional Convolutional Neural Networks

Paper：https://ui.adsabs.harvard.edu/abs/2021arXiv210302096F/abstract

## Requirement

python 3.6.5
pytorch 1.7.0


## Parameter Setting

Operation parameter reference the function "options_func()", and note that in "load_data()" needs to add the data set you need and its storage location.

运行参数参考
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
    
并注意在load_data()中需要添加你需要的数据集及其存放的位置。

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
