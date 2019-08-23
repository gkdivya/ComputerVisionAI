### Advanced Convolutions

Run this [network](https://colab.research.google.com/drive/1STOg33u7haqSptyjUL40FZIxNW4XdBQK). After training the network, whatever accuracy we get is the base accuracy. 


#### Assignment_6A.ipynb
The above network is fixed by
* removing dense layers
* add layers required to reach receptive field
* fixed kernel scaleup and down (1x1)
* all dropouts are properly placed
* used padding with "border_mode='same'

Aim is to get more accuracy than the base network in less than 100 epochs

#### Assignment_6B.ipynb
The above network is re written using the convolutions in the order given below:
* Normal Convolution
* Spatially Separable Convolution  (Conv2d(x, (3,1)) followed by Conv2D(x,(3,1))
* Depthwise Separable Convolution
* Grouped Convolution (use 3x3, 5x5 only)
* Grouped Convolution (use 3x3 only, one with dilation = 1, and another with dilation = 2) 

This model is trained for 50 epochs. 

