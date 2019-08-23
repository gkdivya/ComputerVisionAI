### Advanced Convolutions

Run this [network](https://colab.research.google.com/drive/1STOg33u7haqSptyjUL40FZIxNW4XdBQK) After training the network, whatever accuracy you get is your base accuracy. Epochs = 100
* Fix the network above:
* remove dense
* add layers required to reach RF
* fix kernel scaleup and down (1x1)
* see if all dropouts are properly placed

follow the guidelines we discussed in the class (
Get accuracy more than the base accuracy in less number 100 epochs. Hint, you might want to use "border_mode='same',"
Save File as Assignment 6A

Rewrite it again using these convolutions in the order given below:
* Normal Convolution
* Spatially Separable Convolution  (Conv2d(x, (3,1)) followed by Conv2D(x,(3,1))
* Depthwise Separable Convolution
* Grouped Convolution (use 3x3, 5x5 only)
* Grouped Convolution (use 3x3 only, one with dilation = 1, and another with dilation = 2) 

You must use all of the 5 above at least once
Train this new model for 50 epochs. 
Save File as Assignment 6B

Total Score of 600. 400 for Code (300+100), and 200 for documentation (100+100)
Upload the github folder link which has both the files. 
