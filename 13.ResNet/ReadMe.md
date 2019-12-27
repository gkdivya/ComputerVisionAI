ResNet18 - CIFAR10
===================

* CIFAR-10 with ResNet18
* Model must look like Conv->B1->B2->B3->B4 and not individually called Convs. 
  *  Batch Size 128
  *  Use Normalization values of: (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
  *  Random Crop of 32 with padding of 4px
  *  Horizontal Flip (0.5)
  *  Optimizer: SGD, Weight-Decay: 5e-4
  *  NOT-OneCycleLR
  *  Train for 300 Epochs

Target Accuracy is 90% (it can go till ~93%)
