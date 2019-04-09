
## Problem

Convolutional neural networks: Tensorflow is an environment for building, among other things, neural networks. You can build multilayer networks; convolutional layers; and residual networks using tensorflow. There is a tool called tensorboard, which will display the progress of learning. This homework does not require a GPU, and you will not be graded on the accuracy of your final classifier (other than perhaps losing points if it is wholly inconsistent with a reasonable number).

Obtain and install Tensorflow, here. Be aware the install can be exciting (I had to update a variety of packages to install, discovered that 1.7 caused my mac to barf, and went to 1.5; tensorflow then yelled at me because "Your CPU supports instructions that this TensorFlow binary was not compiled to use" but nothing bad has happened so far).
Go through the MNIST tutorial here. You may find the tensorboard tutorial here. helpful. Insert appropriate lines of code into the tensorflow example to log the accuracy on tensorboard every 100 batches, for at least 2000 batches. You should screen capture the accuracy graph from tensorboard, and submit this.

A. Modify the architecture that is offered in the MNIST tutorial to get the best accuracy you can. I made three convolutional layers of smaller depth (i.e. the 32 went to 8), dropped the max pooling, and used three layers. Submit a screen capture of tensorboard graphs of accuracy. We will make it possible for people to compare graphs anonymously. This is to allow people to show off how well their model is doing, and see how others are doing; it's not required, and won't be graded, but it's been a source of fun and excitement in the past.

B. Here is how to submit graphs for comparision. Go to this Google form and supply what it asks for.
On that page, you'll see a link to results; press that, and you'll get a collection of tensorboard graphs that have been submitted


## Problem

Go through the CIFAR-10 tutorial here, and ensure you can run the code. Note the warning at the top: "This tutorial is intended for advanced users of TensorFlow and assumes expertise and experience in machine learning." Enjoy the sense that you are one of these. Finally, insert appropriate lines of code into the tensorflow example to log the accuracy on tensorboard every 100 batches, for at least 2000 batches. You should screen capture the accuracy graph from tensorboard, and submit this.
Modify the architecture that is offered in the CIFAR-10 tutorial to get the best accuracy you can. Anything better than about 93.5% will be comparable with current research. Be aware that people with bigger computers will likely do better at this exercise (so I won't make grades depend on accuracy). Submit a screen capture of tensorboard graphs of accuracy. We will make it possible for people to compare these graphs anonymously.This is to allow people to show off how well their model is doing, and see how others are doing; it's not required, and won't be graded, but it's been a source of fun and excitement in the past.
Here is how to submit graphs for comparision. Go to this Google form and supply what it asks for.

On that page, you'll see a link to results; press that, and you'll get a collection of tensorboard graphs that have been submitted. There may be two junk PDF's submitted by DAF to test the system; you can ignore those.
