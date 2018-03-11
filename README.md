# Out of distribution detector for NN based classifiers

Neural Networks have acieved great success in classifying images. For example, consider a DenseNet architecture trained on the CIFAR-10 datset.
If trained well, it can achieve accuracy of atleast 92%. That is excellent. Given an image from the CIFAR-10 datset, the Neural Network can 
classify whether it is an airplane, automobile, bird, cat, deer, dog, frog, horse, ship or a truck. 

But what if I input a picture of a house? It is obviously going to run the image through the network and output a probability 
distribution over the 10 classes mentioned above, and going to conclude that its a dog (say) since it had the highest probability.
This appears to be a major limitation of the classifier if it was to be used in real life scenarios. Ofcourse, its not realistic to expect the 
clasifier to detect a house since there are infinitely many classes. A realistic and reasonable behavior is to expect the classifier say 
that it doesn't know what the image is.

This is my attempt at building an enhancer for any image classifier that will empower the network to be able to detect an input image 
which is not from the same sub-space of images(out-of-distribution) that it was trained on. In addition, it should retain the original 
classification power on the in-distribution inputs.

## Hypothesis

The output probability distribution of in-distribution images is "different" from the output probability distribution of 
out-of-distribution images. By "different", I mean that the probability spaces of both the output distributions can be 
distinguished by a well trained function. This can be thought of a binary classifier on the probability distributions.

## Solution

I naturally did the obvious and trained a binary image classifier on the probability distributions. To achieve this, I took a 
sample of in-distribution images and a sample of out-of-distribution images and ran them to the pre-trained network to get their respective 
probability distribtuions over the output classes. This will be the training set for the binary classifier. Note that since I'm taking a sample of out-of-distribution images, this solution 
doesn't necessarily work well when a new image which doesn't belong to neither of the classes is passed through this classifier. 
But atleast it an NN classifier trained on CIFAR-10 will be able to detect MNIST images and vice versa. Note that the original 
pre-trained network is left untouched.

## Results

I will start off by saying I got better than expected results for such a simple strategy. 
* I started off with a pre-trained CIFAR-10 classifier which had an accuracy of around 0.92.
* Training set of binary classifier: 1000 images from CIFAR-10(in-distribution) and 1000 images from Tiny Imagenet(out-of-distribution)
* The binary classifier is a Feed forward NN with 2 hidden layers with ReLU activations. (Actual architecture of this network did not matter much)
* Test set of binary classifier: 9000 images from CIFAR-10(in-distribution) and 9000 images from Tiny Imagenet(out-of-distribution)
* **Result: Classification accuracy on test set: 0.8**

## State-of-the-art benchmarks

I got the idea for implementing this after reading this paper on arXiv: [link](https://arxiv.org/abs/1706.02690)
The employ a completely different approach from what I described above.
The classification accuracy reported in the paper for the same setting of train/test data is 0.94.
