# Tensorflow_Basics
This repository will contain code for some simple problems in tensorflow. This is created with the purpose of revisiting when in doubt.

The below files are added as of the latest update:
1) basics: Basic understanding of Tensors, Variables, Placeholders, Sessions, and Training of some dummy data
2) mnist_conv: A deep CNN with max pooling to classify mnist data. 
   1000 epochs, Minibatch size 50, Dropout 0.3, Adam Optimizer with lr 1e-4 - Training accuracy: 0.9599, Testing Accuracy: 0.9599
3) mnist_summary_checkpoints: Same code as mnist_conv, but with summary for accuracy (after every 100 steps), and checkpoints saved (after every 1k steps)


# To-Do:
1) Make the directory and file paths commandline
2) Add a constants file for hyperparameters
