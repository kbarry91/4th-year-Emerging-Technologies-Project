# Digit recognition  script

# Must import gzip to allow pyhton to read  and uncompress zip files
import gzip
import timeit

# Import numpy as np
import numpy as np
# Import tensorflow as tf
#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data


# https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

def loadMnistDataSet():
    """
    Loads Mnist Dataset
    """
    print("Loading Mnist...")
    (image_train, label_train), (image_test, label_test) = mnist.load_data()

    imageSize = 784

    # i,j act as offsets to offest idx format
    imageOffest = 16
    labelOffset = 8

   
    print("Loading Mnist Complete!")
    # Start timer
   # start_time_train = timeit.default_timer()

    # Stop Timer
   # timeTakenTrain = timeit.default_timer() - start_time_train
    """
    # test for correct values
    i = 0
    for i in range(2):
    
        for x in train_image[i]:
            print()
            for y in x:
                if(y != 255):
                    print("#", end="")
                else:
                    print(".", end="")
   
   """


def buildNeuralNet():
    print("Building Deep Neural Network Classification...")

    # Load the mnist dataset 
   # mnist = input_data.read_data_sets('data')
    #print("\n MNIST successfully Loaded....")

    # return images and labells as a 32 bit integer numpy array
  #  def input(dataset):
     #   return dataset.images, dataset.labels.astype(np.int32)

    # Specify the features
    # MNIST images have shape 28px x 28px, so we can define one feature with shape [28, 28]


def userMenu():
   # print("tf-----",tf.__version__)
   # print("ker---",keras.__version__)
    """
    Launches a User menu
    """
    choice = True
    while choice:
        print("""
        ==== MNIST DATASET DIDIT RECOGNITION ====
        1.loadMnistDataSet
        2.buildNeuralNet
        3.Exit
        """)
        choice = input("Enter Selction? ")
        if choice == "1":
            loadMnistDataSet()
        elif choice == "2":
            buildNeuralNet()
        elif choice == "3":
            print("\n Exiting...")
            exit()
        elif choice != "":
             print("\n Invalid Option Try again")


userMenu()
