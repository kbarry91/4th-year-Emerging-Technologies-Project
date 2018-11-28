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

# load dataset
(image_train, label_train), (image_test, label_test) = mnist.load_data()

def prepareDataSet():
    global image_train
    global image_test
    global label_train
    global label_test 
    """ 
    Loads Mnist Dataset
    """
    print("Preparing Mnist...")
    #  (image_train, label_train), (image_test, label_test) = mnist.load_data()
   

    imageSize = 784
    labelOptions = 10
    # i,j act as offsets to offest idx format
    imageOffest = 16
    labelOffset = 8

    # Reshape image data set
    image_train = image_train.reshape(60000, imageSize)
    image_test = image_test.reshape(10000, imageSize)

    # Convert image data to type float 32
    image_train = image_train.astype('float32')
    image_test = image_test.astype('float32')

    # Values are rgb 0-255 . Convert to  0 OR  1
    image_train /= 255
    image_test /= 255

    # convert class vectors to binary class matrices
    #A binary matrix representation of the input. The classes axis is placed last. 
    # Used for  categorical_crossentropy loss on model
    label_train = keras.utils.to_categorical(label_train, labelOptions)
    label_test = keras.utils.to_categorical(label_test, labelOptions)
    print (label_test)
    
    # Output Statistics
    print(image_train.shape[0], 'training images loaded.')
    print(image_test.shape[0], 'test images loaded.')
    print("Preparing Mnist data Complete!")
    
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


def buildModel():
    print("Building Model...")

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
        print("==== MNIST DATASET DIDIT RECOGNITION ====\n1.Prepare and Setup image dataset\n2.buildNeuralNet\n3.Exit")
        choice = input("Select Option? ")
        if choice == "1":
            prepareDataSet()
            buildModel()
        elif choice == "2":
            buildNeuralNet()
        elif choice == "3":
            print("\n Exiting...")
            print ("In exit s--",label_test)
            exit()
        elif choice != "":
             print("\n Invalid Option Try again")

# Launch the main menu
userMenu()
