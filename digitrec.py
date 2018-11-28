# Digit recognition  script

# Must import gzip to allow pyhton to read  and uncompress zip files
import gzip
import timeit

# Import numpy as np
import numpy as np
# Import tensorflow as tf
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def loadMnistDataSet():
    """
    Loads Mnist Dataset
    """
    print("Loading Mnist !")

    # Unzip and open training image and label file for reading
    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_images = f.read()

    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_labels = f.read()

    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_images = f.read()

    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_labels = f.read()

    imageSize = 784

    # i,j act as offsets to offest idx format
    imageOffest = 16
    labelOffset = 8

    # reshape the images and labels.
    train_images = ~np.array(list(train_images[imageOffest:])).reshape(
        60000, 1, imageSize).astype(np.uint8)
    train_labels = np.array(list(train_labels[labelOffset:])).astype(np.uint8)

    test_images = ~np.array(list(test_images[imageOffest:])).reshape(
        10000, imageSize).astype(np.uint8) / 255.0
    test_labels = np.array(list(test_labels[labelOffset:])).astype(np.uint8)
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


def userMenu():
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
