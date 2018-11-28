# Digit recognition  script

# Must import gzip to allow pyhton to read  and uncompress zip files
import gzip
import timeit
# Import numpy as np
import numpy as np

def loadMnistDataSet():
    """
    Loads Mnist Dataset
    """
    print("Loading Mnist !")

    # Unzip and open training image and label file for reading
    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_images_content = f.read()
    
    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_labels_content = f.read()   

    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_images = f.read()

    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_labels = f.read()

    imageSize = 784
    trainImageSize=len(train_images_content)

    # i,j act as offsets to offest idx format 
    i=16
    j=8

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

loadMnistDataSet()