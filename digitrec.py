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

# load dataset for global use
(image_train, label_train), (image_test, label_test) = mnist.load_data()
labelOptions = 10
imageSize = 784

def prepareDataSet():
    """ 
    Loads Mnist Dataset
    """
    global image_train
    global image_test
    global label_train
    global label_test 
    
    print("Preparing Mnist...")
    #  (image_train, label_train), (image_test, label_test) = mnist.load_data()
   

   
    
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


def buildNeuralNet():
    
    global model
    batch_size = 128
    #num_classes = 10
    epochs = 1

    model = Sequential() # Using Sequental ,Linear stack of layers
    model.add(Dense(512, activation='relu', input_shape=(784,)))# add layer off input shape 784 and output shape of *532
    model.add(Dropout(0.2))# set fraction of input rates to drop during training
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(labelOptions, activation='softmax'))

    # Print a string summary of the model
    model.summary()

    # Configure model for training 
    # loss = name of objective function
    # metrics list of metrics to be evaluated by the model
    """
    From https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b
    The RMSprop optimizer is similar to the gradient descent algorithm with momentum.
    The RMSprop optimizer restricts the oscillations in the vertical direction.
    To prevent the gradients from blowing up, we include a parameter epsilon in the denominator which is set to a small value
    """
    # Compile the model
    # Using sgd gives .86 accur takinhg 18s
    # Using RMSprop() gives .98 taking 23s 
    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    # Start timer and build model
    print("\nBuilding Model...")
    start_time_train = timeit.default_timer()

    # Train the model
    # verbose 1 displays progress bar
    history = model.fit(image_train, label_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(image_test, label_test))
    
    # Stop Timer
    print("\nModel built !")
    end_time_train = timeit.default_timer() - start_time_train

    score = model.evaluate(image_test, label_test, verbose=0)
    print('Test loss    :', score[0])
    print('Test accuracy:', score[1])
    print("Time Taken to train model at ",epochs,"epochs =",end_time_train)


def prediction():
    predictionB = model.predict(np.array([image_test[0]], dtype=float))
    print("Predicted B: ", predictionB)
    print("Actual: B", label_test[0])
    # Make a prediction using Tensorflow and our classifier we created above from our testData
    # prediction = model.predict(np.array([x_test[1]], dtype=float), as_iterable=False)
    #prediction = history.predict(np.array([image_test[1]], dtype=float))
    #bestPrediction= max()
    #model.predict

    # Print our prediction and display the actual image we are trying to predict
    #print("Predicted A: ", prediction.index(max(prediction)))
    #print("Actual: A", label_test[1])
    

def userMenu():
    """
    Launches a menu for user input
    """
    
    # print("tf-----",tf.__version__)
    # print("ker---",keras.__version__)
    netBuilt = False
    choice = True

    while choice:
        print("\n==== MNIST DATASET DIGIT RECOGNITION ====\n1.Prepare and Setup image dataset\n2.Predict an image\n3.Exit")
        choice = input("Select Option? ")
        if choice == "1":
            if netBuilt:
                print("Neural network has already been configured")
            else:
                prepareDataSet()
                buildNeuralNet()
                netBuilt = True
        elif choice == "2":
            if netBuilt:
                prediction()
            else:
                print("Neural network not configured, select option 1 and try again !")
            
        elif choice == "3":
            print("\n Exiting...")
            exit()
        elif choice != "":
             print("\n Invalid Option Try again")

# Launch the main menu
userMenu()
