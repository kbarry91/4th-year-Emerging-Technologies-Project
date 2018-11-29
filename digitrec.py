# Digit recognition  script

# Must import gzip to allow pyhton to read  and uncompress zip files
import gzip
import timeit

# Import numpy as np
import numpy as np

import sys
# Import tensorflow as tf
#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data


# https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
# Import Cv2  and Imagefor image processing
import cv2
from PIL import Image

# load dataset for global use
(image_train, label_train), (image_test, label_test) = mnist.load_data()
labelOptions = 10
imageSize = 784


def prepareDataSet():
    """ 
    Loads Mnist Dataset into memory and prepares images to for generation of Neural Network model.
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
    # A binary matrix representation of the input. The classes axis is placed last.
    # Used for  categorical_crossentropy loss on model
    label_train = keras.utils.to_categorical(label_train, labelOptions)
    label_test = keras.utils.to_categorical(label_test, labelOptions)
    print(label_test)

    # Output Statistics
    print(image_train.shape[0], 'training images loaded.')
    print(image_test.shape[0], 'test images loaded.')
    print("Preparing Mnist data Complete!")

def buildNeuralNet():
    """
    buildNeuralNet allows a user to enter an image to predict using the prebuilt model.
    User may select amount of epoch iterations.
    User must enter image name eg . xxx.png
    """
    menuOption = 0
    # Allow user to enter amount for epoch
    while int(menuOption) < 1 or int(menuOption) > 20:
        menuOption= input("Select amount of steps (epochs 1-20) :")

        # Confirm input is integer
        try :
            menuOption = int(menuOption)
        except ValueError:
            print("(ERROR)--> Value must be an integer")
            menuOption=0
            #buildNeuralNet()

    epochs = int(menuOption)
   
    global model
    global score
    batch_size = 128
    #num_classes = 10
    
   
    # Using Sequental ,Linear stack of layers
    # Add layer off input shape 784 and output shape of *532
    # Set fraction of input rates to drop during training
    model = Sequential()  
    model.add(Dense(512, activation='relu', input_shape=(784,))) 
    model.add(Dropout(0.2))
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
    # rmsprop = RMSprop(lr=learning_rate) gives .28  taking 25s
    rmsprop = RMSprop()
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
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
    print("Time Taken to train model at ", epochs, "epochs =", end_time_train)


def prediction():
    """
    prediction() converts a user entered image to a numpy array and predicts its outcome.
    image to upload must be located in "testImages/" directory.
    """
    menuOption="\nPlease enter an image name to test or (exit) to return: "
    print(menuOption)
    while menuOption != "exit":
        # Get user input
        menuOption= input("Image Name :")

        # check for exit condition
        if menuOption=="exit":
            print("Returning to main menu...")
            break
        
        # Check if file is valid
        try:
            img = np.invert(Image.open("testImages/"+menuOption ))
            '''
                if(len(sys.argv) == 2):
                    img = cv2.imread(sys.argv[1])
                    if(not img.data):
                        print("Could not load image")
                        exit
            '''
            # preprocessing
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            #img = cv2.resize(img, (28, 28), Image.ANTIALIAS)
            img = cv2.bitwise_not(img)
            img = img.reshape(1, 784)
            img = img.astype('float32')
            img /= 255

            # predict the handwritten digit in the input image
            #score = model.predict(img, batch_size=1, verbose=0)
            predictionB = model.predict(np.array(img, dtype=float))

            print("Predicted B: ", predictionB)
            # print(shape(predictionB))
            bestPrediction = max(predictionB[0])
            print("Max Pred _______",bestPrediction) 
            counter = 0
            for pred in predictionB[0]:
                print (counter,"==",pred)
                counter= counter +1

            #print("Actual: B", label_test[0])

            # display scores
            print("\nPrediction score for test input: " + menuOption)
            
            sort = sorted(range(len(predictionB[0])),
                        key=lambda k: predictionB[0][k], reverse=True)
            for index in sort:
                print(str(index) + ": " + str(predictionB[0][index]))
            percent = format(predictionB[0][sort[0]] * 100, '.2f')


            print("\nSystem is" ,str(percent) ," that the image is " ,str(sort[0]))
            menuOption="exit"
        # must catch errors    
        # File not found
        except FileNotFoundError: 
            print("(ERROR)--> ",menuOption," image not found !")
        # Any other error
        except: # If any other error occours
            print("(ERROR)--> Generic error uploading ",menuOption) 
            

def testPrediction():
    """
    testPrediction() is used to test the model against the MNist dataset test images.
    User input of int between 1-10000 required.
    Input is index of image in MNIST test image dataset.
    """
    testIndex = -1
    # Allow user to enter amount for epoch
    while int(testIndex) < 1 or int(testIndex) > 10000:
        testIndex= input("Select MNIST image to predict (index 1-10000) :")

        # Confirm input is integer
        try :
            testIndex = int(testIndex)
        except ValueError:
            print("(ERROR)--> value must be an integer")
            testIndex=-1
            
            #buildNeuralNet()

    testIndex = int(testIndex-1)
    predictionB = model.predict(np.array([image_test[testIndex]], dtype=float))
    print("Predicted B: ", predictionB)
    print("Actual: B", label_test[testIndex])
    # Make a prediction using Tensorflow and our classifier we created above from our testData
    # prediction = model.predict(np.array([x_test[1]], dtype=float), as_iterable=False)
    #prediction = history.predict(np.array([image_test[1]], dtype=float))
    #bestPrediction= max()
    # model.predict

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
        print("\n==== MNIST DATASET DIGIT RECOGNITION ====\n1.Prepare Dataset and Image Recognition Model\n2.Upload an Image to predict\n3.Predict image from MNIST test dataset\n4.Exit")
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
                print("(ERROR)--> Neural network not configured, select option 1 and try again !")
        elif choice == "3":
            if netBuilt:
                testPrediction()
            else:
                print("(ERROR)--> Neural network not configured, select option 1 and try again !")
        elif choice == "4":
            print("\n Exiting...")
            exit()
        elif choice != "":
            print("\n(ERROR)--> Invalid Option Try again")


# Launch the main menu
userMenu()
