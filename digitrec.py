# Digit recognition  script

# Must import gzip to allow pyhton to read  and uncompress zip files
import gzip
import timeit
# Import numpy as np
import numpy as np

def loadMnist():
    """
    Loads Mnist Dataset
    """
    print("Loading Mnist !")

    # Unzip and open training image and label file for reading
    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_images_content = f.read()
    
    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_labels_content = f.read()   
    
    train_image = []
    train_label = []
    imageSize = 784
    trainImageSize=len(train_images_content)

    # i,j act as offsets to offest idx format 
    i=16
    j=8

    # Start timer
   # start_time_train = timeit.default_timer()

    #Performance tweeks
    appendI = train_image.append
    appendL = train_label.append

    # Iterate through file
    while (i < trainImageSize): 
        # Add image to array
        appendI(~np.array(list(train_images_content[i:i+imageSize])).reshape(28,28).astype(np.uint8))
        # Add label to array
        appendL(int.from_bytes(train_labels_content[j:j+1], byteorder="big"))
    
        i += imageSize
        j += 1

    # Stop Timer
   # timeTakenTrain = timeit.default_timer() - start_time_train

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
   
    print("\nCorresponding Label :",train_label[i])  

loadMnist()