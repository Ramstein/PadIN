'''
!pip install wget
from zipfile import ZipFile
import wget
print('Beginning file downlaod with wget module')

url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'
wget.download(url, 'sample_data/')


print('2. Extract all files in ZIP to different directory')

    # Create a ZipFile Object and load sample.zip in it
with ZipFile('sample_data/kagglecatsanddogs_3367a.zip', 'r') as zipObj:
  # Extract all the contents of zip file in different directory
  zipObj.extractall('content/')

'''



import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random
import datetime
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.callbacks import TensorBoard


DATADIR = 'content/PetImages'
CATEGORIES = ['Cat', 'Dog']  #'''categories that we have to deal with'''
img_array= []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)  # path to cats and dogs dir
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
        plt.imshow(img_array, cmap='gray')
        plt.show()

        print(img_array)
        print(img_array.shape)

        break
    break


IMG_SIZE = 299  #every image of 299x299
resized_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(resized_img_array, cmap='gray')  # cmap = hot, plasma, cool,
plt.show()


training_data = []
def create_training_data():  # creating training datasets
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)  # path to cats and dogs dir

        classIndex = CATEGORIES.index(category) # 0 for dog and 1 for cat

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)

                resized_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([resized_img_array, classIndex])
            except Exception as e:
                pass

create_training_data()

print(len(training_data))



'''shuffle training data'''
random.shuffle(training_data)



# for sample in training_data[:10]:
#     print(sample[1])



x=[]
y=[]
for features, label in training_data:
    x.append(features)
    y.append(label)
x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)  #we can't pass a list to keras for training
                                        #'''we have to pass here a numpy array '''

# print(x[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))


pickle_out = open("x.pickle", 'wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out= open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open('x.pickle', 'rb')
x = pickle.load(pickle_in)
pickle_in = open('y.pickle', 'rb')
y = pickle.load(pickle_in)


x = x / 255.0
INPUT_SHAPE = x.shape[1:]#(224, 224, 3)
DROPOUT=0.2
NB_CLASSES=10
NB_EPOCHS=10
BATCH_SIZE=128
VALIDATION_SPLIT=0.2
OPTIMIZER = Adam()


max, min, accIndex , lossIndex=70.0 , 4.0, 1, 1
date = datetime.datetime.now()

dense_layers = [2, 1, 0]  # 0, 1,2
layer_sizes = [512, 256, 128, 64]  #32, 64, 128, 256, 512
conv_layers = [3, 2, 1]  # 1, 2,3

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=INPUT_SHAPE))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (5, 5)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                model.add(Dropout(DROPOUT))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))
                model.add(Dropout(DROPOUT))

            model.add(Dense(NB_CLASSES))
            model.add(Activation('softmax'))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='categorical_crossentropy',
                          optimizer=OPTIMIZER,
                          metrics=['accuracy'],
                          )

            history = model.fit(x, y,
                      batch_size=BATCH_SIZE,
                      epochs=NB_EPOCHS,
                      validation_split=VALIDATION_SPLIT,
                      verbose=1,
                      callbacks=[tensorboard])
            if history.history.get('val_acc')[-1] > max:
                max = history.history.get('val_acc')[-1]
                if accIndex >= 2:
                    os.remove('{}_{}_{}_{}_{}_{}'.format(accIndex-1, round(max, 4), CBP[0], CBP[1], CBP[2], f":{date:%Y-%m-%d-%Hh%Mm%Ss}"))
                val_acc_out = open('{}_{}_{}_{}_{}_{}'.format(accIndex, round(max, 4), CBP[0], CBP[1], CBP[2], f":{date:%Y-%m-%d-%Hh%Mm%Ss}"), "wb")
                pickle.dump(model.save('{}_{}_{}_{}_{}_{}'.format(accIndex, round(max, 4), CBP[0], CBP[1], CBP[2], f":{date:%Y-%m-%d-%Hh%Mm%Ss}")),
                            val_acc_out)
                val_acc_out.close()
                accIndex += 1

                pickle_upload = open('{}_pickle'.format(accIndex - 1), 'rb')
                p_upload = pickle.load(pickle_upload)
                print(p_upload)


            if history.history.get('val_loss')[-1] < min:
                min = history.history.get('val_loss')[-1]
                if lossIndex>=2:
                    os.remove('{}_{}_{}_{}_{}_{}'.format(lossIndex-1, round(min, 4), CBP[0], CBP[1], CBP[2], f":{date:%Y-%m-%d-%Hh%Mm%Ss}"))
                val_loss_out = open('{}_{}_{}_{}_{}_{}'.format(lossIndex, round(min, 4), CBP[0], CBP[1], CBP[2], f":{date:%Y-%m-%d-%Hh%Mm%Ss}"))
                pickle.dump(model.save('{}_{}_{}_{}_{}_{}'.format(lossIndex, round(min, 4), CBP[0], CBP[1], CBP[2], f":{date:%Y-%m-%d-%Hh%Mm%Ss}")),
                            val_loss_out)
                val_loss_out.close()
                lossIndex += 1




model.save('64x3-CNN.model')


CATEGORIES = ["Dog", "Cat"]  # will use this to convert prediction num to string value


def prepare(filepath):
    IMG_SIZE = 299  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)  # read in the image, convert to grayscale
    resized_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return resized_img_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # return the image with shaping that TF wants.


model = tf.keras.models.load_model("64x3-CNN.model")
prediction = model.predict([prepare('dog.jpg')])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
print(prediction)
print(prediction[0][0])

print(CATEGORIES[int(prediction[0][0])])


#We can also test our cat example:

prediction = model.predict([prepare('cat.jpg')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[int(prediction[0][0])])



'''
alpha. Also referred to as the learning rate or step size. The proportion that weights are updated (e.g. 0.001). Larger values (e.g. 0.3) results in faster initial learning before the rate is updated. Smaller values (e.g. 1.0E-5) slow learning right down during training
beta1. The exponential decay rate for the first moment estimates (e.g. 0.9).
beta2. The exponential decay rate for the second-moment estimates (e.g. 0.999). This value should be set close to 1.0 on problems with a sparse gradient (e.g. NLP and computer vision problems).
epsilon. Is a very small number to prevent any division by zero in the implementation (e.g. 10E-8).

We can see that the popular deep learning libraries generally use the default parameters recommended by the paper.

TensorFlow: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08.
Keras: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0.
Blocks: learning_rate=0.002, beta1=0.9, beta2=0.999, epsilon=1e-08, decay_factor=1.
Lasagne: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
Caffe: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
MxNet: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
Torch: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8




'''