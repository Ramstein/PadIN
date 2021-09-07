
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
import datetime, copy
import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard

from tensorflow._api.v2.nn import relu, sigmoid
from tensorflow.python.keras.regularizers import l2
from sklearn.model_selection import train_test_split

DROPOUT=0.3
NB_EPOCHS=10
BATCH_SIZE=256
OPTIMIZER = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  #learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08

dense_layers = [2, 1, 0]  # 0, 1,2
layer_sizes = [1028, 512, 256, 128]  # 128, 256, 512
conv_layers = [6, 5, 4, 3, 2, 1]  # 1, 2,3
date = datetime.datetime.now()

def fitModel(NB_CLASSES, x_train, y_train, datagen, IMG_SIZE, CBP):
    '''
    fitModel() is here for train the model, all of the parameters are described followinig
    :param NB_CLASSES: Number of classes in the dataset
    :param x_train: numpy array of training images for the model
    :param y_train: Labels of the images
    :param datagen: ImageDataGenerator used for augmenting the images and here a datagen
                    reference to thata has been used for fitting x_train and flowing it to
                    the fit_generator() method
    :param IMG_SIZE: IMG_SIZE = [1080, 608]
                    INPUT_SHAPE = IMG_SIZE + [3]
                    instagram landscape picture size [16:9] i.e. [1080, 608] input size image

    :param CBP: CBP = ['CollegeName', 'Batch', 'Professor']
    :return: return max, min, testScore, accIndex, lossIndex, testIndex, date

    '''
    print("Training the model.")
    max, min, testScore, accIndex, lossIndex, testIndex = 70.0, 4.0, 75, 1, 1, 1
    test_score_out = []
    INPUT_SHAPE = IMG_SIZE + [3]  #instagram landscape picture size [16:9] i.e. [1080, 608] input size image

    '''splitting x_train in train , validtion and test sets'''
    # train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.125, random_state=1)

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, f"{datetime.datetime.now():%m-%d-%Hh%Mm%Ss}")

                model = Sequential()
                model.add(Conv2D(layer_size, (8, 8), padding='valid',activation=relu,
                                 input_shape=INPUT_SHAPE, kernel_regularizer=l2(0.001)))
                # model.add(BatchNormalization())
                model.add(Dropout(DROPOUT))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

                for l in range(conv_layer - 1):
                    model.add(Conv2D(layer_size, (8, 8), padding='valid',activation=relu,
                                     kernel_regularizer=l2(0.001)))
                    # model.add(BatchNormalization())
                    model.add(Dropout(DROPOUT))
                    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


                model.add(Flatten())

                for _ in range(dense_layer):
                    model.add(Dense(layer_size, activation=relu, kernel_regularizer=l2(0.001)))
                    model.add(Dropout(DROPOUT))

                model.add(Dense(NB_CLASSES, activation=sigmoid))

                tensorboard = TensorBoard(log_dir="sample_data/{}".format(NAME))

                model.compile(loss='categorical_crossentropy',
                              optimizer=OPTIMIZER,
                              metrics=['accuracy'],
                              )
                # fit the datagen to the x_train
                datagen.fit(x_train)
                # train the model
                history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                                    steps_per_epoch=len(x_train)//BATCH_SIZE,
                                    epochs=NB_EPOCHS,
                                    verbose=1,
                                    callbacks=[tensorboard],
                                    validation_data=(x_val, y_val),
                                    )

                '''saving the comparatively best one'''
                if history.history.get('val_acc')[-1] > max:
                    max = history.history.get('val_acc')[-1]
                    if accIndex >= 2:
                        os.remove('ValAcc{}_{}_{}_{}_{}_{}'.format(accIndex - 1, round(max, 4), CBP[0], CBP[1], CBP[2],
                                                             f":{date:%Y-%m-%d-%Hh%Mm%Ss}"))
                    val_acc_out = open('ValAcc{}_{}_{}_{}_{}_{}'.format(accIndex, round(max, 4), CBP[0], CBP[1], CBP[2],
                                                                  f":{date:%Y-%m-%d-%Hh%Mm%Ss}"), "wb")
                    pickle.dump(model.save('ValAcc{}_{}_{}_{}_{}_{}'.format(accIndex, round(max, 4), CBP[0], CBP[1], CBP[2],
                                                                      f":{date:%Y-%m-%d-%Hh%Mm%Ss}")),
                                val_acc_out)
                    val_acc_out.close()
                    accIndex += 1

                if history.history.get('val_loss')[-1] < min:
                    min = history.history.get('val_loss')[-1]
                    if lossIndex >= 2:
                        os.remove('ValLoss{}_{}_{}_{}_{}_{}'.format(lossIndex - 1, round(min, 4), CBP[0], CBP[1], CBP[2],
                                                             f":{date:%Y-%m-%d-%Hh%Mm%Ss}"))
                    val_loss_out = open('ValLoss{}_{}_{}_{}_{}_{}'.format(lossIndex, round(min, 4), CBP[0], CBP[1], CBP[2],
                                                                   f":{date:%Y-%m-%d-%Hh%Mm%Ss}"))
                    pickle.dump(model.save('ValLoss{}_{}_{}_{}_{}_{}'.format(lossIndex, round(min, 4), CBP[0], CBP[1], CBP[2],
                                                                      f":{date:%Y-%m-%d-%Hh%Mm%Ss}")),
                                val_loss_out)
                    val_loss_out.close()
                    lossIndex += 1

                score = model.evaluate(x_test, y_test, verbose=1)  # score[testScore, testAccuracy]

                if score[1] > testScore:
                    testScore = score[1]
                    if testIndex >= 2:
                        os.remove('TestScore{}_{}_{}_{}_{}_{}'.format(testIndex - 1, round(testScore, 4), CBP[0], CBP[1], CBP[2],
                                                                      f":{date:%Y-%m-%d-%Hh%Mm%Ss}"))
                        os.remove('{}_FinestHistory'.format(testIndex - 1))
                    test_acc_out = open(
                        'TestScore{}_{}_{}_{}_{}_{}'.format(testIndex, round(testScore, 4), CBP[0], CBP[1], CBP[2],
                                                                    f":{date:%Y-%m-%d-%Hh%Mm%Ss}"), "wb")
                    pickle.dump(
                        model.save('TestScore{}_{}_{}_{}_{}_{}'.format(testIndex, round(testScore, 4), CBP[0], CBP[1], CBP[2],
                                                              f":{date:%Y-%m-%d-%Hh%Mm%Ss}")),
                        test_acc_out)
                    test_acc_out.close()
                    '''dumping best scoring history for future use'''
                    test_score_history = open('{}_FinestHistory'.format(testIndex), "wb")
                    pickle.dump(history, test_score_history)
                    test_score_out = copy.copy(score)
                    test_score_history.close()

                    testIndex += 1

    '''drawing the diagram for accuracy and loss for best saved model'''
    test_score_history_out = open('{}_FinestHistory'.format(testIndex), "rb")
    history = pickle.load(test_score_history_out)
    print('Best test score: ', test_score_out[0])
    print('Best test accuracy: ', test_score_out[1])

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return max, min, testScore, accIndex, lossIndex, testIndex, date
