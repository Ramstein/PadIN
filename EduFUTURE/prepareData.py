

import numpy as np
import os, copy, cv2, random, pickle

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import np_utils

np.random.seed(1671) #for reproducibiality

class PrepareData:
    def prepTrainingData(path=None):

        IMG_SIZE = [1080, 608]  #instagram landscape picture size [16:9] input size image
        DATADIR = 'sample_data/'
        NB_CLASSES, i = 0, 0
        NUM_TO_AUGMENT = 5
        CATEGORIES, CBP, img_array, resized_img_array = [], [], [], []
        training_data, x_train, y_train = [], [], []

        for r, d, f in os.walk(DATADIR):
            if i == 1:
                CATEGORIES = copy.copy(d)
                NB_CLASSES = len(d)
                i += 1
            if i == 0:
                CBP = d[i].split('_')
                i += 1

        print('NB_CLASSES: ', NB_CLASSES)
        print("CATEGORIES: ", CATEGORIES)
        print("CBP: ", CBP)

        '''Creating taining data'''
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)  # path to all category images
            classIndex = CATEGORIES.index(category) # 0 for first file of image

            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    resized_img_array = cv2.resize(img_array, IMG_SIZE)
                    training_data.append([resized_img_array, classIndex])
                except Exception as e:
                    pass

        print('Length of training_data: ', len(training_data))

        random.shuffle(training_data)  #shuffle training data

        '''separating the features and the labels as x_train and y_train'''
        for feature, label in training_data:
            x_train.append(feature)
            y_train.append(label)

        x_train = np.array(x_train).reshape(-1, IMG_SIZE, 3)
        x_train = x_train.astype('float32') / 255.0

        y_train = np_utils.to_categorical(y_train, NB_CLASSES)




        '''Augmenting using ImageDataGenerator
        '''
        print("Augmenting training set images....")
        datagen = ImageDataGenerator(
            rotation_range=40,  # 0-180 degree for randomly rotataing pics
            width_shift_range=[-20, 20],  # for randomly tranlating pics vertically or horizontally
            height_shift_range=[-30, 30],
            brightness_range=[0.5, 1.0],
            shear_range=20,
            zoom_range=0.2,  # for randomly zooming pictures
            horizontal_flip=True,  # for randomly flipping half of the images horizantally
            fill_mode='nearest',# is the strategy used for filling in new pixels that can appear after a rotation or a shift
            data_format='channel_last' # s, w, d, c
        )
        xtas, ytas = [], []
        for i in range(x_train.shape[0]):
            num_aug = 0
            x = x_train[i]  # (3, 32, 32)
            x = x.reshape((1,) + x.shape)  # (1, 3, 32, 32)

            for x_aug in datagen.flow(x, batch_size=1, save_to_dir='sample_data/', save_prefix='REK',
                                      save_format='jpeg'):
                if num_aug >= NUM_TO_AUGMENT:
                    break
                xtas.append(x_aug[0])
                num_aug += 1


        '''dumping with the help of pickle'''
        x_train_out, y_train_out = open('x_train_out', 'wb'), open('y_train_out', 'wb')
        pickle.dump(x_train, x_train_out)
        pickle.dump(y_train, y_train_out)

        '''loading again back into the x_train & y-train'''
        x_train_in, y_train_in = open('x_train_out', 'rb'), open('y_train_out', 'rb')
        x_train = pickle.load(x_train_in)
        y_train = pickle.load(y_train_in)


        return NB_CLASSES, CATEGORIES, CBP, x_train, y_train, datagen, IMG_SIZE

