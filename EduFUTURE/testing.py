
'''DownloadingData required imports'''
import wget
from zipfile import ZipFile
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

'''preparing data required imports'''
import numpy as np
import os, copy, cv2, random, pickle, datetime

# from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras.utils import np_utils

# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import np_utils


'''required imports for fittig the model'''
import matplotlib.pyplot as plt

# from tensorflow.python.keras.optimizers import Adam






# url = 'https://drive.google.com/open?id=1mcM9I-Z7NXESXVs0zLAbithwrWUNrmAj'
url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'

fileName = 'Actors_Face_detection.zip'                 #'''for gdrive file the name of the file is not
                                  #  inferrable by the detect_filename()
                                   # , so is filename is not equal to none means
                                    #it's a gDrive file'''

path = r'C:\Users\zeeshan\PycharmProjects\PadIN 1.0\EduFUTURE'




'''preparing training data variables'''
IMG_SIZE = [299, 299]  #instagram landscape picture size [16:9] input size image
DATADIR = r'C:\Users\zeeshan\PycharmProjects\PadIN 1.0\EduFUTURE'
NB_CLASSES, i = 0, 0
NUM_TO_AUGMENT = 5
CATEGORIES, CBP, img_array, resized_img_array = [], [], [], []
training_data, x_train, y_train = [], [], []


'''fitting training data variables'''
DROPOUT=0.3
NB_EPOCHS=10
BATCH_SIZE=256
# OPTIMIZER = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  #learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08

dense_layers = [2, 1, 0]  # 0, 1,2
layer_sizes = [1028, 512, 256, 128]  # 128, 256, 512
conv_layers = [6, 5, 4, 3, 2, 1]  # 1, 2,3




def build():
    '''
    Overall calls all of the methods with required parameters
    :return: FineTuned Model in your drive
    '''
    #authenticating to drive
    # print('----Authenticating to googleDrive----')
    # drive = authenticate()

    #downlaoding data
    extractedTo = download(url=url,
                                   path=path,
                                   # drive=drive,
                           fileName=fileName)

    #preparing Training data
    NB_CLASSES, CATEGORIES, CBP, x_train, y_train, datagen, IMG_SIZE = prepTrainingData(path=extractedTo)


'''Downlaoding the data'''
def download(url, path, fileName):

    if fileName:  # if the filename is given and not a null string then
        if not os.path.isfile(os.path.join(path, fileName)):
            print('----Downloading Data----')

            id = url.split('=')[len(url.split('=')) - 1]

            # last_weight_file = drive.CreateFile({'id': id})
            # last_weight_file.GetContentFile(fileName)

            print('----Extracting the file----')
            with ZipFile(os.path.join(path, fileName), 'r') as zipObj:  # extracts the file in sample_data/examples-master/cpp & dcgan & imagenet
                zipObj.extractall(path)
        else:
            print('File is already available.\n'
                  '----Extracting the file----')
            path = os.path.join(r'C:\Users\zeeshan\PycharmProjects\PadIN 1.0\EduFUTURE', fileName.split('.')[0])

            if not os.path.exists(path):
                os.makedirs(path)

            with ZipFile(path, 'r') as zipObj:  # extracts the file in sample_data/examples-master/cpp & dcgan & imagenet
                zipObj.extractall(path)

    if not fileName:
        fileName = wget.detect_filename(url=url)
        if not os.path.isfile(os.path.join(path, fileName)):
            print('----Downloading Data----')
            wget.download(url, 'sample_data')
            fileName = wget.detect_filename(url=url)

            print('----Extracting the file----')

            with ZipFile(os.path.join(path, fileName), 'r') as zipObj:  # extracts in the sample_data/PetImages/Cat & Dog
                # Extract all the contents of zip file in different directory
                zipObj.extractall(path)
        else:
            print('File is already available.\n----Extracting the file----')
            with ZipFile(os.path.join(path, fileName), 'r') as zipObj:  # extracts the file in sample_data/examples-master/cpp & dcgan & imagenet
                zipObj.extractall(path)
    extractedTo = path
    print('----Extraction Done---- \npathToExtractedData: {}'.format(extractedTo))
    return extractedTo


'''Preparing data'''

def prepTrainingData(path):

    print('----Preparing training data----')
    IMG_SIZE = [299, 299]  # instagram landscape picture size [16:9] input size image
    DATADIR = path
    NB_CLASSES, i = 0, 0
    NUM_TO_AUGMENT = 5
    CATEGORIES, CBP, img_array, resized_img_array = [], [], [], []
    training_data, x_train, y_train = [], [], []
    print('Getting Number of NB_CLASSES, CATEGORIES and CBP.')
    for r, d, f in os.walk(DATADIR):
        if i == 1:
            CATEGORIES = copy.copy(d)
            NB_CLASSES = len(d)
            i += 1
        if i == 0:
            CBP = d[i].split('_')
            i += 1
    newDATADIR = os.path.join(DATADIR, CBP[0])
    print('NB_CLASSES: ', NB_CLASSES)
    print("CATEGORIES: ", CATEGORIES)
    print("CBP: ", CBP)

    '''Creating taining data'''
    print('Creating Training data....')
    for category in CATEGORIES:
        path = os.path.join(newDATADIR, category)  # path to all category images
        classIndex = CATEGORIES.index(category)  # 0 for first file of image

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_img_array = cv2.resize(img_array, (IMG_SIZE[0], IMG_SIZE[1]))
                training_data.append([resized_img_array, classIndex])
            except Exception as e:
                pass

    print('Length of training_data: ', len(training_data),
          '\n len(training_data[0]): ', len(training_data[0]))

    # random.shuffle(training_data)  # shuffle training data

    print('''separating the features and the labels as x_train and y_train''')
    for feature, label in training_data:
        x_train.append(feature)
        y_train.append(label)

    x_train = np.array(x_train).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 3)
    x_train = x_train.astype('float32') / 255.0

    # y_train = np_utils.to_categorical(y_train, NB_CLASSES)

    # '''Augmenting using ImageDataGenerator
    # '''
    # print("Augmenting training set images....")
    # datagen = ImageDataGenerator(
    #     rotation_range=40,  # 0-180 degree for randomly rotataing pics
    #     width_shift_range=[-20, 20],  # for randomly tranlating pics vertically or horizontally
    #     height_shift_range=[-30, 30],
    #     brightness_range=[0.5, 1.0],
    #     shear_range=20,
    #     zoom_range=0.2,  # for randomly zooming pictures
    #     horizontal_flip=True,  # for randomly flipping half of the images horizantally
    #     fill_mode='nearest',
    #     # is the strategy used for filling in new pixels that can appear after a rotation or a shift
    #     data_format='channel_last'  # s, w, d, c
    # )
    # xtas, ytas = [], []
    # for i in range(x_train.shape[0]):
    #     num_aug = 0
    #     x = x_train[i]  # (3, 32, 32)
    #     x = x.reshape((1,) + x.shape)  # (1, 3, 32, 32)
    #
    #     for x_aug in datagen.flow(x, batch_size=1, save_to_dir='sample_data/', save_prefix='REK',
    #                               save_format='jpeg'):
    #         if num_aug >= NUM_TO_AUGMENT:
    #             break
    #         xtas.append(x_aug[0])
    #         num_aug += 1

    date = datetime.datetime.now()

    print('''dumping with the help of pickle''')
    x_train_out, y_train_out = open('{}_x_train.pickle'.format(f":{date:%Y-%m-%d-%Hh%Mm%Ss}"), 'wb'), \
                               open('{}_y_train.pickle'.format(f":{date:%Y-%m-%d-%Hh%Mm%Ss}"), 'wb')
    pickle.dump(x_train, x_train_out)





    pickle.dump(y_train, y_train_out)
    print(NB_CLASSES, CATEGORIES, CBP, x_train, y_train, IMG_SIZE)

    return NB_CLASSES, CATEGORIES, CBP, x_train, y_train, IMG_SIZE


if __name__=='__main__':
    build()
