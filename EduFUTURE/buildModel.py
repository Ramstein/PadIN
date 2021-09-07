!pip install wget pydrive


'''authenticateTOGdrive required imports'''
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from google.colab import auth
from oauth2client.client import GoogleCredentials


'''DownloadingData required imports'''
import wget
from zipfile import ZipFile

'''preparing data required imports'''
import numpy as np
import os, copy, cv2, random, pickle, datetime

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import np_utils

'''required imports for fittig the model'''
import matplotlib.pyplot as plt
import tensorflow as tf


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split

from tensorflow.python.keras.models import load_model


'''Starting Build ALL important changing parametera are here'''

'''1. Downloading parameters'''

# url = 'https://drive.google.com/open?id=1mcM9I-Z7NXESXVs0zLAbithwrWUNrmAj'
url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'

fileName = ''                 #'''for gdrive file the name of the file is not
                                  #  inferrable by the detect_filename()
                                   # , so is filename is not equal to none means
                                    #it's a gDrive file'''

path = 'sample_data'
pathTOExtractedData = ''



'''preparing training data variables'''
IMG_SIZE = [299, 299]  #instagram landscape picture size [16:9] input size image
DATADIR = 'sample_data/'
NB_CLASSES, i = 0, 0
NUM_TO_AUGMENT = 5
CATEGORIES, CBP, img_array, resized_img_array = [], [], [], []
training_data, x_train, y_train = [], [], []


'''fitting training data variables'''
DROPOUT=0.3
NB_EPOCHS=10
BATCH_SIZE=256
OPTIMIZER = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  #learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08

dense_layers = [2, 1, 0]  # 0, 1,2
layer_sizes = [1028, 512, 256, 128]  # 128, 256, 512
conv_layers = [6, 5, 4, 3, 2, 1]  # 1, 2,3

def build():
    '''
    Overall calls all of the methods with required parameters
    :return: FineTuned Model in your drive
    '''
    #authenticating to drive
    print('----Authenticating to googleDrive----')
    drive = authenticate()

    #downlaoding data
    extractedTo = download(url=url,
                                   path=path,
                                   drive=drive, fileName=fileName)

    #preparing Training data
    NB_CLASSES, CATEGORIES, CBP, x_train, y_train, datagen, IMG_SIZE = prepTrainingData(
        path=extractedTo)



    '''fitting the model on the training data'''
    max, min, testScore, accIndex, lossIndex, testIndex, date = fitModel(NB_CLASSES=NB_CLASSES,
                                                                         x_train=x_train,
                                                                         y_train=y_train,
                                                                         datagen=datagen,
                                                                         IMG_SIZE=IMG_SIZE,
                                                                         CBP=CBP)
    '''uploading model to the gDrive'''
    uploadFineModel(drive=drive, max=max, min=min, testScore=testScore,
                    accIndex=accIndex, lossIndex=lossIndex, testIndex=testIndex,
                    date=date, CBP=CBP)


'''authentication process to gDrive'''
def authenticate():
    '''
    authenticates to google drive using the pydrive module
    :return: drive : a reference to the GoogleDrive() for doing any type
            of CreateFile() or GetContentFile() operations
    '''
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    print('-'*4+'Authnticated to GoogleDrive, Take care of adc.json file.\n it can compromise your account.'+'-'*4)
    return drive


'''Downlaoding the data'''
def download(url, path, drive, fileName):
    '''
    download() method is here for downloading the dataset or images in the
    form of a zip or whatever and extracts those files.
    :param url:
    :param path:    path = 'sample_data'

    :param drive:
    :param fileName:
    :return: path to the extracted data
    '''
    if fileName:  # if the filename is given and not a null string then
        if not os.path.isfile(os.path.join(path, fileName)):
            print('----Downloading Data----')

            id = url.split('=')[len(url.split('=')) - 1]

            last_weight_file = drive.CreateFile({'id': id})
            last_weight_file.GetContentFile(fileName)

            print('----Extracting the file----')
            with ZipFile(os.path.join(path, fileName), 'r') as zipObj:  # extracts the file in sample_data/examples-master/cpp & dcgan & imagenet
                zipObj.extractall(path)
        else:
            print('File is already available.\n----Extracting the file----')
            with ZipFile(os.path.join(path, fileName), 'r') as zipObj:  # extracts the file in sample_data/examples-master/cpp & dcgan & imagenet
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
    print('----Extraction Done---- \npathToExtractedData: {}'.
          format(extractedTo))
    return extractedTo


'''Preparing data'''

def prepTrainingData(path):
    '''

    :param path: this is the to which the compressed file has been extracted to
    :return:      returns NB_CLASSES, CATEGORIES, CBP,
            x_train, y_train, datagen, IMG_SIZE

    '''
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
        fill_mode='nearest',
        # is the strategy used for filling in new pixels that can appear after a rotation or a shift
        data_format='channel_last'  # s, w, d, c
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

    print('''dumping with the help of pickle''')
    x_train_out, y_train_out = open('x_train_out', 'wb'), open('y_train_out', 'wb')
    pickle.dump(x_train, x_train_out)
    pickle.dump(y_train, y_train_out)

    print('''loading again back into the x_train & y-train''')
    x_train_in, y_train_in = open('x_train_out', 'rb'), open('y_train_out', 'rb')
    x_train = pickle.load(x_train_in)
    y_train = pickle.load(y_train_in)

    return NB_CLASSES, CATEGORIES, CBP, x_train, y_train, datagen, IMG_SIZE


'''compiling and training the model'''

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
    date = datetime.datetime.now()
    max, min, testScore, accIndex, lossIndex, testIndex = 70.0, 4.0, 75, 1, 1, 1
    test_score_out = []
    INPUT_SHAPE = IMG_SIZE + [3]  # instagram landscape picture size [16:9] i.e. [1080, 608] input size image

    print('''splitting x_train in train , validtion and test sets''')
    # train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.125, random_state=1)

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer,
                                                             f"{datetime.datetime.now():%m-%d-%Hh%Mm%Ss}")

                model = Sequential()
                model.add(Conv2D(layer_size, (8, 8), padding='valid',
                                 input_shape=INPUT_SHAPE, kernel_regularizer=l2(0.001)))
                model.add(Activation('relu'))
                # model.add(BatchNormalization())
                model.add(Dropout(DROPOUT))
                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

                for l in range(conv_layer - 1):
                    model.add(Conv2D(layer_size, (8, 8), padding='valid',
                                     kernel_regularizer=l2(0.001)))
                    model.add(Activation('relu'))
                    # model.add(BatchNormalization())
                    model.add(Dropout(DROPOUT))
                    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

                model.add(Flatten())

                for _ in range(dense_layer):
                    model.add(Dense(layer_size, kernel_regularizer=l2(0.001)))
                    model.add(Activation('relu'))
                    model.add(Dropout(DROPOUT))

                model.add(Dense(NB_CLASSES))
                model.add(Activation('sigmoid'))
                tensorboard = TensorBoard(log_dir="sample_data/{}".format(NAME))

                model.compile(loss='categorical_crossentropy',
                              optimizer=OPTIMIZER,
                              metrics=['accuracy'],
                              )
                # fit the datagen to the x_train
                datagen.fit(x_train)
                # train the model
                history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                                              steps_per_epoch=len(x_train) // BATCH_SIZE,
                                              epochs=NB_EPOCHS,
                                              verbose=1,
                                              callbacks=[tensorboard],
                                              validation_data=(x_val, y_val),
                                              )

                '''saving the comparatively best one'''
                if history.history.get('val_acc')[-1] > max:
                    if accIndex >= 2:
                        os.remove('ValAcc{}_{}_{}_{}_{}_{}'.format(accIndex - 1, round(max, 4), CBP[0], CBP[1], CBP[2],
                                                                   f":{date:%Y-%m-%d-%Hh%Mm%Ss}"))
                    max = history.history.get('val_acc')[-1]
                    val_acc_out = open('ValAcc{}_{}_{}_{}_{}_{}'.format(accIndex, round(max, 4), CBP[0], CBP[1], CBP[2],
                                                                        f":{date:%Y-%m-%d-%Hh%Mm%Ss}"), "wb")
                    pickle.dump(
                        model.save('ValAcc{}_{}_{}_{}_{}_{}'.format(accIndex, round(max, 4), CBP[0], CBP[1], CBP[2],
                                                                    f":{date:%Y-%m-%d-%Hh%Mm%Ss}")),
                        val_acc_out)
                    val_acc_out.close()
                    accIndex += 1

                if history.history.get('val_loss')[-1] < min:
                    if lossIndex >= 2:
                        os.remove(
                            'ValLoss{}_{}_{}_{}_{}_{}'.format(lossIndex - 1, round(min, 4), CBP[0], CBP[1], CBP[2],
                                                              f":{date:%Y-%m-%d-%Hh%Mm%Ss}"))
                    min = history.history.get('val_loss')[-1]
                    val_loss_out = open(
                        'ValLoss{}_{}_{}_{}_{}_{}'.format(lossIndex, round(min, 4), CBP[0], CBP[1], CBP[2],
                                                          f":{date:%Y-%m-%d-%Hh%Mm%Ss}"))
                    pickle.dump(
                        model.save('ValLoss{}_{}_{}_{}_{}_{}'.format(lossIndex, round(min, 4), CBP[0], CBP[1], CBP[2],
                                                                     f":{date:%Y-%m-%d-%Hh%Mm%Ss}")),
                        val_loss_out)
                    val_loss_out.close()
                    lossIndex += 1

                score = model.evaluate(x_test, y_test, verbose=1)  # score[testScore, testAccuracy]

                if score[1] > testScore:
                    if testIndex >= 2:
                        os.remove(
                            'TestScore{}_{}_{}_{}_{}_{}'.format(testIndex - 1, round(testScore, 4), CBP[0], CBP[1],
                                                                CBP[2],
                                                                f":{date:%Y-%m-%d-%Hh%Mm%Ss}"))
                        os.remove('{}_FinestHistory'.format(testIndex - 1))
                    testScore = score[1]
                    test_acc_out = open(
                        'TestScore{}_{}_{}_{}_{}_{}'.format(testIndex, round(testScore, 4), CBP[0], CBP[1], CBP[2],
                                                            f":{date:%Y-%m-%d-%Hh%Mm%Ss}"), "wb")
                    pickle.dump(
                        model.save(
                            'TestScore{}_{}_{}_{}_{}_{}'.format(testIndex, round(testScore, 4), CBP[0], CBP[1], CBP[2],
                                                                f":{date:%Y-%m-%d-%Hh%Mm%Ss}")),
                        test_acc_out)
                    test_acc_out.close()
                    '''dumping best scoring history for future use'''
                    test_score_history = open('{}_FinestHistory'.format(testIndex), "wb")
                    pickle.dump(history, test_score_history)
                    test_score_out = copy.copy(score)
                    test_score_history.close()

                    testIndex += 1

    print('''drawing the diagram for accuracy and loss for best saved model''')
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


'''Uploding the model to gDrive'''
def uploadFineModel(drive, max, min, testScore, accIndex, lossIndex, testIndex, date, CBP):
    '''

    :param drive:
    :param max:
    :param min:
    :param testScore:
    :param accIndex:
    :param lossIndex:
    :param testIndex:
    :param date:
    :param CBP:
    :return:
    '''
    print("Uploading the fine model to drive.")

    # uploading best accuracy acquired model
    val_acc_upload = open(
        '{}_{}_{}_{}_{}_{}'.format(accIndex, round(max, 4), CBP[0], CBP[1], CBP[2], f":{date:%Y-%m-%d-%Hh%Mm%Ss}"),
        'rb')
    val_acc_up = pickle.load(val_acc_upload)

    model_acc = drive.CreateFile({"title": '{}_{}_{}_{}_{}_{}'.format(accIndex, round(min, 4), CBP[0], CBP[1], CBP[2],
                                                                       f":{date:%Y-%m-%d-%Hh%Mm%Ss}")})
    model_acc.SetContentFile(val_acc_up)
    model_acc.Upload()
    accFileLink = 'https://drive.google.com/open?id='+str(model_acc.get('id'))


    # Uploading lowest loss acquired model
    val_loss_upload = open(
        '{}_{}_{}_{}_{}_{}'.format(lossIndex, round(min, 4), CBP[0], CBP[1], CBP[2], f":{date:%Y-%m-%d-%Hh%Mm%Ss}"),
        'rb')
    val_loss_up = pickle.load(val_loss_upload)
    model_loss = drive.CreateFile({"title": '{}_{}_{}_{}_{}_{}'.format(lossIndex, round(min, 4), CBP[0], CBP[1], CBP[2], f":{date:%Y-%m-%d-%Hh%Mm%Ss}")})
    model_loss.SetContentFile(val_loss_up)
    model_loss.Upload()
    lossFileLink = 'https://drive.google.com/open?id='+str(model_loss.get('id'))


    # Uploading the best scoring model on the fresh unseen data

    val_score_upload = open(
        '{}_{}_{}_{}_{}_{}'.format(testIndex, round(testScore, 4), CBP[0], CBP[1], CBP[2],
                                   f":{date:%Y-%m-%d-%Hh%Mm%Ss}"),
        'rb')
    val_score_up = pickle.load(val_score_upload)
    model_score = drive.CreateFile({"title": '{}_{}_{}_{}_{}_{}'.format(testIndex, round(testScore, 4), CBP[0],
                                                                        CBP[1], CBP[2],
                                                                        f":{date:%Y-%m-%d-%Hh%Mm%Ss}")})
    model_score.SetContentFile(val_score_up)
    model_score.Upload()
    scoreFileLink = 'https://drive.google.com/open?id=' + str(model_score.get('id'))

    # Printing the links of those all file to load them into the continuously evaluating model
    print('val_acc model link: ', accFileLink)
    print('val_loss model link: ', lossFileLink)
    print('test_score model link: ', scoreFileLink)