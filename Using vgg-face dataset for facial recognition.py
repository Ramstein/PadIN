from keras.models import Model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.core import Activation
from keras.optimizers import Adam


from PIL import Image
import numpy as np
import datetime

from keras.preprocessing.image import ImageDataGenerator

'''Downloading datasets from the gdrive''' #https://drive.google.com/file/d/0B4ChsjFJvew3NkF0dTc1OGxsOFU/view?usp=sharing
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='0B4ChsjFJvew3NkF0dTc1OGxsOFU',
                                    dest_path='contents/vggface_weights_tensorflow.h5',
                                    unzip=True)

def bottleneck():
    datagen = ImageDataGenerator(rescale=1.)
    generator = datagen.flow_from_directory(train_data_dir,
                                            target_size=(img_width, img_height),
                                            batch_size=32,
                                            class_mode=None,
                                            shuffle=False)

    pad1_1 = ZeroPadding2D(padding=(1, 1), name='in_train')(img)
    conv1_1 = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(pad1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu', name='conv1_2')(pad1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

    pad2_1 = ZeroPadding2D((1, 1), trainable=False)(pool1)
    conv2_1 = Convolution2D(128, 3, 3, activation='relu', name='conv2_1')(pad2_1)
    pad2_2 = ZeroPadding2D((1, 1), trainable=False)(conv2_1)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu', name='conv2_2')(pad2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), trainable=False)(conv2_2)

    pad3_1 = ZeroPadding2D((1, 1))(pool2)
    conv3_1 = Convolution2D(256, 3, 3, activation='relu', name='conv3_1')(pad3_1)
    pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
    conv3_2 = Convolution2D(256, 3, 3, activation='relu', name='conv3_2')(pad3_2)
    pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu', name='conv3_3')(pad3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), trainable=False)(conv3_3)

    pad4_1 = ZeroPadding2D((1, 1))(pool3)
    conv4_1 = Convolution2D(512, 3, 3, activation='relu', name='conv4_1')(pad4_1)
    pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
    conv4_2 = Convolution2D(512, 3, 3, activation='relu', name='conv4_2')(pad4_2)
    pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu', name='conv4_3')(pad4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

    pad5_1 = ZeroPadding2D((1, 1))(pool4)
    conv5_1 = Convolution2D(512, 3, 3, activation='relu', name='conv5_1')(pad5_1)
    pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu', name='conv5_2')(pad5_2)
    pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu', name='conv5_3')(pad5_3)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)
    fc6 = Convolution2D(4096, 7, 7, activation='relu', name='fc6')(pool5)
    fc6_drop = Dropout(0.5)(fc6)

    model = Model(input=img, output=fc6_drop)
    bottleneck_features_train = model.predict_generator(generator, nb_train_samples)
    np.save(open('features.npy', 'w'), bottleneck_features_train)


def entrenamos_modelo(weights_path=None):
    train_data = np.load(open('features.npy'))
    print(train_data.shape)

    train_labels = np.array(
        [0] * (nb_train_samples / 8) + [1] * (nb_train_samples / 8) + [2] * (nb_train_samples / 8) + [3] * (
                nb_train_samples / 8) + [4] * (nb_train_samples / 8) + [5] * (nb_train_samples / 8) + [6] * (
                nb_train_samples / 8) + [7] * (nb_train_samples / 8))

    lbl1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0], ] * 197)
    lbl2 = np.array([[0, 1, 0, 0, 0, 0, 0, 0], ] * 197)
    lbl3 = np.array([[0, 0, 1, 0, 0, 0, 0, 0], ] * 197)
    lbl4 = np.array([[0, 0, 0, 1, 0, 0, 0, 0], ] * 197)
    lbl5 = np.array([[0, 0, 0, 0, 1, 0, 0, 0], ] * 197)
    lbl6 = np.array([[0, 0, 0, 0, 0, 1, 0, 0], ] * 197)
    lbl7 = np.array([[0, 0, 0, 0, 0, 0, 1, 0], ] * 197)
    lbl8 = np.array([[0, 0, 0, 0, 0, 0, 0, 1], ] * 197)
    label = np.concatenate([lbl1, lbl2, lbl3, lbl4, lbl5, lbl6, lbl7, lbl8])
    '''train_labels --> loss='sparse_categorical_crossentropy'
       labels --> loss='categorical_crossentropy'
    '''

    # MODEL VGG (the old model)
    pad1_1 = ZeroPadding2D(padding=(1, 1), trainable=False, input_shape=(4096, 1, 1), name='in_train')(img)
    conv1_1 = Convolution2D(64, 3, 3, activation='relu', name='conv1_1', trainable=False)(pad1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1), trainable=False)(conv1_1)
    conv1_2 = Convolution2D(64, 3, 3, activation='relu', name='conv1_2', trainable=False)(pad1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), trainable=False)(conv1_2)

    pad2_1 = ZeroPadding2D((1, 1), trainable=False)(pool1)
    conv2_1 = Convolution2D(128, 3, 3, activation='relu', name='conv2_1', trainable=False)(pad2_1)
    pad2_2 = ZeroPadding2D((1, 1), trainable=False)(conv2_1)
    conv2_2 = Convolution2D(128, 3, 3, activation='relu', name='conv2_2', trainable=False)(pad2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), trainable=False)(conv2_2)

    pad3_1 = ZeroPadding2D((1, 1), trainable=False)(pool2)
    conv3_1 = Convolution2D(256, 3, 3, activation='relu', name='conv3_1', trainable=False)(pad3_1)
    pad3_2 = ZeroPadding2D((1, 1), trainable=False)(conv3_1)
    conv3_2 = Convolution2D(256, 3, 3, activation='relu', name='conv3_2', trainable=False)(pad3_2)
    pad3_3 = ZeroPadding2D((1, 1), trainable=False)(conv3_2)
    conv3_3 = Convolution2D(256, 3, 3, activation='relu', name='conv3_3', trainable=False)(pad3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), trainable=False)(conv3_3)

    pad4_1 = ZeroPadding2D((1, 1), trainable=False)(pool3)
    conv4_1 = Convolution2D(512, 3, 3, activation='relu', name='conv4_1', trainable=False)(pad4_1)
    pad4_2 = ZeroPadding2D((1, 1), trainable=False)(conv4_1)
    conv4_2 = Convolution2D(512, 3, 3, activation='relu', name='conv4_2', trainable=False)(pad4_2)
    pad4_3 = ZeroPadding2D((1, 1), trainable=False)(conv4_2)
    conv4_3 = Convolution2D(512, 3, 3, activation='relu', name='conv4_3', trainable=False)(pad4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), trainable=False)(conv4_3)

    pad5_1 = ZeroPadding2D((1, 1), trainable=False)(pool4)
    conv5_1 = Convolution2D(512, 3, 3, activation='relu', name='conv5_1', trainable=False)(pad5_1)
    pad5_2 = ZeroPadding2D((1, 1), trainable=False)(conv5_1)
    conv5_2 = Convolution2D(512, 3, 3, activation='relu', name='conv5_2', trainable=False)(pad5_2)
    pad5_3 = ZeroPadding2D((1, 1), trainable=False)(conv5_2)
    conv5_3 = Convolution2D(512, 3, 3, activation='relu', name='conv5_3', trainable=False)(pad5_3)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), trainable=False)(conv5_3)

    fc6 = Convolution2D(4096, 7, 7, activation='relu', name='fc6', trainable=False)(pool5)
    fc6_drop = Dropout(0.5)(fc6)

    # We TRAIN this layer
    fc7 = Convolution2D(4096, 1, 1, activation='relu', name='fc7', trainable=False)(fc6_drop)
    fc7_drop = Dropout(0.5)(fc7)
    fc8 = Convolution2D(2622, 1, 1, name='fc8', trainable=False)(fc7_drop)
    flat = Flatten()(fc8)
    out = Activation('softmax')(flat)
    model = Model(input=img, output=out)

    # We load the weight of the old model so when we construct ours we dont have to retrain all of it.
    if weights_path:
        model.load_weights(weights_path)

    # We construct our new model: first 14 layers of the old + two new ones. The new FC has to be trained and the Softmax layer too.
    fc7_n = Convolution2D(4096, 1, 1, activation='relu', name='fc7_n', trainable=True,
                          input_shape=train_data.shape[1:])(fc6_drop)
    fc7_drop_n = Dropout(0.5)(fc7_n)
    fc8_n = Convolution2D(8, 1, 1, name='fc8_n', trainable=False)(fc7_drop_n)
    flat_n = Flatten(name='flat_n')(fc8_n)
    out_n = Activation('softmax')(flat_n)
    model2 = Model(input=img, output=out_n)
    # model2.summary()

    sgd = Adam(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    '''train_labels --> loss='sparse_categorical_crossentropy'
       labels --> loss='categorical_crossentropy'
    '''

    model2.fit(train_data, label, nb_epoch=nb_epoch, batch_size=64)
    print('Model Trained')

    # We save the weights so we can load them in our model
    date = datetime.datetime.now()
    model2.save_weights(f"M:{date:%Y-%m-%d_%Hh%Mm%Ss}")  # always save your weights after training or during training

    # We have two options: 1) Return the model here or in the vgg_trained_model Function
    return model2


if __name__ == "__main__":
    im = Image.open('A.J._Buckley.jpg')
    im = im.resize((224, 224))
    im = np.array(im).astype(np.float32)
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)

    # For the training stage
    img_width, img_height = 224, 224
    img = Input(shape=(3, img_height, img_width))
    # validation_data_dir = 'data/validation'
    nb_train_samples = 1576  # 197 per class and we have 8 classes (8 emotions)
    nb_validation_samples = 0
    nb_epoch = 20

    # Stages to construct the model
    model = entrenamos_modelo('vggface_weights_tensorflow.h5')  # Construction of the model

    # model.summary()

    out = model.predict(im)
    print(out[0][0])