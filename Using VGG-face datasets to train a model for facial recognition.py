

from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.core import Activation
from keras.optimizers import Adam

from PIL import Image
import numpy as np
import datetime

from keras.preprocessing.image import ImageDataGenerator

# '''Downloading datasets from the gdrive'''  # https://drive.google.com/file/d/0B4ChsjFJvew3NkF0dTc1OGxsOFU/view?usp=sharing
# from google_drive_downloader import GoogleDriveDownloader as gdd
#
# gdd.download_file_from_google_drive(file_id='0B4ChsjFJvew3NkF0dTc1OGxsOFU',
#                                     dest_path='contents/vggface_weights_tensorflow.h5',
#                                     unzip=True)

new_model = load_model('nn4.small2.v2.h5')
model = new_model.output





if __name__ == "__main__":
    im = Image.open('vlcsnap-2019-02-24-21h26m43s779.png')
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


    # model.summary()

    out = new_model.predict(im)
    print(out[0][0])