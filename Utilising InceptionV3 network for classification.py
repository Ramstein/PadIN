from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
import numpy as np
import cv2

NDARRAY = np.expand_dims(im, axis=0)

'''prebuilt model eith prebuilt weights on imagenet'''
model = InceptionV3(weights='imagenet', include_top=True)
Adam = Adam(lr=0.1, decay=1e-6, epsilon=1e-6) #lr=10**-3, decay=1e-7, epsilon=1e-6
model.compile(optimizer=Adam, loss='categorical_crossentropy')

'''resize into VGG16 trained images format'''
im = cv2.resize(cv2.imread('vlcsnap-2019-02-24-21h26m43s779.png', cv2.IMREAD_COLOR), (299,299))
im = NDARRAY

'''predict'''
out = model.predict(im)
plt.plot(out.ravel())
plt.show()
print(np.argmax(out))


