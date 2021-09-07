from keras.models import Model
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
import cv2


'''prebuilt model eith prebuilt weights on imagenet'''
model = VGG16(weights='imagenet', include_top=True)
Adam = Adam(lr=0.1, decay=1e-6, epsilon=1e-6) #lr=10**-3, decay=1e-7, epsilon=1e-6
model.compile(optimizer=Adam, loss='categorical_crossentropy')

'''resize into VGG16 trained images format'''
im = cv2.resize(cv2.imread('vlcsnap-2019-02-24-21h26m43s779.png', cv2.IMREAD_COLOR), (224,224))
im = np.expand_dims(im, axis=0)

'''predict'''
out = model.predict(im)
plt.plot(out.ravel())
plt.show()
print(np.argmax(out))


