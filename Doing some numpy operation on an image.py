import cv2, os
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 299
path = r'C:\Users\zeeshan\PycharmProjects\PadIN 1.0'
img = 'vlcsnap-2019-02-24-21h26m43s779.png'

img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
# print(img_array)

# plt.imshow(img_array, cmap='plasma')
# plt.show()

training_data , classIndex= [], [1]

resized_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# print(resized_img_array)

training_data.append([resized_img_array, classIndex[0]])

print(training_data)

print('Length of training data: ',len(training_data))

x_train, y_train= [], []

for feature, label in training_data:
    x_train.append(feature)
    y_train.append(label)


x_train = np.array(x_train).astype('float32')#.reshape(IMG_SIZE, IMG_SIZE)
print(x_train)
# x_train = x_train[:, np.newaxis, : , : ]

x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

print(x_train)

















