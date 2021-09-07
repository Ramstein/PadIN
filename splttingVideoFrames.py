import cv2
import math

videoFile = "people-walking.mp4"
imagesFolder = r'C:\Users\zeeshan\Desktop\Project\New folder'
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
print(frameRate)
frameRate = 0
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    print(frameId)
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        print(frameId % math.floor(frameRate))
        filename = imagesFolder + "/image_" +  str(int(frameId)) + ".jpg"
        cv2.imwrite(filename, frame)
cap.release()
cv2.destroyAllWindows()

