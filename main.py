import cv2
import os
from matplotlib import pyplot as plt
img=cv2.imread('picture_detection.jpg')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cascade_path=os.path.join(cv2.data.haarcascades,"haarcascade_frontalface_default.xml")
eyes_path=os.path.join(cv2.data.haarcascades,"haarcascade_eye.xml")
face_data=cv2.CascadeClassifier(cascade_path)
eyes_data=cv2.CascadeClassifier(eyes_path)

result=face_data.detectMultiScale(img_gray,minSize=(20,20))
print(result)
for(x,y,w,h) in result:
    cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(255,0,0),20)
    face_roi=img_gray[y:y+h,x:x+w]
    eyes=eyes_data.detectMultiScale(face_roi,minSize=(20,20))
    for (ex,ey,ew,eh) in eyes:
        cx=x+ex+ew//2
        cy=y+ey+eh//2
        cv2.circle(img_rgb,(cx,cy),max(ew,eh)//2,(0,255,0),10)

plt.imshow(img_rgb)
plt.title('Найденые лица')
plt.show()