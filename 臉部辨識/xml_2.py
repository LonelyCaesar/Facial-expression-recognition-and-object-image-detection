import  cv2
import  matplotlib.pyplot as plt
eye_cascade='./cascade_files/haarcascade_eye_tree_eyeglasses.xml'
classifier=cv2.CascadeClassifier(eye_cascade)
smile_cascade='./cascade_files/haarcascade_smile.xml'
smile_classifier=cv2.CascadeClassifier(smile_cascade)
image=cv2.imread('./images_face/classmates.jpg')
img_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
bboxes=classifier.detectMultiScale(
    image,scaleFactor=1.1,
    minNeighbors=20
)
print(bboxes)
for x,y,w,h in bboxes:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('show',image)

bboxes2=smile_classifier.detectMultiScale(
    image,scaleFactor=2.5,
    minNeighbors=20,
    minSize=(55,55)#數字可以改
)
print(bboxes2)
for x,y,w,h in bboxes2:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imshow('show2',image)
cv2.waitKey()
cv2.destroyAllWindows()