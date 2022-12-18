import  cv2
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier=cv2.CascadeClassifier('haarcascade_eye.xml')
img=cv2.imread('two_people.jpg')
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
faces=face_classifier.detectMultiScale(
    img,scaleFactor=1.1,
    minNeighbors=6,
    minSize=(3,3)
)
print(faces)
for x,y,w,h in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('show',img)

cv2.putText(img,'Find'+str(len(faces))+'face!',(30,img.shape[0]-5),
            cv2.FONT_HERSHEY_PLAIN,5,(255,255,255),2)
cv2.imshow('show2',img)
print('找出人臉之後再找尋眼睛這個部分語法')
roi_gray=gray[y:y+h,x:x+w]
roi_color=img[y:y+h,x:x+w]
eyes=eye_classifier.detectMultiScale(
    roi_gray,scaleFactor=1.1,
    minNeighbors=16
)
print(eyes)
for ex,ey,ew,eh in eyes:
    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)#數字可以改
cv2.imshow('show2',img)
cv2.waitKey()
cv2.destroyAllWindows()