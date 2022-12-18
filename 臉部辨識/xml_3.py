import  cv2
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_haar(filename):
    img=cv2.imread(filename)
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    faces=face_classifier.detectMultiScale(
    img,scaleFactor=1.1,
    minNeighbors=6,
    minSize=(3,3)
    )
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('show', img)

print('加入dlib')
import  dlib
from  imutils import  face_utils
print('產生一個dlib偵測臉部特徵物件')
detect=dlib.get_frontal_face_detector()
print('產生一個dlib臉部特徵標示物件')
predictoer=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def detect_face_landmarks(filename):
    img=cv2.imread(filename)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #不一樣的地方
    faces=detect(gray,1)
    for face in faces:
        #print(face)
        shape=predictoer(gray,face)
        #print(shape)
        print('產生數值資訊')
        shape=face_utils.shape_to_np(shape)
        print(shape)
        print('繪製臉部資訊')
        x,y,w,h=face_utils.rect_to_bb(face)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,0, 0),3)
        print('將數值資訊轉換為座標位置')
        for x,y in shape:
            cv2.circle(img,(x,y),5,(0,0,255),-1)

    cv2.imshow('show2', img)
#filename='two_people.jpg'
filename='./images_face/classmates.jpg'
detect_haar(filename)
detect_face_landmarks(filename)
cv2.waitKey()
cv2.destroyAllWindows()