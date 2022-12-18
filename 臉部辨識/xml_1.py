import  cv2
casc_path='haarcascade_frontalface_default.xml'
faceCascade=cv2.CascadeClassifier(casc_path)
imagename=cv2.imread('group_photos.jpg')
faces=faceCascade.detectMultiScale(
    imagename,scaleFactor=1.1,
    minNeighbors=4,
    minSize=(3,3),
    maxSize=(57,57)
)
print(faces)
for x,y,w,h in faces:
    cv2.rectangle(imagename,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('show',imagename)

cv2.putText(imagename,'Find'+str(len(faces))+'face!',(30,imagename.shape[0]-5),
            cv2.FONT_HERSHEY_PLAIN,5,(255,255,255),2)
cv2.imshow('show2',imagename)
cv2.waitKey()
cv2.destroyAllWindows()
print('進行儲存')
from PIL import Image
count=1
for x,y,w,h in faces:
    cv2.rectangle(imagename,(x,y),(x+w,y+h),(128,255,0),2)
    filename='media//face'+str(count)+'.jpg'
    image1=Image.open('group_photos.jpg')
    image2=image1.crop((x,y,x+w,y+h))
    #image3=image2.resize((70,70),Image.ANTIALIAS)
    image2.save(filename)
    count+=1