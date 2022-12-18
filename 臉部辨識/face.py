import  face_recognition,cv2
video_capture=cv2.VideoCapture('0612.mp4')
#video_capture=cv2.VideoCapture('short_hamilton_clip.mp4')
frames=[]
frame_count=0
while video_capture.isOpened():
    ret,frame=video_capture.read()
    #print(ret)
    if not ret: #False
        break
    frame=frame[:,:,::-1]
    frame_count+=1
    frames.append(frame)
    if len(frames)==128:
        print('ok')
        batch_of_face_locations=face_recognition.batch_face_locations(
            frames,number_of_times_to_upsample=0
        )
        for frame_number_in_batch,face_locations in enumerate(batch_of_face_locations):
            number_of_faces_in_frame=len(face_locations)
            frames_number=frame_count-128+frame_number_in_batch
            print('我在',frames_number,'影格內找到',number_of_faces_in_frame)
            for face_location in face_locations:
                top,right,bottom,left=face_location
                print('top:',top,'left:',left)
                print('bottom:',bottom,'right:',right)
