import cv2

#define the cascades......this is just loading of cascades ..xml file
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface16.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye16.xml')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces will be find
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #to mark a rectangle
    for (x,y,w,h) in faces:
        #x,y and x+w ,y+h are co ordinates
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        #roi of face
        #since eyes are in faces
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]         #to impose it on color frame
        #ro detect eye ......eye_cascade
        eyes = eye_cascade.detectMultiScale(roi_gray)

        #to mark a rectangle for eye

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

    cv2.imshow('frame',frame)

    k= cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()