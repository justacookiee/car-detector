import cv2

video = cv2.VideoCapture('dataset_video1.avi')
car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
    ret, frame = video.read()
    if (type(frame) == type(None)):
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    car = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x, y, w, h) in car:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (150,0,0), 2)
    cv2.imshow('Car Detector', frame)
    if cv2.waitKey(30) == 27:
        break
cv2.destroyAllWindows()
