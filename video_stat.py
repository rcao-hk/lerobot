import numpy as np
import cv2 as cv
 
cap = cv.VideoCapture('data/imitation_data/data1/Kinect.avi')

# print(cap)
# cap.set(cv.CAP_PROP_FPS, 1/0.035)
# get the FPS of the video
# fps = cap.get(cv.CAP_PROP_FPS)
# print(fps)
fps = 28.0

count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        # print("No frame")
        break
    else:
        count += 1

print(count)
# print(frame)
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
#     cv.imshow('frame', gray)
#     if cv.waitKey(1) == ord('q'):
#         break
 
# cap.release()
# cv.destroyAllWindows()