# import os
# import cv2
# import numpy as np
#
# INPUT_VIDEO = 'test.mp4'
# OUTPUT_IMG = 'out_my_video'
# os.makedirs(OUTPUT_IMG, exist_ok=True)
#
#
# def print_image(img, frame_diff):
#     """
#     Place images side-by-side
#     """
#     new_img = np.zeros([img.shape[0], img.shape[1] * 2, img.shape[2]])  # [height, width*2, channel]
#     new_img[:, :img.shape[1], :] = img  # place color image on the left side
#     new_img[:, img.shape[1]:, 0] = frame_diff  # place gray image on the right side
#     new_img[:, img.shape[1]:, 1] = frame_diff
#     new_img[:, img.shape[1]:, 2] = frame_diff
#     return new_img
#
#
# def main(video_path):
#     cap = cv2.VideoCapture(video_path)  # https://docs.opencv.org/4.0.0/d8/dfe/classcv_1_1VideoCapture.html
#     last_gray = None
#     idx = -1
#     while (True):
#         ret, frame = cap.read()  # read frames
#         idx += 1
#         if not ret:
#             print('Stopped reading the video (%s)' % video_path)
#             break
#
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert color image to gray
#
#         if last_gray is None:
#             last_gray = gray
#             continue
#
#         diff = cv2.absdiff(gray,
#                            last_gray)  # frame difference! https://docs.opencv.org/4.0.0/d2/de8/group__core__array.html#ga6fef31bc8c4071cbc114a758a2b79c14
#         cv2.imwrite(os.path.join(OUTPUT_IMG, 'img_%06d.jpg' % idx), print_image(frame, diff))
#         last_gray = gray
#         print('Done image @ %d...' % idx)
#         pass
#     pass
#
#
# if __name__ == "__main__":
#     print('Running frame difference algorithm on %s' % INPUT_VIDEO)
#     main(video_path=INPUT_VIDEO)
#     print('* Follow me @ ' + "\x1b[1;%dm" % (34) + ' https://www.facebook.com/minhng.info/' + "\x1b[0m")
#     print('* Join GVGroup for discussion @ ' + "\x1b[1;%dm" % (
#         34) + 'https://www.facebook.com/groups/ip.gvgroup/' + "\x1b[0m")
#     print('* Thank you ^^~')
#
#     print('[NOTE] Run the following command to turn you images in to video:')
#     print(
#         'ffmpeg -framerate 24 -f image2 -start_number 1 -i out_my_video/img_%*.jpg -crf 10 -q:v 5  -pix_fmt yuv420p out_video.mp4')
# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
# parameters for loading data and images
detection_model_path = 'haarcascade/haarcascade_frontalface_default.xml'
# loading models

# load model facial_expression
model_facial_expression = model_from_json(open("model/fer.json", "r").read())
# load weights facial_expression
model_facial_expression.load_weights('model/fer.h5')

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

face_detection = cv2.CascadeClassifier(detection_model_path)
path_video = "democlassroom.mp4"
video = cv2.VideoCapture(path_video)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = width, height
# Print out the resolution
print(repr(size))

# Set the number of frames and the background
FPS_SMOOTHING = 0.9

# ret, frame1 = video.read()
ret, frame2 = video.read()
frame1 = None
next_frame = 0
fps = 0.0
prev = time.time()


while video.isOpened():

    status, color = "No Movement", (0, 255, 0)
    no_movement_check = False
    now = time.time()
    fps = (fps * FPS_SMOOTHING + (1 / (now - prev)) * (1.0 - FPS_SMOOTHING))
    print("fps: {:.1f}".format(fps))

    ret, frame2 = video.read()
    if frame2 is None:
        break

    if frame1 is None:
        frame1 = frame2

    difference = cv2.absdiff(frame1, frame2)
    thresh = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]


    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)


    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilate = cv2.dilate(threshold, None, iterations=3)
    contour, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnts= cv2.findContours(dilate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    # for c in cnts:
    #     # if the contour is too small, ignore it
    #     # if cv2.contourArea(c) < args["min_area"]:
    #     #     continue
    #     # compute the bounding box for the contour, draw it on the frame,
    #     # and update the text
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     status = "Occupied"
    #     no_movement_check = True
    if cnts is not None:
        status = "Occupied"
        no_movement_check = True
    if next_frame %2 == 0 and no_movement_check:
        gray_face = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            if y+w >10 and x+h >10:
            # cv2.rectangle(frame1, (x_f, y_f), (x_f + w_f, y_f + h_f), (255, 0, 0), 2)
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                preds = model_facial_expression.predict(roi)[0]
                # emotion_probability = np.max(preds)
                label = EMOTIONS[preds.argmax()]
                cv2.putText(frame1, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.drawContours(frame1, contour, -1, (0, 0, 255), 2)
    cv2.putText(frame1, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame1, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame1.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    # cv2.putText(frame1, "Fps: " + str(difference), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("image", frame1)
    frame1 = frame2

    next_frame +=1
    if cv2.waitKey(40) == ord('q'):
        break

video.release()