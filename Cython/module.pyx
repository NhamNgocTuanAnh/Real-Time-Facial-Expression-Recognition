
import numpy
import cv2,os
import time

from libc.stdio cimport *
cimport numpy
cimport cython
from libc.stdint cimport (
  uint8_t, uint16_t, uint32_t, uint64_t,
  int8_t, int16_t, int32_t, int64_t,
  uintptr_t
)

import logging


from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import threading,queue
input_image_time_buffer = queue.Queue(50)



cdef int number = 100000
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# load model facial_expression
model_facial_expression = model_from_json(open("model/fer.json", "r").read())
# load weights facial_expression
model_facial_expression.load_weights('model/fer.h5')



EMOTIONS = numpy.array(["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"])
cdef :
    str log_system = 'Error: '
    bint paused = False
    bint finished = False
    numpy.ndarray frame_temp, gray_temp, preds

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cls():
    os.system('cls' if os.name=='nt' else 'clear')

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef clickListener(event, x, y, flags, param):
    global paused
    if event==cv2.EVENT_LBUTTONUP:
        print ("%s video" % ("Resume" if paused else "Pause"))
        paused = not paused
        
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef smooth_emotions():
    global gray_temp, EMOTIONS, preds, finished
    total_preds = numpy.array([])
    cdef :
        float last_time = 0
        float current_time = 0
        int number_face = 0
        str label = ''
        numpy.ndarray roi
    while not finished:
       # cls()
        #
        
        try:

            number_face += 1
            
            (gray_temp,current_time) = input_image_time_buffer.get(timeout=1)
            #current_time = input_time_buffer.get(timeout=1)
            
            #if current_time != last_time:
               # last_time = current_time
                #total_preds = total_preds / number_face
                #number_face = 0
            
            
            roi = numpy.expand_dims(img_to_array(gray_temp.astype("float") / 255.0), axis=0)
            #print(str(type(roi)))
            preds = model_facial_expression.predict(roi)[0]
            
            label_temp = int(preds.argmax())
            path_face_save = str("face_database/"+str(EMOTIONS[label_temp])+"/"+str(EMOTIONS[label_temp])+'_'+ str(int(current_time))+"_"+str(number_face) + ".jpg")
            cv2.imwrite(path_face_save, gray_temp)
        except queue.Empty:
            logging.warning("Empty memory!")
        #cv2.imshow("Probabilities", canvas)

# Có một số yếu tố khiến mã chậm hơn như đã thảo luận trong tài liệu Cython đó là:
#
# Kiểm tra giới hạn để đảm bảo các chỉ số nằm trong phạm vi của mảng.
# Sử dụng các chỉ số âm để truy cập các phần tử mảng.

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef show():
    # detection_buffer = Queue()
    global paused, finished, EMOTIONS, preds, log_system
    cdef str text = ''
    #cdef bint video_true = True
    cascade = []
    # Load_Cascades(cascade)
    cap = cv2.VideoCapture('democlassroom.mp4')
    #cap = cv2.VideoCapture('test.mp4')
    
    cdef float fps = cap.get(cv2.CAP_PROP_FPS)
    cdef float time_frame = 1.0 / fps

    cdef bint ret = True
    # In order to define boolean objects in cython, they need to be defined as bint.
    # According to here: The bint of "boolean int" object is compiled to a c int,
    # but get coerced to and from Cython as booleans.


    cdef int i, high_level_emotion
    cdef int increment = 0

    cdef int increment_times = 0
    cdef int count = 0
    cdef int max_emotion_position = 0 
    cdef numpy.uint32_t x, y, w, h
    cdef float end_time, start_time, frame_number, frame_delay
    frame_number = 0
    frame_delay = 0
    # cdef Rectangle face
    # cdef int x, y, w, h
    
    cdef numpy.ndarray frame, gray
    while True :

       
        try:
            ret, frame = cap.read()
            # Kieru gì <ret 'bool'>
            # Kieru gì <frame 'numpy.ndarray'>

            if (cv2.waitKey(1) & 0xFF == ord('q') )or(not (ret is  True)):
                finished = True
                break
                
            start_time = time.time()    
            count += 1  
                
            if count % 5 == 0:
                # Our operations on the frame come here

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                        flags=cv2.CASCADE_SCALE_IMAGE)

                for i in range(0,len(faces)):
          
                    x = faces[i][0]
                    y = faces[i][1]
                    w = faces[i][2]
                    h = faces[i][3]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.imshow('frame', frame)
                    
                    roi = gray[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (48, 48))
                    input_image_time_buffer.put((roi,time.time()), timeout=1)
                 
        
                        
                
            if not(preds is None):
                #label = EMOTIONS[preds.argmax()]
                max_emotion_position = preds.argmax()
                for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

                    #construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
               
                    color = (255, 0, 0)          
                    if max_emotion_position == i:
                        color = (225,225,225)
                    high_level_emotion = int(prob * 300)
                    cv2.rectangle(frame, (7, (i * 35) + 5),
                                  (high_level_emotion, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(frame, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.imshow('frame', frame)
            
            end_time = time.time()
            if (end_time - start_time) > 0:
                fpsInfo = "FPS: " + str(1.0 / (end_time - start_time))  # FPS = 1 / time to process loop
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, fpsInfo, (20, 30), font, 0.4, (255, 0, 0), 1)
            time.sleep(0.025) 
        except queue.Full:
            logging.warning("full memory!")
            pass


        
    cap.release()
    cv2.destroyAllWindows()

@cython.boundscheck(False)
@cython.wraparound(False)
# Using cython compiler directives to remove some of the checks that numpy usually has to make
# Use typed memoryviews so that I can specify memory layout (and sometimes they are faster in general compared to the older buffer interface)
# Unrolled the loops so that we don't use numpy's slice machinary:
def Main():
    global gray_temp, EMOTIONS
    assure_path_exists("face_database/")
    for emotion_index in range(0, len(EMOTIONS)):
        assure_path_exists("face_database/"+EMOTIONS[emotion_index]+"/")
    tReadFile = threading.Thread(target=show)
    tProcessingFile = threading.Thread(target=smooth_emotions)

    tReadFile.start()
    tProcessingFile.start()

    tProcessingFile.join()
    tReadFile.join()
    print("Bye !!!")
