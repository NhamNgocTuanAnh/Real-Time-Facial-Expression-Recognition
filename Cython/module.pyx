
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



from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import threading,queue
input_image_time_buffer = queue.Queue(50)
input_time_buffer = queue.Queue(50)
input_preds_buffer = queue.Queue(50)


cdef int number = 100000
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# load model facial_expression
model_facial_expression = model_from_json(open("model/fer.json", "r").read())
# load weights facial_expression
model_facial_expression.load_weights('model/fer.h5')

# cdef public void Load_Cascades(cascade):
#   cascade.append(cv2.CascadeClassifier('haarcascade_frontalface_default.xml'))
#   # cascade.append(cv2.CascadeClassifier('haarcascade_eye.xml'))
#   print("Cascades loaded!!")

cdef Py_UNICODE* EMOTIONS[7]
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
cdef numpy.ndarray frame_temp, gray_temp, preds

cdef bint paused = False
cdef bint finished = False

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cls():
    os.system('cls' if os.name=='nt' else 'clear')

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
    cdef float last_time = 0  
    cdef float current_time = 0
    cdef int number_face = 0 
    cdef str label = ''
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
            
            roi = gray_temp.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = numpy.expand_dims(roi, axis=0)
            preds = model_facial_expression.predict(roi)[0]
            
            #if number_face == 0:
             #   total_preds = preds
           # else:
             #   total_preds += preds
                
            label = EMOTIONS[preds.argmax()]
            print(str(type(label)))
            #print(str(total_preds)+" time: "+ str(current_time))
            #input_preds_buffer.put(preds, timeout=1)
            #
        except queue.Empty:
            print(" ")
        #cv2.imshow("Probabilities", canvas)

# Có một số yếu tố khiến mã chậm hơn như đã thảo luận trong tài liệu Cython đó là:
#
# Kiểm tra giới hạn để đảm bảo các chỉ số nằm trong phạm vi của mảng.
# Sử dụng các chỉ số âm để truy cập các phần tử mảng.

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef show():
    # detection_buffer = Queue()
    global paused, finished, EMOTIONS
    cdef bint video_true = True
    cascade = []
    # Load_Cascades(cascade)
    cap = cv2.VideoCapture('democlassroom.mp4')
    cdef numpy.ndarray frame, gray

    cdef float fps = cap.get(cv2.CAP_PROP_FPS)
    cdef float time_frame = 1.0 / fps

    cdef bint ret = True
    # In order to define boolean objects in cython, they need to be defined as bint.
    # According to here: The bint of "boolean int" object is compiled to a c int,
    # but get coerced to and from Cython as booleans.


    cdef int i
    cdef int increment = 0

    cdef int increment_times = 0
    cdef int count = 0
    cdef numpy.uint32_t x, y, w, h
    cdef float end_time, start_time, frame_number, frame_delay
    frame_number = 0
    frame_delay = 0
    # cdef Rectangle face
    # cdef int x, y, w, h

    
    while (video_true) :
        canvas = numpy.zeros((550, 300, 3), dtype="uint8")
        if not paused:
            start_time = time.time()
            try:
                    # print("Kiểu gì EMOTIONS" + str(type(EMOTIONS)))
                # Capture frame-by-frame
                ret, frame = cap.read()
                # print("Kieru gì" + str(type(ret)))
                # print("Kieru gì" + str(type(frame)))
                # Kieru gì <class 'bool'>
                # Kieru gì <class 'numpy.ndarray'>
                if not (ret is  True):
                    finished = True
                    break
                if not(frame is None):
                    if count % 5 == 0:
                        # Our operations on the frame come here
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                                flags=cv2.CASCADE_SCALE_IMAGE)

                        i = 0
                        for i in range(0,len(faces)):
                            # print(faces[i][0])
                            # print(face.dtype)
                            # print(str(type(face)))
                            x = faces[i][0]
                            y = faces[i][1]
                            w = faces[i][2]
                            h = faces[i][3]

                            # print("Kiểu gì x" + str(type(x)))
                            # if y+w >20 and x+h >20:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                            
                            roi = gray[y:y + h, x:x + w]
                            roi = cv2.resize(roi, (48, 48))
                            input_image_time_buffer.put((roi,time.time()), timeout=1)
                            
                            i+=1
                        end_time = time.time()
                        if (end_time - start_time) > 0:
                            fpsInfo = "FPS: " + str(1.0 / (end_time - start_time))  # FPS = 1 / time to process loop
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(frame, fpsInfo, (10, 20), font, 0.4, (255, 0, 0), 1)
                        cv2.imshow('frame', frame)
                        i = 0

                    count += 1
 


            except queue.Full:
                print("Full memory")
                pass
        #try:
            #preds = input_preds_buffer.get(timeout=0)
            #label = EMOTIONS[preds.argmax()]
            #for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
                #text = "{}: {:.2f}%".format(emotion, prob * 100)
                #cv2.putText(canvas, text, (10, (i * 35) + 23),
                            #cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            #(255, 255, 255), 2)
        #except queue.Empty:
            #print(" ")
        #cv2.imshow("Probabilities", canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            finished = True
            break
        
    cap.release()
    cv2.destroyAllWindows()

@cython.boundscheck(False)
@cython.wraparound(False)
# Using cython compiler directives to remove some of the checks that numpy usually has to make
# Use typed memoryviews so that I can specify memory layout (and sometimes they are faster in general compared to the older buffer interface)
# Unrolled the loops so that we don't use numpy's slice machinary:
cpdef public Main():
    tReadFile = threading.Thread(target=show)
    tProcessingFile = threading.Thread(target=smooth_emotions)

    tReadFile.start()
    tProcessingFile.start()

    tProcessingFile.join()
    tReadFile.join()
    print("Bye !!!")
