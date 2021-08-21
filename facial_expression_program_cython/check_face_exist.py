from tkinter import *
import cv2,os
import csv
import numpy
import imutils
import pandas as pd
import datetime
import time
from time import sleep

import glob
from PIL import Image, ImageTk
import face_recognition
from imutils import face_utils
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
import cv2,skimage
from skimage import data, io, filters
import dlib
import matplotlib as mpl
from scipy.spatial import ConvexHull
rootdir = 'face_database/'
def cls():
    os.system('cls' if os.name=='nt' else 'clear')
def check(path):
    count_img = 0
    count_img_moved = 0
    detector = dlib.get_frontal_face_detector() #Load face detector
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    #path = 'face_database/'
    print(path)
    i = 0
    for r, d, f in os.walk(path):
        for file in f:
            count_img +=1
            if '.jpg' in file:
                cls()
                print ("Processing...")
                print ("Total img: " + str(count_img))
                print ("Total img deleted: " + str(count_img_moved))
                # print("Kieru " + os.path.join(r, file) + str(i))
                img = cv2.imread(os.path.join(r, file), cv2.IMREAD_GRAYSCALE)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                img = cv2.resize(img, (71, 71))
                check = True
                if check:
                    name_file = os.path.splitext(os.path.join(r, file))[0]
                    rects = detector(img, 0)
                    i+= 1
                    for (i, rect) in enumerate(rects):
                        if numpy.shape(rect) != ():

                            # determine the facial landmarks for the face region, then
                            # convert the facial landmark (x, y)-coordinates to a NumPy
                            # array
                            shape = predictor(img, rect)
                            landmark_tuple = []
                            for n in range(0, 27):
                                x = shape.part(n).x
                                y = shape.part(n).y
                                landmark_tuple.append((x, y))
                            routes = []

                            for i in range(15, -1, -1):
                                from_coordinate = landmark_tuple[i + 1]
                                to_coordinate = landmark_tuple[i]
                                routes.append(from_coordinate)

                            from_coordinate = landmark_tuple[0]
                            to_coordinate = landmark_tuple[17]
                            routes.append(from_coordinate)

                            for i in range(17, 20):
                                from_coordinate = landmark_tuple[i]
                                to_coordinate = landmark_tuple[i + 1]
                                routes.append(from_coordinate)

                            from_coordinate = landmark_tuple[19]
                            to_coordinate = landmark_tuple[24]
                            routes.append(from_coordinate)

                            for i in range(24, 26):
                                from_coordinate = landmark_tuple[i]
                                to_coordinate = landmark_tuple[i + 1]
                                routes.append(from_coordinate)

                            from_coordinate = landmark_tuple[26]
                            to_coordinate = landmark_tuple[16]
                            routes.append(from_coordinate)
                            routes.append(to_coordinate)

                            mask = numpy.zeros((image.shape[0], image.shape[1]))
                            mask = cv2.fillConvexPoly(mask, numpy.array(routes), 1)
                            mask = mask.astype(numpy.bool)

                            out = numpy.zeros_like(image)
                            out[mask] = image[mask]
                            (x, y, w, h) = face_utils.rect_to_bb(rect)
                            roi = out[y:y + h, x:x + w]


                            if os.path.exists(os.path.join(r, file)):
                                os.remove(os.path.join(r, file))
                            else:
                                print("Can not delete the file as it doesn't exists" + os.path.join(r, file))

                            cv2.imwrite(os.path.join(r, file), roi)
                        else:
                            if os.path.exists(os.path.join(r, file)):
                                os.remove(os.path.join(r, file))
                            cv2.imwrite(str(path)+'background/'+str(name_file)+'.jpg', img)
                            count_img_moved +=1    

if __name__ == '__main__':
    check(rootdir)
    #remove_slow(rootdir)
    #remove_in_json("surprised","my_duplicates.json")
