import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from PIL import Image
from os import listdir
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import backend as K
import os
from os import listdir
from utils import *

cam_capture = cv2.VideoCapture(0)
upper_left = (50, 50)
bottom_right = (300, 300)

while True:
    _, image_frame = cam_capture.read()
    r = cv2.rectangle(image_frame, upper_left, bottom_right, (100, 50, 200), 5)
    rect_img = image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    
    sketcher_rect = rect_img
    template = cv2.imread('template.png',0)
    w, h = template.shape[::-1]
    img_gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(rect_img, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
        cv2.imwrite('card_pic/part.jpg',rect_img)
    #Replacing the sketched image on Region of Interest
    cv2.imshow('Detected',image_frame)
    if len(listdir('card_pic')) != 0:
        imported_images = []
        directory_name = "target"
        image_names = listdir(directory_name)
    #resize all images in the dataset for processing
        for image_name in image_names:
            foo = cv2.imread("target/" + image_name)
            foo = cv2.resize(foo, (115, 115))
            imported_images.append(foo)

        image_set = np.array([process_image(image, width = 115) for image in  imported_images])
        image_width = 115
        pixels = image_width * image_width * 3
        hidden = 1000
        corruption_level = 0.25
        X = tf.placeholder("float", [None, pixels], name = 'X')
        mask = tf.placeholder("float", [None, pixels], name = 'mask')
        weight_max = 4 * np.sqrt(4 / (6. * (pixels + hidden)))
        weight_initial = tf.random_uniform(shape = [pixels, hidden], minval = -weight_max, maxval = weight_max)
        W = tf.Variable(weight_initial, name = 'W')
        b = tf.Variable(tf.zeros([hidden]), name = 'b')
        W_prime = tf.transpose(W)
        b_prime = tf.Variable(tf.zeros([pixels]), name = 'b_prime')
        out = model(X, mask, W, b, W_prime, b_prime)
        cost = tf.reduce_sum(tf.pow(X-out, 2))
        optimization = tf.train.GradientDescentOptimizer(0.02).minimize(cost)
        x_train, x_test = train_test_split(image_set)
        sess = tf.Session()
    #you need to initialize all variables
        sess.run(tf.global_variables_initializer())
        for start, end in zip(range(0, len(x_train), 128), range(128, len(x_train), 128)):
            input_ = x_train[start:end]
            mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
            sess.run(optimization, feed_dict = {X: input_, mask: mask_np})
        mask_np = np.random.binomial(1, 1 - corruption_level, x_test.shape)
        masknp = np.random.binomial(1, 1 - corruption_level, image_set.shape)
        image_name = "card_pic/part.jpg" #Image to be used as query
        image = cv2.imread(image_name)
        image = cv2.resize(image, (115, 115))
        image = process_image(image, 115)
        results = search(imported_images,corruption_level,pixels,image,image_set,sess,X, mask, W, b)
        image_frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]] = sketcher_rect
        cv2.imwrite('card_pic/closest.jpg',cv2.cvtColor(results[-1], cv2.COLOR_BGR2RGB))

        directory_name = "target"
        image_names = listdir(directory_name)
        template2 = cv2.imread('card_pic/closest.jpg',0)
        w, h = template2.shape[::-1]

        for image_name in image_names:
            foo = cv2.imread("target/" + image_name)
            foo = cv2.resize(foo, (115, 115))
            img_gray = cv2.cvtColor(foo, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(img_gray,template2,cv2.TM_CCOEFF_NORMED)
            threshold = 0.99
            loc = np.where( res >= threshold)
            if len(loc[0]) != 0:
                print(image_name)
    #Replacing the sketched image on Region of Interest
        cv2.imshow('Detected',image_frame)
        os.remove("card_pic/closest.jpg") 
        os.remove("card_pic/part.jpg") 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 
cam_capture.release()
cv2.destroyAllWindows()
