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


def process_image(image, width):
    out = []
    for x in range(width):
        for y in range(width):
            for z in range(3): #account for RGB
                out.append(image[x][y][z])
    return np.asarray(out)

def reparameterize(W, b):
    epsilon = K.random_normal(shape = (1, hidden), mean = 0.)
    return W + K.exp(b/2) * epsilon

def model(X, mask, W, b, W_prime, b_prime):
    tilde_X = mask * X

    hidden_state = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)
    out = tf.nn.sigmoid(tf.matmul(hidden_state, W_prime) + b_prime)
    return out

def hidden_state(X, mask, W, b):
    tilde_X = mask * X
    
    state = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)
    return state

def euclidean_distance(arr1, arr2):
    x = 0
    for i in range(len(arr1)):
        x += pow(arr1[i] - arr2[i], 2)
    return np.sqrt(x)

def search(imported_images,corruption_level,pixels,image,image_set,sess,X, mask, W, b):
    hidden_states = [sess.run(hidden_state(X, mask, W, b),
                              feed_dict={X: im.reshape(1, pixels), mask: np.random.binomial(1, 1-corruption_level, (1, pixels))}) for im in image_set]
    query = sess.run(hidden_state(X, mask, W, b),
                              feed_dict={X: image.reshape(1,pixels), mask: np.random.binomial(1, 1-corruption_level, (1, pixels))})
    starting_state = int(np.random.random()*len(hidden_states)) #choose random starting state
    best_states = [imported_images[starting_state]]
    distance = euclidean_distance(query[0], hidden_states[starting_state][0]) #Calculate similarity between hidden states
    for i in range(len(hidden_states)):
        dist = euclidean_distance(query[0], hidden_states[i][0])
        if dist <= distance:
            distance = dist #as the method progresses, it gets better at identifying similiar images
            best_states.append(imported_images[i])
    if len(best_states)>0:
        return best_states
    else:
        return best_states[len(best_states)-101:]
