#!/usr/bin/python

import os
import cv2
import numpy as np
import pickle

def pickle_keypoints(keypoints, descriptors):
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id, descriptors[i])
        i = i+1

        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)

if __name__ == "__main__":

    path='C:\\Users\\sebas\\Documents\\Universidad\\Máster\\KTH\\Project Course in Robotics\\PnPRansac\\objects\\'
    orb = cv2.ORB_create()
    images=['airport.png','dangerous_curve_left.png', 'dangerous_curve_right.png', 'follow_left.png','follow_right.png','junction.png',
            'no_bicycle.png','no_heavy_truck.png','no_parking.png','no_stopping_and_parking.png', 'residential.png', 'road_narrows_from_left.png',
            'road_narrows_from_right.png','roundabout_warning.png','stop.png']
    temp_a = []
    for l in images:
        img = cv2.imread(l, cv2.IMREAD_GRAYSCALE)
        kp1, des1 = orb.detectAndCompute(img,mask = None)
        temp = pickle_keypoints(kp1[:20], des1[:20])
        temp_a.append(temp)

    pickle.dump(temp_a, open("keypoints_database.p", "wb"))

    keypoints_database = pickle.load(open("keypoints_database.p", "rb"))
    kp1, desc1 = unpickle_keypoints(keypoints_database[7])
    img = cv2.imread(images[7], cv2.IMREAD_GRAYSCALE)
    kp2, desc2 = orb.detectAndCompute(img, mask = None)

    """
    index = []
    for point in kp1:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        index.append(temp)
    pickle.dump(temp, open("keypoints_database.p", "wb"))

    #f = open(path+"airport_keypoints.txt", 'wb')
    #f.write(pickle.dumps(index))
    #f.close()

    pickle_off = open(path+"airport_keypoints.txt", 'rb')
    index2 = pickle.loads(pickle_off)
    kp = []

    for point in index2:
        temp = cv2.KeyPoint(x = point[0][0], y = point[0][1], _size = point[1], _angle = point[2], _response = point[3],
                            _octave = point[4], _class_id = point[5])
        kp.append(temp)
    """

    # Draw the keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = False)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key = lambda x:x.distance)

    img2 = cv2.drawMatches(img, kp1, img, kp2, matches[:10], None, flags = 2)

    cv2.imshow("Test", img2)

    cv2.waitKey(0)




