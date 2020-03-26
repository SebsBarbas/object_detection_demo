#!/usr/bin/python

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import pickle



def unpickle_keypoints(array):
    keypoints = []
    descriptors = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
        temp_descriptor = point[6]
        keypoints.append(temp_feature)
        descriptors.append(temp_descriptor)
    return keypoints, np.array(descriptors)


def main():
    mtx = np.array([[225.93036228, 0.0, 323.39508531], [0.0, 223.29467374, 234.78851706], [0.0, 0.0, 1.0]],
                   dtype = np.float32)
    dist = np.array([0.18928948, -0.18577923, -0.00287653, -0.00739736, 0.04164822], dtype = np.float32)
    tensorflowNet = cv2.dnn.readNetFromTensorflow("C:\\Trained Network\\frozen_inference_graph.pb",
                                                  "C:\\Trained Network\\all_signs.pbtxt")

    classNames = {1: 'airport', 2: 'dangerous_curve_left', 3: 'dangerous_curve_right', 4: 'follow_left',
                  5: 'follow_right', 6: 'junction', 7: 'no_bicycle', 8: 'no_heavy_truck', 9: 'no_parking',
                  10: 'no_stopping_and_parking', 11: 'residential', 12: 'road_narrows_from_left',
                  13: 'road_narrows_from_right',
                  14: 'roundabout_warning', 15: 'stop'}

    images = ['airport.jpg', 'dangerous_curve_left.jpg', 'dangerous_curve_right.jpg', 'follow_left.jpg',
              'follow_right.jpg', 'junction.jpg',
              'no_bicycle.jpg', 'no_heavy_truck.png', 'no_parking.jpg', 'no_stopping_and_parking.jpg',
              'residential.jpg', 'road_narrows_from_left.jpg',
              'road_narrows_from_right.jpg', 'roundabout_warning.jpg', 'stop.jpg']


    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    orb = cv2.ORB_create()

    cap = cv2.VideoCapture(0)

    keypoints_database = pickle.load(open("keypoints_database.p", "rb"))


    while 1:

        _, img3 = cap.read()
        rows, cols, channels = img3.shape
        tensorflowNet.setInput(cv2.dnn.blobFromImage(img3, 1.3, size = (300, 300), swapRB = True, crop = True))

        # Runs a forward pass to compute the net output

        detections = tensorflowNet.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]  # Confidence of prediction
            if confidence > 0.9:  # Filter prediction
                class_id = int(detections[0, 0, i, 1])  # Class label

                # Object location

                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)
                # Factor for scale to original size of frame
                heightFactor = img3.shape[0] / 300.0
                widthFactor = img3.shape[1] / 300.0
                # Scale object detection to frame
                # Draw location of object

                roi = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
                roi = roi[yLeftBottom:yRightTop, xLeftBottom:xRightTop]
                th, thresh = cv2.threshold(roi, 100, 255, cv2.THRESH_OTSU)

                try:
                    edged = cv2.Canny(roi, th, 250)
                except:
                    continue

                '''

                try:

                    edged = cv2.Canny(roi,0, 200)

                except:

                    pass

                '''
                contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
                                               offset = (xLeftBottom, yLeftBottom))
                #   for cnt in contours:

                #      area=cv2.contourArea(cnt)

                #     if area>5000:

                #       approx=cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)

                #        cv2.drawContours(img3,cnt,-1,(0,255,255),8,cv2.LINE_AA)

                # if 2<=len(approx)<=4:

                #   cv2.drawContours(img3,[approx],-1,(0,255,255),8,cv2.LINE_AA)

                # cv2.drawContours(img3,contours,-1,(0,255,255),8,cv2.LINE_AA)

                # Draw label and confidence of prediction in frame resized
                cv2.rectangle(img3, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                              (0, 255, 0), thickness = 3)

                if class_id in classNames:

                    for cnt in contours:
                        area = cv2.contourArea(cnt)

                        if area > 500:
                            approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
                            # print(len(approx))

                            if len(approx):
                                cv2.drawContours(img3, [approx], 0, (0, 255, 0), 2, cv2.LINE_AA)

                                for k in approx:
                                    cv2.putText(img3, str((k[0][0], k[0][1])), (k[0][0], k[0][1]),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                                    # print(k)
                                    img3 = cv2.circle(img3, (k[0][0], k[0][1]), 5, (0, 0, 255), 1)
                                    img3[k[0][1], k[0][0]] = [0, 0, 255]

                                kp_model, desc_model = unpickle_keypoints(keypoints_database[class_id-1])
                                curr_grey_img = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)


                                cv2.imshow("Prueba", roi)
                                kp_image, desc_image = orb.detectAndCompute(roi, mask = None)
                                img5=cv2.drawKeypoints(roi, kp_image, roi)
                                cv2.imshow("ORB IMAGEN", img5)
                                matches = bf.match(desc_model, desc_image)
                                matches = sorted(matches, key = lambda x: x.distance)[:4]

                                img4 = cv2.drawMatches(cv2.imread(images[class_id-1], cv2.IMREAD_GRAYSCALE), kp_model, roi, kp_image, matches, None, flags = 2)
                                cv2.imshow("Prueba3", img4)
                                list_kpmodel=[]
                                list_kpimage=[]
                                for mat in matches:
                                    img1_idx = mat.queryIdx
                                    img2_idx = mat.trainIdx

                                    (x1, y1) = kp_model[img1_idx].pt
                                    (x2, y2) = kp_image[img2_idx].pt

                                # Append to each list

                                    list_kpmodel.append((x1, y1))
                                    list_kpimage.append((x2, y2))




                                image_points = np.array(list_kpimage, dtype = np.float32)
                                model_points = np.array(list_kpmodel, dtype = np.float32)
                                print(image_points)
                                print(model_points)
                                success, rmat, tmat = cv2.solvePnPRansac(model_points, image_points, mtx, dist,
                                                                         flags = 1)[:3]

                                print("Rotation Vector:\n {0}".format(rmat))
                                print("Translation Vector:\n {0}".format(tmat))

                    label = classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(img3, (xLeftBottom, yLeftBottom - labelSize[1]),
                                  (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                  (0, 255, 0), cv2.FILLED)
                    cv2.putText(img3, label, (xLeftBottom, yLeftBottom),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))
                cv2.namedWindow('ORB stream', cv2.WINDOW_NORMAL)
                cv2.imshow('ORB stream', img3)
        k = cv2.waitKey(1) & 0xff

        if k == 32:
            p += 1
            img = images[p]
            if p == 14:
                p = 0

        cv2.namedWindow('ORB', cv2.WINDOW_NORMAL)
        cv2.imshow('ORB', img3)


if __name__ == '__main__':
    main()

'''

def load_images_from_folder(folder):

    images = []

    for filename in os.listdir(folder):

        img = cv2.imread(os.path.join(folder,filename))

        if img is not None:

            images.append(img)

    return images



def main():

    tensorflowNet =  cv2.dnn.readNetFromTensorflow('/home/alsarmi/Desktop/tensorflow_training/All_signs_inference_graph/frozen_inference_graph.pb', '/home/alsarmi/Desktop/tensorflow_training/all_signs.pbtxt')

    classNames = { 1: 'airport', 2: 'dangerous_curve_left', 3: 'dangerous_curve_right', 4: 'follow_left',

                    5: 'follow_right', 6: 'junction', 7: 'no_bicycle', 8: 'no_heavy_truck', 9: 'no_parking',

                    10: 'no_stopping_and_parking', 11: 'residential', 12: 'road_narrows_from_left', 13: 'road_narrows_from_right',

                    14: 'roundabout_warning', 15: 'stop' }

    rootdir = "/home/alsarmi/Desktop/tensorflow_training/DataSetGenerator-master/objects/"

    cap = cv2.VideoCapture('/dev/video0')

    images=load_images_from_folder(rootdir)

    # FLANN parameters

    FLANN_INDEX_LSH = 6

    index_params= dict(algorithm = FLANN_INDEX_LSH,

                   table_number = 6, # 12

                   key_size = 12,     # 20

                   multi_probe_level = 1) #2

    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)



    des2=None

    # Initiate ORB detector

    orb = cv2.ORB_create()

    p=0

    img=images[0]

    while 1:

        _, img3 = cap.read()

        rows, cols, channels = img3.shape

        tensorflowNet.setInput(cv2.dnn.blobFromImage(img3, 1.3,size=(300, 300), swapRB=True, crop=True))

        # Runs a forward pass to compute the net output

        detections = tensorflowNet.forward()

        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2] #Confidence of prediction 

            if confidence > 0.9: # Filter prediction 

                class_id = int(detections[0, 0, i, 1]) # Class label



                # Object location 

                xLeftBottom = int(detections[0, 0, i, 3] * cols) 

                yLeftBottom = int(detections[0, 0, i, 4] * rows)

                xRightTop   = int(detections[0, 0, i, 5] * cols)

                yRightTop   = int(detections[0, 0, i, 6] * rows)

                # Factor for scale to original size of frame

                heightFactor = img3.shape[0]/300.0  

                widthFactor = img3.shape[1]/300.0 

                # Scale object detection to frame



                # Draw location of object  

                mask = np.zeros(img3.shape[:2], dtype=np.uint8)

                cv2.rectangle(mask, (xLeftBottom+100, yLeftBottom+100), (xRightTop-100, yRightTop-100), (255,255,255), thickness = -1)





                kp2 = orb.detect(img3,mask)

                kp2, des2 = orb.compute(img3, kp2)

                roi=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY) 

                roi=roi[yLeftBottom:yRightTop,xLeftBottom:xRightTop]



                try:

                    edged = cv2.Canny(roi,0, 200)

                except:

                    pass

                contours,_= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,offset=(xLeftBottom,yLeftBottom))

                for cnt in contours:

                    area=cv2.contourArea(cnt)

                    if area>5000:

                        approx=cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)



                        cv2.drawContours(img3,[approx],-1,(0,0,0),8,cv2.LINE_AA)



                # Draw label and confidence of prediction in frame resized

                cv2.rectangle(img3, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),

                            (0, 255, 0),thickness=3)

                if class_id in classNames:

                    label = classNames[class_id] + ": " + str(confidence)

                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])

                    cv2.rectangle(img3, (xLeftBottom, yLeftBottom - labelSize[1]),

                                        (xLeftBottom + labelSize[0], yLeftBottom + baseLine),

                                        (0, 255, 0), cv2.FILLED)

                    cv2.putText(img3, label, (xLeftBottom, yLeftBottom),

                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))

                img4 = cv2.drawKeypoints(img3, kp2, mask, color=(0,255,0), flags=0)

                cv2.namedWindow('ORB stream',cv2.WINDOW_NORMAL)

                cv2.imshow('ORB stream',img4)



            #draw a red rectangle around detected objects        





        # find the keypoints with ORB

        k=cv2.waitKey(1) & 0xff

        if k==32:

            p+=1

            img=images[p]

            if p==14:

                p=0



        kp = orb.detect(img,None)



        # compute the descriptors with ORB

        kp, des = orb.compute(img, kp)

        img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)



        cv2.namedWindow('ORB',cv2.WINDOW_NORMAL)

        cv2.imshow('ORB',img2)









if __name__ == '__main__':

    main()

'''

'''

import numpy as np

import cv2

from matplotlib import pyplot as plt



import os

import cv2







def load_images_from_folder(folder):

    images = []

    for filename in os.listdir(folder):

        img = cv2.imread(os.path.join(folder,filename))

        if img is not None:

            images.append(img)

    return images



def main():

    tensorflowNet =  cv2.dnn.readNetFromTensorflow('/home/alsarmi/Desktop/tensorflow_training/All_signs_inference_graph/frozen_inference_graph.pb', '/home/alsarmi/Desktop/tensorflow_training/all_signs.pbtxt')

    classNames = { 1: 'airport', 2: 'dangerous_curve_left', 3: 'dangerous_curve_right', 4: 'follow_left',

                    5: 'follow_right', 6: 'junction', 7: 'no_bicycle', 8: 'no_heavy_truck', 9: 'no_parking',

                    10: 'no_stopping_and_parking', 11: 'residential', 12: 'road_narrows_from_left', 13: 'road_narrows_from_right',

                    14: 'roundabout_warning', 15: 'stop' }

    rootdir = "/home/alsarmi/Desktop/tensorflow_training/DataSetGenerator-master/objects/"

    cap = cv2.VideoCapture('/dev/video0')

    images=load_images_from_folder(rootdir)

    # FLANN parameters

    FLANN_INDEX_LSH = 6

    index_params= dict(algorithm = FLANN_INDEX_LSH,

                   table_number = 6, # 12

                   key_size = 12,     # 20

                   multi_probe_level = 1) #2

    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)



    des2=None

    # Initiate ORB detector

    orb = cv2.ORB_create()

    p=0

    img=images[0]

    while 1:

        _, img3 = cap.read()

        rows, cols, channels = img3.shape

        tensorflowNet.setInput(cv2.dnn.blobFromImage(img3, 1.3,size=(300, 300), swapRB=True, crop=True))

        # Runs a forward pass to compute the net output

        detections = tensorflowNet.forward()

        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2] #Confidence of prediction 

            if confidence > 0.9: # Filter prediction 

                class_id = int(detections[0, 0, i, 1]) # Class label



                # Object location 

                xLeftBottom = int(detections[0, 0, i, 3] * cols) 

                yLeftBottom = int(detections[0, 0, i, 4] * rows)

                xRightTop   = int(detections[0, 0, i, 5] * cols)

                yRightTop   = int(detections[0, 0, i, 6] * rows)

                # Factor for scale to original size of frame

                heightFactor = img3.shape[0]/300.0  

                widthFactor = img3.shape[1]/300.0 

                # Scale object detection to frame

                #xLeftBottom = int(widthFactor * xLeftBottom) 

                #yLeftBottom = int(heightFactor * yLeftBottom)

                #xRightTop   = int(widthFactor * xRightTop)

                #yRightTop   = int(heightFactor * yRightTop)

                # Draw location of object  

                mask = np.zeros(img3.shape[:2], dtype=np.uint8)

                cv2.rectangle(mask, (xLeftBottom+4, yLeftBottom+4), (xRightTop-4, yRightTop-4), (255,255,255), thickness = -1)





                kp2 = orb.detect(img3,mask)

                kp2, des2 = orb.compute(img3, kp2)

                roi=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY) 

                roi=roi[yLeftBottom:yRightTop,xLeftBottom:xRightTop]

                try:

                    edged = cv2.Canny(roi,0, 200)

                except:

                    cv2.imshow('test',edged)

                    pass

                contours,_= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,offset=(xLeftBottom,yLeftBottom))

                for cnt in contours:

                    area=cv2.contourArea(cnt)

                    if area>5000:

                        approx=cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)



                        cv2.drawContours(img3,[approx],-1,(0,0,0),8,cv2.LINE_AA)



                # Draw label and confidence of prediction in frame resized

                cv2.rectangle(img3, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),

                            (0, 255, 0),thickness=3)

                if class_id in classNames:

                    label = classNames[class_id] + ": " + str(confidence)

                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    yLeftBottom = max(yLeftBottom, labelSize[1])

                    cv2.rectangle(img3, (xLeftBottom, yLeftBottom - labelSize[1]),

                                        (xLeftBottom + labelSize[0], yLeftBottom + baseLine),

                                        (0, 255, 0), cv2.FILLED)

                    cv2.putText(img3, label, (xLeftBottom, yLeftBottom),

                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))

                img4 = cv2.drawKeypoints(img3, kp2, mask, color=(0,255,0), flags=0)

                cv2.namedWindow('ORB stream',cv2.WINDOW_NORMAL)

                cv2.imshow('ORB stream',img4)



            #draw a red rectangle around detected objects        





        # find the keypoints with ORB

        k=cv2.waitKey(1) & 0xff

        if k==32:

            p+=1

            img=images[p]

            if p==14:

                p=0



        kp = orb.detect(img,None)



        # compute the descriptors with ORB

        kp, des = orb.compute(img, kp)

        img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)



        if not des2.any()==None:



            matches = flann.knnMatch(des,des2,k=2)

            # Need to draw only good matches, so create a mask

            matchesMask = [[0,0] for i in range(len(matches))]

            # ratio test as per Lowe's paper

  #          print(enumerate(matches))

            try:

                for i,(m,n) in enumerate(matches):

                    if m.distance < 0.7*n.distance:

                        matchesMask[i]=[1,0]



                draw_params = dict(matchColor = (0,0,255),

                        singlePointColor = (255,0,0),

                        matchesMask = matchesMask,

                        flags = cv2.DrawMatchesFlags_DEFAULT)

                img6 = cv2.drawMatchesKnn(img2,kp,img4,kp2,matches[:10],None,matchColor=[0,0,255], singlePointColor=[255,0,0])#**draw_params)

            # draw only keypoints location,not size and orientation

            except:

                pass

            cv2.namedWindow('ORB',cv2.WINDOW_NORMAL)

            cv2.imshow('ORB',img6)

if __name__ == '__main__':
    main()
    '''





