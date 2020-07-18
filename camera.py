"""
Created on Sun Jun  7 18:39:20 2020

@author: RahulReddyGajjada
"""
import cv2
import numpy as np

class VideoCamera():
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        ret, frame = self.video.read()
        
        MIN_MATCH_COUNT = 15

        detector = cv2.xfeatures2d.SIFT_create()
        
        FLANN_INDEX_KDTREE = 0
        flannParams = dict(algorithm=FLANN_INDEX_KDTREE, tree=5)
        
        flann = cv2.FlannBasedMatcher(flannParams, {})
        
        trainImg = cv2.imread('aadhar.jpg', 0)

        trainKP, trainDesc = detector.detectAndCompute(trainImg, None)
        
        QueryImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)

        matches = flann.knnMatch(queryDesc, trainDesc, k=2)
        
        goodMatch = []
        for m, n in matches:
            if(m.distance < 0.7 * n.distance):
                goodMatch.append(m) 
                
        if len(goodMatch) > MIN_MATCH_COUNT:
            tp = []
            qp = []
            for m in goodMatch:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp, qp = np.float32((tp, qp))
            
            H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
            h, w = trainImg.shape
            trainBorder = np.float32([[[0,0], [0,h-1], [w-1, h-1], [w-1, 0]]])
            
            if H is not None:
                queryBorder = cv2.perspectiveTransform(trainBorder, H)
                cv2.polylines(frame, [np.int32(queryBorder)], True, (0, 255, 0), 3)
        else:
            print(f"Matched only by {len(goodMatch)/MIN_MATCH_COUNT}%")
        
        ret, jpeg = cv2.imencode('.jpg', frame)

        return jpeg.tobytes()
