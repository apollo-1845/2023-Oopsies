import time
import cv2
import math
from picamera import PiCamera
import os

image_1 = 'photo_01929.jpg'
image_1 = 'photo_01930.jpg'

def checkTimeDiff(start):
    return time.process_time()-start
    
def convertToCV(image1,image2):
    cv_image1 = cv2.imread(image1,0)
    cv_image2 = cv2.imread(image2,0)
    return cv_image1, cv_image2
def calculateFeatures(image1,image2, feature_number):
    orb=cv2.ORB_create(nfeatures= feature_number)
    keypoints1, descriptors1=orb.detectAndCompute(image1,None)
    keypoints2, descriptors2=orb.detectAndCompute(image2,None)
    return keypoints1, keypoints2, descriptors1,descriptors2
    
def calculateMatches(descriptors_1, descriptors_2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches
    
def convertToCoordinates(keypoints_1, keypoints_2, matches):
    coordinates_1 = []
    coordinates_2 = []
    for match in matches:
        image_1_idx = match.queryIdx
        image_2_idx = match.trainIdx
        (x1,y1) = keypoints_1[image_1_idx].pt
        (x2,y2) = keypoints_2[image_2_idx].pt
        coordinates_1.append((x1,y1))
        coordinates_2.append((x2,y2))
    return coordinates_1, coordinates_2
    
def getMeanDistance(coordinates_1, coordinates_2):
    totalDistance = 0.0
    numberOfCoordinates=len(coordinates_1)
    merged_coordinates = list(zip(coordinates_1, coordinates_2))
    for i in range(numberOfCoordinates):
        deltaX = coordinates_1[0][0] - coordinates_2[0][0]
        deltaY = coordinates_1[0][1] - coordinates_2[0][1]
        distance = math.hypot(deltaX, deltaY)
        totalDistance += distance
    return totalDistance / numberOfCoordinates
    
def getSpeed(feature_distance, time_difference):
    distance = feature_distance * 0.12648
    speed = distance / time_difference
    return speed
    
start_time=time.process_time
total_time=0
begin_time=0
end_time=0
time_between_photos=0
camera=PiCamera()
total_average_speeds=0
total_pictures=0
while total_time<600:
    begin_time=time.process_time()
    camera.capture("Photo1.jpg")
    image_1="Photo1.jpg"
    sleep(5)
    camera.capture("Photo2.jpg")
    total_time+=5
    end_time=time.process_time()+5-begin_time()
    image_1_cv, image_2_cv = convertToCV(image_1, image_2)
    keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculateFeatures(image_1_cv, image_2_cv, 1000)
    matches = calculateMatches(descriptors_1, descriptors_2)
    coordinates_1, coordinates_2 = convertToCoordinates(keypoints_1, keypoints_2, matches)
    pixel_Distance=getMeanDistance(coordinates_1,coordinates_2)
    total_average_speed+=getSpeed(pixel_Distance, end_time)
    total_pictures+=1
    total_time+=time.process_time-start_time
    start_time=time.process_time
mean_speed=total_average_speed/total_pictures
#rounding mean speed to 5 s.f.
string_speed=str(mean_speed)
rounded_speed=""
significant_figures=1
index=0

while significant_figures<6:
    if string_speed[index]!=".":
        rounded_speed+=string_speed[index]
        significant_figures+=1
    index+=1
with open("result.txt","w") as file:
    file.write(rounded_speed)

