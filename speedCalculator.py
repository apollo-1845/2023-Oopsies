import time
import cv2
import math
# we will be using time instead of the tutorial's suggested EXIF data because the time function returns the time elapsed
# with significantly greater precision than the time recorded in the EXIF data.
start=time.process_time()
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
    # making this more efficient is going to be a priority. Possibly by implementing a binary search
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute_force.match(descriptors_1, descriptors_2)
    # do we even need to sort the matches?
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
    # considering that there will always be as many coordinates in coordinates 1 and coordinates 2, we don't actually need
    # to merge the lists
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
    # I merged the calculation of *GSD/100000 into a single constant
    distance = feature_distance * 0.12648
    speed = distance / time_difference
    return speed
image_1 = 'photo_07464.jpg'
image_2 = 'photo_07465.jpg'
image_1_cv, image_2_cv = convertToCV(image_1, image_2)
keypoints_1, keypoints_2, descriptors_1, descriptors_2 = calculateFeatures(image_1_cv, image_2_cv, 1000)
matches = calculateMatches(descriptors_1, descriptors_2)
# for whatever reason, the program isn't finding any matches
coordinates_1, coordinates_2 = convertToCoordinates(keypoints_1, keypoints_2, matches)
pixel_Distance=getMeanDistance(coordinates_1,coordinates_2)
print(getSpeed(pixel_Distance, 9))

