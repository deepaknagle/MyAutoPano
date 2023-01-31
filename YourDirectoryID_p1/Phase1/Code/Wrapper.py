#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

# Code starts here:

import numpy as np
import cv2
import copy
import argparse
import glob
import matplotlib.pyplot as plt
import os
# Add any python libraries here


def anms(cornerMap, Nbest):
    cornerMap = np.squeeze(cornerMap)
    lc = {}
    for i in range(1, len(cornerMap)):
        dist = cornerMap[:i] - cornerMap[i]
        dist = dist * dist
        dist = np.sum(dist, axis=1)
        mindist = np.min(list(dist))
        idx, = np.where(dist==mindist)
        location = cornerMap[idx]
        x, y = location[0]
        location = (x, y)
        if location in lc:
            if mindist > lc[location]:
                continue
        lc.update({location : mindist})
    sorted_lc = dict(sorted(lc.items(), key=lambda a:a[1]))
    z = list(sorted_lc.keys())
    z.reverse()
    return z[:Nbest]

def featureDescriptor(image, map):
    image = (image - np.mean(image))/np.std(image)
    ymax, xmax = image.shape
    listofVectors = []
    listxy = []
    for i in range(len(map)):
        xi, yi = map[i]
        xlower = xi - 20
        xupper = xi + 20
        ylower = yi - 20
        yupper = yi + 20
        if xi < 20:
            xlower = 0
            xupper = xi + 20
        if xi >= (xmax - 20):
            xlower = xi - 20
            xupper = xmax
        if yi < 20:
            ylower = 0
            yupper = yi + 20
        if yi >= (ymax - 20):
            ylower = yi - 20
            yupper = ymax
        imagepatch = image[ylower:yupper + 1, xlower:xupper + 1]
        blurredOutput = cv2.GaussianBlur(imagepatch, (5, 5), 1)
        subsampledOutput = cv2.resize(blurredOutput, (8, 8))
        vector = subsampledOutput.ravel()
        vector = (vector - np.mean(vector))/np.std(vector)
        listxy.append((xi, yi))
        listofVectors.append(vector)
    return listofVectors, listxy


def featureMatching(vectorlist1, listxy1, vectorlist2, listxy2, threshhold):
    pair1 = []
    pair2 = []
    distancls = []
    o = 0
    for vi in vectorlist1:
        distancarr = vectorlist2 - vi
        sq_distancarr = distancarr**2
        sum = np.sum(sq_distancarr, axis=1)
        min1 = np.min(sum)
        pos1 = np.where(sum==min1)
        sum[pos1[0][0]] = 10000000000000
        min2 = np.min(sum)
        if min1/min2 > threshhold:
            continue
        pair1.append(listxy1[o])
        pair2.append(listxy2[pos1[0][0]])
        distancls.append(min1)
        o += 1
    return np.array(pair1, dtype=np.float32), np.array(pair2, dtype=np.float32), np.array(distancls, dtype=np.float32)


def draw_matches(distances1):
    matchlist = []
    for idx, val in enumerate(distances1):
        matchlist.append(cv2.DMatch(idx, idx, float(val)))
    return matchlist

def keypts(kp):
    z = []
    for i in kp:
        x = i[0]
        y = i[1]
        z.append(cv2.KeyPoint(int(x), int(y), 3))
    return z

def ransac(pointlist1, pointlist2, threshhold, Nmax, percentage):
    idx = []
    totalinliners = 0
    updatedH = np.zeros((3,3))
    for num in range(Nmax):
        fourfeatures1 = []
        fourfeatures2 = []
        for i in range(4):
            idx = np.random.randint(0, len(pointlist1))
            fourfeatures1.append(tuple(pointlist1[idx]))
            fourfeatures2.append(tuple(pointlist2[idx]))
        H = cv2.getPerspectiveTransform(np.float32(fourfeatures1), np.float32(fourfeatures2))
        inlinersindexlist = []
        for i in range(len(pointlist1)):
            p_dash = np.matmul(H, [pointlist1[i][0], pointlist1[i][1], 1])
            if p_dash[2] == 0:
                p_dash[2] = 0.000001
            px = np.float32(p_dash[0]/p_dash[2])
            py = np.float32(p_dash[1]/p_dash[2])
            point = np.array((pointlist2[i][0], pointlist2[i][1]))
            p_dash = np.array((px, py))
            ssd = np.sum((point-p_dash)**2)
            if ssd < threshhold:
                inlinersindexlist.append(i)
        if totalinliners < len(inlinersindexlist):
            totalinliners = len(inlinersindexlist)
            z = copy.deepcopy(inlinersindexlist)
            updatedH = H
            if len(inlinersindexlist) > percentage*len(pointlist1):
                break
    return z

def ransac_executor(image1pts, image2pts, distancelist, threshhold, Nmax, percentage):
    idx = ransac(image1pts, image2pts, threshhold, Nmax, percentage)
    keypoints1 = []
    keypoints2 = []
    matchlist = []
    for i in idx:
        keypoints1.append(tuple(image1pts[i]))
        keypoints2.append(tuple(image2pts[i]))
        matchlist.append(distancelist[i])
    kptconverted1 = keypts(keypoints1)
    kptconverted2 = keypts(keypoints2)
    distancesconverted = draw_matches(matchlist)


    return kptconverted1, kptconverted2, distancesconverted

def blend_images(*images):
    first = images[0]
    o = 1
    for image in images[1:]:
        H, variable = homography_matrix(first, image)
        if variable < 30:
            print(f"Image Number {o} has too less matches")
            continue




def main():
    # Add any Command Line arguments here
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--Path', default=str(parent_dir)+ '\Data\Train\Set1', help = 'Input Image Location')
    Args = Parser.parse_args()
    NumFeatures = Args.NumFeatures
    Path = Args.Path
    """
    Read a set of images for Panorama stitching
    """
    img_location = Path + "\*.jpg"
    imgset = []
    for filename in glob.glob(img_location):
        img = cv2.imread(filename)
        imgset.append(img)
    """
    Corner Detection
    Save Corner detection output as corners.png
    """

    vectorbank = []
    positionbank = []
    for img in imgset[:2]:
        img1 = copy.deepcopy(img)
        gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        cornerMap = cv2.goodFeaturesToTrack(gray_img, 10000, 0.001, 15)
        cornerMap = np.int32(cornerMap)

        # for point in cornerMap:
        #     x, y = point.reshape(-1)
        #     cv2.circle(img1, (x, y), 2, 255, -1)
        # plt.imshow(img1)
        # plt.show()

        """
        Perform ANMS: Adaptive Non-Maximal Suppression
        Save ANMS output as anms.png
        """

        anmsMap = anms(cornerMap, 1000)
        img2 = copy.deepcopy(img)
        for point in anmsMap:
            x, y = point
            cv2.circle(img2, (x, y), 2, 255, -1)
        plt.imshow(img2)
        plt.show()

        """
        Feature Descriptors
        Save Feature Descriptor output as FD.png
        """

        img3 = copy.deepcopy(gray_img)
        vl, pl = featureDescriptor(img3, anmsMap)
        vectorbank.append(np.array(vl, dtype=np.float32))
        positionbank.append(np.array(pl, dtype=np.float32))


    """
    Feature Matching
    Save Feature Matching output as matching.png
    """

    image1pointlist, image2pointlist, distancelst = featureMatching(vectorbank[0], positionbank[0], vectorbank[1], positionbank[1], 0.7)
    pointmatches = draw_matches(distancelst)
    kpt1 = keypts(image1pointlist)
    kpt2 = keypts(image2pointlist)
    outputimg1 = np.array([])
    matchedImage = cv2.drawMatches(imgset[0], kpt1, imgset[1], kpt2, pointmatches, outputimg1, 1)
    plt.imshow(matchedImage)
    plt.show()

    """
    Refine: RANSAC, Estimate Homography
    """

    outputimg2 = np.array([])
    k1, k2, mimg = ransac_executor(image1pointlist, image2pointlist, distancelst, threshhold=30, Nmax=100, percentage=0.9)
    ransacedImage = cv2.drawMatches(imgset[0], k1, imgset[1], k2, mimg, outputimg2, 1)
    plt.imshow(ransacedImage)
    plt.show()


    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
