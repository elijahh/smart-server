#!/usr/bin/env python

# When user uploads video, process keyframes and extract features

from scipy import ndimage

import cv2
import ffms
import numpy as np
import sys

SHOT_CORRELATION = 0.7 # minimum color histogram correlation within shot
BLANK_THRESHOLD = 50 # number of keypoints needed for image not to be "blank"


def keyframe_index(vs, width, height, i):
    return vs.get_frame(vs.track.keyframes[i]).planes[0].reshape((height, width, 3)).copy()


def calc_histogram(frame):
    hist = [] # list of histograms for all channels (colors)
    for channel in range(3):
        hist_item = cv2.calcHist([frame],[channel],None,[256],[0,255])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist += [hist_item]
    return hist


def correlation(hist1, hist2):
    return min([cv2.compareHist(hist1[i], hist2[i], cv2.cv.CV_COMP_CORREL) for i in range(3)]) # min over all channels (colors)


# grab only one keyframe per shot
def get_keyframes(filename):
    vs = ffms.VideoSource(filename)
    vs.set_output_format([ffms.get_pix_fmt('bgr24')])
    width = vs.get_frame(0).EncodedWidth
    height = vs.get_frame(0).EncodedHeight
    cached_hist = []
    keyframes = []
    for i in range(len(vs.track.keyframes)-1):
        img1 = keyframe_index(vs, width, height, i)
        img2 = keyframe_index(vs, width, height, i+1)
        if len(cached_hist) == 0:
            hist1 = calc_histogram(img1)
        else:
            hist1 = cached_hist
        hist2 = calc_histogram(img2)
        cached_hist = hist2
        correl = correlation(hist1, hist2)
        if correl < SHOT_CORRELATION:
            keyframes += [img1, img2]
    if len(keyframes) == 0:
        keyframes = [vs.get_frame(vs.track.keyframes[0]).planes[0].reshape((height, width, 3))]
    return keyframes


def discard_blanks(frames):
    # Initiate ORB detector
    orb = cv2.ORB(nfeatures=1500,edgeThreshold=10)
    
    frames_to_discard = []
    for i in range(len(frames)):
        kp, des = orb.detectAndCompute(frames[i],None)
        if len(kp) < BLANK_THRESHOLD: # discard frames with fewer keypoints than threshold
            frames_to_discard += [i]
    for i in range(len(frames_to_discard)):
        frames.pop(frames_to_discard[i] - i)


def denoise(frame):
    return ndimage.median_filter(frame, 3)


def remove_borders(frame):
    width = len(frame[0])
    height = len(frame)
    color = frame[0][0] # border color inferred from top-left pixel
    # detect top and bottom borders
    border_size_y = 0
    end_border = False
    for j in range(height / 2):
        for i in range(width):
            if (frame[j][i] != color).any() or (frame[-j-1][i] != color).any():
                end_border = True
        if end_border == True:
            break
        border_size_y += 1
    # detect left and right borders
    border_size_x = 0
    end_border = False
    for i in range(width / 2):
        for j in range(height):
            if (frame[j][i] != color).any() or (frame[j][-i-1] != color).any():
                end_border = True
        if end_border == True:
            break
        border_size_x += 1
    return frame[border_size_y:height-border_size_y,border_size_x:width-border_size_x]


def normalize_aspect_ratio(frame):
    pass


def equalize_histogram(frame):
    pass


def preprocess(frame):
    frame = denoise(frame)
    frame = remove_borders(frame)
    normalize_aspect_ratio(frame)
    equalize_histogram(frame)


def extract_feature(frame):
    pass


if __name__ == "__main__":
    frames = get_keyframes(sys.argv[1])
    discard_blanks(frames)
    for frame in frames:
        preprocess(frame)
        extract_feature(frame)
