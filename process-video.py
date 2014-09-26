#!/usr/bin/env python

# When user uploads video, process keyframes and extract features

from scipy import misc
from scipy import ndimage

import cv2
import ffms
import numpy as np
import sys

import save-dataset

SHOT_CORRELATION = 0.7 # minimum color histogram correlation within shot
BLANK_THRESHOLD = 50 # number of keypoints needed for image not to be "blank"
NORMAL_SIZE = (180, 320) # resize all frames to this size (height, width)
CORNER_MASK_SIZE = 20 # number of pixels to mask off corners to focus on central frame
NEIGHBOR_DISTANCE = 7 # distance in pixels between neighbors for computing correlogram


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
    return misc.imresize(frame, NORMAL_SIZE)


def equalize_histogram(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.cv.CV_BGR2HSV)
    frame_hsv[:,:,2] = cv2.equalizeHist(frame_hsv[:,:,2]) # equalize V channel
    return cv2.cvtColor(frame_hsv, cv2.cv.CV_HSV2BGR)


def preprocess(frame):
    frame = denoise(frame)
    frame = remove_borders(frame)
    frame = normalize_aspect_ratio(frame)
    frame = equalize_histogram(frame)
    return frame
    

# HSV quantization to 166 dimensions:
# 18 hues * 3 saturations * 3 values + 4 grays (162 + 4 bins)
# In OpenCV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
def quantize(frame):
    width = len(frame[0])
    height = len(frame)
    frame_quantized = []
    for j in range(height):
        row_quantized = []
        for i in range(width):
            (h, s, v) = frame[j][i]
            bin_q = 0
            if (s <= 25 or v <= 25) and not (v >= 200 and s >= 13): # gray (saturation or value lower than 10% -- but if value is higher than 80%, then saturation must be lower than 5%)
                v_q = int(round(v / 85.0)) # 0 - 3
                bin_q = 162 + v_q
            else:
                h_q = h / 10 # 0 - 17
                s_q = max(0, int(round(s / 85.0) - 1)) # 0 - 2
                v_q = max(0, int(round(v / 85.0) - 1)) # 0 - 2
                bin_q = h_q * 9 + s_q * 3 + v_q
            row_quantized += [bin_q]
        frame_quantized += [row_quantized]
    return frame_quantized


def get_neighborhood(frame, j, i):
    width = len(frame[0])
    height = len(frame)
    left_edge = max(0, i - NEIGHBOR_DISTANCE)
    right_edge = min(width, i + NEIGHBOR_DISTANCE)
    top_edge = max(0, j - NEIGHBOR_DISTANCE)
    bottom_edge = min(height, j + NEIGHBOR_DISTANCE)
    return list(frame[top_edge:bottom_edge,left_edge:right_edge].flatten())


def calc_autocorrelogram(frame):
    width = len(frame[0])
    height = len(frame)
    num_same_color = np.zeros(166)
    num_neighbors = np.zeros(166)
    for j in range(height):
        for i in range(width):
            color = frame[j][i]
            neighbors = get_neighborhood(frame, j, i)
            num_same_color[color] += neighbors.count(color)
            num_neighbors[color] += float(len(neighbors))
    autocorrelogram = []
    for i in range(166):
        if num_neighbors[i] != 0.0: # avoid division by zero
            autocorrelogram += [num_same_color[i] / num_neighbors[i]]
        else:
            autocorrelogram += [0.0]
    return autocorrelogram


def extract_feature(frame):
    frame_hsv = cv2.cvtColor(frame, cv2.cv.CV_BGR2HSV)
    frame_q = np.array(quantize(frame_hsv))
    # extract color auto-correlogram from central horizontal and vertical strips => 332-d feature
    width = len(frame_q[0])
    height = len(frame_q)
    horizontal_strip = frame_q[CORNER_MASK_SIZE:height-CORNER_MASK_SIZE,:]
    vertical_strip = frame_q[:,CORNER_MASK_SIZE:width-CORNER_MASK_SIZE]
    component1 = calc_autocorrelogram(horizontal_strip)
    component2 = calc_autocorrelogram(vertical_strip)
    return component1 + component2


if __name__ == "__main__":
    frames = get_keyframes(sys.argv[1])
    discard_blanks(frames)
    features = []
    for frame in frames:
        frame = preprocess(frame)
        features += [extract_feature(frame)]
    save-dataset.save_correlograms(sys.argv[1], features)
