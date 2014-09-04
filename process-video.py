#!/usr/bin/env python

# When user uploads video, process keyframes and extract features

import cv2
import ffms
import sys


def get_keyframes(filename):
    return []


def discard_blanks(frames):
    pass


def remove_borders(frame):
    pass


def normalize_aspect_ratio(frame):
    pass


def denoise(frame):
    pass


def contrast_correct(frame):
    pass


def preprocess(frame):
    remove_borders(frame)
    normalize_aspect_ratio(frame)
    denoise(frame)
    contrast_correct(frame)


def extract_feature(frame):
    pass


if __name__ == "__main__":
    frames = get_keyframes(sys.argv[1])
    discard_blanks(frames)
    for frame in frames:
        preprocess(frame)
        extract_feature(frame)