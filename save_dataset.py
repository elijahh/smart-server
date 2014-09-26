# Save features to HDF5 dataset

import h5py
import numpy as np

import os.path
import time


def save_correlograms(video_id, features_to_save):
    # save to dataset file for persistence
    num_features = len(features_to_save)
    f = h5py.File("videos.hdf5")
    g = f.require_group("correlograms") # data grouped by type of feature
    videoID = g.get("videoID") # map frame features -> video IDs
    features = g.get("features")
    starting_index = 0
    # resize dataset to fit new data
    if videoID == None:
        dt = h5py.special_dtype(vlen=bytes)
        videoID = g.create_dataset("videoID", (num_features,1), dtype=dt, maxshape=(None,1))
        features = g.create_dataset("features", (num_features,332), maxshape=(None,332))
    else:
        dataset_size = videoID.shape[0]
        new_size = dataset_size + num_features
        videoID.resize(new_size, axis=0)
        features.resize(new_size, axis=0)
        starting_index = dataset_size
    videoID[starting_index:] = video_id
    features[starting_index:] = features_to_save
    f.close()


def search_dataset(video_id, features_to_save):
    # save to temp file so that active daemon can add points and search
    f = open("uploads/" + video_id, "w")
    for row in features_to_save:
        for element in row:
            f.write(str(element) + " ")
        f.write("\n")
    f.close()

    # searcher daemon does work here

    # parse results
    while not os.path.exists("results/" + video_id):
        time.sleep(1)
    f = h5py.File("results/" + video_id)
    results = f.get("result")
    v = h5py.File("videos.hdf5")
    g = v.get("correlograms")
    d = g.get("videoID")
    similar_videos = {}
    for row in results:
        for ID in row:
            similar_videos[d[ID][0]] = similar_videos.get(d[ID][0], 0) + 1
    print similar_videos
    f.close()
    v.close()
