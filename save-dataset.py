# Save features to HDF5 dataset

import h5py
import numpy as np


def save_correlograms(video_id, features_to_save):
    num_features = len(features_to_save)
    f = h5py.File("videos.hdf5")
    g = f.require_group("correlograms") # data grouped by type of feature
    videoID = g.get("videoID") # map frame features -> video IDs
    features = g.get("features")
    starting_index = 0
    # resize dataset to fit new data
    if videoID == None:
        videoID = g.create_dataset("videoID", (num_features,1), maxshape=(None,1))
        features = g.create_dataset("features", (num_features,332), maxshape=(None,332))
    else:
        dataset_size = videoID.shape[0]
        new_size = dataset_size + num_features
        videoID.resize(new_size, axis=0)
        features.resize(new_size, axis=0)
        starting_index = dataset_size - 1
    videoID[starting_index:] = video_id
    features[starting_index:] = features_to_save
    f.close()