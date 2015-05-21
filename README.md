# smart-server
When uploading a video, save color features and search for matches.

This code made up the server of my thesis project, "YouTrace: A Smartphone System for Tracking Video Modifications".

To integrate into a hosting service (I used MediaDrop), uploading calls "process_video.py", which in turn calls
"save_dataset.py". The searcher daemon ("searcher.cpp", compiled using h5c++) needs to be running, because it
maintains the index for all video features.
