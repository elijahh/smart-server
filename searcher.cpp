// Process that maintains an index and handles searches
// Written in C++ to take advantage of FLANN

#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

#include "boost/filesystem.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = boost::filesystem;


flann::Matrix<float> featuresFromFile(fs::path full_path) {
  std::ifstream infile(full_path.string().c_str());
  if (infile.is_open()) {
    std::vector<float> features[332];
    std::string line;
    while (std::getline(infile, line)) {
      std::istringstream iline(line);
      std::istream_iterator<float> itr(iline);
      for (int i = 0; i < 332; ++i, ++itr) {
	features[i].push_back(*itr);
      }
    }
    int numFeatures = features[0].size();
    flann::Matrix<float> featuresMat(new float[numFeatures*332], numFeatures, 332);
    for (int j = 0; j < numFeatures; ++j) {
      for (int i = 0; i < 332; ++i) {
	featuresMat[j][i] = features[j][i];
      }
    }
    return featuresMat;
  }
  return flann::Matrix<float>(0, 0, 0);
}


flann::Matrix<int> indexAndSearch(flann::Index<flann::L2<float> > index, flann::Matrix<float> query) {
  index.addPoints(query);

  // nearest-neighbor search with 3 neighbors
  int nn = 3;
  flann::Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
  flann::Matrix<float> dists(new float[query.rows*nn], query.rows, nn);
  index.knnSearch(query, indices, dists, nn, flann::SearchParams());

  return indices;
}


void checkDirectory(flann::Index<flann::L2<float> > index, fs::path full_path) {
  // check directory for files containing features to add,
  // output file with nearest neighbor indices for each feature
  // (indices will then be mapped to video IDs by the python program)
  fs::directory_iterator end_iter;
  for (fs::directory_iterator dir_itr(full_path); dir_itr != end_iter; ++dir_itr) {
    if (fs::is_regular_file(dir_itr->status())) {
      std::ofstream outfile;
      outfile.open(std::string(std::string("results/") + dir_itr->path().filename().string()).c_str());
      outfile << indexAndSearch(index, featuresFromFile(dir_itr->path()));
      outfile.close();
    }
  }
}


void handleSearches(flann::Index<flann::L2<float> > index) {
  fs::path full_path(fs::initial_path<fs::path>());
  full_path = fs::absolute(fs::path("uploads/"), full_path);
  if (!fs::exists(full_path) || !fs::is_directory(full_path)) {
    std::cout << "Upload directory " << full_path.string() << " cannot be found or is not a directory." << std::endl;
    return;
  }
  while (true) {
    checkDirectory(index, full_path);
  }
}


int main(int argc, char** argv) {
  flann::Matrix<float> dataset;
  flann::load_from_file(dataset, "videos.hdf5", "correlograms/features");
  flann::Index<flann::L2<float> > index(dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();
  handleSearches(index);
  return 0;
}
