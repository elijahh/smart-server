// Process that maintains an index and handles searches
// Written in C++ to take advantage of FLANN

#include <flann/flann.hpp>
#include <flann/io/hdf5.h>

#include "boost/filesystem.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <ctime>
#include <sys/stat.h>
#include <unistd.h>

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
        featuresMat[j][i] = features[i][j];
      }
    }
    return featuresMat;
  }
  return flann::Matrix<float>(0, 0, 0);
}


void indexAndSearch(flann::Matrix<int> indices, flann::Matrix<float> dists, int nn, flann::Index<flann::L2<float> > index, flann::Matrix<float> query, bool pointsInIndex) {
  if (!pointsInIndex) index.addPoints(query);

  index.knnSearch(query, indices, dists, nn, flann::SearchParams());

  delete[] query.ptr();
}


void checkDirectory(flann::Index<flann::L2<float> > index, fs::path full_path, std::time_t dataset_accessed) {
  // check directory for files containing features to add,
  // output file with nearest neighbor indices for each feature
  // (indices will then be mapped to video IDs by the python program)
  fs::directory_iterator end_iter;
  for (fs::directory_iterator dir_itr(full_path); dir_itr != end_iter; ++dir_itr) {
    if (fs::is_regular_file(dir_itr->status())) {
      // prepare search
      bool pointsInIndex = (fs::last_write_time(dir_itr->path()) < dataset_accessed);
      int nn = 3; // 3 neighbors
      flann::Matrix<float> query = featuresFromFile(dir_itr->path());
      flann::Matrix<int> indices(new int[query.rows*nn], query.rows, nn);
      flann::Matrix<float> dists(new float[query.rows*nn], query.rows, nn);
      indexAndSearch(indices, dists, nn, index, query, pointsInIndex);

      // save results
      const std::string outfile = std::string("results_") + dir_itr->path().filename().string();
      flann::save_to_file(indices, outfile, "indices");
      flann::save_to_file(dists, outfile, "dists");
      const std::string newfile = std::string("results/") + dir_itr->path().filename().string();
      fs::rename(fs::path(outfile), fs::path(newfile));
      chmod(newfile.c_str(), S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);

      delete[] indices.ptr();
      delete[] dists.ptr();
      fs::remove(dir_itr->path());
    }
  }
}


void handleSearches(flann::Index<flann::L2<float> > index, std::time_t dataset_accessed) {
  fs::path full_path(fs::initial_path<fs::path>());
  full_path = fs::absolute(fs::path("uploads/"), full_path);
  if (!fs::exists(full_path) || !fs::is_directory(full_path)) {
    std::cout << "Upload directory " << full_path.string() << " cannot be found or is not a directory." << std::endl;
    return;
  }
  while (true) {
    checkDirectory(index, full_path, dataset_accessed);
  }
}


int main(int argc, char** argv) {
  chdir("/home/elijahhoule/youtrace/");
  flann::Matrix<float> dataset;
  flann::load_from_file(dataset, "videos.hdf5", "correlograms/features");
  std::time_t dataset_accessed = std::time(NULL);
  flann::Index<flann::L2<float> > index(dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();
  handleSearches(index, dataset_accessed);
  return 0;
}
