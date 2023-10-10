/* 
 * Elsa Tamara
 * A2 - Content Based Imaged Retrieval
 * CS 5330 - Fall 2023
*/

#include <iostream>
#include <vector>
#include <filesystem>
#include "opencv2/opencv.hpp"
#include "csv_util.h"
#include "helpers.h"

using namespace std;

// sorts the match score
bool compareSecond(const pair<char*, float>& a, const pair<char*, float>& b) {
    return a.second < b.second;
}

// main method for baseline matcher
int main(int argc, char *argv[]) {
    string target = "../data/pic_1016.jpg";
    string csv = "../data/output_feature_vectors.csv";
    char *targetImage = new char[256];
    char *outputCSV = new char[256];
    strcpy(outputCSV, csv.c_str());
    strcpy(targetImage, target.c_str());

    vector<char *> filenames;
    vector<vector<float>> fileFeatureVectors;
    int echo_file = 1;

    int result = read_image_data_csv(outputCSV, filenames, fileFeatureVectors, echo_file);
 
    if (result != 0) {
        printf("Error reading CSV file.");
        return 1;
    }

    vector<pair<char*, float>> matches;

    vector<float> featureVectorTargetImg = calculate9x9ImageFeature(targetImage);
    size_t N = 4;

    for (size_t i = 0; i < fileFeatureVectors.size(); ++i) {
        float distance = euclideanDistance(featureVectorTargetImg, fileFeatureVectors[i]);
        matches.push_back({filenames[i], distance});
    }
    stable_sort(matches.begin(), matches.end(), compareSecond);
    for (size_t i = 0; i < N; i++) {
         std::cout << "Match " << i+1 << ": File name " << matches[i].first << ", Distance " << matches[i].second << std::endl;
    }
}

