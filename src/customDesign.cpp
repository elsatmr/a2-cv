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
#include <numeric>


using namespace std;
using namespace cv;

//sorts the pair in an array
bool compareSecond(const pair<char*, float>& a, const pair<char*, float>& b) {
    return a.second > b.second;
}

//main program to compare 2 images based on the histograms of its chopped up
// 3x3 grid version of an image
int main(int argc, char *argv[]) {
    string target = "../data/puppycat.jpg";
    string imageDirectory = "../data/picz";
    int N = 4;

    char *targetImage = new char[256];
    strcpy(targetImage, target.c_str());
    Mat src = imread(target);
    if (src.data == NULL) {
        printf("error: unable to read image %s\n", targetImage);
        return(-2);
    }

    vector<Mat> gridMats;
    divideToGrid3x3(src, gridMats);
    vector<Mat> histTargets;
    for (int i = 0; i < gridMats.size(); i++) {
        Mat histTarget = Mat::zeros(Size(256, 256), CV_32FC1);
        buildHistogram(gridMats[i], histTarget);
        histTargets.push_back(histTarget);
    }

    vector<char*> imagePaths;
    vector<pair<char*, double>> matches;
    

    for (const auto& entry : __fs::filesystem::directory_iterator(imageDirectory)) {
        char* imagePath = new char[entry.path().string().size() + 1];
        strcpy(imagePath, entry.path().string().c_str());
        imagePaths.push_back(imagePath);
    }

    for (char* &imagePath : imagePaths) {
        if (strstr(imagePath, ".jpg") == nullptr) {
            continue;
        }
        Mat imgCompare = imread(imagePath);
        if (imgCompare.data == NULL) {
            printf("error: unable to read image %s\n", imagePath);
            return(-2);
        }

        vector<Mat> compareMats;
        divideToGrid3x3(imgCompare, compareMats);
        vector<Mat> histCompares;
        vector<double> distances;
        for (int i = 0; i < compareMats.size(); i++) {
            Mat histCompare = Mat::zeros(Size(256, 256), CV_32FC1);
            buildHistogram(compareMats[i], histCompare);
            double dist = histogramIntersectionDistance(histTargets[i], histCompare);
            distances.push_back(dist);
        }

        double weightedDistance = distances[0] +  distances[1] + distances[2] + distances[3] + distances[4] + distances[5] + 1 * distances[6] + 2 * distances[7] + 1 * distances[8];
        
        matches.push_back({imagePath, weightedDistance});
    }

    stable_sort(matches.begin(), matches.end(), compareSecond);
    for (size_t i = 0; i < N; i++) {
         std::cout << "Match " << i+1 << ": File name " << matches[i].first << ", Distance " << matches[i].second << std::endl;
    }

}