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
using namespace cv;

//sorts the pair in an array
bool compareSecond(const pair<char*, float>& a, const pair<char*, float>& b) {
    return a.second > b.second;
}

// main program to compare two images on multiple 2d histograms
int main(int argc, char *argv[]) {
    string target = "../data/pic.0274.jpg";
    string imageDirectory = "../data/olympus";
    int N = 4;

    char *targetImage = new char[256];
    strcpy(targetImage, target.c_str());

    Mat topHalfMat, bottomHalfMat;
    divideIntoTopBottomHalves(target, topHalfMat, bottomHalfMat);

    Mat topHalfTargetHist = Mat::zeros(Size(256, 256), CV_32FC1);
    Mat bottomHalfTargetHist = Mat::zeros(Size(256, 256), CV_32FC1);
    buildHistogram(topHalfMat, topHalfTargetHist);
    buildHistogram(bottomHalfMat, bottomHalfTargetHist);

    vector<pair<char*, double>> matches;
    vector<char*> imagePaths;

    for (const auto& entry : __fs::filesystem::directory_iterator(imageDirectory)) {
        char* imagePath = new char[entry.path().string().size() + 1];
        strcpy(imagePath, entry.path().string().c_str());
        imagePaths.push_back(imagePath);
    }

    for (char* &imagePath : imagePaths) {
        if (strstr(imagePath, ".jpg") == nullptr) {
            continue;
        }
        Mat topCompare, bottomCompare;
        divideIntoTopBottomHalves(imagePath, topCompare, bottomCompare);

        Mat topHistCompare = Mat::zeros(Size(256, 256), CV_32FC1);
        Mat bottomHistCompare = Mat::zeros(Size(256, 256), CV_32FC1);
        buildHistogram(topCompare, topHistCompare);
        buildHistogram(bottomCompare, bottomHistCompare);
        double topDist = histogramIntersectionDistance(topHalfTargetHist, topHistCompare);
        double bottomDist = histogramIntersectionDistance(bottomHalfTargetHist, bottomHistCompare);
        double combinedDistance = (0.7 * topDist) + (0.3 * bottomDist);
        matches.push_back({imagePath, combinedDistance});
    }

    sort(matches.begin(), matches.end(), compareSecond);

    for (size_t i = 0; i < N; i++) {
         std::cout << "Match " << i+1 << ": File name " << matches[i].first << ", Distance " << matches[i].second << std::endl;
    }
}