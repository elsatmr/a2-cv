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

// main program to compare two images on a single 2d histogram
int main(int argc, char *argv[]) {
    string target = "../data/pic.0164.jpg";
    string imageDirectory = "../data/olympus";
    int N = 4;

    char *targetImage = new char[256];
    strcpy(targetImage, target.c_str());

    Mat src = imread(target);
    if (src.data == NULL) {
        printf("error: unable to read image %s\n", targetImage);
        return(-2);
    }

    Mat histTarget = Mat::zeros(Size(256, 256), CV_32FC1);
    buildHistogram(src, histTarget);

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
        Mat imgCompare = imread(imagePath);
        if (imgCompare.data == NULL) {
            printf("error: unable to read image %s\n", imagePath);
            return(-2);
        }
        Mat hist = Mat::zeros(Size(256, 256), CV_32FC1);
        buildHistogram(imgCompare, hist);
        matches.push_back({imagePath, histogramIntersectionDistance(histTarget, hist)});
    }

    stable_sort(matches.begin(), matches.end(), compareSecond);
    for (size_t i = 0; i < N; i++) {
         std::cout << "Match " << i+1 << ": File name " << matches[i].first << ", Distance " << matches[i].second << std::endl;
    }
}