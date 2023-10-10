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

// main program to compare two images based on its color and texture
// (edges detection by sobel filter and its gradient magnitude)
int main(int argc, char *argv[]) {
    string target = "../data/pic.0535.jpg";
    string imageDirectory = "../data/olympus";
    int N = 4;

    char *targetImage = new char[256];
    strcpy(targetImage, target.c_str());

    Mat src = imread(target);
    if (src.data == NULL) {
        printf("error: unable to read image %s\n", targetImage);
        return(-2);
    }

    vector<char*> imagePaths;
    for (const auto& entry : __fs::filesystem::directory_iterator(imageDirectory)) {
        char* imagePath = new char[entry.path().string().size() + 1];
        strcpy(imagePath, entry.path().string().c_str());
        imagePaths.push_back(imagePath);
    }
    
    Mat xSrc16, ySrc16, gMagSrc;
    sobelX3x3(src, xSrc16);
    sobelY3x3(src, ySrc16);
    magnitude(xSrc16, ySrc16, gMagSrc);

    Mat histMagTarget = Mat::zeros(Size(256, 256), CV_32FC1);
    Mat histTarget = Mat::zeros(Size(256, 256), CV_32FC1);
    buildHistogram(gMagSrc, histMagTarget);
    buildHistogram(src, histTarget);

    vector<pair<char*, double>> matches;

    for (char* &imagePath : imagePaths) {
        if (strstr(imagePath, ".jpg") == nullptr) {
            continue; // Skip if ".jpg" is found in imagePath
        }
        Mat imgCompare = imread(imagePath);
        if (imgCompare.data == NULL) {
            printf("error: unable to read image %s\n", imagePath);
            return(-2);
        }
        Mat xCompare16, yCompare16, gMagCompare;
        sobelX3x3(imgCompare, xCompare16);
        sobelY3x3(imgCompare, yCompare16);
        magnitude(xCompare16, yCompare16, gMagCompare);
        Mat histMagCompare = Mat::zeros(Size(256, 256), CV_32FC1);
        Mat histCompare = Mat::zeros(Size(256, 256), CV_32FC1);
        buildHistogram(gMagCompare, histMagCompare);
        buildHistogram(imgCompare, histCompare);
        double magDist = histogramIntersectionDistance(histMagTarget, histMagCompare);
        double dist = histogramIntersectionDistance(histTarget, histCompare);
        double combinedDistance = (0.5 * magDist) + (0.5 * dist);
        matches.push_back({imagePath, combinedDistance});
    }

    stable_sort(matches.begin(), matches.end(), compareSecond);

    for (size_t i = 0; i < N; i++) {
         std::cout << "Match " << i+1 << ": File name " << matches[i].first << ", Distance " << matches[i].second << std::endl;
    }

}