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

// calculates the weighted histogram intersection distance
double weightedHistogramIntersection(const Mat& hist1, const Mat& hist2, double sigma) {
    Mat gausX = getGaussianKernel(hist1.rows, sigma, CV_32F);
    Mat gausY = getGaussianKernel(hist1.cols, sigma, CV_32F);
    Mat transposed = gausX * gausY.t();
    Mat weightedHist1, weightedHist2;
    multiply(hist1, transposed, weightedHist1);
    multiply(hist2, transposed, weightedHist2);

    Scalar sumWeightedHist1 = sum(weightedHist1);
    Scalar sumWeightedHist2 = sum(weightedHist2);

    return sumWeightedHist1[0] - sum(weightedHist1.mul(min(hist1, hist2)))[0]
         + sumWeightedHist2[0] - sum(weightedHist2.mul(min(hist1, hist2)))[0];
}

// sorts the pair in an array
bool compareSecond(const pair<char*, float>& a, const pair<char*, float>& b) {
    return a.second < b.second;
}

// main program to find similar images to the blue trash can
int main(int argc, char *argv[]) {
    string target = "../data/pic_0920.jpg";
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
            double dist = weightedHistogramIntersection(histTargets[i], histCompare, 32.0);
            distances.push_back(dist);
        }

        double sumDistance = std::accumulate(distances.begin(), distances.end(), 0.0);
        matches.push_back({imagePath, sumDistance});
    }

    stable_sort(matches.begin(), matches.end(), compareSecond);
    for (size_t i = 0; i < N; i++) {
         std::cout << "Match " << i+1 << ": File name " << matches[i].first << ", Distance " << matches[i].second << std::endl;
    }
}