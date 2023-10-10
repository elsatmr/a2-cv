/* A2 - Content Based Image Retrieval
 * Elsa Tamara
 * CS5330 - Fall 2023
*/

#include <iostream>
#include <vector>
#include <filesystem>
#include "opencv2/opencv.hpp"
#include "csv_util.h"
#include "helpers.h"

using namespace std;

// main program to read an image, calculate the image feature, and then append it to the csv
int main() {
    string imageDirectory = "../data/olympus";
    string csv = "../data/output_feature_vectors.csv";
    char *outputCSV = new char[256];
    strcpy(outputCSV, csv.c_str());
    int resetFile = 1; 

    vector<char*> imagePaths;

    for (const auto& entry : __fs::filesystem::directory_iterator(imageDirectory)) {
        char* imagePath = new char[entry.path().string().size() + 1];
        strcpy(imagePath, entry.path().string().c_str());
        imagePaths.push_back(imagePath);
    }

    for (char* &imagePath : imagePaths) {
        if (strstr(imagePath, ".jpg") == nullptr) {
            continue; // Skip if ".jpg" is found in imagePath
        }
        vector<float> featureVector = calculate9x9ImageFeature(imagePath);
        append_image_data_csv(outputCSV, imagePath, featureVector, resetFile);
        resetFile = 0;
    }

    return 0;
}

