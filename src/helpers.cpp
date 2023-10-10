/* 
 * Elsa Tamara
 * A2 - Content Based Imaged Retrieval
 * CS 5330 - Fall 2023
*/
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <filesystem>

using namespace std;
using namespace cv;

//extracts the 9x9 feature in the center of an image
vector<float> calculate9x9ImageFeature(char* &imagePath) {
    Mat src = imread(imagePath);

    if (src.empty()) {
        printf("Error: no image is found");
        return {-1};
    }

    if (src.rows < 9 || src.cols < 9) {
        printf("Error: Image size is smaller than 9x9.");
        return {-1};
    }

    Rect centerRegion((src.cols - 9) / 2, (src.rows - 9) / 2, 9, 9);
    
    Mat centerPixels = src(centerRegion);
    vector<float> featureVector;

    for (int i = 0; i < centerPixels.rows; i++) {
        Vec3b *row = centerPixels.ptr<Vec3b>(i);
        for (int j = 0; j < centerPixels.cols; j++) {
            for (int c = 0; c < 3; c++) {
                featureVector.push_back(row[j][c]);
            }
        }
    }

    return featureVector;
}

//calculates the euclidean distance between two vector<float>
float euclideanDistance(vector<float>& vec1, vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        printf("Feature vectors must have the same dimensionality.");
        return -1.0;
    }

    float sum = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        sum += pow(vec1[i] - vec2[i], 2);
    }

    return sqrt(sum);
}

//builds a histogram of an image
int buildHistogram(Mat &src, Mat &hist) {
    const int histsize = 256;
    float max;

    max = 0;
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            float B = src.at<Vec3b>(i, j)[0];
            float G = src.at<Vec3b>(i, j)[1];
            float R = src.at<Vec3b>(i, j)[2];
            
            

            // compute rg standard chromaticity
            float divisor = (R + G + B);
            divisor = divisor > 0.0 ? divisor : 1.0;
            float r = R / divisor;
            float g = G / divisor;
            

            // compute indexes, r, g are in [0, 1]
            int rindex = (int)( r * (histsize-1) + 0.5); // rounds to nearest value
            int gindex = (int)( g * (histsize-1) + 0.5);

            //increment the histogram
            hist.at<float>(rindex, gindex)++;
            float newvalue = hist.at<float>(rindex, gindex);
            max = newvalue > max ? newvalue : max;
        }
    }
    hist /= (src.rows * src.cols);
    return 0;
}

// calculates the histogram intersection distance between two histograms
double histogramIntersectionDistance(const Mat& hist1, const Mat& hist2) {
    if(hist1.size() != hist2.size() && hist1.type() != hist2.type()) {
        printf("error: histograms have to be the same size and type");
        return(-1);
    }

    Mat minValues;
    min(hist1, hist2, minValues);

    return sum(minValues)[0];
}

// divides an image to top and bottom part equally
int divideIntoTopBottomHalves(string imgPath, Mat &top, Mat &bottom) {
    Mat src = imread(imgPath);
    if (src.empty()) {
        std::cerr << "Error: Unable to read image." << std::endl;
        return -1;
    }

    int midpoint = src.rows / 2;

    top = src(Rect(0, 0, src.cols, midpoint)).clone();
    bottom = src(Rect(0, midpoint, src.cols, midpoint)).clone();
    return 0;
}

// divides an image into 9 equal boxes or 3x3 grid
int divideToGrid3x3(Mat &src, vector<Mat> &gridMats) {
    int cellHeight = src.rows / 3;
    int cellWidth = src.cols / 3;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Rect grid(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
            Mat cell = src(grid).clone();
            gridMats.push_back(cell);
        }
    }
    return 0;
}

//applies sobelx filter
int sobelX3x3( cv::Mat &src, cv::Mat &dst16 ) {
    if (src.empty()) {
        printf("Invalid input image");
        return -1;
    }

    dst16 = Mat(src.size(), CV_16SC3, Scalar(0, 0, 0));

    for (int i = 1; i < src.rows - 1; i++) {
        Vec3b *rptrm1 = src.ptr<Vec3b>(i-1);
        Vec3b *rptr = src.ptr<Vec3b>(i);
        Vec3b *rptrp1 = src.ptr<Vec3b>(i+1);
        Vec3s *dptr = dst16.ptr<Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
                dptr[j][c] = (rptr[j-1][c] - rptr[j+1][c]);
            }
        }
    }
    return 0;
}

//applies sobely filter
int sobelY3x3( cv::Mat &src, cv::Mat &dst16 ) {
    if (src.empty()) {
        printf("Invalid input image");
        return -1;
    }

    dst16 = Mat(src.size(), CV_16SC3, Scalar(0, 0, 0)); 

    for (int i = 1; i < src.rows - 1; i++) {
        Vec3b *rptrm1 = src.ptr<Vec3b>(i-1);
        Vec3b *rptr = src.ptr<Vec3b>(i);
        Vec3b *rptrp1 = src.ptr<Vec3b>(i+1);
        Vec3s *dptr = dst16.ptr<Vec3s>(i);

        for (int j = 1; j < src.cols - 1; j++) {
            for (int c = 0; c < 3; c++) {
               dptr[j][c] = rptrm1[j][c] - rptrp1[j][c];
            }
        }
    }

    return 0;
}

// calculate the gradient magnitude of an image
int magnitude(Mat &sx, Mat &sy, Mat &dst) {
    if (sx.empty() || sy.empty()) {
        printf("Either input x sobel or y sobel is empty");
        return -1;
    }

    if (sx.size() != sy.size()) {
        printf("x sobel's size is not equal to y sobel's size");
        return -2;
    }

    if (sx.type() != CV_16SC3 || sy.type() != CV_16SC3) {
        printf("Either x sobel or y sobel's type is not CV_16SC3");
        return -3;
    }


    dst = cv::Mat(sx.size(), CV_8UC3);

    for (int i = 0; i < sx.rows; i++) {
        Vec3s *sx_ptr = sx.ptr<Vec3s>(i);
        Vec3s *sy_ptr = sy.ptr<Vec3s>(i);
        Vec3b *d_ptr = dst.ptr<Vec3b>(i);

        for (int j = 0; j < sx.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int magnitude = sqrt(sx_ptr[j][c] * sx_ptr[j][c] + sy_ptr[j][c] * sy_ptr[j][c]);
                d_ptr[j][c] = static_cast<uchar>(magnitude);
            }
        }
    }

    return 0;
}
