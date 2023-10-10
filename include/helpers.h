#ifndef helpers
#define helpers

#include <vector>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

vector<float> calculate9x9ImageFeature(char* &imagePath);
float euclideanDistance(vector<float>& vec1, vector<float>& vec2);
int buildHistogram(Mat &src, Mat &hist);
double histogramIntersectionDistance(const Mat& hist1, const Mat& hist2);
int divideIntoTopBottomHalves(string imgPath, Mat &top, Mat &bottom);
int magnitude(Mat &sx, Mat &sy, Mat &dst);
int sobelY3x3( cv::Mat &src, cv::Mat &dst16);
int sobelX3x3( cv::Mat &src, cv::Mat &dst16);
int divideToGrid3x3(Mat &src, vector<Mat> &gridMats);

#endif