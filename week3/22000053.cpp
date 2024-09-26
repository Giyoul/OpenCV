#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    // moon
    Mat moon, moon_filtered, temp, laplacian, convertabs, moon_result;

    // moon = imread("../3_week_dataset/moon.png", 0);
    // moon_result = imread("../3_week_dataset/moon.png", 0);
    moon = imread("moon.png", 0);
    moon_result = imread("moon.png", 0);

    GaussianBlur(moon, temp, Size(3,3), 0, 0, BORDER_DEFAULT);
    Laplacian(temp, laplacian, CV_16S, 1, 1, 0);
    convertScaleAbs(laplacian, convertabs);

    moon_filtered = moon + convertabs;

    int height = moon.size().height;
    int width = moon.size().width;

    // moon은 오른쪽 절반만 필터링
    for(int i = 0; i < height; i++){
        for(int j = width/2; j < width; j++){
            moon_result.at<uchar>(i, j) = moon_filtered.at<uchar>(i, j);
        }
    }

    // saltnpepper
    Mat src, dst, saltnpepper_result;
    int val = 9;
    
    // src = imread("../3_week_dataset/saltnpepper.png", 0);
    // saltnpepper_result = imread("../3_week_dataset/saltnpepper.png", 0);
    src = imread("saltnpepper.png", 0);
    saltnpepper_result = imread("saltnpepper.png", 0);

    medianBlur(src, dst, val);

    height = src.size().height;
    width = src.size().width;

    // saltnpepper는 좌측 절반만 필터링
    for(int i = 0; i < height; i++){
        for(int j = 0; j < width/2; j++){
            saltnpepper_result.at<uchar>(i, j) = dst.at<uchar>(i, j);
        }
    }

    imshow("moon", moon);
    imshow("moon_filtered", moon_result);
    imshow("saltnpepper", src);
    imshow("saltnpepper_filtered", saltnpepper_result);

    waitKey(0);
    return 0;
}