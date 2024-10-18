#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    Mat finger, adap, adap1;

    finger = imread("finger_print.png", 0);
    adap = imread("adaptive.png", 0);
    adap1 = imread("adaptive_1.jpg", 0);

    adaptiveThreshold(finger, finger, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 10);
    adaptiveThreshold(adap, adap, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 10);
    adaptiveThreshold(adap1, adap1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 10);

    namedWindow("finger_print");
    namedWindow("adaptive");
    namedWindow("adaptive_1");

    moveWindow("adaptive", 200, 0);
    moveWindow("adaptive_1", 400, 0);   

    imshow("finger_print", finger);
    imshow("adaptive", adap);
    imshow("adaptive_1", adap1);
    waitKey(0);
    return 0;
}