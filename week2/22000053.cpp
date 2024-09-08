#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

int main(){
    Mat gray_image, result;
    // gray_image = imread("../lena.png", 0);  // it's for test in my pc
    gray_image = imread("lena.png", 0);
    result = gray_image.clone();

    int height = gray_image.size().height;
    int width = gray_image.size().width;
    int pixel;

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            pixel = gray_image.at<uchar>(i, j);

            if(pixel < 127){
                pixel = 255 - pixel;
            } else {
                pixel = saturate_cast<uchar>(pow((float)(pixel / 255.0), 10) * 255.0f);
            }
            result.at<uchar>(width - j - 1 , i) = pixel;
        }
    }

    
    imshow("gray image", gray_image);
    imshow("result", result);

    waitKey(0);
    return 0;
}