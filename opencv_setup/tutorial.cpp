#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    Mat frame;
    int fps;
    int delay;
    VideoCapture cap;

    if(cap.open("../background.mp4") == 0){
        cout << "no such file" << endl;
        waitKey(0);
    }

    fps = cap.get(CAP_PROP_FPS);
    delay = 1000 / fps;

    while(1){
        cap >> frame;
        if(frame.empty()){
            cout << "end of video" << endl;
            break;
        }
        imshow("video", frame);
        waitKey(delay);
    }

    cap.release();
    destroyAllWindows();

    return 0;
}