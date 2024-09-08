#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    Mat frame;
    int fps, maxFrame;
    int delay;
    VideoCapture cap;

    if(cap.open("background.mp4") == 0){
        cout << "no such file" << endl;
        waitKey(0);
    }

    fps = cap.get(CAP_PROP_FPS);
    delay = 1000 / fps;

    maxFrame = cap.get(CAP_PROP_FRAME_COUNT);

    while(1){
        cap >> frame;
        int currentFrame = cap.get(CAP_PROP_POS_FRAMES);
        cout << "frames: " << currentFrame << " / " << maxFrame << "\n";
        if(currentFrame == fps * 3) break; // 프레임 수 x 3초
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