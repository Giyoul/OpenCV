#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

struct MouseParams {
    Mat img;
    vector<Point2f> in, out;
    int inCount;
};

static void onMouse(int event, int x, int y, int, void* param) {
    MouseParams* mp = (MouseParams*)param;

    if (event == EVENT_LBUTTONDOWN) {
        mp->in.push_back(Point2f(x, y));
        mp->inCount = mp->in.size();
    }

    if (event == EVENT_RBUTTONDOWN) {
        mp->in.clear();
        mp->inCount = 0;
    }
}

int main() {
    VideoCapture mainVideo("Timesquare.mp4");
    VideoCapture overlayVideo("contest.mp4");

    if (!mainVideo.isOpened() || !overlayVideo.isOpened()) {
        cerr << "비디오 파일을 열 수 없습니다." << endl;
        return -1;
    }

    namedWindow("background", WINDOW_AUTOSIZE);
    namedWindow("input", WINDOW_AUTOSIZE);

    moveWindow("background", 0, 100);
    moveWindow("input", 900, 100);

    Mat mainFrame, overlayFrame;
    
    MouseParams mp;

    int overlayWidth = overlayVideo.get(CAP_PROP_FRAME_WIDTH);
    int overlayHeight = overlayVideo.get(CAP_PROP_FRAME_HEIGHT);

    mp.out.push_back(Point2f(0, 0));
    mp.out.push_back(Point2f(overlayWidth, 0));
    mp.out.push_back(Point2f(overlayWidth, overlayHeight));
    mp.out.push_back(Point2f(0, overlayHeight));

    if (!mainVideo.read(mainFrame)) {
        cerr << "첫 번째 비디오 프레임을 읽을 수 없습니다." << endl;
        return -1;
    }

    mp.img = mainFrame;

    setMouseCallback("background", onMouse, (void*)&mp);

    while (true) {
        if (!mainVideo.read(mainFrame) || !overlayVideo.read(overlayFrame)) {
            cerr << "비디오 프레임을 읽을 수 없습니다." << endl;
            break;
        }

        if (mainFrame.empty() || overlayFrame.empty()) {
            cerr << "비디오 프레임이 비어있습니다." << endl;
            break;
        }

        Mat result = mainFrame.clone();

        for (size_t i = 0; i < mp.in.size(); i++) {
            circle(result, mp.in[i], 3, Scalar(0, 0, 255), 5);
        }

        if (mp.in.size() == 4) {
            Mat homo_mat = getPerspectiveTransform(mp.out, mp.in);
            Mat transformedOverlay;

            warpPerspective(overlayFrame, transformedOverlay, homo_mat, mainFrame.size());

            Mat mask = Mat::zeros(mainFrame.size(), CV_8UC1);

            Point intPoints[1][4];
            for (int i = 0; i < 4; i++) {
                intPoints[0][i] = Point(mp.in[i].x, mp.in[i].y);
            }

            const Point* pts[1] = { intPoints[0] };
            int npt[] = {4};
            fillPoly(mask, pts, npt, 1, Scalar(255));

            transformedOverlay.copyTo(result, mask);
        }

        imshow("background", result);
        imshow("input", overlayFrame);

        if (waitKey(30) == 27) {
            break;
        }
    }

    mainVideo.release();
    overlayVideo.release();
    destroyAllWindows();

    return 0;
}
