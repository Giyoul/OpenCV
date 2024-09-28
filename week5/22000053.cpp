#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    Mat frame, left_roi, right_roi, result;
    int fps, maxFrame;
    int delay;
    Point p1, p2;
    float rho, theta, a, b, x0, y0;
    VideoCapture cap;
    vector<Vec2f> lines1, lines2;
    Rect left_roi_rect(200, 400, 400, 200);
    Rect right_roi_rect(600, 400, 400, 200);

    if(cap.open("Road.mp4") == 0){
        cout << "no such file" << endl;
        waitKey(0);
    }

    fps = cap.get(CAP_PROP_FPS);
    delay = 1000 / fps;

    // maxFrame = cap.get(CAP_PROP_FRAME_COUNT);

    while(1){
        cap >> frame;
        int currentFrame = cap.get(CAP_PROP_POS_FRAMES);
        // to confirm second;
        // cout << "frames: " << currentFrame << " / " << maxFrame << " - fps " << fps << "\n";
        if(currentFrame == fps * 20) break; // 프레임 수 x 20초
        if(frame.empty()){
            cout << "end of video" << endl;
            break;
        }

        result = frame.clone();
        left_roi = frame.clone();
        right_roi = frame.clone();
        left_roi = left_roi(left_roi_rect);
        right_roi = right_roi(right_roi_rect);
        cvtColor(left_roi, left_roi, COLOR_BGR2GRAY);
        cvtColor(right_roi, right_roi, COLOR_BGR2GRAY);
        blur(left_roi, left_roi, Size(5,5));
        blur(right_roi, right_roi, Size(5,5));
        // GaussianBlur(left_roi, left_roi, Size(5,5), 30, 5, BORDER_DEFAULT);
        // GaussianBlur(right_roi, right_roi, Size(5,5), 5, 5, BORDER_DEFAULT);
        Canny(left_roi, left_roi, 10, 60, 3);
        Canny(right_roi, right_roi, 10, 60, 3);

        /*
            무슨 버그인지는 모르겠는데, 파라미터로 필터링 하면 각도를 잘 인식하지 못하는 경우가 있음.
            차라리 파이 말고 degree로 바꾼다음에 직접 계산 하고 조건문으로 걸러줘야 할듯.
        */
        HoughLines(left_roi, lines1, 1, CV_PI/180, 100);
        HoughLines(right_roi, lines2, 1, CV_PI/180, 100);

        // 왼쪽 선 병합
        float sum_rho_left = 0, sum_theta_left = 0;
        int count_left = 0;
        for (int i = 0; i < lines1.size(); i++) {
            theta = lines1[i][1] * 180 / CV_PI; // 라디안을 각도로 변환
            if (theta >= 30 && theta <= 60) { // 필터링
                sum_rho_left += lines1[i][0];
                sum_theta_left += lines1[i][1];
                count_left++;
            }
        }

        if (count_left > 0) {   // 이거는 0일 경우를 대비하기 위함.
            float avg_rho_left = sum_rho_left / count_left;
            float avg_theta_left = sum_theta_left / count_left;
            a = cos(avg_theta_left);
            b = sin(avg_theta_left);
            x0 = a * avg_rho_left;
            y0 = b * avg_rho_left;
            p1 = Point(cvRound(x0 + 1000 * (-b)) + 200, cvRound(y0 + 1000 * a) + 400);
            p2 = Point(cvRound(x0 - 1000 * (-b)) + 200, cvRound(y0 - 1000 * a) + 400);
            line(result, p1, p2, Scalar(0, 0, 255), 3, 8);
        }

        // 오른쪽 선 병합
        float sum_rho_right = 0, sum_theta_right = 0;
        int count_right = 0;
        for (int i = 0; i < lines2.size(); i++) {
            theta = lines2[i][1] * 180 / CV_PI; // 라디안을 각도로 변환
            if (theta >= 120 && theta <= 150) { // 필터링
                sum_rho_right += lines2[i][0];
                sum_theta_right += lines2[i][1];
                count_right++;
            }
        }

        if (count_right > 0) {
            float avg_rho_right = sum_rho_right / count_right;
            float avg_theta_right = sum_theta_right / count_right;
            a = cos(avg_theta_right);
            b = sin(avg_theta_right);
            x0 = a * avg_rho_right;
            y0 = b * avg_rho_right;
            p1 = Point(cvRound(x0 + 1000 * (-b)) + 600, cvRound(y0 + 1000 * a) + 400);
            p2 = Point(cvRound(x0 - 1000 * (-b)) + 600, cvRound(y0 - 1000 * a) + 400);
            line(result, p1, p2, Scalar(0, 0, 255), 3, 8);
        }

        namedWindow("Left canny");
        namedWindow("Right canny");
        namedWindow("Frame");
        moveWindow("Left canny", 200, 0);
        moveWindow("Right canny", 600, 0);
        moveWindow("Frame", 0, 300);
        imshow("Left canny", left_roi);
        imshow("Right canny", right_roi);
        imshow("Frame", result);
        waitKey(delay);
    }

    cap.release();
    destroyAllWindows();
    
    return 0;
}