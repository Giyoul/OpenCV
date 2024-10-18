#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    Mat frame, result, rois, image;
    int fps, delay;
    Point p1, p2;
    float rho, theta, a, b, x0, y0;
    VideoCapture cap;
    vector<Vec2f> lines;  // 전체 프레임에서 검출된 선 저장
    Rect roi_rect(230, 200, 260, 280);

    if (cap.open("Project2_video.mp4") == 0) {
        cout << "no such file" << endl;
        waitKey(0);
    }

    fps = cap.get(CAP_PROP_FPS);
    delay = 300 / fps; // FPS에 따른 지연 시간

    int currentFrame = 0;
    int displayUntilFrame = -1; // 2초 후 프레임을 저장할 변수

    while (true) {
        cap >> frame;
        currentFrame = cap.get(CAP_PROP_POS_FRAMES);
        if (frame.empty()) {
            cout << "end of video" << endl;
            break;
        }

        result = frame.clone();  // 결과를 표시할 이미지
        rois = frame.clone();
        rois = rois(roi_rect);

        cvtColor(rois, rois, COLOR_BGR2GRAY);  // ROI를 그레이스케일로 변환
        blur(rois, rois, Size(5, 5));  // 블러 처리
        Canny(rois, rois, 10, 60, 3);  // 엣지 검출

        HoughLines(rois, lines, 1, CV_PI / 180, 100);  // ROI에서 HoughLines 검출

        float sum_rho = 0, sum_theta = 0;
        int count = 0;
        for (int i = 0; i < lines.size(); i++) {
            theta = lines[i][1] * 180 / CV_PI;  // 라디안을 각도로 변환
            if ((theta >= 155 && theta <= 177) || (theta >= 3 && theta <= 40)) {
                sum_rho += lines[i][0];
                sum_theta += lines[i][1];
                count++;
                cout << "Detected theta: " << theta << endl; // 디버깅용 출력
            }
        }

        // count > 0인 경우 텍스트 출력 설정
        if (count > 0) {
            displayUntilFrame = currentFrame + (2 * fps); // 2초 후 프레임 업데이트
        }

        // 현재 프레임이 displayUntilFrame 이하인 경우 텍스트 출력
        if (currentFrame <= displayUntilFrame) {
            putText(result, format("Lane Departure!"), Point(20, 80), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 4);
        }


        Ptr<BackgroundSubtractor> bg_model = createBackgroundSubtractorMOG2();
        Mat foregroundMask, backgroundImg, foregroundImg;

        image = frame.clone();
        resize(image, image, Size(480, 360));
        if (foregroundMask.empty()){
            foregroundMask.create(image.size(), image.type());
        }
        bg_model->apply(image, foregroundMask);
        GaussianBlur(foregroundMask, foregroundMask, Size(11, 11), 3.5, 3.5);
        threshold(foregroundMask, foregroundMask, 10, 255, THRESH_BINARY);
        foregroundImg = Scalar::all(0);
        image.copyTo(foregroundImg, foregroundMask);
        bg_model->getBackgroundImage(backgroundImg);

        imshow("foreground mask", foregroundMask);
        imshow("foreground image", foregroundImg);
        if (!backgroundImg.empty()) {
            imshow("mean background image", backgroundImg);
        }

        namedWindow("Project2");
        moveWindow("Project2", 600, 0);
        imshow("Project2", result);  // 최종 결과 화면 출력
        waitKey(delay);
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
