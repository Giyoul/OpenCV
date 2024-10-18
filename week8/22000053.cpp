#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    Mat frame, result, rois, image;
    int fps, delay;
    VideoCapture cap;
    vector<Vec2f> lines;  // 전체 프레임에서 검출된 선 저장
    Rect roi_rect(230, 200, 260, 280);
    Rect stop_moving_rect(280, 270, 160, 160);

    if (cap.open("Project2_video.mp4") == 0) {
        cout << "no such file" << endl;
        return -1; // 에러 발생 시 종료
    }

    fps = cap.get(CAP_PROP_FPS);
    delay = 300 / fps; // FPS에 따른 지연 시간

    int currentFrame = 0;
    int displayUntilFrame = -1; // 2초 후 프레임을 저장할 변수
    int delaytext = -1;
    int delaytextEnd = -1;
    int stationaryFrameCount = 0; // 정지 상태 카운트
    int backgroundUpdateInterval = 70;
    bool able = true;

    // 배경 차감에 사용할 변수 초기화
    Mat background, gray, foregroundMask, foregroundImg;
    cap >> background; // 첫 번째 프레임을 배경으로 설정
    cvtColor(background, background, COLOR_BGR2GRAY); // 그레이스케일로 변환

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
            float theta = lines[i][1] * 180 / CV_PI;  // 라디안을 각도로 변환
            if ((theta >= 155 && theta <= 177) || (theta >= 3 && theta <= 40)) {
                sum_rho += lines[i][0];
                sum_theta += lines[i][1];
                count++;
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

        // 배경 차감 처리
        cvtColor(frame, gray, COLOR_BGR2GRAY); // 현재 프레임을 그레이스케일로 변환
        absdiff(background, gray, foregroundMask); // 배경과 현재 프레임 차감
        threshold(foregroundMask, foregroundMask, 50, 255, THRESH_BINARY); // 이진화
        foregroundMask = foregroundMask(stop_moving_rect);
        foregroundMask.copyTo(foregroundImg); // 포그라운드 마스크 복사

        // 픽셀 값 변화 감지
        double nonZeroSum = 0; // 포그라운드 마스크의 모든 픽셀 값의 합
        for (int y = 0; y < foregroundMask.rows; y++) {
            for (int x = 0; x < foregroundMask.cols; x++) {
                nonZeroSum += foregroundMask.at<uchar>(y, x); // 각 픽셀 값을 누적
            }
        }

        if (nonZeroSum > 3000000) { // 임계값 설정
            if (able) {
                delaytext = currentFrame + (2 * fps);
                delaytextEnd = delaytext + (2 * fps);
                able = false;
            }
        }

        if (currentFrame >= delaytext && currentFrame <= delaytextEnd) {
            putText(result, format("Start Moving!"), Point(20, 150), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 4);
        }
        if (currentFrame > delaytextEnd) {
            able = true;
        }

        // 일정 프레임 동안 변화가 없으면 배경 업데이트
        if (currentFrame % backgroundUpdateInterval == 0) {
            background = gray.clone(); // 새로운 배경으로 업데이트
        }

        // 결과 출력
        namedWindow("Foreground Image");
        moveWindow("Foreground Image", 600, 0);
        imshow("Foreground Image", foregroundImg); // 포그라운드 이미지 출력
        imshow("Foreground Mask", foregroundMask); // 포그라운드 마스크 출력
        imshow("Project2", result);  // 최종 결과 화면 출력
        waitKey(delay);
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
