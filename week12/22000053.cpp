#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

struct MouseParams {
    Mat img;
    vector<Point2f> in, out;
    int inCount;  // 클릭된 점의 개수를 추적
};

// 마우스 클릭 이벤트 처리 함수
static void onMouse(int event, int x, int y, int, void* param) {
    MouseParams* mp = (MouseParams*)param;

    if (event == EVENT_LBUTTONDOWN) {  // 왼쪽 버튼 클릭
        mp->in.push_back(Point2f(x, y)); // 클릭된 점을 저장
        mp->inCount = mp->in.size();  // 클릭된 점 개수 업데이트
    }

    // 우클릭 시, 선택한 점들 초기화
    if (event == EVENT_RBUTTONDOWN) {
        mp->in.clear();
        mp->inCount = 0;  // 초기화
    }
}

int main() {
    // Timesquare.mp4 비디오와 contest.mp4 비디오 파일을 불러옴
    VideoCapture mainVideo("Timesquare.mp4");
    VideoCapture overlayVideo("contest.mp4");

    if (!mainVideo.isOpened() || !overlayVideo.isOpened()) {
        cerr << "비디오 파일을 열 수 없습니다." << endl;
        return -1;
    }

    // "background"와 "input" 윈도우 생성
    namedWindow("background", WINDOW_AUTOSIZE);
    namedWindow("input", WINDOW_AUTOSIZE);

    // 윈도우 위치 조정
    moveWindow("background", 0, 100);  // "background" 창을 왼쪽에 배치
    moveWindow("input", 900, 100);       // "input" 창을 오른쪽에 배치

    Mat mainFrame, overlayFrame;
    
    // MouseParams 설정
    MouseParams mp;

    // contest.mp4 영상의 width와 height 가져오기
    int overlayWidth = overlayVideo.get(CAP_PROP_FRAME_WIDTH);
    int overlayHeight = overlayVideo.get(CAP_PROP_FRAME_HEIGHT);

    // contest.mp4의 크기에 맞춰서 out의 좌표 설정
    mp.out.push_back(Point2f(0, 0));  // (0,0) 위치
    mp.out.push_back(Point2f(overlayWidth, 0)); // (overlayWidth, 0)
    mp.out.push_back(Point2f(overlayWidth, overlayHeight)); // (overlayWidth, overlayHeight)
    mp.out.push_back(Point2f(0, overlayHeight)); // (0, overlayHeight)

    // 첫 번째 비디오 프레임을 읽어 img에 저장
    if (!mainVideo.read(mainFrame)) {
        cerr << "첫 번째 비디오 프레임을 읽을 수 없습니다." << endl;
        return -1;
    }

    mp.img = mainFrame;

    // "background" 창에서 마우스 이벤트 처리
    setMouseCallback("background", onMouse, (void*)&mp);

    // 비디오를 계속해서 읽고 처리
    while (true) {
        // 비디오 프레임을 읽을 때마다 체크
        if (!mainVideo.read(mainFrame) || !overlayVideo.read(overlayFrame)) {
            cerr << "비디오 프레임을 읽을 수 없습니다." << endl;
            break;
        }

        // 비디오 프레임이 비어있는지 체크
        if (mainFrame.empty() || overlayFrame.empty()) {
            cerr << "비디오 프레임이 비어있습니다." << endl;
            break;
        }

        // "Timesquare.mp4" 비디오를 "background" 창에 표시
        Mat result = mainFrame.clone();

        // 클릭된 점들에 빨간 점 표시
        for (size_t i = 0; i < mp.in.size(); i++) {
            circle(result, mp.in[i], 3, Scalar(0, 0, 255), 5); // 점 표시
        }

        // 사용자가 4개의 점을 클릭한 경우 퍼스펙티브 변환을 적용
        if (mp.in.size() == 4) {
            // 퍼스펙티브 변환 행렬 계산
            Mat homo_mat = getPerspectiveTransform(mp.out, mp.in);  // 변환 방향이 반대로 되어야 합니다.
            Mat transformedOverlay;

            // "contest.mp4" 비디오를 변환하여 "Timesquare.mp4" 비디오와 같은 크기로 맞추기
            warpPerspective(overlayFrame, transformedOverlay, homo_mat, mainFrame.size());

            // 변환된 비디오를 선택된 영역에 덮어씌움
            Mat mask = Mat::zeros(mainFrame.size(), CV_8UC1);

            // Point2f를 Point로 변환하여 이중 배열에 저장
            Point intPoints[1][4];  // 2D 배열 선언
            for (int i = 0; i < 4; i++) {
                intPoints[0][i] = Point(mp.in[i].x, mp.in[i].y);
            }

            // fillPoly로 다각형을 채움
            const Point* pts[1] = { intPoints[0] };  // 첫 번째 행에 있는 포인터를 전달
            int npt[] = {4};
            fillPoly(mask, pts, npt, 1, Scalar(255));

            // 변환된 비디오를 메인 비디오에 덮어씌움
            transformedOverlay.copyTo(result, mask);
        }

        // "background" 창에 결과 출력
        imshow("background", result);

        // "contest.mp4" 비디오를 "input" 창에 표시
        imshow("input", overlayFrame);

        // ESC 키를 누르면 종료
        if (waitKey(30) == 27) {
            break;
        }
    }

    // 비디오 파일을 해제하고 윈도우를 닫음
    mainVideo.release();
    overlayVideo.release();
    destroyAllWindows();

    return 0;
}
