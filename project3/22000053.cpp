#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    // 각 이미지 로드 (이름 고정)
    Mat img1 = imread("pano1.JPG");
    Mat img2 = imread("pano2.JPG");
    Mat img3 = imread("pano3.JPG");
    Mat img4 = imread("pano4.JPG");

    if (img1.empty() || img2.empty() || img3.empty() || img4.empty()) {
        cout << "Error loading images!" << endl;
        return -1;
    }

    // 해상도 조정
    resize(img1, img1, Size(640, 480));
    resize(img2, img2, Size(640, 480));
    resize(img3, img3, Size(640, 480));
    resize(img4, img4, Size(640, 480));

    // ORB와 매칭기 생성
    Ptr<ORB> orb = ORB::create(1000);
    BFMatcher matcher(NORM_HAMMING);

    // 특징점, 디스크립터를 위한 변수들
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    // Homography 기반 워핑 결과 저장용 캔버스
    Mat canvas = img1.clone();

    // 이미지 연결에 사용할 순차적인 쌍 리스트
    vector<Mat> imageList = {img2, img3, img4};

    for (const auto& nextImg : imageList) {
        // 현재 기준 이미지와 다음 이미지를 비교
        orb->detectAndCompute(canvas, noArray(), keypoints1, descriptors1);
        orb->detectAndCompute(nextImg, noArray(), keypoints2, descriptors2);

        // 매칭 수행 (KNN 매칭)
        vector<vector<DMatch>> knnMatches;
        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

        // 좋은 매칭 필터링
        vector<DMatch> goodMatches;
        float nndr = 0.6f;  // NNDR 비율
        for (int i = 0; i < knnMatches.size(); i++) {
            if (knnMatches.at(i).size() == 2 && knnMatches.at(i).at(0).distance < nndr * knnMatches.at(i).at(1).distance) {
                goodMatches.push_back(knnMatches[i][0]);
            }
        }

        if (goodMatches.size() < 4) {
            cout << "Not enough good matches to compute Homography!" << endl;
            continue;
        }

        // 매칭된 특징점 추출
        // 이게 들어가야 되는 이유가, 우리가 지금 keyPoints의 generic을 KeyPoint로 설정해 놨는데, findHomography는 Point2f로 설정되어있어서 바꿔줘야 함.
        vector<Point2f> points1, points2;
        for (int i = 0; i < goodMatches.size(); i++) {
            points1.push_back(keypoints1[goodMatches[i].queryIdx].pt);  // KeyPoint에서 query는 첫 번쨰 이미지를 뜻함
            points2.push_back(keypoints2[goodMatches[i].trainIdx].pt);  // KeyPoint에서 train은 두 번쨰 이미지를 뜻함
        }

        // Homography 계산
        Mat H = findHomography(points2, points1, RANSAC);    // dist가 point1 이 되어야 함.

        // 현재 이미지 기준으로 다음 이미지를 워핑
        Mat warpedImage;
        warpPerspective(nextImg, warpedImage, H, Size(canvas.cols + nextImg.cols, canvas.rows + nextImg.rows));

        // 캔버스를 확장하여 워핑된 이미지를 병합
        Mat tempCanvas;
        tempCanvas = Mat::zeros(Size(canvas.cols * 2, canvas.rows * 2), 16);

        imshow("wrapimage", warpedImage);
        waitKey(1000);
        destroyAllWindows();
        
        warpedImage.copyTo(tempCanvas(Rect(0, 0, warpedImage.cols, warpedImage.rows)));  // warpedImage 복사

        canvas.copyTo(tempCanvas(Rect(0, 0, canvas.cols, canvas.rows)));  // canvas의 이미지를 tempCanvas에 덮어 씌우기

        imshow("tempCanvas", tempCanvas);
        waitKey(1000);
        destroyAllWindows();

        canvas = tempCanvas;    // tempCanvas를 canvas로 대체

        imshow("Canvas", canvas);
        waitKey(1000);
        destroyAllWindows();
        // Rect roi(0, 0, canvas.cols * 0.6, canvas.rows * 0.6);
        // canvas = canvas(roi);
    }

    Rect roi(0, 0, 1920, 720);
    Mat cropped = canvas(roi);

    imshow("Panorama", cropped);
    waitKey(0);

    return 0;
}
