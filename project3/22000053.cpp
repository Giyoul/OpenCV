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
    // int maxXarr[3] = {880, 1280, 1800};
    int count = 0;

    for (const auto& nextImg : imageList) {

        cout << "count << " << count << ", canvas.cols << " << canvas.cols << endl; 

        // 현재 기준 이미지와 다음 이미지를 비교
        orb->detectAndCompute(canvas, noArray(), keypoints1, descriptors1);
        orb->detectAndCompute(nextImg, noArray(), keypoints2, descriptors2);

        // 매칭 수행 (KNN 매칭)
        vector<vector<DMatch>> knnMatches;
        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

        // 좋은 매칭 필터링
        vector<DMatch> goodMatches;
        float nndr = 0.8f;  // NNDR 비율
        // for (int i = 0; i < knnMatches.size(); i++) {
        //     if (knnMatches.at(i).size() == 2 && knnMatches.at(i).at(0).distance < nndr * knnMatches.at(i).at(1).distance) {
        //         goodMatches.push_back(knnMatches[i][0]);
        //     }
        // }

        int canvasMiddle = canvas.cols / 2;

        for (int i = 0; i < knnMatches.size(); i++) {
            if (knnMatches.at(i).size() == 2 && knnMatches.at(i).at(0).distance < nndr * knnMatches.at(i).at(1).distance) {
                // 매칭된 특징점이 canvas의 중간을 기준으로 오른쪽에 있는지 확인
                if (keypoints1[knnMatches[i][0].queryIdx].pt.x > canvasMiddle) {
                    goodMatches.push_back(knnMatches[i][0]);
                }
            }
        }

        if (goodMatches.size() < 4) {
            cout << "Not enough good matches to compute Homography!" << endl;
            continue;
        }

        // 매칭된 특징점 추출
        vector<Point2f> points1, points2;
        for (int i = 0; i < goodMatches.size(); i++) {
            points1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
            points2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
        }

        // Homography 계산
        Mat H = findHomography(points2, points1, LMEDS);

        // 현재 이미지 기준으로 다음 이미지를 워핑
        Mat warpedImage;
        warpPerspective(nextImg, warpedImage, H, Size(canvas.cols + nextImg.cols, canvas.rows + nextImg.rows));

        // canvas의 네 모서리 좌표 정의
        vector<Point2f> canvasCorners = { 
            Point2f(0, 0), 
            Point2f(canvas.cols, 0), 
            Point2f(canvas.cols, canvas.rows), 
            Point2f(0, canvas.rows) 
        };

        cout << H << endl;

        // Homography를 통해 canvasCorners를 warpedImage의 좌표계로 변환
        vector<Point2f> transformedCorners;
        for (const Point2f& pt : canvasCorners) {
            // 동차 좌표 (x, y, 1)로 변환
            float x = pt.x;
            float y = pt.y;
            float w = 1.0;

            // Homography 변환 수식
            // perspectiveTransform 쓰면 이런거 안해도 될텐데 ppt에 perspectiveTransform가 없으니까 ppt에 나와있는 행렬식을 손으로 계산하면 아래 식임
            float x_prime = (H.at<double>(0, 0) * x + H.at<double>(0, 1) * y + H.at<double>(0, 2)) /
                            (H.at<double>(2, 0) * x + H.at<double>(2, 1) * y + H.at<double>(2, 2));
            float y_prime = (H.at<double>(1, 0) * x + H.at<double>(1, 1) * y + H.at<double>(1, 2)) /
                            (H.at<double>(2, 0) * x + H.at<double>(2, 1) * y + H.at<double>(2, 2));

            transformedCorners.push_back(Point2f(x_prime, y_prime));
        }

        // 변환된 좌표의 범위 계산 (warp된 영역의 유효 폭)
        // float maxX = canvas.cols + H.at<double>(0, 2); // 우측 경계
        // float maxX = transformedCorners[1].x;

        float maxX = max((float)canvas.cols, transformedCorners[1].x); // 하단 경계
        // if(count == 1){
        //     maxX = max((float)canvas.cols, transformedCorners[1].x - canvas.cols -120); // 하단 경계
        // } else if (count == 2){
        //     maxX = maxX + 300;
        // }
        // // if(count == 1){
        // //     maxX = max((float)canvas.cols, transformedCorners[1].x - canvas.cols); // 하단 경계
        // // }
        float maxY = max((float)canvas.rows, transformedCorners[1].y); // 하단 경계

        cout << "count << " << count << ", maxX << " << maxX << endl;


        // 필요한 크기로 캔버스를 확장
        Mat tempCanvas = Mat::zeros(Size(canvas.cols * 6, canvas.rows * 6), 16);

        // canvas 복사
        canvas.copyTo(tempCanvas(Rect(0, 0, canvas.cols, canvas.rows)));

        // warpedImage와 canvas 병합
        warpedImage.copyTo(tempCanvas(Rect(0, 0, warpedImage.cols, warpedImage.rows))); // Homography로 이동한 좌표

        canvas = tempCanvas;    // tempCanvas를 canvas로 대체

        // 캔버스를 확장하여 워핑된 이미지를 병합
        // Rect roi(0, 0, maxXarr[count++], maxY);
        if(count != 2){
            Rect roi(0, 0, maxX, maxY);
            canvas = canvas(roi);
        }
    
        cout << "count << " << count << ", canvas.cols << " << canvas.cols << endl;

        count++;

        imshow("wrapimage", warpedImage);
        waitKey(1000);
        destroyAllWindows();

        imshow("tempCanvas", tempCanvas);
        waitKey(1000);
        destroyAllWindows();

        imshow("Canvas", canvas);
        waitKey(1000);
        destroyAllWindows();
    }

    Rect roi(0, 0, 1700, 720);
    canvas = canvas(roi);

    imshow("Panorama", canvas);
    waitKey(0);

    return 0;
}
