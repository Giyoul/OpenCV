#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    vector<string> fileNames;
    Mat queryImage;
    String imageName;
    cout << "Enter query image name : ";
    cin >> imageName;

    queryImage = imread(imageName, 1);
    if(queryImage.empty()){
        cout << "No file!" << endl;
        waitKey(0);
        return -1;
    }

    resize(queryImage, queryImage, Size(640, 480));

    Ptr<ORB> orb = ORB::create(1000);
    BFMatcher matcher(NORM_HAMMING);
    vector<KeyPoint> queryKeypoints;
    Mat queryDescriptors;
    orb->detectAndCompute(queryImage, noArray(), queryKeypoints, queryDescriptors);

    double bestMatchRatio = 0.0;
    String bestMatchFile;
    Mat bestMatchImage;
    vector<DMatch> bestMatches;
    vector<DMatch> bestGoodMatches;
    vector<KeyPoint> bestKeypoints;

    vector<Mat> images;
    cout << "Sample image Load Size : 10\n";
    glob("Handong*_1.*", fileNames);  // Handong1_1부터 Handong10_1까지의 파일 패턴
    int n_count = 0;
    
    for (const auto& file : fileNames) {
        n_count++;
        Mat dbImage = imread(file, 1);
        if (dbImage.empty()) {
            cout << "Failed to load image: " << file << endl;
            continue;
        }
        resize(dbImage, dbImage, Size(640, 480));
        
        vector<KeyPoint> dbKeypoints;
        Mat dbDescriptors;
        orb->detectAndCompute(dbImage, noArray(), dbKeypoints, dbDescriptors);

        vector<vector<DMatch>> knnMatches;
        matcher.knnMatch(queryDescriptors, dbDescriptors, knnMatches, 2);

        vector<DMatch> goodMatches;
        float nndr = 0.6f;  // NNDR 비율
        for (const auto& m : knnMatches) {
            if (m.size() == 2 && m[0].distance <= nndr * m[1].distance) {
                goodMatches.push_back(m[0]);
            }
        }

        // 각 이미지의 매칭 수 출력
        cout << "Image number " << n_count << " Matching: " << goodMatches.size() << endl;

        double matchRatio = static_cast<double>(goodMatches.size()) / queryKeypoints.size();
        if (matchRatio > bestMatchRatio) {
            bestMatchRatio = matchRatio;
            bestMatchFile = file;
            bestMatchImage = dbImage;  // 베스트 매칭 이미지를 저장
            bestGoodMatches = goodMatches; // 좋은 매칭 결과 저장
            bestKeypoints = dbKeypoints;
        }        
    }

    imshow("Query", queryImage);
    if (!bestMatchImage.empty() && bestGoodMatches.size() >= 4 && !bestKeypoints.empty() && !queryKeypoints.empty()) {
        Mat imgMatches; // 매칭 결과를 그릴 이미지 변수
        drawMatches(bestMatchImage, bestKeypoints, queryImage, queryKeypoints, bestGoodMatches, imgMatches, Scalar::all(-1), Scalar(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        imshow("Best_matching", imgMatches); // 매칭 결과를 보여줌
    } else {
        cout << "No good match found!" << endl;
    }

    waitKey(0);
    destroyAllWindows();
    return 0;
}
