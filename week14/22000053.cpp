#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

// PSNR 계산 함수
double calculatePSNR(const Mat& original, const Mat& reconstructed) {
    Mat diff;
    absdiff(original, reconstructed, diff); // 차이 계산
    diff.convertTo(diff, CV_32F); // float로 변환
    diff = diff.mul(diff); // 제곱

    Scalar s = sum(diff); // 모든 픽셀의 합
    double mse = s[0] / (double)(original.total()); // 평균 제곱 오차

    if (mse == 0) return INFINITY; // 완벽한 복원일 경우 무한대

    // 8x8 블록의 최대값 찾기
    double max_I;
    minMaxLoc(original, nullptr, &max_I); // 최대값을 max_I에 저장

    double psnr = 20.0 * log10(max_I) - 10.0 * log10(mse); // PSNR 계산
    return psnr;
}


void applyDCTAndQuantization(const Mat& src, const Mat& quantMatrix, Mat& dst) {
    dst = Mat::zeros(src.size(), CV_8UC1); // 출력 이미지 초기화
    for (int row = 0; row < src.rows; row += 8) {
        for (int col = 0; col < src.cols; col += 8) {
            Mat block = src(Rect(col, row, 8, 8)); // 8x8 블록 추출
            Mat blockFloat;
            block.convertTo(blockFloat, CV_32F); // float 형식으로 변환

            // DCT 변환
            Mat dctBlock;
            dct(blockFloat, dctBlock, 0);

            // 양자화 행렬을 32F로 변환
            Mat quantMatrixFloat;
            quantMatrix.convertTo(quantMatrixFloat, CV_32F);

            // 양자화: dctBlock과 quantMatrixFloat을 나눔
            Mat quantized;
            divide(dctBlock, quantMatrixFloat, quantized, 1, CV_32F); // 결과 타입을 명시적으로 CV_32F로 지정

            // 역 DCT
            Mat inverseDCT;
            dct(quantized, inverseDCT, 1);

            // 결과 블록을 8U로 변환하여 저장
            Mat resultBlock;
            inverseDCT.convertTo(resultBlock, CV_8U);
            resultBlock.copyTo(dst(Rect(col, row, 8, 8)));
        }
    }
}


int main(int argc, char* argv[]) {
    Mat image = imread("lena.png", IMREAD_COLOR);
    if (image.empty()) {
        cerr << "Image not found!" << endl;
        return -1;
    }

    Mat image_Ycbcr, Ycbcr_channels[3];
    cvtColor(image, image_Ycbcr, COLOR_BGR2YCrCb); // YCrCb로 변환
    split(image_Ycbcr, Ycbcr_channels); // 채널 분리
    Mat originalY = Ycbcr_channels[0]; // Y 채널 선택

    // 양자화 행렬 정의
    Mat quantization_mat1 = (Mat_<float>(8, 8) <<
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99);
    
    Mat quantization_mat2 = (Mat_<double>(8, 8) <<
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1
        );
    Mat quantization_mat3 = (Mat_<double>(8, 8) <<
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100,
        100, 100, 100, 100, 100, 100, 100, 100
        );

    // 결과 이미지 저장용
    Mat resultQM1, resultQM2, resultQM3;

    // 각 양자화 행렬로 처리
    applyDCTAndQuantization(originalY, quantization_mat1, resultQM1);
    applyDCTAndQuantization(originalY, quantization_mat2, resultQM2);
    applyDCTAndQuantization(originalY, quantization_mat3, resultQM3);

    // PSNR 계산
    cout << "QM1: psnr = " << calculatePSNR(originalY, resultQM1) << endl;
    cout << "QM2: psnr = " << calculatePSNR(originalY, resultQM2) << endl;
    cout << "QM3: psnr = " << calculatePSNR(originalY, resultQM3) << endl;

    // 결과 표시
    imshow("Original Y", originalY);
    imshow("QM1", resultQM1);
    imshow("QM2", resultQM2);
    imshow("QM3", resultQM3);

    waitKey(0);
    return 0;
}
