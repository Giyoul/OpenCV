#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

double calculatePSNR(const Mat& original, const Mat& reconstructed) {
    Mat diff;
    absdiff(original, reconstructed, diff); // 차이 계산
    diff.convertTo(diff, CV_32F);           // float로 변환

    // 차이값을 직접 제곱
    for (int i = 0; i < diff.rows; i++) {
        for (int j = 0; j < diff.cols; j++) {
            diff.at<float>(i, j) = diff.at<float>(i, j) * diff.at<float>(i, j); // 제곱
        }
    }

    // 모든 요소의 합 직접 계산
    double totalSum = 0.0;
    for (int i = 0; i < diff.rows; i++) {
        for (int j = 0; j < diff.cols; j++) {
            totalSum += diff.at<float>(i, j);
        }
    }

    double mse = totalSum / (double)(original.total()); // 평균 제곱 오차

    if (mse == 0) return INFINITY; // 완벽한 복원일 경우 무한대

    double max_I = 255;

    // cmath도 쓰지 말라고 했으니 log10(max_I) 직접 계산하기 위해서는 테일러 급수 써야함.
    double x = max_I;
    double ln_x = 0.0;
    double term = (x - 1) / (x + 1);
    double term_squared = term * term;
    for (int i = 1; i < 100; i += 2) {
        ln_x += (1.0 / i) * term; // 테일러 급수
        term *= term_squared;     // term^(i+2)
    }
    ln_x *= 2;
    double log10_max_I = ln_x / 2.302585; // ln(10) ≈ 2.302585 근사치를 고정값으로 계산해야 함.

    // 동일하게 테일러 급수 사용해서 log10(mse) 직접 계산
    x = mse;
    ln_x = 0.0;
    term = (x - 1) / (x + 1);
    term_squared = term * term;
    for (int i = 1; i < 100; i += 2) {
        ln_x += (1.0 / i) * term;
        term *= term_squared;
    }
    ln_x *= 2;
    double log10_mse = ln_x / 2.302585;

    double psnr = 20.0 * log10_max_I - 10.0 * log10_mse; // PSNR 계산
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

            // 양자화: 직접 나눗셈 연산
            Mat quantized = dctBlock.clone();
            for (int i = 0; i < quantized.rows; i++) {
                for (int j = 0; j < quantized.cols; j++) {
                    quantized.at<float>(i, j) = dctBlock.at<float>(i, j) / quantMatrixFloat.at<float>(i, j);
                }
            }

            // 양자화된 값을 반올림
            for (int i = 0; i < quantized.rows; i++) {
                for (int j = 0; j < quantized.cols; j++) {
                    quantized.at<float>(i, j) = round(quantized.at<float>(i, j));
                }
            }

            // 역 양자화
            Mat dequantized = quantized.clone();
            for (int i = 0; i < dequantized.rows; i++) {
                for (int j = 0; j < dequantized.cols; j++) {
                    dequantized.at<float>(i, j) *= quantMatrixFloat.at<float>(i, j);
                }
            }

            // 역 DCT
            Mat inverseDCT;
            dct(dequantized, inverseDCT, DCT_INVERSE);

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
    
    Mat quantization_mat2 = (Mat_<float>(8, 8) <<
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1
        );
    Mat quantization_mat3 = (Mat_<float>(8, 8) <<
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
