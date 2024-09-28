#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void white_balacing(Mat img) { Mat bgr_channels[3]; split(img, bgr_channels);
    double avg;
    int sum,temp,i, j, c;
    for (c = 0; c < img.channels(); c++) {
        sum = 0;
        avg = 0.0f;
        for (i = 0; i < img.rows; i++) {
            for (j = 0; j < img.cols; j++) {
                sum += bgr_channels[c].at<uchar>(i, j);
            }
        }
        avg = sum / (img.rows * img.cols);
        for (i = 0; i < img.rows; i++) {
            for (j = 0; j < img.cols; j++) {
                temp = (128 / avg) * bgr_channels[c].at<uchar>(i, j);
                if (temp>255) bgr_channels[c].at<uchar>(i, j) = 255;
                else bgr_channels[c].at<uchar>(i, j) = temp;
            }
        }
    }
    merge(bgr_channels, 3, img);
}

int main() {
    Mat frame, out_frame;
    int fps, maxFrame;
    int delay, mode = 0;
    VideoCapture cap;

    if(cap.open("video.mp4") == 0) {
        cout << "no such file" << endl;
        waitKey(0);
    }

    fps = cap.get(CAP_PROP_FPS);
    delay = 1000 / fps;
    maxFrame = cap.get(CAP_PROP_FRAME_COUNT);

    while (1) {
        cap >> frame;
        if(frame.empty()){
            cout << "end of video" << endl;
            break;
        }

        out_frame = frame.clone();
        char c = (char)waitKey(1);
        if (c == 27) break; // ESC로 종료

        if (c == 110) mode = 1; // Negative transformation ('n')
        else if (c == 103) mode = 2; // Gamma transformation ('g')
        else if (c == 104) mode = 3; // Histogram equalization ('h')
        else if (c == 115) mode = 4; // Color slicing ('s')
        else if (c == 99) mode = 5; // Color conversion ('c')
        else if (c == 97) mode = 6; // Average filtering ('a')
        else if (c == 117) mode = 7; // Unsharp masking ('u')
        else if (c == 119) mode = 8; // White balancing ('w')
        else if (c == 114) mode = 0; // Reset ('r')

        if (mode == 0) {
            out_frame = frame.clone();
        }
        else if (mode == 1) { 
            // Negative transformation
            Mat hsv;
            cvtColor(frame, hsv, COLOR_BGR2HSV); // BGR에서 HSV로 변환

            // HSV 채널을 반전
            for (int j = 0; j < hsv.rows; j++) {
                for (int i = 0; i < hsv.cols; i++) {
                    Vec3b &pixel = hsv.at<Vec3b>(j, i);
                    pixel[2] = 255 - pixel[2]; // Value 채널 반전 (0~255 범위)
                }
            }

            cvtColor(hsv, out_frame, COLOR_HSV2BGR); // 다시 BGR로 변환
        }
        else if (mode == 2) {
            // Gamma transformation (gamma = 2.5)
            Mat gamma_img = frame.clone();
            float gamma = 2.5;  // 감마 값은 2.5
            unsigned char pix[256];

            // 감마 변환 테이블을 미리 계산
            for (int i = 0; i < 256; i++) {
                pix[i] = saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
            }

            // 각 BGR 채널에 감마 변환 적용
            for (int j = 0; j < frame.rows; j++) {
                for (int i = 0; i < frame.cols; i++) {
                    // 각 픽셀에 대해 B, G, R 채널 값을 개별적으로 변환
                    // ppt 2_2 pointer access 방법 사용, pixel access에서 사용한 Vec3b 사용.
                    Vec3b &color = gamma_img.at<Vec3b>(j, i);
                    color[0] = pix[color[0]];  // Blue 채널
                    color[1] = pix[color[1]];  // Green 채널
                    color[2] = pix[color[2]];  // Red 채널
                }
            }

            out_frame = gamma_img.clone();  // 결과 이미지를 out_frame에 저장
        }

        else if (mode == 3) {
            // Histogram equalization
            Mat hsv;
            cvtColor(frame, hsv, COLOR_BGR2HSV);
            // ppt 4_3
            vector<Mat> hsv_channels;
            split(hsv, hsv_channels);
            equalizeHist(hsv_channels[2], hsv_channels[2]); // Equalize only the V channel
            merge(hsv_channels, hsv);
            cvtColor(hsv, out_frame, COLOR_HSV2BGR);
        }
        else if (mode == 4) {
            Mat HSV, mask_out;
            vector<Mat> mo(3);
            cvtColor(frame, HSV, COLOR_BGR2HSV); // BGR -> HSV 변환

            // 채널 분리
            split(HSV, mo);

            // 새로운 채널을 생성하여 hue 값이 9 ~ 23인 경우에만 값을 유지하고 나머지는 흑백으로 설정
            for (int j = 0; j < HSV.rows; j++) {
                uchar* h = mo[0].ptr<uchar>(j); // hue 채널 접근
                uchar* s = mo[1].ptr<uchar>(j); // saturation 채널 접근
                uchar* v = mo[2].ptr<uchar>(j); // value 채널 접근

                for (int i = 0; i < HSV.cols; i++) {
                    // Hue 값이 9와 23 사이일 경우 해당 픽셀을 유지
                    if (h[i] >= 9 && h[i] <= 23) {
                        // Hue, Saturation, Value 값을 그대로 유지
                        s[i] = s[i]; // Saturation 값 유지
                        v[i] = v[i]; // Value 값 유지
                    } else {
                        // 나머지 색상은 흑백으로 변환
                        s[i] = 0; // Saturation 값을 0으로 설정하여 흑백으로 만들기
                    }
                }
            }

            // 변환된 채널을 다시 합치기
            merge(mo, mask_out);
            cvtColor(mask_out, out_frame, COLOR_HSV2BGR);
        }
        else if (mode == 5) {
            // Color conversion (Hue + 50)
            Mat hsv;
            cvtColor(frame, hsv, COLOR_BGR2HSV);
            for (int j = 0; j < hsv.rows; j++) {
                for (int i = 0; i < hsv.cols; i++) {
                    int hue = hsv.at<Vec3b>(j, i)[0];
                    if (hue > 129) hue -= 129;
                    else hue += 50;
                    hsv.at<Vec3b>(j, i)[0] = hue;
                }
            }
            cvtColor(hsv, out_frame, COLOR_HSV2BGR);
        }
        else if (mode == 6) {
            // Average filtering (blur with 9x9)
            blur(frame, out_frame, Size(9, 9));
        }
        else if (mode == 7) {
            // Sharpening by unsharp masking (blur with 9x9)
            Mat blurred;
            blur(frame, blurred, Size(9, 9));
            out_frame = frame.clone() + (frame.clone() - blurred);
        }
        else if (mode == 8) {
            // White balancing using gray world assumption
            out_frame = frame.clone();
            white_balacing(out_frame);
        }

        imshow("video", out_frame);
        waitKey(delay);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

