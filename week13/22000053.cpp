#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include<iostream>
using namespace cv;
using namespace std;
int main(){
    // open vidoe webcam
    VideoCapture cap = VideoCapture(0);
    int successes = 0;
    int numBoards = 70; // total num of corners
    int numCornersHor = 10; // num of horizon corners
    int numCornersVer = 7; // num of vertical corners
    int Rect_size = 20; // length of one side of the rectangle
    int numSquares = (numCornersHor - 1) * (numCornersVer - 1);
    Size board_sz = Size(numCornersHor, numCornersVer);
    // Container
    vector<vector<Point3f> > object_points;
    vector<vector<Point2f> > image_points;
    vector<Point2f> corners;
    vector<Point3f> obj;
    for (int i = 0; i < numCornersHor; i++){
        for (int j = 0; j < numCornersVer; j++){
            obj.push_back(Point3f(i*Rect_size, j*Rect_size, 0.0f));
        }
    }
    Mat img;
    Mat gray;
    cap >> img;
    cout << "Image size:"<<img.size() << endl;
    while (successes < numBoards)
    {
        cap >> img;
        cvtColor(img, gray, cv::COLOR_RGB2GRAY);
        if (img.empty()) break; // end of video stream
        if (waitKey(1) == 27) break; // stop capturing by pressing ESC
        // Finds the positions of internal corners of the chessboard.
        bool found = findChessboardCorners( gray, // Source chessboard view. It must be an 8-bit grayscale or color image.
            board_sz, // Number of inner corners per a chessboard row and column
            corners, // Output array of detected corners.
            CALIB_CB_ADAPTIVE_THRESH
            // Various operation flags that can be zero or a combination
            // Use adaptive thresholding to convert the image to black and white, rather than a fixed threshold level
        );
        if (found == 1){
            // Refines the corner locations.
            cornerSubPix(gray, // Input single-channel, 8-bit or float image.
                corners, // Initial coordinates of the input corners and refined coordinates provided for output.
                Size(11, 11), // Half of the side length of the search window
                Size(-1, -1), // Half of the size of the dead region in the middle of the search. zone over which the summation in the formula below is not done
                // The class defining termination criteria for iterative algorithms
                TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.1)
            );
            // Renders the detected chessboard corners.
            drawChessboardCorners( img, // Destination image. It must be an 8-bit color image.
                board_sz, // Number of inner corners per a chessboard row and column
                corners, // Array of detected corners, the output of findChessboardCorners.
                found // Parameter indicating whether the complete board was found or not
                // The return value of findChessboardCorners should be passed here.
            );
            image_points.push_back(corners);
            object_points.push_back(obj);
            printf("Snap stored!\n");
            successes++;
        }
        imshow("win1", img);
        imshow("win2", gray);
        waitKey(1000);
    }
    cout << "Complete!" << "\n";
    Mat intrinsic = Mat(3, 3, CV_32FC1);
    Mat distCoeffs;
    vector<Mat> rvecs;
    vector<Mat> tvecs;
    intrinsic.ptr<float>(0)[0] = 1;
    intrinsic.ptr<float>(1)[1] = 1;
    // Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern
    calibrateCamera( object_points, // The outer vector contains as many elements as the number of the pattern views.
        image_points, // A vector of vectors of the 2D image points.
        img.size(), // Size of the image
        intrinsic, // Intrinsic camera matrix
        distCoeffs, // Lens distortion coefficients. These coefficients will be explained in a future post.
        rvecs, // Rotation specified as a 3×1 vector.
        // The direction of the vector specifies the axis of rotation and the magnitude of the vector specifies the angle of rotation.
        tvecs // 3×1 Translation vector.
    );
    Mat imageUndistorted;
    // Print result
    cout << "==================Intrinsic Parameter=====================" << "\n";
    for (int i = 0; i < intrinsic.rows; i++){
        for (int j = 0; j < intrinsic.cols; j++) {
            cout << intrinsic.at<double>(i, j)<<"\t\t";
        }
        cout << endl;
    }
    cout << "====================================================" << "\n";
    cap.release();
    waitKey();
    return 0;
}