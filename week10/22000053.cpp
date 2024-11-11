#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    CascadeClassifier face_classifier;
    if (!face_classifier.load("haarcascade_frontalface_alt.xml")) {
        cerr << "Error loading face cascade\n";
        return -1;
    }

    VideoCapture cap("Faces.mp4");
    if (!cap.isOpened()) {
        cerr << "Could not open video file\n";
        return -1;
    }

    Mat frame, grayframe;
    vector<Rect> faces;
    Rect trackedFace;
    bool isTracking = false;
    bool hasDetected = false;
    string currentMode = "";
    bool showMessage = false;
    int messageDuration = 2000;
    int64 messageStartTime = 0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        if (!currentMode.empty()) {
            int minSize, maxSize;
            if (currentMode == "n") { minSize = 76; maxSize = 90; }
            else if (currentMode == "m") { minSize = 54; maxSize = 55; }
            else if (currentMode == "f") { minSize = 30; maxSize = 39; }
            else { minSize = maxSize = 0; }

            if (minSize > 0 && maxSize > 0) {
                cvtColor(frame, grayframe, COLOR_BGR2GRAY);
                face_classifier.detectMultiScale(grayframe, faces, 1.1, 3, 0, Size(minSize, minSize), Size(maxSize, maxSize));

                if (!faces.empty()) {
                    hasDetected = true;
                    trackedFace = faces[0];
                    if (!isTracking) {
                        rectangle(frame, trackedFace, Scalar(0, 255, 0), 2);
                    }
                }
            }
        }

        if (showMessage) {
            int64 elapsedTime = (getTickCount() - messageStartTime) * 1000 / getTickFrequency();
            if (elapsedTime < messageDuration) {
                putText(frame, "Detect before tracking", Point(10, 20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            } else {
                showMessage = false;
            }
        }

        imshow("Faces", frame);
        char key = (char)waitKey(30);

        if (key == 'n' || key == 'm' || key == 'f' || key == 'F' || key == 'N' || key == 'M') {
            if (currentMode != "t" && currentMode != "T") {
                currentMode = string(1, tolower(key));
            }
        }
        else if (key == 't' || key == 'T') {
            if (hasDetected) {
                isTracking = !isTracking;
                if (!isTracking) {
                    destroyWindow("tracking");
                }
            } else {
                showMessage = true;
                messageStartTime = getTickCount();
            }
        }
        else if (key == 'r' || key == 'R') {
            hasDetected = false;
            isTracking = false;
            currentMode = "";
            destroyWindow("tracking");
        }

        if (isTracking && hasDetected) {
            Mat trackingWindow(frame.size(), frame.type(), Scalar(0, 0, 0));
            trackingWindow.setTo(Scalar(255, 0, 0));

            int margin = 20;
            Rect expandedFaceRect = trackedFace;
            expandedFaceRect.x = max(trackedFace.x - margin, 0);
            expandedFaceRect.y = max(trackedFace.y - margin, 0);
            expandedFaceRect.width = min(trackedFace.width + 2 * margin, frame.cols - expandedFaceRect.x);
            expandedFaceRect.height = min(trackedFace.height + 2 * margin, frame.rows - expandedFaceRect.y);

            Mat faceROI = frame(expandedFaceRect);
            faceROI.copyTo(trackingWindow(expandedFaceRect));

            imshow("tracking", trackingWindow);
        }

        if (key == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
