#include <opencv2/opencv.hpp>
#include <iostream>
#include <windows.h>

using namespace cv;
using namespace std;

// Function to check if the file exists using Windows API
bool fileExists(const wstring& filename) {
    DWORD fileAttr = GetFileAttributesW(filename.c_str());
    return (fileAttr != INVALID_FILE_ATTRIBUTES && !(fileAttr & FILE_ATTRIBUTE_DIRECTORY));
}

// Serial Sobel filter function
void applySobelFilter(const cv::Mat& src, cv::Mat& dst) {
    cv::Mat grad_x, grad_y;
    cv::Sobel(src, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(src, grad_y, CV_64F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, dst);
}

// Optimized parallel Sobel filter function
void applySobelFilterParallelOptimized(const cv::Mat& src, cv::Mat& dst) {
    dst = cv::Mat::zeros(src.size(), CV_64F);
#pragma omp parallel for schedule(dynamic)
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            double gx = -src.at<uchar>(i - 1, j - 1) - 2 * src.at<uchar>(i, j - 1) - src.at<uchar>(i + 1, j - 1) +
                src.at<uchar>(i - 1, j + 1) + 2 * src.at<uchar>(i, j + 1) + src.at<uchar>(i + 1, j + 1);
            double gy = -src.at<uchar>(i - 1, j - 1) - 2 * src.at<uchar>(i - 1, j) - src.at<uchar>(i - 1, j + 1) +
                src.at<uchar>(i + 1, j - 1) + 2 * src.at<uchar>(i + 1, j) + src.at<uchar>(i + 1, j + 1);
            dst.at<double>(i, j) = sqrt(gx * gx + gy * gy);
        }
    }
}

int main() {
    // Path to the image
    wstring imagePath = L"C:\\Users\\panka\\Downloads\\im.jpg";

    // Print the image path to verify it's correct
    wcout << L"Image path: " << imagePath << endl;

    // Check if the file exists
    if (!fileExists(imagePath)) {
        wcerr << L"File does not exist at the specified path: " << imagePath << endl;
        return -1;
    }
    else {
        wcout << L"File exists. Proceeding to load the image." << endl;
    }

    // Load the image in grayscale using cv::imread with a wide string path
    cv::Mat image = cv::imread(cv::String(imagePath.begin(), imagePath.end()), cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        cerr << "Could not open or find the image!\n";
        return -1;
    }

    // Apply Sobel filters
    cv::Mat edgesSerial, edgesParallel;
    applySobelFilter(image, edgesSerial);
    applySobelFilterParallelOptimized(image, edgesParallel);

    // Compare the two images
    cv::Mat diff;
    cv::absdiff(edgesSerial, edgesParallel, diff);
    double maxDiff = cv::norm(diff, cv::NORM_INF);

    if (maxDiff < 1e-6) {
        cout << "The parallel implementation is correct!\n";
    }
    else {
        cout << "There are differences between the serial and parallel implementations.\n";
    }

    // Display the original and edge-detected images
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);

    namedWindow("Serial Sobel", WINDOW_AUTOSIZE);
    imshow("Serial Sobel", edgesSerial);

    namedWindow("Parallel Sobel", WINDOW_AUTOSIZE);
    imshow("Parallel Sobel", edgesParallel);

    waitKey(0);

    return 0;
}
