#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem; // Ensure the correct namespace for filesystem

// Function to apply the Sobel filter serially
void sobelFilterSerial(const Mat& input, Mat& output) {
    CV_Assert(input.channels() == 1);

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        {0,  0,  0},
        {1,  2,  1}
    };

    output = Mat::zeros(input.size(), CV_8U);

    for (int y = 1; y < input.rows - 1; y++) {
        for (int x = 1; x < input.cols - 1; x++) {
            int sumX = 0;
            int sumY = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    sumX += Gx[i + 1][j + 1] * input.at<uchar>(y + i, x + j);
                    sumY += Gy[i + 1][j + 1] * input.at<uchar>(y + i, x + j);
                }
            }

            int magnitude = static_cast<int>(sqrt(sumX * sumX + sumY * sumY));
            magnitude = magnitude > 255 ? 255 : magnitude;
            magnitude = magnitude < 0 ? 0 : magnitude;

            output.at<uchar>(y, x) = static_cast<uchar>(magnitude);
        }
    }
}

// Function to apply the Sobel filter in parallel using OpenMP
void sobelFilterParallel(const Mat& input, Mat& output) {
    CV_Assert(input.channels() == 1);

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        {0,  0,  0},
        {1,  2,  1}
    };

    output = Mat::zeros(input.size(), CV_8U);

#pragma omp parallel for
    for (int y = 1; y < input.rows - 1; y++) {
        for (int x = 1; x < input.cols - 1; x++) {
            int sumX = 0;
            int sumY = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    sumX += Gx[i + 1][j + 1] * input.at<uchar>(y + i, x + j);
                    sumY += Gy[i + 1][j + 1] * input.at<uchar>(y + i, x + j);
                }
            }

            int magnitude = static_cast<int>(sqrt(sumX * sumX + sumY * sumY));
            magnitude = magnitude > 255 ? 255 : magnitude;
            magnitude = magnitude < 0 ? 0 : magnitude;

            output.at<uchar>(y, x) = static_cast<uchar>(magnitude);
        }
    }
}

int main() {
    cout << "Starting the Sobel edge detection program." << endl;
    cout << "Current working directory: " << fs::current_path() << endl;

    // Path to the image file. Ensure the image is placed in the specified directory.
    string imagePath = "C:/Users/panka/Downloads/111.jpg";
    cout << "Attempting to load image from: " << imagePath << endl;

    // Load the image
    Mat image = imread(imagePath, IMREAD_COLOR);
    if (!image.data) {
        cerr << "Could not open or find the image at " << imagePath << endl;
        cerr << "Please check if the file exists and the path is correct." << endl;
        return -1;
    }
    cout << "Image loaded successfully." << endl;

    // Convert the image to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Apply Sobel filters and display results
    Mat edgesSerial, edgesParallel;
    double start, end;

    // Serial implementation
    start = omp_get_wtime();
    sobelFilterSerial(grayImage, edgesSerial);
    end = omp_get_wtime();
    cout << "Serial Sobel filter applied in " << (end - start) << " seconds." << endl;

    // Parallel implementation
    start = omp_get_wtime();
    sobelFilterParallel(grayImage, edgesParallel);
    end = omp_get_wtime();
    cout << "Parallel Sobel filter applied in " << (end - start) << " seconds." << endl;

    // Display the results
    namedWindow("Original Image", WINDOW_AUTOSIZE);
    imshow("Original Image", image);

    namedWindow("Sobel Edge Detection - Serial", WINDOW_AUTOSIZE);
    imshow("Sobel Edge Detection - Serial", edgesSerial);

    namedWindow("Sobel Edge Detection - Parallel", WINDOW_AUTOSIZE);
    imshow("Sobel Edge Detection - Parallel", edgesParallel);

    waitKey(0);

    cout << "Program finished successfully." << endl;

    return 0;
}
