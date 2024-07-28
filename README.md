# Explanation of the Project 

---

**Header Files and Namespaces**

- `#include <opencv2/opencv.hpp>`: Includes core OpenCV functionalities.
- `#include <iostream>`: For input and output streams.
- `#include <windows.h>`: For Windows API functions.
- `using namespace cv;` and `using namespace std;`: To avoid prefixing OpenCV and standard library functions with `cv::` and `std::`.

**Helper Function: File Existence Check**

- `fileExists`: Checks if a file exists at the given path using Windows API.
- `GetFileAttributesW`: Retrieves file attributes. Returns `INVALID_FILE_ATTRIBUTES` if the file does not exist.

**Serial Sobel Filter Function**

- `applySobelFilter`: Applies the Sobel filter serially to detect edges.
- Uses `cv::Sobel` to compute gradients in the x and y directions.
- Uses `cv::magnitude` to combine these gradients into an edge magnitude image.

**Optimized Parallel Sobel Filter Function**

- `applySobelFilterParallelOptimized`: Applies the Sobel filter in parallel using OpenMP for optimization.
- Manually computes gradients in x and y directions using Sobel kernel.
- Combines these gradients into an edge magnitude image using `sqrt(gx * gx + gy * gy)`.

**Main Function**

- Defines the image path and checks if the file exists using `fileExists`.
- Loads the image in grayscale using `cv::imread`.
- Applies both the serial and parallel Sobel filters.
- Compares the results using `cv::absdiff` and `cv::norm` to check for differences.
- Displays the original and edge-detected images using `imshow`.
- Waits for a key press using `waitKey`.

**Output and Visualization**

- Displays messages indicating the existence of the file and the correctness of the parallel implementation.
- Shows windows with the original image and the results of the serial and parallel Sobel filters.
