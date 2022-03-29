#include <opencv.hpp>
using namespace cv;
using namespace std;

std::string windowName = "MeanF VS MedianF";
void onChange(int, void*);

int main()
{
    std::string image_path = "../miku.jpg";
    Mat orgImg = imread(image_path);

    Mat outImg;

    int kernelSize = 1;
    namedWindow(windowName);
    createTrackbar("Size", windowName, &kernelSize, 32, onChange, &orgImg);
    onChange(kernelSize, &orgImg);


    waitKey(0);

    return 0;
}

void onChange(int curSize, void* orgMat)
{
    int i = 2 * curSize + 1;
    Mat outMatMean, outMatMedian;
    blur(*(Mat*)orgMat, outMatMean, Size(i, i)); //均值滤波
    medianBlur(*(Mat*)orgMat, outMatMedian, i);  //中值滤波

    imshow("MeanF out", outMatMean);
    imshow("MedianF out", outMatMedian);
    /*imwrite("MeanF out"+to_string(i)+".jpg", outMatMean);
    imwrite("MedianF out"+to_string(i)+".jpg", outMatMedian);*/
}