#include <opencv.hpp>
using namespace cv;
using namespace std;

std::string windowName = "Mean Blur";
void onChange(int, void*);

int main()
{
    std::string image_path = "../miku.jpg";
    Mat orgImg = imread(image_path);

    Mat outImg;

    int kernelSize = 1;
    namedWindow(windowName);
    createTrackbar("Size", windowName, &kernelSize, 16, onChange, &orgImg);
    onChange(kernelSize,&orgImg);

    waitKey(0);

    return 0;
}

void onChange(int curSize, void* orgMat)
{
    int i = 2 * curSize + 1;
    Mat outMat;
    blur(*(Mat*)orgMat, outMat, Size(i,i)); //¾ùÖµÂË²¨
    imshow(windowName+" out", outMat);
    //imwrite(windowName + " out"+to_string(i)+".jpg", outMat);
}