#include <opencv.hpp>
//#include <cmath>
using namespace cv;
using namespace std;


int main()
{
    std::string imagePath = "../miku.jpg";
    Mat orgImg = imread(imagePath);
    Mat grayImg;
    cvtColor(orgImg, grayImg, COLOR_BGR2GRAY);  //转成灰度图
    imshow("org", grayImg);//输出原始灰度图
    //imwrite("grayImg.jpg", grayImg);

    Mat grayLayer[8];
    for (int i = 0; i < 8; i++) grayLayer[i] = Mat::zeros(grayImg.size(), CV_8UC1);  //初始化8层图的内存空间

    for (int r = 0; r < grayImg.rows; r++)
        for (int c = 0; c < grayImg.cols; c++) //对原图遍历
            for (int i = 0; i < 8; i++) //处理一个点的每个bit位
                grayLayer[i].at<uint8_t>(r, c) = -((grayImg.at<uint8_t>(r, c) &1<<i)>>i);
                //用按位与、移位，得到一位的bit位，如果是1，则直接赋值-1（即8b11111111）
    
    for (int i = 0; i < 8; i++)
    {
        string imgName("grayLayer");
        imshow(imgName + to_string(i), grayLayer[i]);//输出8张图
        //imwrite(imgName+to_string(i) + ".jpg", grayLayer[i]);
    }

    waitKey(0);
    return 0;
}