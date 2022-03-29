#include <opencv.hpp>
//#include <cmath>
using namespace cv;
using namespace std;


int main()
{
    std::string imagePath = "../miku.jpg";
    Mat orgImg = imread(imagePath);
    Mat grayImg;
    cvtColor(orgImg, grayImg, COLOR_BGR2GRAY);  //ת�ɻҶ�ͼ
    imshow("org", grayImg);//���ԭʼ�Ҷ�ͼ
    //imwrite("grayImg.jpg", grayImg);

    Mat grayLayer[8];
    for (int i = 0; i < 8; i++) grayLayer[i] = Mat::zeros(grayImg.size(), CV_8UC1);  //��ʼ��8��ͼ���ڴ�ռ�

    for (int r = 0; r < grayImg.rows; r++)
        for (int c = 0; c < grayImg.cols; c++) //��ԭͼ����
            for (int i = 0; i < 8; i++) //����һ�����ÿ��bitλ
                grayLayer[i].at<uint8_t>(r, c) = -((grayImg.at<uint8_t>(r, c) &1<<i)>>i);
                //�ð�λ�롢��λ���õ�һλ��bitλ�������1����ֱ�Ӹ�ֵ-1����8b11111111��
    
    for (int i = 0; i < 8; i++)
    {
        string imgName("grayLayer");
        imshow(imgName + to_string(i), grayLayer[i]);//���8��ͼ
        //imwrite(imgName+to_string(i) + ".jpg", grayLayer[i]);
    }

    waitKey(0);
    return 0;
}