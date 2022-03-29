import cv2
import numpy as np


## 补充一下添加各种噪声的函数




# 算术均值滤波器
def ArithmeticMean(img, kernelSize):
    AImg = np.zeros(img.shape)
    k = int((kernelSize-1)/2) # 模板中心

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 不在滤波核范围内
            if i<k or i>(img.shape[0]-k-1) or j<k or j>(img.shape[1]-k-1):
                AImg[i][j] = img[i][j] # 像素值不变
            else: # 范围内
                for n in range(kernelSize):
                    for m in range(kernelSize):
                        # 范围内像素值求和取平均
                        AImg[i][j] += 1.0/(kernelSize*kernelSize)*img[i-k+n][j-k+m]

    AImg = np.uint8(AImg)
    return AImg

# 几何均值滤波器
def GeometricMean(img, kernelSize):
    GImg = np.ones(img.shape)
    k = int((kernelSize-1)/2) # 模板中心

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 不在滤波核范围内
            if i<k or i>(img.shape[0]-k-1) or j<k or j>(img.shape[1]-k-1):
                GImg[i][j] = img[i][j] # 像素值不变
            else: # 范围内
                for n in range(kernelSize):
                    for m in range(kernelSize):
                        # 范围内像素值求乘积后开根号
                        GImg[i][j] *= img[i-k+n][j-k+m]
                GImg[i][j] = pow(GImg[i][j], 1/(kernelSize*kernelSize))

    GImg = np.uint8(GImg)
    return GImg

# 谐波均值滤波器
def HarmonicMean(img, kernelSize):
    HImg = np.zeros(img.shape)
    k = int((kernelSize-1)/2)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 不在滤波核范围内
            if i<k or i>(img.shape[0]-k-1) or j<k or j>(img.shape[1]-k-1):
                HImg[i][j] = img[i][j] # 像素值不变
            else:
                for n in range(kernelSize):
                    for m in range(kernelSize):
                        if all(img[i-k+n][j-k+m]) == 0:
                            HImg[i][j] = 0
                            break
                        else:
                            HImg[i][j] += 1/img[i-k+n][j-k+m]
                    else:
                        continue
                    break

                if all(HImg[i][j]) != 0:
                    HImg[i][j] = (kernelSize*kernelSize)/HImg[i][j]

    HImg = np.uint8(HImg)
    return HImg

# 逆谐波均值滤波器
def IHarmonicMean(img, kernelSize, Q):
    IHImg = np.zeros(img.shape)
    # print(IHImg)
    # print(img[0][0])
    k = int((kernelSize-1)/2)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 不在滤波核范围内
            if i<k or i>(img.shape[0]-k-1) or j<k or j>(img.shape[1]-k-1):
                IHImg[i][j] = img[i][j] # 像素值不变
            else:
                res_top = 0
                res_bottom = 0
                for n in range(kernelSize):
                    for m in range(kernelSize):
                        if Q>0:
                            res_top += pow(img[i-k+n][j-k+m], Q+1)
                            res_bottom += pow(img[i-k+n][j-k+m], Q)
                # print(res_top)
                        else:
                            if all(img[i-k+n][j-k+m]) == 0:
                                IHImg[i][j] = 0
                                break
                            else:
                                res_top += pow(img[i-k+n][j-k+m], Q+1)
                                res_bottom += pow(img[i-k+n][j-k+m], Q)
                    else:
                        continue
                    break
                else:
                    if all(res_bottom) != 0:
                        IHImg[i][j] = res_top/res_bottom

    HImg = np.uint8(IHImg)
    return HImg

img = cv2.imread('D:\Study\digital image processing/lena1.jpg')
cv2.imshow("img", img)
# res1 = ArithmeticMean(img, 3)
# cv2.imshow("AImg", res1)
# res2 = GeometricMean(img, 3)
# cv2.imshow("GImg", res2)
# res3 = HarmonicMean(img, 3)
# cv2.imshow("HImg", res3)
# res4 = IHarmonicMean(img, 3, -1.5)
# cv2.imshow("Q=-1.5", res4)
# res5 = IHarmonicMean(img, 3, 1.5)
# cv2.imshow("Q=1.5", res5)

cv2.waitKey(0)
cv2.destroyAllWindows()






# 方法一
# 调用 medianBlur() 函数实现中值滤波
img = cv2.imread('D:\Study\digital image processing/lena2.jpg')
#常用来去除椒盐噪声
#卷积核使用奇数
res = cv2.medianBlur(img, 3)
cv2.imshow("Input", img)
cv2.imshow("Median", res)
cv2.waitKey()
cv2.destroyAllWindows()

# 方法二
# 填充方式是无填充
# 对图像边缘，上下左右处忽略掉不进行滤波，只对可以容纳下一个滤波模板的区域滤波
# def MedianFilter(image, k=3, padding=None):
#     img = image
#     height = img.shape[0]
#     width = img.shape[1]
#     if not padding:
#         edge = int((k-1)/2)
#         if height-1-edge <= edge or width-1-edge <= edge:
#             print("The parameter k is to large.")
#             return None
#         res = np.zeros((height, width), dtype="uint8")
#         for i in range(edge, height-edge):
#             for j in range(edge, width-edge):
#                 # 调用np.median求取中值
#                 res[i, j] = np.median(img[i-edge:i+edge+1, j-edge:j+edge+1])
#     return res
# 
# img = cv2.imread('D:\Study\digital image processing/lena2.jpg')
# res = MedianFilter(img)
# cv2.imshow("Input", img)
# cv2.imshow("Median", res)
# cv2.waitKey()
# cv2.destroyAllWindows()
