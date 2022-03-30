import cv2
import numpy as np

LOW_PASS = 0
HIGH_PASS = 1

IDEAL_FILTER = 0
BUTTERWORTH_FILTER = 1
GAUSSIAN_FILTER = 2

def combine_images(images, axis=1):
    '''
    合并图像。
    @param images: 图像列表(图像成员的维数必须相同)
    @param axis: 合并方向。 
    axis=0时，图像垂直合并;
    axis = 1 时， 图像水平合并。
    @return 合并后的图像
    '''
    ndim = images[0].ndim
    shapes = np.array([mat.shape for mat in images])
    assert np.all(map(lambda e: len(e) == ndim, shapes)
                  ), 'all images should be same ndim.'
    if axis == 0:  # 垂直方向合并图像
        cols = np.max(shapes[:, 1])# 合并图像的 cols
        copy_imgs = [cv2.copyMakeBorder(img, 0, 0, 0, cols - img.shape[1],# 扩展各图像 cols大小，使得 cols一致
                                        cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
        return np.vstack(copy_imgs)# 垂直方向合并
    else:  # 水平方向合并图像
        rows = np.max(shapes[:, 0])# 合并图像的 rows
        copy_imgs = [cv2.copyMakeBorder(img, 0, rows - img.shape[0], 0, 0, # 扩展各图像rows大小，使得 rows一致
                                        cv2.BORDER_CONSTANT, (0, 0, 0)) for img in images]
        return np.hstack(copy_imgs)# 水平方向合并
 
 
def fft(img):
    '''对图像进行傅立叶变换，并返回移位位后的频率矩阵'''
    assert img.ndim == 2, 'img should be gray.'
    rows, cols = img.shape[:2]
    # nrows = cv2.getOptimalDFTSize(rows)    # 计算最优尺寸
    # ncols = cv2.getOptimalDFTSize(cols)
    # nimg = np.zeros((nrows, ncols))# 根据新尺寸，建立新变换图像
    # nimg[:rows, :cols] = img
    nimg = img
    fftMat = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT) # opencv的傅立叶变换
    fftShiftMat = np.fft.fftshift(fftMat) # 移位，低频部分移到中间，高频部分移到四周
    return fftShiftMat
 
def ifft(fftShiftMat):
    '''傅立叶反变换，返回反变换图像'''
    f_ishift_mat = np.fft.ifftshift(fftShiftMat)# 反移位，低频部分回到顶角
    img_back = cv2.idft(f_ishift_mat)# 傅立叶反变换
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])# 将复数转换为幅度, sqrt(re^2 + im^2)
    return img_back
 
def depth_quantize(mat):
    '''对输入的图像矩阵做0~255的归一化和8bit量化'''
    top = np.max(mat)
    bot = np.min(mat)
    mat = 255*(mat-bot)/(top-bot) # 均匀归一化到0~255之间
    mat = np.uint8(np.around(mat))   # 量化到8bit深度
    return mat

def fft_image(fftMat):
    '''将频率矩阵转换为可视图像'''
    logMat = np.log(1+cv2.magnitude(np.float32(fftMat[:, :, 0]),np.float32(fftMat[:, :, 1]))) # 转成对数 log(1+np.abs(fftMat))
    logMat = depth_quantize(logMat) 
    return logMat
 
def fft_distances(rows, cols):
    '''计算m,n矩阵每一点距离中心的距离'''
    crow,ccol = rows/2 , cols/2 # 取出中心原点
    u = np.array([np.abs(i-crow) for i in range(rows)],dtype=np.float32).reshape(rows,1)#生成一维的距离向量
    v = np.array([np.abs(j-ccol) for j in range(cols)],dtype=np.float32)# 生成一维的距离向量
    Duv = np.sqrt(u**2 + v**2) # 用行和列的距离向量去张成距离矩阵
    return Duv
 
 
def lpfilter(flag, rows, cols, D0, n):
    '''低通滤波器
    @param flag: 滤波器类型
    @param rows: 被滤波的矩阵高度
    @param cols: 被滤波的矩阵宽度
    @param D0: 滤波器大小 D0
    @param n: 巴特沃斯阶数 
    @return 滤波器矩阵 
    '''
    assert D0 > 0, 'D0 should be more than 0.' #断言滤波核规模

    filterMat = None # 生成滤波核对象矩阵
   
    if flag == IDEAL_FILTER:  # 理想低通滤波
        filterMat = np.zeros((rows, cols), np.uint8)
        cv2.circle(filterMat,(int(cols/2), int(rows/2)) ,D0, (1, 1, 1), thickness=-1) # 生成通带的部分
        #filterMat = np.ones((rows, cols, 2), np.uint8) # 调试用的一个全通滤波器
        # 注：这circle里rows和cols的顺序反过来了，因为是原点坐标(x,y)，不是前两维(row,col) # 坑b玩意，我在这里懵逼了好久
        #filterMat = np.zeros((rows, cols), np.complex64)
        #real = np.zeros(filterMat.real.shape,np.float32)
        # cv2.circle(real, (int(rows/2),int(cols/2)),D0, (1, 1), thickness=-1) # 生成通带的部分
        # filterMat.real = real
        
    elif flag == BUTTERWORTH_FILTER: # 巴特沃斯低通
        Duv = fft_distances(rows,cols)
        filterMat = 1 / (1 + np.power(Duv/D0, 2*n)) # 巴特沃斯核的数学公式
    
    elif flag == GAUSSIAN_FILTER: # 高斯低通
        Duv = fft_distances(rows,cols)
        filterMat = np.exp(-(Duv*Duv)/(2*D0*D0)) # 高斯核的数学公式
       
    filterMat = cv2.merge((filterMat, filterMat))# fliter_mat 也需要2个通道（频域有实部和虚部）
    return filterMat
 
 
def hpfilter(flag, rows, cols, D0, n):
    '''高通滤波器
    @param flag: 滤波器类型
    @param rows: 被滤波的矩阵高度
    @param cols: 被滤波的矩阵宽度
    @param D0: 滤波器大小 D0
    @param n: 巴特沃斯阶数 
    @return 滤波器矩阵 
    '''
    assert D0 > 0, 'D0 should be more than 0.'
    filterMat = None
    
    if flag == IDEAL_FILTER:
        filterMat = np.ones((rows, cols), np.uint8)
        cv2.circle(filterMat, (int(cols/2), int(rows/2)),D0, (0, 0, 0), thickness=-1)

    elif flag == BUTTERWORTH_FILTER:
        Duv = fft_distances(rows, cols)
        Duv[int(rows/2),int(cols/2)] = 0.000001# Duv有 0 值(中心距离中心为0)， 为避免0到分母上了，设中心为 0.000001
        filterMat = 1/(1 + np.power(D0/Duv, 2*n))

    elif flag == GAUSSIAN_FILTER:
        Duv = fft_distances(*fftShiftMat.shape[:2])
        filterMat = 1-np.exp(-(Duv**2)/(2*D0**2))
        
    filterMat = cv2.merge((filterMat, filterMat))
    return filterMat
 
 
def filtering(_=None):
    '''生成滤波核，并返回滤波后的时域图像'''
    kernelSize = cv2.getTrackbarPos('kernel size', filterWin)
    flag = cv2.getTrackbarPos('filter type', filterWin)
    n = cv2.getTrackbarPos('Butterworth n', filterWin)
    lh = cv2.getTrackbarPos('low/high pass', filterWin)
    
    filterMat = None# 生成一个滤波核对象

    if lh == LOW_PASS:
        filterMat = lpfilter(flag, fftShiftMat.shape[0], fftShiftMat.shape[1], kernelSize, n)
    elif lh == HIGH_PASS:
        filterMat = hpfilter(flag, fftShiftMat.shape[0], fftShiftMat.shape[1], kernelSize, n)
    
    filteredMat =  fftShiftMat*filterMat  # 滤波变换的操作
    #print(filteredMat == fftShiftMat)
    img_back = ifft(filteredMat) # 傅里叶反变换得到滤波后的图像
    #img_back = np.uint8(np.around(img_back))
    # filteredMat = np.fft.ifftshift(filteredMat)
    # img_back = np.fft.ifft2(filteredMat)
    # img_back = np.abs(img_back)

    img_back = depth_quantize(img_back) # 量化回8bit深度
    # 显示图像  [滤波后的时域图像   滤波核图像   滤波后的频域叠加图像]
    cv2.imshow(imageWin, combine_images([img_back, fft_image(filterMat),fft_image(filteredMat)]))


if __name__ == '__main__':
    img = cv2.imread('./miku.jpg',0)
    rows, cols = img.shape[:2] #获取前两维的长度
    filterWin = 'Filter Parameters'# 滤波器窗口名称
    imageWin = 'Filtered Image'# 图像窗口名称

    
    cv2.namedWindow(imageWin)
    cv2.namedWindow(filterWin)# 创建窗体对象

    cv2.createTrackbar('kernel size', filterWin, 20, int(min(rows, cols)/4), filtering)# 创建滑动条
    cv2.createTrackbar('filter type', filterWin, BUTTERWORTH_FILTER, 2, filtering)# 创建flag 滚动条
     #0 ideal    #1 Butterworth    #2 Gaussian
    cv2.createTrackbar('Butterworth n', filterWin, 1, 5, filtering) # 巴特沃斯的阶数 滚动条
    cv2.createTrackbar('low/high pass', filterWin, LOW_PASS, 1, filtering)# lh: 滤波器是低通还是高通， 0 为低通， 1为高通
   
    #cv2.imshow(imageWin,img)

    fftShiftMat = fft(img) # 得到频谱

    

    filtering() #执行操作
    cv2.resizeWindow(filterWin, 512, 20) # 这句话没啥卵用，只是让滚动条窗口看着好调一点
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()