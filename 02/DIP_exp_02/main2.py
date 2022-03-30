import cv2
import numpy as np
import random

# 一些显示处理的函数
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

def depth_quantize(mat):
    '''对输入的图像矩阵做0~255的归一化和8bit量化'''
    top = np.max(mat)
    bot = np.min(mat)
    eps = 1e-8
    mat = 255.0*(mat-bot)/(top-bot+eps) # 均匀归一化到0~255之间
    mat = np.around(mat).astype(np.uint8) # 量化到8bit深度
    return mat

# 噪声种类宏定义
GAUSSIAN_NOISE = 0
RAYLEIGH_NOISE = 1
GAMMA_NOISE = 2
EXPONENT_NOISE = 3
UNIFORM_NOISE = 4
SALT_PEPPER_NOISE = 5

## 添加各种噪声的函数
def gaussian_noise(img, mean=10, sigma=30):
    '''
    添加高斯噪声
    @param img: 原图
    @param mean: 均值
    @param sigma: 标准差
    @return gaussian_out: 噪声处理后的图片
    '''
    noise = np.random.normal(mean, sigma, img.shape)# 产生高斯 noise
    gaussian_out = img + noise# 将噪声和图片叠加
    #gaussian_out = np.clip(gaussian_out, 0, 1)# 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = depth_quantize(gaussian_out)# 将图片灰度范围的恢复为 0-255
    return gaussian_out

def rayleigh_noise(img,a=20):
    '''添加瑞利噪声'''
    #a = 30.0
    noise = np.random.rayleigh(a, size=img.shape)
    rayleigh_out = img + noise
    rayleigh_out = depth_quantize(rayleigh_out)
    return rayleigh_out

def gamma_noise(img,a=10,b=2.5):
    '''添加Gamma噪声'''
    #a = 30.0
    #a, b = 10.0, 2.5
    assert a>b,"a should be larger than b"
    noise = np.random.gamma(shape=b, scale=a, size=img.shape)
    gm_out = img + noise
    gm_out = depth_quantize(gm_out)
    return gm_out

def exponent_noise(img,a=10):
    '''添加指数噪声'''
    #a = 10.0
    noise = np.random.exponential(scale=a, size=img.shape)
    exp_out = img + noise
    exp_out = depth_quantize(exp_out)
    return exp_out

def uniform_noise(img,a=10,b=150):
    '''
    添加均匀噪声
    @param image: 需要加噪的图片
    @param a: 均匀噪声直方图上的取值下限
    @param b: 均匀噪声直方图上的取值上限
    @return uf_out
    '''
    assert a<b,"a should be smaller than b"
    noise = np.random.uniform(low=a,high=b,size=img.shape) #生成均匀噪声矩阵
    np.random.uniform
    uf_out = img+noise #叠加噪声
    uf_out = depth_quantize(uf_out) #灰度量化
    return uf_out

def salt_pepper_noise(img, proportion=0.1):
    '''
    添加椒盐噪声
    @param proportion: 加入噪声的百分比
    @return img with salt and pepper
    '''
    sp_out = img
    rows, cols = sp_out.shape#获取高度宽度像素值
    num = int(rows*cols*proportion) #一个准备加入多少噪声小点
    for i in range(num):
        c = random.randint(0, cols - 1)
        r = random.randint(0, rows - 1)
        if random.randint(0, 1) == 0:
            sp_out[r, c] = 0  #撒胡椒
        else:
            sp_out[r, c] = 255 # 撒盐
    return sp_out



# 滤波器种类宏定义
ARITHMETIC_MEAN_FILTER = 0
GEOMETRIC_MEAN_FILTER = 1
HARMONIC_MEAN_FILTER = 2
INV_HARMONIC_MEAN_FILTER = 3
MEDIAN_FILTER = 4
MAX_FILTER = 5
MIN_FILTER = 6
MID_FILTER = 7
COR_ALPHA_FILTER = 8
ADAPTIVE_LOCAL_FILTER = 9

# 均值滤波器系
def arithmetic_mean_filter(img, kernel_size):
    '''算术均值滤波器'''
    img_out = np.zeros(img.shape)#初始化对象
    kernel_mask = np.ones((kernel_size,kernel_size),np.float32)/(kernel_size*kernel_size)  # 生成归一化盒式核
    img_out = cv2.filter2D(img, -1, kernel_mask) # 均值滤波
    img_out = depth_quantize(img_out) #归一化和量化
    return img_out

def geometric_mean_filter(img, kernel_size):
    '''几何均值滤波器'''
    img_out = np.zeros(img.shape)
    rows,cols = img.shape
    pad = int(kernel_size/2)
    img_pad = np.pad(img.copy(),((pad,kernel_size-pad-1),(pad,kernel_size-pad-1)), mode="edge").astype(np.float64) # 原图的边缘填充

    eps = 1e-8
    for i in range(pad,rows+pad):
        for j in range(pad,cols+pad):
            prod = np.prod(img_pad[i-pad:i+pad+1, j-pad:j+pad+1]+eps)  #求核内乘积
            img_out[i-pad][j-pad] = np.power(prod,1/(kernel_size*kernel_size)) #

    img_out = depth_quantize(img_out)
    return img_out

def harmonic_mean_filter(img, kernel_size):
    '''谐波均值滤波器'''
    img_out = np.zeros(img.shape)
    rows,cols = img.shape
    pad = int(kernel_size/2)
    img_pad = np.pad(img.copy(),((pad,kernel_size-pad-1),(pad,kernel_size-pad-1)), mode="edge").astype(np.float64)

    eps = 1e-8
    for i in range(pad,rows+pad):
        for j in range(pad,cols+pad):
            kernel_temp = (1.0/(img_pad[i-pad:i+pad+1, j-pad:j+pad+1]+eps))
            kernel_sum = kernel_temp.sum()
            img_out[i-pad][j-pad] = (kernel_size*kernel_size)/(kernel_sum+eps)

    img_out = depth_quantize(img_out)
    return img_out

def inv_harmonic_mean_filter(img, kernel_size, Q = 1.5):
    '''逆谐波均值滤波器'''
    Q = cv2.getTrackbarPos('Q(inv harmonic)', filterWin)
    img_out = np.zeros(img.shape)
    rows,cols = img.shape
    pad = int(kernel_size / 2)
    img_pad = np.pad(img.copy(), ((pad, kernel_size-pad-1), (pad, kernel_size-pad-1)), mode="edge").astype(np.float64)

    eps = 1e-8 # 一个很小的值，防止0在分母上
    for i in range(pad,rows+pad):
        for j in range(pad,cols+pad):
            kernel_temp = img_pad[i-pad:i+pad+1, j-pad:j+pad+1]+eps # 中间量
            numerator = np.sum(np.power(kernel_temp,Q+1)) #分子
            denominator = np.sum(np.power(kernel_temp,Q)) #分母
            img_out[i-pad][j-pad] = numerator/(denominator+eps)

    img_out =depth_quantize(img_out)
    return img_out



# 统计排序滤波器系
def median_filter(img, kernel_size):
    '''中值滤波器'''
    img_out = np.zeros(img.shape)
    rows,cols = img.shape
    pad = int(kernel_size / 2)
    img_pad = np.pad(img.copy(), ((pad, kernel_size-pad-1), (pad, kernel_size-pad-1)), mode="edge")

    for i in range(pad,rows+pad):
        for j in range(pad,cols+pad):
            img_out[i-pad][j-pad] = np.median(img_pad[i-pad:i+pad+1, j-pad:j+pad+1])

    img_out =depth_quantize(img_out)
    return img_out

def max_filter(img, kernel_size):
    '''最大值滤波器'''
    img_out = np.zeros(img.shape)
    rows,cols = img.shape
    pad = int(kernel_size / 2)
    img_pad = np.pad(img.copy(), ((pad, kernel_size-pad-1), (pad, kernel_size-pad-1)), mode="edge")

    for i in range(pad,rows+pad):
        for j in range(pad,cols+pad):
            img_out[i-pad][j-pad] = np.max(img_pad[i-pad:i+pad+1, j-pad:j+pad+1])

    img_out =depth_quantize(img_out)
    return img_out

def min_filter(img, kernel_size):
    '''最小值滤波器'''
    img_out = np.zeros(img.shape)
    rows,cols = img.shape
    pad = int(kernel_size / 2)
    img_pad = np.pad(img.copy(), ((pad, kernel_size-pad-1), (pad, kernel_size-pad-1)), mode="edge")

    for i in range(pad,rows+pad):
        for j in range(pad,cols+pad):
            img_out[i-pad][j-pad] = np.min(img_pad[i-pad:i+pad+1, j-pad:j+pad+1])

    img_out =depth_quantize(img_out)
    return img_out

def mid_filter(img, kernel_size):
    '''中点值滤波器'''
    img_out = np.zeros(img.shape)
    rows,cols = img.shape
    pad = int(kernel_size/2)
    img_pad = np.pad(img.copy(), ((pad, kernel_size-pad-1), (pad, kernel_size-pad-1)), mode="edge").astype(np.float64)

    for i in range(pad,rows+pad):
        for j in range(pad,cols+pad):
            kernel_temp = img_pad[i-pad:i+pad+1, j-pad:j+pad+1]
            img_out[i-pad][j-pad] = (kernel_temp.min()+kernel_temp.max())/2

    img_out =depth_quantize(img_out)
    return img_out

def cor_alpha_filter(img,kernel_size,d=2):
    '''修正阿尔法均值滤波器''' # 统计排序和算数均值的结合
    d = cv2.getTrackbarPos('d(cor alpha)', filterWin)
    kernel_size = max(int(np.sqrt(2*d+1)),kernel_size)
    img_out = np.zeros(img.shape)
    rows,cols = img.shape
    pad = int(kernel_size/2)
    img_pad = np.pad(img.copy(), ((pad, kernel_size-pad-1), (pad, kernel_size-pad-1)), mode="edge")
    eps = 1e-8
    for i in range(pad,rows+pad):
        for j in range(pad,cols+pad):
            kernel_temp = img_pad[i-pad:i+pad+1, j-pad:j+pad+1]
            kernel_temp = np.sort(kernel_temp.flatten())  # 对邻域像素按灰度值排序
            kernel_sum = np.sum(kernel_temp[d:-d-1])  # 删除 d 个最大灰度值, d 个最小灰度值
            img_out[i-pad][j-pad] = kernel_sum/(kernel_size*kernel_size-2*d+eps)  # 对剩余像素进行算术平均

    img_out =depth_quantize(img_out)
    return img_out

# 自适应滤波器系
def adaptive_local_filter(img,kernel_size):
    '''自适应局部降噪滤波器'''
    img_out = np.zeros(img.shape)
    rows,cols = img.shape
    pad = int(kernel_size/2)
    img_pad = np.pad(img.copy(), ((pad, kernel_size-pad-1), (pad, kernel_size-pad-1)), mode="edge")

    # 估计原始图像的噪声方差
    noise_mean, noise_stddev = cv2.meanStdDev(img)
    noise_var = noise_stddev ** 2 # 标准差转成方差
    #print(variance_noise)

    # 自适应局部降噪
    eps = 1e-8
    for i in range(pad,rows+pad):
        for j in range(pad,cols+pad):
            kernel_temp = img_pad[i-pad:i+pad+1, j-pad:j+pad+1] 
            g_xy = img_pad[i,j]  # 含噪声图像的原本像素点
            z_Sxy = np.mean(kernel_temp)  # 局部平均灰度
            kernel_var = np.var(kernel_temp)  # 灰度的局部方差
            residual = min(noise_var/(kernel_var+eps), 1.0) # 加性噪声假设：sigma_eta<=sigma_Sxy
            img_out[i-pad][j-pad] = g_xy - residual * (g_xy - z_Sxy)

    img_out =depth_quantize(img_out)
    return img_out    



def noising(img,noiseFlag):
    '''添加噪声'''
    img_noised = np.zeros(img.shape)
    if noiseFlag == GAUSSIAN_NOISE:
        img_noised = gaussian_noise(img_org.copy())
    elif noiseFlag == RAYLEIGH_NOISE:
        img_noised = rayleigh_noise(img_org.copy())
    elif noiseFlag == GAMMA_NOISE:
        img_noised = gamma_noise(img_org.copy())
    elif noiseFlag == EXPONENT_NOISE:
        img_noised = exponent_noise(img_org.copy())
    elif noiseFlag == UNIFORM_NOISE:
        img_noised = uniform_noise(img_org.copy())
    elif noiseFlag == SALT_PEPPER_NOISE:
        img_noised = salt_pepper_noise(img_org.copy())
    return img_noised

def filtering(img_noised,filterFlag,kernelSize):
    '''过滤波器'''
    img_filtered = np.zeros(img_noised.shape)
    if filterFlag == ARITHMETIC_MEAN_FILTER:
        img_filtered = arithmetic_mean_filter(img_noised,kernelSize);
    elif filterFlag == GEOMETRIC_MEAN_FILTER:
         img_filtered = geometric_mean_filter(img_noised,kernelSize);
    elif filterFlag == HARMONIC_MEAN_FILTER:
        img_filtered = harmonic_mean_filter(img_noised,kernelSize);
    elif filterFlag == INV_HARMONIC_MEAN_FILTER:
        img_filtered = inv_harmonic_mean_filter(img_noised,kernelSize);
    elif filterFlag == MEDIAN_FILTER:
        img_filtered = median_filter(img_noised,kernelSize);
    elif filterFlag == MAX_FILTER:
        img_filtered = max_filter(img_noised,kernelSize);
    elif filterFlag == MIN_FILTER:
        img_filtered = min_filter(img_noised,kernelSize);
    elif filterFlag == MID_FILTER:
        img_filtered = mid_filter(img_noised,kernelSize);
    elif filterFlag == COR_ALPHA_FILTER:
        img_filtered = cor_alpha_filter(img_noised,kernelSize);
    elif filterFlag == ADAPTIVE_LOCAL_FILTER:
        img_filtered = adaptive_local_filter(img_noised,kernelSize);
    return img_filtered

# 服务函数
img_noised = None
preNoiseFlag = -1
def onChange(_=None):
    '''添加噪声 进行复原'''
    noiseFlag = cv2.getTrackbarPos('noise type', filterWin)
    kernelSize = cv2.getTrackbarPos('kernel size', filterWin)+1
    filterFlag = cv2.getTrackbarPos('filter type', filterWin)
    curNoiseFlag = noiseFlag
    global preNoiseFlag,img_noised
    if curNoiseFlag != preNoiseFlag: # 判断噪声要不要刷新
        img_noised = noising(img_org.copy(),noiseFlag)
        preNoiseFlag = curNoiseFlag
    img_filtered = filtering(img_noised,filterFlag,kernelSize)
    cv2.imshow(imageWin, combine_images([img_org,img_noised,img_filtered]))


MAX_d = 50
MAX_Q = 5
NOISE_MAX_TYPE = SALT_PEPPER_NOISE
FILTER_MAX_TYPE = ADAPTIVE_LOCAL_FILTER
if __name__ == '__main__':
    img_org = cv2.imread('./miku.jpg',0)
    rows, cols = img_org.shape[:2] #获取前两维的长度
    filterWin = 'Filter Parameters'# 滤波器窗口名称
    imageWin = 'Filtered Image'# 图像窗口名称
    
    cv2.namedWindow(imageWin)
    cv2.namedWindow(filterWin,cv2.WINDOW_AUTOSIZE)# 创建窗体对象

    cv2.createTrackbar('noise type', filterWin, GAUSSIAN_NOISE, NOISE_MAX_TYPE, onChange)# 创建滑动条
    cv2.createTrackbar('filter type', filterWin, ADAPTIVE_LOCAL_FILTER, FILTER_MAX_TYPE, onChange)
    cv2.createTrackbar('kernel size', filterWin, 3, 10, onChange)
    cv2.createTrackbar('Q(inv harmonic)', filterWin, 1, MAX_Q, onChange) #除核规模之外的参数（反谐波的Q）
    cv2.createTrackbar('d(cor alpha)', filterWin, 1, MAX_d, onChange) #除核规模之外的参数（修阿尔法的d）
  
    onChange() #执行操作
    #cv2.resizeWindow(filterWin, 1024, 20) # 这句话没啥卵用，只是让滚动条窗口看着好调一点
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()