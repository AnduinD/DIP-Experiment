%path='AyanamiRei.jpg';

rgb_R = zeros(128, 128);rgb_R(1:64, 1:64) = 255;
rgb_G = zeros(128, 128);rgb_G(1:64, 65:128)= 255;
rgb_B = zeros(128, 128);rgb_B(65:128, 1:64) = 255;
I_RGB=cat(3, rgb_R, rgb_G, rgb_B);
%figure, imshow(rgb), title('RGB彩色图像'); 



%I=imread(path);
I=I_RGB;
%figure(20);
%imshow(I);title('org RGB');%显示原始彩色图像

% 取单通道RGB
[r,g,b] = deal(I(:,:,1),I(:,:,2),I(:,:,3)); %通道分离split channel
figure(1);
subplot(2,2,1);imshow(cat(3,r,zeros(size(r)),zeros(size(r))));title('R');
subplot(2,2,2);imshow(cat(3,zeros(size(g)),g,zeros(size(g))));title('G');
subplot(2,2,3);imshow(cat(3,zeros(size(b)),zeros(size(b)),b));title('B');
subplot(2,2,4);imshow(I);title('org RGB');%显示原始彩色图像

%%  任务1：HSI颜色空间
I_HSI=rgb2hsi(I);
figure(2);
% hhh = hsi2rgb(cat(3,I_HSI(:,:,1),zeros(size(I_HSI(:,:,2))),zeros(size(I_HSI(:,:,3)))));
% sss = hsi2rgb(cat(3,zeros(size(I_HSI(:,:,1))),I_HSI(:,:,2),zeros(size(I_HSI(:,:,3)))));
% iii = hsi2rgb(cat(3,zeros(size(I_HSI(:,:,1))),zeros(size(I_HSI(:,:,2))),I_HSI(:,:,3)));
subplot(2,2,1);imshow(I_HSI(:,:,1),[]);title('H');
subplot(2,2,2);imshow(I_HSI(:,:,2),[]);title('S');
subplot(2,2,3);imshow(I_HSI(:,:,3),[]);title('I');
rgb_from_hsi = hsi2rgb(I_HSI); % 这个公式返回的是0~1的RGB
subplot(2,2,4);imshow(rgb_from_hsi);title("rgb from hsi");

%%  任务2：YUV颜色空间
I_YUV = rgb2yuv(I);
yyy = yuv2rgb(cat(3,I_YUV(:,:,1),zeros(size(I_YUV(:,:,2))),zeros(size(I_YUV(:,:,3)))));
uuu = yuv2rgb(cat(3,zeros(size(I_YUV(:,:,1))),I_YUV(:,:,2),zeros(size(I_YUV(:,:,3)))));
vvv = yuv2rgb(cat(3,zeros(size(I_YUV(:,:,1))),zeros(size(I_YUV(:,:,2))),I_YUV(:,:,3)));
figure(3);
subplot(2,2,1);imshow(yyy,[]);title('Y');
subplot(2,2,2);imshow(uuu,[]);title('U');
subplot(2,2,3);imshow(vvv,[]);title('V');
rgb_from_yuv = yuv2rgb(I_YUV);
subplot(2,2,4);imshow(rgb_from_yuv);title("rgb from yuv");


%%  任务3：HSV颜色空间
I_HSV = rgb2hsv(I);   % 这里用matlab内建的转换函数
figure(4);
subplot(2,2,1);imshow(I_HSV(:,:,1),[]);title('H');
subplot(2,2,2);imshow(I_HSV(:,:,2),[]);title('S');
subplot(2,2,3);imshow(I_HSV(:,:,3),[]);title('V');
rgb_from_hsv = hsv2rgb(I_HSV);
subplot(2,2,4);imshow(rgb_from_hsv);title("rgb from hsv");



%% 任务4：I通道亮度的图像增强
rgb_enhance = img_enhance(I);
figure(10);imshow(rgb_enhance);title("image enhancement"); 




%%  RGB-HSI
function [hsi] = rgb2hsi(rgb)
    rgb=im2double(rgb);
    [r,g,b]=deal(rgb(:,:,1),rgb(:,:,2),rgb(:,:,3));
    
    numerator=(r-g)+(r-b);
    denominator=2*sqrt((r-g).^2 + (r-b).*(g-b));
    theta=acos(numerator./(denominator+eps))/(2*pi);
    hhh=theta.*(b<=g)+(1-theta).*(b>g);
    sss=1-3.*min(cat(3,r,g,b),[],3)./(r+g+b+eps);
    iii=(r+g+b)/3;
    
    %hhh=(hhh-min(hhh,[],'all'))./(max(hhh,[],'all')-min(hhh,[],'all')+eps);
    %sss=(sss-min(sss,[],'all'))./(max(sss,[],'all')-min(sss,[],'all')+eps);
    hsi=cat(3,hhh,sss,iii);
end

function [rgb] = hsi2rgb(hsi)
  hsi=im2double(hsi);
  [hhh,sss,iii]=deal(hsi(:,:,1),hsi(:,:,2),hsi(:,:,3));% md怎么i是个内建的关键字 md怎么ss也是淦
  [r,g,b]=deal(zeros(size(hhh)));
  for m = 1:size(hhh,1)
    for n = 1:size(hhh,2)
        if (hhh(m,n)>=0&&hhh(m,n)< 1/3)
          r(m,n) = iii(m,n)*(1+(sss(m,n)*cos(2*pi*hhh(m,n)))/cos(pi/3-2*pi*hhh(m,n)));
          b(m,n) = iii(m,n)*(1-sss(m,n));
          g(m,n) = 3*iii(m,n)-(b(m,n)+r(m,n));
          continue; 
        end
        if (hhh(m,n)>=1/3&&hhh(m,n)< 2/3)
          g(m,n) = iii(m,n)*(1+(sss(m,n)*cos(2*pi*hhh(m,n)-2/3*pi))/cos(pi-2*pi*hhh(m,n)));
          r(m,n) = iii(m,n)*(1-sss(m,n));
          b(m,n) = 3*iii(m,n)-(g(m,n)+r(m,n));
          continue;
        end
        if (hhh(m,n)>=2/3&&hhh(m,n)<=1)
          b(m,n) = iii(m,n)*(1+(sss(m,n)*cos(2*pi*hhh(m,n)-4/3*pi))/cos(5/3*pi-2*pi*hhh(m,n)));
          g(m,n) = iii(m,n)*(1-sss(m,n));
          r(m,n) = 3*iii(m,n)-(b(m,n)+g(m,n));
          continue;
        end
    end
  end
  rgb = rescale(cat(3,r,g,b));
end

%%  RGB-YUV
function [yuv] = rgb2yuv(rgb)  
    cvtMat = [ 0.299  0.587  0.114;
              -0.147 -0.289  0.436; 
               0.615 -0.515 -0.100];
    cvtY=repmat(reshape(cvtMat(1,:),[1,1,3]),[size(rgb,[1,2]),1]);
    cvtU=repmat(reshape(cvtMat(2,:),[1,1,3]),[size(rgb,[1,2]),1]);
    cvtV=repmat(reshape(cvtMat(3,:),[1,1,3]),[size(rgb,[1,2]),1]);
    yuv = fix(cat(3,dot(cvtY,double(rgb),3),dot(cvtU,double(rgb),3),dot(cvtV,double(rgb),3)));
end

function [rgb] = yuv2rgb(yuv)  
  cvtMat = [ 1.00  0.00  1.14;
             1.00 -0.39 -0.58; 
             1.00  2.03  0.00];
  cvtR=repmat(reshape(cvtMat(1,:),[1,1,3]),[size(yuv,[1,2]),1]);
  cvtG=repmat(reshape(cvtMat(2,:),[1,1,3]),[size(yuv,[1,2]),1]);
  cvtB=repmat(reshape(cvtMat(3,:),[1,1,3]),[size(yuv,[1,2]),1]);
  rgb = fix(cat(3,dot(cvtR,double(yuv),3),dot(cvtG,double(yuv),3),dot(cvtB,double(yuv),3)));
  rgb = uint8(max(rgb,0));
end

%% image enhancement
function [rgb_out] = img_enhance(rgb_in)
    hsv_tmp = rgb2hsv(rgb_in);

    % 增强i通道的亮度
    hsv_tmp(:,:,3) = histeq(hsv_tmp(:,:,3));
    hsv_tmp(:,:,3) = 0.6*hsv_tmp(:,:,3);
    hsv_tmp(:,:,2) = 0.6*hsv_tmp(:,:,2);

    rgb_out = hsv2rgb(hsv_tmp);
end


