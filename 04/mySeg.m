I = rgb2gray(imread("RaspberryPi.jpg"));
I_filtered = imgaussfilt(I,3);

figure(1);
subplot(1,2,1);imshow(I);title('org');
subplot(1,2,2);imshow(I_filtered);title('Gaussian');

figure(4);
% 设计Roberts算子
Roberts_x=[-1,0;
            0,1];
Roberts_y=[0,-1;
           1, 0];
Gx_Roberts=conv2(I_filtered,Roberts_x,'same');
Gy_Roberts=conv2(I_filtered,Roberts_y,'same');% 全卷积提取边缘
%G_Roberts=sqrt(gradx_Roberts.^2+grady_Roberts.^2);% 总梯度幅值
G_Roberts = abs(Gx_Roberts)+abs(Gy_Roberts);
%grad_direction_roberts = atan(grady_Roberts./gradx_Roberts); % 梯度方向矩阵
thresh_Roberts = 3; % 阈值
G_Roberts = G_Roberts>thresh_Roberts;
subplot(2,2,1);imshow(G_Roberts);title('Roberts');

% 设计Sobel算子
Sobel_x=[-1,0,1;
         -2,0,2;
         -1,0,1];
Sobel_y=[-1,-2,-1;
          0, 0, 0;
          1, 2, 1];
Gx_Sobel=conv2(I_filtered,Sobel_x,'same');
Gy_Sobel=conv2(I_filtered,Sobel_y,'same');% 全卷积提取边缘
G_Sobel = abs(Gx_Sobel)+abs(Gy_Sobel);% 四领域距离
%G_Sobel=sqrt(Gx_Sobel.^2+Gy_Sobel.^2);
% %Gdir_Sobel = atan(Gy_Sobel./Gx_Sobel);
thresh_Sobel = 15;
G_Sobel = G_Sobel>thresh_Sobel;
subplot(2,2,2);imshow(G_Sobel);title('Sobel');


% 设计Prewitt算子
Prewitt_x=[-1,0,1;
           -1,0,1;
           -1,0,1];
Prewitt_y=[-1,-1,-1;
            0, 0, 0;
            1, 1, 1];
Gx_Prewitt=conv2(I_filtered,Prewitt_x,'same');
Gy_Prewitt=conv2(I_filtered,Prewitt_y,'same');
%G_prewitt=sqrt(Gx_Prewitt.^2+Gy_Prewitt.^2);
G_Prewitt = abs(Gx_Prewitt)+abs(Gy_Prewitt);
%Gdir_Prewitt = atan(Gy_Prewitt./Gx_Prewitt);
thresh_Prewitt = 15;
Gbw_Prewitt = G_Prewitt>thresh_Prewitt;
subplot(2,2,3);imshow(Gbw_Prewitt);title('Prewitt');

% 设计Laplacian算子
Laplace_1=[0, 1,0;
           1,-4,1;
           0, 1,0];
Laplace_2=[1, 1,1;
           1,-8,1;
           1, 1,1];
G1_Laplace=conv2(I_filtered,Laplace_1,'same');
G2_Laplace=conv2(I_filtered,Laplace_2,'same');
G_Laplace=abs(G2_Laplace);  %G2_Laplace
%G_Laplace = G2_Laplace;
thresh_Laplacian = 3;
G_Laplace = G_Laplace>thresh_Laplacian;
subplot(2,2,4);imshow(G_Laplace);title('Laplacian');



figure(5);
subplot(2,2,1);imshow(G_Prewitt> 0);title('Prewitt 0');
subplot(2,2,2);imshow(G_Prewitt> 4);title('Prewitt 4');
subplot(2,2,3);imshow(G_Prewitt> 8);title('Prewitt 8');
subplot(2,2,4);imshow(G_Prewitt>12);title('Prewitt 12');
