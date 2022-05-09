I = rgb2gray(imread("RaspberryPi.jpg"));
I_filtered = imgaussfilt(I,3);

figure(1);
subplot(1,2,1);imshow(I);title('org');
subplot(1,2,2);imshow(I_filtered);title('Gaussian');

figure(2);
% edge1=edge(I_filtered,'Roberts',0);  subplot(2,3,1);imshow(edge1);title('Roberts');
% edge2=edge(I_filtered,'Sobel',  0);  subplot(2,3,2);imshow(edge2);title('Sobel');
% edge3=edge(I_filtered,'Prewitt',0);  subplot(2,3,3);imshow(edge3);title('Prewitt');
% edge4=edge(I_filtered,'LOG',    0);  subplot(2,3,4);imshow(edge4);title('Laplacian');
% edge5=edge(I_filtered,'Canny',  0);  subplot(2,3,5);imshow(edge5);title('Canny');

edge1=edge(I_filtered,'Roberts',0.0028);  subplot(2,3,1);imshow(edge1);title('Roberts');
edge2=edge(I_filtered,'Sobel',  0.0018);  subplot(2,3,2);imshow(edge2);title('Sobel');
edge3=edge(I_filtered,'Prewitt',0.0015);  subplot(2,3,3);imshow(edge3);title('Prewitt');
edge4=edge(I_filtered,'LOG',    0.00005); subplot(2,3,4);imshow(edge4);title('Laplacian');
edge5=edge(I_filtered,'Canny',  0.013);  subplot(2,3,5);imshow(edge5);title('Canny');

