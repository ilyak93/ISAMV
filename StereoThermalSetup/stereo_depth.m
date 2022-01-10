left_images = string(ls("G:\Vista_project\cur\left\"));
left_images = left_images(3:end);
left_images = arrayfun(@(s) append("G:\Vista_project\cur\left\", s), left_images);

right_images = string(ls("G:\Vista_project\cur\right\"));
right_images = right_images(3:end);
right_images = arrayfun(@(s) append("G:\Vista_project\cur\right\", s), right_images);


[imagePoints,boardSize] = detectCheckerboardPoints(left_images, right_images);

squareSize = 20;
worldPoints = generateCheckerboardPoints(boardSize,squareSize);

I = imread("G:\Vista_project\cur\left\Vista_project0tc10020.jpg"); 
imageSize = [size(I,1),size(I,2)];
stereoParams = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);

frameLeft = imread("E:\meavrer_tov\0tc1.jpg");
frameRight = imread("E:\meavrer_tov\0tc2.jpg");

[frameLeftRect, frameRightRect] = rectifyStereoImages(frameLeft, frameRight, stereoParams);

figure;
imshow(stereoAnaglyph(frameLeftRect, frameRightRect));
title('Rectified Video Frames');

frameLeftGray  = frameLeftRect;
frameRightGray = frameRightRect;
    
disparityMap = disparitySGM(frameLeftGray, frameRightGray);
figure;
imshow(disparityMap, [0, 64]);
title('Disparity Map');
colormap jet
colorbar

points3D = reconstructScene(disparityMap, stereoParams);

% Convert to meters and create a pointCloud object
points3D = points3D ./ 1000;

% visualization - not mandatory

[m,n,r]=size(frameLeftRect);
frame_viz=zeros(m,n,3); 
rgb(:,:,1)=frameLeftRect;
rgb(:,:,2)=rgb(:,:,1);
rgb(:,:,3)=rgb(:,:,1);

ptCloud = pointCloud(points3D, 'Color', rgb);

% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-3, 3], [0, 8], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down');

% Visualize the point cloud
view(player3D, ptCloud);