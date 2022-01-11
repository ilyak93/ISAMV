left_images = string(ls("G:\Vista_project\cur\left_used\"));
left_images = left_images(3:end);
left_images = arrayfun(@(s) append("G:\Vista_project\cur\left_used\", s), left_images);

right_images = string(ls("G:\Vista_project\cur\right_used\"));
right_images = right_images(3:end);
right_images = arrayfun(@(s) append("G:\Vista_project\cur\right_used\", s), right_images);


[imagePoints,boardSize] = detectCheckerboardPoints(left_images, right_images);

squareSize = 20;
worldPoints = generateCheckerboardPoints(boardSize,squareSize);

I = imread("G:\Vista_project\cur\left_used\Vista_project0tc10239.jpg"); 
imageSize = [size(I,1),size(I,2)];
stereoParams = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
[~, errors] = showReprojectionErrors(stereoParams);
figure;
showExtrinsics(stereoParams);

frameLeft = imread("G:\Vista_project\me_left - Copy.png");
% frameLeft = double(frameLeft);
% mx = max(max(frameLeft));
% mn = min(min(frameLeft));
% frameLeft = (frameLeft - mn) ./ (mx - mn) * 256;
% frameLeft = uint8(round(frameLeft));

frameRight = imread("G:\Vista_project\me_right - Copy.png");
% frameRight = double(frameRight);
% mx = max(max(frameRight));
% mn = min(min(frameRight));
% frameRight = (frameRight - mn) ./ (mx - mn) * 256;
% frameRight = uint8(round(frameRight));

[frameLeftRect, frameRightRect] = rectifyStereoImages(frameLeft, frameRight, stereoParams);

imtool(stereoAnaglyph(frameLeftRect, frameRightRect));

frameLeftGray  = frameLeftRect;
frameRightGray = frameRightRect;
    
disparityMap = disparityBM(frameLeftGray, frameRightGray, 'DisparityRange',[0 32], 'BlockSize', 5);
%disparityMap(isnan(disparityMap))=0; 
figure;
imshow(disparityMap, [0, 32]);
title('Disparity Map');
colormap jet
colorbar


points3D = reconstructScene(disparityMap, stereoParams);
z = points3D(:, :, 3) ./ (-1000);
imshow(z);




points3D = reconstructScene(disparityMap, stereoParams);

% Convert to meters and create a pointCloud object
points3D = points3D ./ 1000;

% visualization - not mandatory

[m,n,r]=size(frameLeftRect);
frame_viz=zeros(m,n,3); 
frame_viz(:,:,1)=frameLeftRect;
frame_viz(:,:,2)=frame_viz(:,:,1);
frame_viz(:,:,3)=frame_viz(:,:,1);

ptCloud = pointCloud(points3D, 'Color', frame_viz);

% Create a streaming point cloud viewer
player3D = pcplayer([-3, 3], [-3, 3], [-30, 30], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down');

% Visualize the point cloud
view(player3D, ptCloud);