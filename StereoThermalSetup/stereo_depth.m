left_images = string(ls("G:\Vista_project\calib_new_long\left\"));
left_images = left_images(3:end);
left_images_path = arrayfun(@(s) append("G:\Vista_project\calib_new_long\left\", s), left_images);
%left_images_path1 = left_images_path(80:90);
left_images_path2 = left_images_path;
%left_images_path = [left_images_path1.', left_images_path2.'];

right_images = string(ls("G:\Vista_project\calib_new_long\right\"));
right_images = right_images(3:end);
right_images_path = arrayfun(@(s) append("G:\Vista_project\calib_new_long\right\", s), right_images);
%right_images_path1 = right_images_path(80:90);
right_images_path2 = right_images_path;
%right_images_path = [right_images_path1.', right_images_path2.'];


[imagePoints,boardSize] = detectCheckerboardPoints(left_images_path2, right_images_path2);

squareSize = 20;
worldPoints = generateCheckerboardPoints(boardSize,squareSize);

imageSize = [512, 640];
params = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
showReprojectionErrors(params);
figure;
showExtrinsics(params);

frameLeft = imread("G:\Vista_project\2p_left.png");
frameLeft = double(frameLeft);
mx = max(max(frameLeft));
mn = min(min(frameLeft));
frameLeft = (frameLeft - mn) ./ (mx - mn) * 256;
frameLeft = uint8(round(frameLeft));

frameRight = imread("G:\Vista_project\2p_right.png");
frameRight = double(frameRight);
mx = max(max(frameRight));
mn = min(min(frameRight));
frameRight = (frameRight - mn) ./ (mx - mn) * 256;
frameRight = uint8(round(frameRight));

[frameLeftRect, frameRightRect] = rectifyStereoImages(frameLeft, frameRight, params);

imtool(stereoAnaglyph(frameLeftRect, frameRightRect));


%%    
%frameLeftGray  = rgb2gray(frameLeftRect);
%frameRightGray = rgb2gray(frameRightRect);
frameLeftGray  = frameLeftRect;
frameRightGray = frameRightRect;

%disparityMap = disparitySGM(frameLeftGray, frameRightGray,'DisparityRange', [0,16]); %'BlockSize', 21, 'UniquenessThreshold',1
%disparityMap = ;
disparityMap = disparity(1:426,1:495,1);
figure;
imshow(disparityMap, [0 64]);
title('Disparity Map');
colormap jet
colorbar
%%

points3D = reconstructScene(disparityMap, params);

% Convert to meters and create a pointCloud object
points3D = points3D ./ -1000;
%points3D(points3D(:,:,3) > 10) = NaN;
%points3D(:,:, 1) = points3D(:,:, 2) ./ -1;
%points3D(:,:, 2) = points3D(:,:, 3) ./ -1;
%points3D(:, :, 3) = points3D(:, :, 3) + 3;

% visualization - not mandatory

[m,n,r]=size(frameRightGray);
frame_viz=zeros(m,n,3); 
frame_viz(:,:,1)=frameRightGray;
frame_viz(:,:,2)=frame_viz(:,:,1);
frame_viz(:,:,3)=frame_viz(:,:,1);

ptCloud = pointCloud(points3D, 'Color', frame_viz);


% pcshow(ptCloud);
% title('3D');
% xlabel('X');
% ylabel('Y');
% zlabel('Z');

% Create a streaming point cloud viewer
player3D = pcplayer([-100, 100], [-100, 100], [-100, 100], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down');

% Visualize the point cloud
view(player3D, ptCloud);