%{
left_images = string(ls("G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_4_therm_2_rs\best_left - Copy\"));
left_images = left_images(3:end);
left_images = sort_nat(left_images);
left_images = arrayfun(@(s) append("G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_4_therm_2_rs\best_left - Copy\", s), left_images);
left_images = left_images(1:200);

right_images = string(ls("G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_4_therm_2_rs\best_right - Copy\"));
right_images = right_images(3:end);
right_images = sort_nat(right_images);
right_images = arrayfun(@(s) append("G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_4_therm_2_rs\best_right - Copy\", s), right_images);
right_images = right_images(1:200);

squareSize = 20;
imageSize = [720, 1280];

[imagePoints,boardSize] = detectCheckerboardPoints(left_images);
worldPoints = generateCheckerboardPoints(boardSize,squareSize);

[camera1Params, ~, errors1] = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize, 'NumRadialDistortionCoefficients', 3, 'EstimateTangentialDistortion', false);
showReprojectionErrors(camera1Params);

[imagePoints,boardSize] = detectCheckerboardPoints(right_images);
worldPoints = generateCheckerboardPoints(boardSize,squareSize);
[camera2Params, ~, errors2] = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize, 'NumRadialDistortionCoefficients', 3, 'EstimateTangentialDistortion', false);
showReprojectionErrors(camera2Params);
%}
%camera1Params = load('G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_single_6_therm_6_rs_stereo_4\best_stereo_300_best_single\single_cam_calib\camera1p.mat').camera1Params;
%camera2Params = load('G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_single_6_therm_6_rs_stereo_4\best_stereo_300_best_single\single_cam_calib\camera2p.mat').camera2Params;
%showReprojectionErrors(camera1Params);
%showReprojectionErrors(camera2Params);


left_images = string(ls("G:\Vista_project\finish_deep\calibration\best_right_therm_rs_rt_sorted\"));
left_images = left_images(3:end);
left_images = sort_nat(left_images);
left_images = arrayfun(@(s) append("G:\Vista_project\finish_deep\calibration\best_right_therm_rs_rt_sorted\\", s), left_images);
left_images = left_images(1:200);

right_images = string(ls("G:\Vista_project\finish_deep\calibration\best_rs_rs_rt_sorted\"));
right_images = right_images(3:end);
right_images = sort_nat(right_images);
right_images = arrayfun(@(s) append("G:\Vista_project\finish_deep\calibration\best_rs_rs_rt_sorted\", s), right_images);
right_images = right_images(1:200);

squareSize = 20;
[imagePoints,boardSize] = detectCheckerboardPoints(left_images, right_images);
worldPoints = generateCheckerboardPoints(boardSize,squareSize);

imageSize = [720, 1280];
params = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
%params = load("StereoParamsNew250.mat");
%params = params.stereoParamsNew250;
showReprojectionErrors(params);
figure;
showExtrinsics(params);


%{
figure
img_idx = 5;
imshow(imread(left_images(img_idx)));
hold on

plot(params.CameraParameters1.ReprojectedPoints(:,1,img_idx),params.CameraParameters1.ReprojectedPoints(:,2,img_idx),'r+','MarkerSize',8)
legend('Detected Points','ReprojectedPoints')
hold off

imshow(imread("G:\Vista_project\cccalib2 - Copy\aligned_rs_t\4.png"));
[J,newOrigin] = undistortImage(imread("G:\Vista_project\cccalib2 - Copy\aligned_rs_t\4.png"), params.CameraParameters2);
imshow(J);
%}

source_path = "G:/Vista_project/finish_ipc\";
left_files = ls(source_path + "left");
left_files = left_files(3:end, :);
left_files = string(left_files);
left_files = sort_nat(left_files);

right_files = ls(source_path + "right");
right_files = right_files(3:end, :);
right_files = string(right_files);
right_files = sort_nat(right_files);

dest_path = source_path + "\stereo-ptc\";
%params = load("G:\Vista_project\finish_ipc\calibration\stereo-therm.mat").stereoParams;
params.CameraParameters1.ImageSize = [512,640];
params.CameraParameters2.ImageSize = [512,640];

for ind = 50:length(left_files);
    frameLeft = imread("G:\Vista_project\fusion\1\fus\26_left.png");
    %frameLeft = double(frameLeft);
    %mx = max(max(frameLeft));
    %mn = min(min(frameLeft));
    %frameLeft = (frameLeft - mn) ./ (mx - mn) * 256;
    %frameLeft = uint8(round(frameLeft))
    
    frameRight = imread("G:\Vista_project\fusion\1\fus\26_right.png");
    %frameRight = double(frameRight);
    %mx = max(max(frameRight));
    %mn = min(min(frameRight));
    %frameRight = (frameRight - mn) ./ (mx - mn) * 256;
    %frameRight = uint8(round(frameRight));

    

    [frameLeftRect, frameRightRect] = rectifyStereoImages(frameLeft, frameRight, params);

    imtool(stereoAnaglyph(frameLeftRect, frameRightRect));


    %%    
    %frameLeftGray  = rgb2gray(frameLeftRect);
    %frameRightGray = rgb2gray(frameRightRect);
    frameLeftGray  = frameLeftRect;
    frameRightGray = frameRightRect;

    disparityMap = disparitySGM(frameLeftGray, frameRightGray,'DisparityRange', [0,64]); %'BlockSize', 21, 'UniquenessThreshold',1
    %disparityMap = ;
    %dis = open("dis.mat");
    disparityMap = disparityMap(1:468,1:574,1);
    %disparityMap(disparityMap>0) = 64 - disparityMap(disparityMap>0);
    figure;
    imshow(disparityMap, [0 64]);
    title('Disparity Map');
    colormap jet
    colorbar
    %%

    points3D = reconstructScene(disparityMap, params);

    % Convert to meters and create a pointCloud object
    points3D = points3D ./ 1000;
    %points3D(points3D(:,:,3) > 10) = NaN;
    points3D(:,:, 1) = points3D(:,:, 1);
    points3D(:,:, 2) = points3D(:,:, 2);
    %points3D(:, :, 3) = points3D(:, :, 3) + 3;

    % visualization - not mandatory

    [m,n,r]=size(frameLeftGray);
    frame_viz= uint8(zeros(m,n,3)); 
    tmp = uint8(frameLeftGray / 256 - 1);


    frame_viz(:,:,1) = tmp;
    frame_viz(:,:,2)=frame_viz(:,:,1);
    frame_viz(:,:,3)=frame_viz(:,:,1);

    ptCloud = pointCloud(points3D, 'Color', frame_viz);

    % pcshow(ptCloud);
    % title('3D');
    % xlabel('X');
    % ylabel('Y');
    % zlabel('Z');

    % Create a streaming point cloud viewer
    player3D = pcplayer([-10, 10], [-10, 10], [-100, 100], 'VerticalAxis', 'y', ...
        'VerticalAxisDir', 'up');

    % Visualize the point cloud
    view(player3D, ptCloud);

    pcwrite(ptCloud, dest_path+string(ind), 'PLYFormat','ascii');

    %ptCloud = pcread('80_frame.ply');

    %player3D = pcplayer([-10, 10], [-10, 10], [-20, 20], 'VerticalAxis', 'y', ...
    %    'VerticalAxisDir', 'up');

    % Visualize the point cloud
    %view(player3D, ptCloud);
end
