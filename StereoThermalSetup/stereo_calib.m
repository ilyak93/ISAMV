left_images = string(ls("G:\Vista_project\CALIB_NEW\left_best\left\"));
left_images = left_images(3:end);
left_images_path = arrayfun(@(s) append("G:\Vista_project\CALIB_NEW\left_best\left\", s), left_images);
left_images_path1 = left_images_path;
%left_images_path2 = left_images_path(20:39);
%left_images_path = [left_images_path1.', left_images_path2.'];

right_images = string(ls("G:\Vista_project\CALIB_NEW\right_best\right\"));
right_images = right_images(3:end);
right_images_path = arrayfun(@(s) append("G:\Vista_project\CALIB_NEW\right_best\right\", s), right_images);
right_images_path1 = right_images_path;
%right_images_path2 = right_images_path(60:39);
%right_images_path = [right_images_path1.', right_images_path2.'];


[imagePoints,boardSize] = detectCheckerboardPoints(left_images_path1, right_images_path1);

squareSize = 20;
worldPoints = generateCheckerboardPoints(boardSize,squareSize);

imageSize = [512, 640];
params = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
showReprojectionErrors(params);
figure;
showExtrinsics(params);