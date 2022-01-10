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
params = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
showReprojectionErrors(params);
figure;
showExtrinsics(params);