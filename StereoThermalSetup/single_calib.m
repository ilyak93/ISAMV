images = string(ls("G:\Vista_project\CALIB_NEW\left_best\left\"));
images = images(3:end);
images_path = arrayfun(@(s) append("G:\Vista_project\CALIB_NEW\left_best\left\", s), images);


[imagePoints,boardSize] = detectCheckerboardPoints(images_path);

squareSize = 20;
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

imageSize = [512, 640];
params = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
showReprojectionErrors(params);
figure;
showExtrinsics(params);

drawnow;
figure; 
imshow(imageFileNames{1}); 
hold on;
plot(imagePoints(:,1,1), imagePoints(:,2,1),'go');
plot(params.ReprojectedPoints(:,1,1),params.ReprojectedPoints(:,2,1),'r+');
legend('Detected Points','ReprojectedPoints');
hold off;