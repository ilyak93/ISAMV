images = string(ls("G:\Vista_project\cur\"));
images = images(3:end);
images = arrayfun(@(s) append("G:\Vista_project\cur\", s), images);
%images = ["E:\0tc1.jpg", "E:\0tc1n.jpg"];
[imagePoints, boardSize] = detectCheckerboardPoints(images);
imagePoints = squeeze(imagePoints);
squareSizeInMM = 20;
worldPoints = generateCheckerboardPoints(boardSize,squareSizeInMM);

I = imread("G:\Vista_project\cur\Vista_project8tc10153.jpg"); 
imageSize = [size(I, 1),size(I, 2)];
params = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
showReprojectionErrors(params);
figure;
showExtrinsics(params);
drawnow;

figure; 
imshow(I); 
hold on;
plot(imagePoints(:,1,1), imagePoints(:,2,1),'go');
plot(params.ReprojectedPoints(:,1,1),params.ReprojectedPoints(:,2,1),'r+');
legend('Detected Points','ReprojectedPoints');
hold off;