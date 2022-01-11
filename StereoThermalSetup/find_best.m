left_images = string(ls("G:\Vista_project\cur\left\"));
left_images = left_images(3:end);
left_images_fp = arrayfun(@(s) append("G:\Vista_project\cur\left\", s), left_images);

right_images = string(ls("G:\Vista_project\cur\right\"));
right_images = right_images(3:end);
right_images_fp = arrayfun(@(s) append("G:\Vista_project\cur\right\", s), right_images);

errors = zeros(1,length(left_images));

for i=1:length(left_images)
    left = [left_images_fp(i), left_images_fp(i)];
    right = [right_images_fp(i), right_images_fp(i)];
    [imagePoints,boardSize] = detectCheckerboardPoints(left, right);
    
    squareSize = 20;
    worldPoints = generateCheckerboardPoints(boardSize,squareSize);

    I = imread(left_images_fp(i)); 
    imageSize = [size(I,1),size(I,2)];
    stereoParams = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
    errors(i) = stereoParams.MeanReprojectionError;
end

[sorted_errors, indices] = sort(errors);

for i=1:10
    movefile(left_images_fp(indices(i)), "G:\Vista_project\cur\left_used\"+left_images(indices(i)))
    movefile(right_images_fp(indices(i)), "G:\Vista_project\cur\right_used\"+right_images(indices(i)))
end
