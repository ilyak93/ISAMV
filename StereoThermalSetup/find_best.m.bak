left_images = string(ls("G:\Vista_project\CALIB_NEW\left_best\"));
left_images = left_images(3:end-1);
left_images_path = arrayfun(@(s) append("G:\Vista_project\CALIB_NEW\left_best\", s), left_images);


right_images = string(ls("G:\Vista_project\CALIB_NEW\right_best\"));
right_images = right_images(3:end-1);
right_images_path = arrayfun(@(s) append("G:\Vista_project\CALIB_NEW\right_best\", s), right_images);


[imagePoints,boardSize] = detectCheckerboardPoints(left_images_path, right_images_path);

squareSize = 20;
worldPoints = generateCheckerboardPoints(boardSize,squareSize);

imageSize = [512,640];
stereoParams = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
[all_errors] = myComputeReprojectionErrors(stereoParams);
 mean_pair_error = mean(sum(all_errors,2), 2);
[mn, min_idx] = sort(mean_pair_error);

left_dest = "G:\Vista_project\CALIB_NEW\left_best\left\";
right_dest = "G:\Vista_project\CALIB_NEW\right_best\right\";

n_min = 60;
for n=1:length(min_idx(1:n_min))
    cur_idx = min_idx(n);
    copyfile(left_images_path(cur_idx), left_dest);
    copyfile(right_images_path(cur_idx), right_dest);
end