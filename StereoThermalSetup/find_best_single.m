left_images = string(ls("G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_right\"));
left_images = left_images(3:end);
left_images = sort_nat(left_images);
left_images_path = arrayfun(@(s) append("G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_right\", s), left_images);

[imagePoints,boardSize] = detectCheckerboardPoints(left_images_path);

squareSize = 20;
worldPoints = generateCheckerboardPoints(boardSize,squareSize);

imageSize = [720, 1280];
camera1Params = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
[all_errors] = myComputeReprojectionErrors(camera1Params);
showReprojectionErrors(camera1Params);
figure;
showExtrinsics(camera1Params);
mean_pair_error = mean(sum(all_errors,2), 2);
[mn, min_idx] = sort(mean_pair_error);

left_dest = "G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_right_sorted\";

mkdir(left_dest);


n_min = length(min_idx);
for n=1:length(min_idx(1:n_min))
    cur_idx = min_idx(n);
    copyfile(left_images_path(cur_idx), left_dest);
    movefile(left_dest+left_images(cur_idx), left_dest+n+"m.png")
end
