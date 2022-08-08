camera1Params = load('G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_single_6_therm_6_rs_stereo_4\best_stereo_300_best_single\single_cam_calib\camera1p.mat').camera1Params;
camera2Params = load('G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_single_6_therm_6_rs_stereo_4\best_stereo_300_best_single\single_cam_calib\camera2p.mat').camera2Params;

left_images = string(ls("G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_single_6_therm_6_rs_stereo_4\best_stereo_300_best_single\best_stereo\best_left\"));
left_images = left_images(3:end);
left_images = sort_nat(left_images);
left_images_path = arrayfun(@(s) append("G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_single_6_therm_6_rs_stereo_4\best_stereo_300_best_single\best_stereo\best_left\", s), left_images);


right_images = string(ls("G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_single_6_therm_6_rs_stereo_4\best_stereo_300_best_single\best_stereo\best_sync\"));
right_images = right_images(3:end);
right_images = sort_nat(right_images);
right_images_path = arrayfun(@(s) append("G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_single_6_therm_6_rs_stereo_4\best_stereo_300_best_single\best_stereo\best_sync\", s), right_images);


[imagePoints,boardSize] = detectCheckerboardPoints(left_images_path, right_images_path);

squareSize = 20;
worldPoints = generateCheckerboardPoints(boardSize,squareSize);

stereoParams = estimateStereoBaseline(imagePoints, worldPoints, camera1Params, camera2Params);
[all_errors] = myComputeReprojectionErrors(stereoParams);
showReprojectionErrors(stereoParams);
figure;
showExtrinsics(stereoParams);
mean_pair_error = mean(sum(all_errors,2), 2);
[mn, min_idx] = sort(mean_pair_error);

left_dest = "G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_single_6_therm_6_rs_stereo_4\best_stereo_300_best_single\best_stereo\best_left_sorted\";
right_dest = "G:\Vista_project\finish_deep\calibration\single_camera_callibration\best_single_6_therm_6_rs_stereo_4\best_stereo_300_best_single\best_stereo\best_sync_sorted\";
mkdir(left_dest);
mkdir(right_dest);

n_min = length(min_idx);
for n=1:length(min_idx(1:n_min))
    cur_idx = min_idx(n);
    copyfile(left_images_path(cur_idx), left_dest);
    movefile(left_dest+left_images(cur_idx), left_dest+n+"m.png")
    copyfile(right_images_path(cur_idx), right_dest);
    movefile(right_dest+right_images(cur_idx), right_dest+n+"m.png")
end