all_path = 'G:\Vista_project\finish_deep\calibration\\single_camera_callibration\';
all_folders = ls(all_path);
all_folders = string(all_folders);
all_folders = all_folders(3:end-1);
folder_start = 6;
folders_num = length(all_folders);
for i=1:folders_num
    cur_folder = folder_start+i-1;
    left_images = string(ls(string(all_path + string(cur_folder) + '\right_resized\')));
    left_images = left_images(3:end);
    filenum = cellfun(@(x)sscanf(x,'%d.png'), left_images);
    [~, left_idx] = sort(filenum);
    left_images_path = arrayfun(@(s) append(all_path + string(cur_folder) + '\right_resized\', s), left_images);
    left_images_path = left_images_path(left_idx);


    left_dest = all_path + string(cur_folder) + '\best_right\';

    mkdir(left_dest);

    batch_sz = 20;
    last_idx = floor(length(left_images_path) / batch_sz);
    rm = rem(length(left_images_path), batch_sz);
    for k=1:(length(left_images_path) / batch_sz)
        if k == last_idx
            [imagePoints,boardSize] = detectCheckerboardPoints(left_images_path((k-1)*batch_sz+1:k*batch_sz+rm));
        else
            [imagePoints,boardSize] = detectCheckerboardPoints(left_images_path((k-1)*batch_sz+1:k*batch_sz+batch_sz));
        end
        squareSize = 20;
        worldPoints = generateCheckerboardPoints(boardSize,squareSize);
        

        imageSize = [720,1280];
        cameraParams = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
        [all_errors] = myComputeReprojectionErrors(cameraParams);
        [mn, min_idx] = sort(all_errors);

        n_min = 6;
        for n=1:length(min_idx(1:n_min))
            cur_idx = min_idx(n);
            copyfile(left_images_path((k-1)* batch_sz + cur_idx), left_dest);
        end
    end
end


