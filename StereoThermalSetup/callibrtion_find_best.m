all_path = 'G:\Vista_project\finish_deep\calibration\';
all_folders = ls(all_path);
all_folders = string(all_folders);
all_folders = all_folders(1:end-5);
folder_start = 6;
folders_num = length(all_folders)-2;
for i=1:folders_num
    cur_folder = folder_start+i-1;
    left_images = string(ls(string(all_path + string(cur_folder) + '\left_resized\')));
    left_images = left_images(3:end);
    filenum = cellfun(@(x)sscanf(x,'%d.png'), left_images);
    [~, left_idx] = sort(filenum);
    left_images_path = arrayfun(@(s) append(all_path + string(cur_folder) + '\left_resized\', s), left_images);
    left_images_path = left_images_path(left_idx);


    right_images = string(ls(all_path + string(cur_folder) + '\sync\'));
    right_images = right_images(3:end);
    filenum = cellfun(@(x)sscanf(x,'%d.png'), right_images);
    [~, right_idx] = sort(filenum);
    right_images_path = arrayfun(@(s) append(all_path + string(cur_folder) + '\sync\', s), right_images);
    right_images_path = right_images_path(right_idx);

    left_dest = all_path + string(cur_folder) + '\best_right_therm_rs_lt\';
    right_dest = all_path + string(cur_folder) + '\best_rs_rs_lt\';
    mkdir(left_dest);
    mkdir(right_dest);

    batch_sz = 20;
    last_idx = floor(length(left_images_path) / batch_sz);
    rm = rem(length(left_images_path), batch_sz);
    for k=1:(length(left_images_path) / batch_sz)
        if k == last_idx
            [imagePoints,boardSize] = detectCheckerboardPoints(left_images_path((k-1)*batch_sz+1:k*batch_sz+rm), right_images_path((k-1)*batch_sz+1:k*batch_sz+rm));
        else
            [imagePoints,boardSize] = detectCheckerboardPoints(left_images_path((k-1)*batch_sz+1:k*batch_sz+batch_sz), right_images_path((k-1)*batch_sz+1:k*batch_sz+batch_sz));
        end
        squareSize = 20;
        worldPoints = generateCheckerboardPoints(boardSize,squareSize);
        

        imageSize = [720,1280];
        stereoParams = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
        [all_errors] = myComputeReprojectionErrors(stereoParams);
         mean_pair_error = mean(sum(all_errors,2), 2);
        [mn, min_idx] = sort(mean_pair_error);

        n_min = 4;
        for n=1:length(min_idx(1:n_min))
            cur_idx = min_idx(n);
            copyfile(left_images_path((k-1)* batch_sz + cur_idx), left_dest);
            copyfile(right_images_path((k-1) * batch_sz + cur_idx), right_dest);
        end
    end
end


