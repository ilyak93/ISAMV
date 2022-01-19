
left_images = string(ls("G:\Vista_project\left\"));
left_images = left_images(3:end);
filenum = cellfun(@(x)sscanf(x,'%d.png'), left_images);
[~, left_idx] = sort(filenum);
left_images_path = arrayfun(@(s) append("G:\Vista_project\left\", s), left_images);
left_images_path = left_images_path(left_idx);


right_images = string(ls("G:\Vista_project\right\"));
right_images = right_images(3:end);
filenum = cellfun(@(x)sscanf(x,'%d.png'), right_images);
[~, right_idx] = sort(filenum);
right_images_path = arrayfun(@(s) append("G:\Vista_project\right\", s), right_images);
right_images_path = right_images_path(right_idx);

left_dest = "G:\Vista_project\left_best\";
right_dest = "G:\Vista_project\right_best\";


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

    I = imread("G:\Vista_project\cur\left_used\167_left.png"); 
    imageSize = [size(I,1),size(I,2)];
    stereoParams = estimateCameraParameters(imagePoints,worldPoints,'ImageSize',imageSize);
    [all_errors] = myComputeReprojectionErrors(stereoParams);
     mean_pair_error = mean(sum(all_errors,2), 2);
    [mn, min_idx] = sort(mean_pair_error);
    
    n_min = 3;
    for n=1:length(min_idx(1:n_min))
        cur_idx = min_idx(n);
        copyfile(left_images_path((k-1)* batch_sz + cur_idx), left_dest);
        copyfile(right_images_path((k-1) * batch_sz + cur_idx), right_dest);
    end
end


