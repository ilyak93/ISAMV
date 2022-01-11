images = string(ls("G:\Vista_project\cur\right_used\"));
images = images(3:end);
images_path = arrayfun(@(s) append("G:\Vista_project\cur\right_used\", s), images);

for i=1:length(images)
    image = imread(images_path(i));
    image = double(image);
    mx = max(max(image));
    mn = min(min(image));
    image = (image - mn) ./ (mx - mn) * 256;
    image = uint8(round(image));
    imwrite(image, "G:\Vista_project\cur\right_used\norm_"+images(i), 'png');
end
