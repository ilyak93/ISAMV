images = string(ls("G:\Vista_project\cur\left_used\"));
images = images(3:end);
images_path = arrayfun(@(s) append("G:\Vista_project\cur\left_used\", s), images);

for i=1:length(images)
    image = imread(images_path(i));
    mx = max(max(image));
    image = mx - image;
    imwrite(image, "G:\Vista_project\cur\left_used\inv_"+images(i), 'png');
end
