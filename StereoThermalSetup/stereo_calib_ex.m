left_images = string(ls("G:\Vista_project\calib_brown\left\"));
left_images = left_images(3:end);
left_images = arrayfun(@(s) append("G:\Vista_project\calib_brown\left\", s), left_images);

right_images = string(ls("G:\Vista_project\calib_brown\right\"));
right_images = right_images(3:end);
right_images = arrayfun(@(s) append("G:\Vista_project\calib_brown\right\", s), right_images);

for i=1:length(left_images)
    [~,boardSize] = detectCheckerboardPoints(left_images(i), right_images(i));
    if (boardSize(1) ~= 8 || boardSize(2) ~= 10)
        delete(left_images(i));
        delete(right_images(i));
    end
end 