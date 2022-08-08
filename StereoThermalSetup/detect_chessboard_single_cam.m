for t=6:13
    left_images = string(ls("G:\Vista_project\finish_deep\calibration\single_camera_callibration\"+t+"\right_resized\"));
    left_images = left_images(3:end);
    left_images = sort_nat(left_images);
    left_images = arrayfun(@(s) append("G:\Vista_project\finish_deep\calibration\single_camera_callibration\"+t+"\right_resized\", s), left_images);
    

    for i=1:length(left_images)
        [~,boardSize] = detectCheckerboardPoints(left_images(i));
        if (boardSize(1) ~= 8 || boardSize(2) ~= 10)
            delete(left_images(i));
        end
    end 
end