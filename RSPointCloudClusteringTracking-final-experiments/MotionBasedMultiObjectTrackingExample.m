function MotionBasedMultiObjectTrackingExample()
    
    % Create System objects used for reading video, detecting moving objects,
    % and displaying the results.
    obj = mySetupSystemObjects();

    %envCfg = coder.gpuEnvConfig('host');
    %envCfg.DeepLibTarget = 'cudnn';
    %envCfg.DeepCodegen = 1;
    %envCfg.Quiet = 1;
    %coder.checkGpuInstall(envCfg);
    
    tracks = initializeTracks(); % Create an empty array of tracks.
    
    nextId = 1; % ID of the next track
    
    color_keys = arrayfun(@(x) num2str(x), 1:147, 'UniformOutput', false);
    
    color_values = {[0.9411764705882353, 0.9725490196078431, 1.0], [0.9803921568627451, 0.9215686274509803, 0.8431372549019608], [0.0, 1.0, 1.0], [0.4980392156862745, 1.0, 0.8313725490196079], [0.9411764705882353, 1.0, 1.0], [0.9607843137254902, 0.9607843137254902, 0.8627450980392157], [1.0, 0.8941176470588236, 0.7686274509803922], [1.0, 0.9215686274509803, 0.803921568627451], [0.0, 0.0, 1.0], [0.5411764705882353, 0.16862745098039217, 0.8862745098039215], [0.6470588235294118, 0.16470588235294117, 0.16470588235294117], [0.8705882352941177, 0.7215686274509804, 0.5294117647058824], [0.37254901960784315, 0.6196078431372549, 0.6274509803921569], [0.4980392156862745, 1.0, 0.0], [0.8235294117647058, 0.4117647058823529, 0.11764705882352941], [1.0, 0.4980392156862745, 0.3137254901960784], [0.39215686274509803, 0.5843137254901961, 0.9294117647058824], [1.0, 0.9725490196078431, 0.8627450980392157], [0.8627450980392157, 0.0784313725490196, 0.23529411764705882], [0.0, 1.0, 1.0], [0.0, 0.0, 0.5450980392156862], [0.0, 0.5450980392156862, 0.5450980392156862], [0.7215686274509804, 0.5254901960784314, 0.043137254901960784], [0.6627450980392157, 0.6627450980392157, 0.6627450980392157], [0.0, 0.39215686274509803, 0.0], [0.6627450980392157, 0.6627450980392157, 0.6627450980392157], [0.7411764705882353, 0.7176470588235294, 0.4196078431372549], [0.5450980392156862, 0.0, 0.5450980392156862], [0.3333333333333333, 0.4196078431372549, 0.1843137254901961], [1.0, 0.5490196078431373, 0.0], [0.6, 0.19607843137254902, 0.8], [0.5450980392156862, 0.0, 0.0], [0.9137254901960784, 0.5882352941176471, 0.47843137254901963], [0.5607843137254902, 0.7372549019607844, 0.5607843137254902], [0.2823529411764706, 0.23921568627450981, 0.5450980392156862], [0.1843137254901961, 0.30980392156862746, 0.30980392156862746], [0.1843137254901961, 0.30980392156862746, 0.30980392156862746], [0.0, 0.807843137254902, 0.8196078431372549], [0.5803921568627451, 0.0, 0.8274509803921568], [1.0, 0.0784313725490196, 0.5764705882352941], [0.0, 0.7490196078431373, 1.0], [0.4117647058823529, 0.4117647058823529, 0.4117647058823529], [0.4117647058823529, 0.4117647058823529, 0.4117647058823529], [0.11764705882352941, 0.5647058823529412, 1.0], [0.6980392156862745, 0.13333333333333333, 0.13333333333333333], [1.0, 0.9803921568627451, 0.9411764705882353], [0.13333333333333333, 0.5450980392156862, 0.13333333333333333], [1.0, 0.0, 1.0], [0.8627450980392157, 0.8627450980392157, 0.8627450980392157], [0.9725490196078431, 0.9725490196078431, 1.0], [1.0, 0.8431372549019608, 0.0], [0.8549019607843137, 0.6470588235294118, 0.12549019607843137], [0.5019607843137255, 0.5019607843137255, 0.5019607843137255], [0.0, 0.5019607843137255, 0.0], [0.6784313725490196, 1.0, 0.1843137254901961], [0.5019607843137255, 0.5019607843137255, 0.5019607843137255], [0.9411764705882353, 1.0, 0.9411764705882353], [1.0, 0.4117647058823529, 0.7058823529411765], [0.803921568627451, 0.3607843137254902, 0.3607843137254902], [0.29411764705882354, 0.0, 0.5098039215686274], [1.0, 1.0, 0.9411764705882353], [0.9411764705882353, 0.9019607843137255, 0.5490196078431373], [0.9019607843137255, 0.9019607843137255, 0.9803921568627451], [1.0, 0.9411764705882353, 0.9607843137254902], [0.48627450980392156, 0.9882352941176471, 0.0], [1.0, 0.9803921568627451, 0.803921568627451], [0.6784313725490196, 0.8470588235294118, 0.9019607843137255], [0.9411764705882353, 0.5019607843137255, 0.5019607843137255], [0.8784313725490196, 1.0, 1.0], [0.9803921568627451, 0.9803921568627451, 0.8235294117647058], [0.8274509803921568, 0.8274509803921568, 0.8274509803921568], [0.5647058823529412, 0.9333333333333333, 0.5647058823529412], [0.8274509803921568, 0.8274509803921568, 0.8274509803921568], [1.0, 0.7137254901960784, 0.7568627450980392], [1.0, 0.6274509803921569, 0.47843137254901963], [0.12549019607843137, 0.6980392156862745, 0.6666666666666666], [0.5294117647058824, 0.807843137254902, 0.9803921568627451], [0.4666666666666667, 0.5333333333333333, 0.6], [0.4666666666666667, 0.5333333333333333, 0.6], [0.6901960784313725, 0.7686274509803922, 0.8705882352941177], [1.0, 1.0, 0.8784313725490196], [0.0, 1.0, 0.0], [0.19607843137254902, 0.803921568627451, 0.19607843137254902], [0.9803921568627451, 0.9411764705882353, 0.9019607843137255], [1.0, 0.0, 1.0], [0.5019607843137255, 0.0, 0.0], [0.4, 0.803921568627451, 0.6666666666666666], [0.0, 0.0, 0.803921568627451], [0.7294117647058823, 0.3333333333333333, 0.8274509803921568], [0.5764705882352941, 0.4392156862745098, 0.8588235294117647], [0.23529411764705882, 0.7019607843137254, 0.44313725490196076], [0.4823529411764706, 0.40784313725490196, 0.9333333333333333], [0.0, 0.9803921568627451, 0.6039215686274509], [0.2823529411764706, 0.8196078431372549, 0.8], [0.7803921568627451, 0.08235294117647059, 0.5215686274509804], [0.09803921568627451, 0.09803921568627451, 0.4392156862745098], [0.9607843137254902, 1.0, 0.9803921568627451], [1.0, 0.8941176470588236, 0.8823529411764706], [1.0, 0.8941176470588236, 0.7098039215686275], [1.0, 0.8705882352941177, 0.6784313725490196], [0.0, 0.0, 0.5019607843137255], [0.9921568627450981, 0.9607843137254902, 0.9019607843137255], [0.5019607843137255, 0.5019607843137255, 0.0], [0.4196078431372549, 0.5568627450980392, 0.13725490196078433], [1.0, 0.6470588235294118, 0.0], [1.0, 0.27058823529411763, 0.0], [0.8549019607843137, 0.4392156862745098, 0.8392156862745098], [0.9333333333333333, 0.9098039215686274, 0.6666666666666666], [0.596078431372549, 0.984313725490196, 0.596078431372549], [0.6862745098039216, 0.9333333333333333, 0.9333333333333333], [0.8588235294117647, 0.4392156862745098, 0.5764705882352941], [1.0, 0.9372549019607843, 0.8352941176470589], [1.0, 0.8549019607843137, 0.7254901960784313], [0.803921568627451, 0.5215686274509804, 0.24705882352941178], [1.0, 0.7529411764705882, 0.796078431372549], [0.8666666666666667, 0.6274509803921569, 0.8666666666666667], [0.6901960784313725, 0.8784313725490196, 0.9019607843137255], [0.5019607843137255, 0.0, 0.5019607843137255], [0.4, 0.2, 0.6], [1.0, 0.0, 0.0], [0.7372549019607844, 0.5607843137254902, 0.5607843137254902], [0.2549019607843137, 0.4117647058823529, 0.8823529411764706], [0.5450980392156862, 0.27058823529411763, 0.07450980392156863], [0.9803921568627451, 0.5019607843137255, 0.4470588235294118], [0.9568627450980393, 0.6431372549019608, 0.3764705882352941], [0.1803921568627451, 0.5450980392156862, 0.3411764705882353], [1.0, 0.9607843137254902, 0.9333333333333333], [0.6274509803921569, 0.3215686274509804, 0.17647058823529413], [0.7529411764705882, 0.7529411764705882, 0.7529411764705882], [0.5294117647058824, 0.807843137254902, 0.9215686274509803], [0.41568627450980394, 0.35294117647058826, 0.803921568627451], [0.4392156862745098, 0.5019607843137255, 0.5647058823529412], [0.4392156862745098, 0.5019607843137255, 0.5647058823529412], [1.0, 0.9803921568627451, 0.9803921568627451], [0.0, 1.0, 0.4980392156862745], [0.27450980392156865, 0.5098039215686274, 0.7058823529411765], [0.8235294117647058, 0.7058823529411765, 0.5490196078431373], [0.0, 0.5019607843137255, 0.5019607843137255], [0.8470588235294118, 0.7490196078431373, 0.8470588235294118], [1.0, 0.38823529411764707, 0.2784313725490196], [0.25098039215686274, 0.8784313725490196, 0.8156862745098039], [0.9333333333333333, 0.5098039215686274, 0.9333333333333333], [0.9607843137254902, 0.8705882352941177, 0.7019607843137254], [1.0, 1.0, 1.0], [0.9607843137254902, 0.9607843137254902, 0.9607843137254902], [1.0, 1.0, 0.0], [0.6039215686274509, 0.803921568627451, 0.19607843137254902]};
    color_values = flip(color_values);
    colors_dict= containers.Map(color_keys, color_values);
    % Detect moving objects, and track them across video frames.
    idxxx = 1;
    while hasFrame(obj.reader)
        [orig_image, depth, xyz_matrix_st] = readFrame(obj.reader);
        xyz_matrix = xyz_matrix_st.xyz_matrix;
        my_zero_mask = depth ~= 0;
        mx = max(max(xyz_matrix(:,:,3)));
        %xyz_rev = depth;
        xyz_rev = mx - xyz_matrix;
        %ptc = pointCloud(xyz_matrix,'Color',orig_image);
        %indices_actual_points = find(sum(xyz_vector ~= 0,2) == 3);
        frame = orig_image;
        %frame = cat(3, xyz_matrix, sc_image);
        %[centroids, bboxes, mask] = detectObjects(frame, my_zero_mask);
        [centroids, bboxes, mask] = detectObjects2(orig_image, xyz_matrix, my_zero_mask);
        
        %mask = my_zero_mask & mask;
        %pcshow(ptc)
        %[centroids, bboxes] = detectObjectsHorizontalCam(orig_image, depth, ptc);
        %frame2 = reshape(frame, [sz(1)*sz(2),3]);
        
        predictNewLocationsOfTracks();
        [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment();
    
        updateAssignedTracks();
        updateUnassignedTracks();
        deleteLostTracks();
        createNewTracks();
        displayTrackingResults();
        %myDisplayTrackingResults(orig_image);
        pause(1);
    end


function obj = setupSystemObjects()
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the tracked
        % objects in each frame, and playing the video.

        % Create a video reader.
        obj.reader = VideoReader('atrium.mp4');

        % Create two video players, one to display the video,
        % and one to display the foreground mask.
        
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);

        % Create System objects for foreground detection and blob analysis

        % The foreground detector is used to segment moving objects from
        % the background. It outputs a binary mask, where the pixel value
        % of 1 corresponds to the foreground and the value of 0 corresponds
        % to the background.

        obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);

        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.  The blob analysis System object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, centroid, and the bounding box.

        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 400);
end

function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
end

 function [centroids, bboxes, mask] = detectObjects(frame, zero_mask)

        % Detect foreground.
        mask = obj.detector.step(frame);
        %mask = reshape(mask, 720, 1280, 2);
        
        % Apply morphological operations to remove noise and fill in holes.
        mask = imopen(mask, strel('rectangle', [3,3]));
        mask = imclose(mask, strel('rectangle', [15, 15]));
        mask = imfill(mask, 'holes');
        mask = mask & zero_mask;
        

        % Perform blob analysis to find connected components.
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
 end

 function [centroids, bboxes, mask] = detectObjects2(frame, depth, zero_mask)

        % Detect foreground.
        rgb_mask = obj.detector.step(frame);
        depth_mask = obj.detector.step(frame);
        mask = rgb_mask & depth_mask;
        mask = zero_mask & mask;
        %mask = reshape(mask, 720, 1280, 2);
        
        % Apply morphological operations to remove noise and fill in holes.
        mask = imopen(mask, strel('rectangle', [3,3]));
        mask = imclose(mask, strel('rectangle', [15, 15]));
        mask = imfill(mask, 'holes');
        

        % Perform blob analysis to find connected components.
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
 end

    function [ptCloudOut,obstacleIndices,groundIndices] = removeGroundPlane(ptCloudIn,maxGroundDist,referenceVector,maxAngularDist, cur_indices)
        % This method removes the ground plane from point cloud using
        % pcfitplane.
        [~,groundIndices,outliers] = pcfitplane(ptCloudIn,maxGroundDist,referenceVector,maxAngularDist);
        ptCloudOut = select(ptCloudIn,outliers);
        obstacleIndices = cur_indices(outliers);
        groundIndices = cur_indices(groundIndices);
    end

    function [centroids, bboxes] = detectObjectsHorizontalCam(frame, depth_frame, ptc)

        GroundMaxDistance = 1;
        % GroundReferenceVector Reference vector of ground plane
        GroundReferenceVector = [0 0 1];
        % GroundMaxAngularDistance Maximum angular distance of point to reference vector
        GroundMaxAngularDistance = 5;

        % Detect objects by clustering.
        epsilon = 0.001;
        min_points = 2;

        pcseg = 1;
        dwsmpl = 0;
        
        gridStep = 0.1;
        %zero_pixels = sum(frame ~= 0, 3) > 0;
        [ds_ptc, ds_indices] = pcdownsample(ptc,'random',gridStep);
        %[ds_ptc, ds_indices] = removeGroundPlane(ds_ptc, GroundMaxDistance, GroundReferenceVector,  GroundMaxAngularDistance, ds_indices);
        %pcshow(ds_ptc)
        ds_xyz = ds_ptc.Location ; 
        nx = ds_ptc.Location(:, 1);
        ny = ds_ptc.Location(:, 2);
        nz = ds_ptc.Location(:, 3);
        ncolor = double(ds_ptc.Color);
        nxyz = cat(2, nx,ny,nz);
        xyz_rgb = cat(2, nx,ny,nz, ncolor);
        xyz_rgb = reshape(xyz_rgb, [], 6);
        non_zero_idx = nz ~= 0;
        xyz_rgb = xyz_rgb(non_zero_idx);
        
        color = double(ds_ptc.Color);
        %objects = dbscan(xyz_rgb, epsilon, min_points);
        %objects = spectralcluster(xyz_rgb, 40);
        %xyz_rgb = cat(2, ds_xyz, double(ds_ptc.Color) / 255);
 
        
        %objects = kmeans(nxyz, 100);
        
        %objects = clusterdata(xyz_rgb, 30);
        %objects = clusterdata(xyz_rgb,'Linkage','ward','SaveMemory','off','maxclust',40);
        %objects = knncluster(ds_xyz, 20);
        %gmfit = fitgmdist(ds_xyz, 20, 'SharedCov', true(1));
        %gmfit = fitgmdist(xyz_rgb, 40, 'Regularize', 0.0000001);
        %objects = cluster(gmfit, xyz_rgb);
        
        
        notnan = ~isnan(ptc.Location(:,:,3));
        locs = ptc.Location(:,:,3);
        locs = locs(notnan);
        minDistance = 0.055;
        minPoints = 2;
        
        %xds = datastats(reshape(locs, [], 1));
        %q = quantile(reshape(locs, [], 1),5);
        %far_idx = ptc.Location(:,:,3) < 15;
        %far_ptc = select(ptc, far_idx);
        %pcshow(far_ptc)
        [objects, numClusters] = pcsegdist(ds_ptc, minDistance, "NumClusterPoints", minPoints);
        %far_objects = zeros(720*1280, 1);
        %far_objects(reshape(far_idx,1, [])) = objects;
        %objects = far_objects;
        

        %figure
        %colormap(hsv(numClusters))
        %pcshow(ptc.Location, objects)
        %title('Point Cloud Clusters')
        %}
        unq = length(unique(objects(~isnan(objects))));
        mod_objects = objects;
        if pcseg ~= 1
            mod_objects(mod_objects == 0) = unq + 1;
            mod_objects(isnan(mod_objects)) = 0;
            
            objs = zeros(720 * 1280, 1) ;
            objs(ds_indices(non_zero_idx), :) = mod_objects ;
            objs = reshape(objs, 720, 1280, 1) ;
            %nonzero_pixels2 = objs == 0;
        else
            objs = zeros(720 * 1280, 1) ;
            objs(ds_indices(non_zero_idx), :) = mod_objects(non_zero_idx) ;
            objs = reshape(objs, 720, 1280, 1) ;
        end
        %{
        upsampled_objs = zeros(720, 1280);
        if dwsmpl == 1
            upsampled_objs(ds_indices) = objs;
            objs = upsampled_objs;
        end
        %}
        %pcshow(ds_ptc)
        %{
        ds_xyz = ptc.Location;
        ncolor = double(ptc.Color) / 200.;
        %nxyz = cat(2, nx,ny,nz);
        full_xyz_rgb = cat(3, ds_xyz, ncolor);
        full_xyz_rgb = reshape(full_xyz_rgb, [], 3);
        not_clustered_points = full_xyz_rgb(zero_pixels & nonzero_pixels2, :);
        clustered_points = full_xyz_rgb(ds_indices, :);
        idx = knnsearch(clustered_points, not_clustered_points);
        objs(zero_pixels & nonzero_pixels2) = objs(idx);
        %}
        stats = regionprops(objs, 'Centroid', 'BoundingBox', 'Area');
        areas = cat(1,stats.Area);
        
        [sorted_areas, cluster_indices] = sort(areas, 'descend');
        clusters_show = zeros(720*1280, 3);
        for i=1:min(length(colors_dict), unq-1)
            color = colors_dict(num2str(i));
            clusters_show(objs == cluster_indices(i), 1) = color(1);
            clusters_show(objs == cluster_indices(i), 2) = color(2);
            clusters_show(objs == cluster_indices(i), 3) = color(3);
        end
        imshow(reshape(clusters_show, 720,1280,3));
        clusters_img = reshape(clusters_show, 720,1280,3);
        imwrite(clusters_img, "./f/"+string(idxxx)+"c.png");
        
        stats = stats((areas / (length(ds_xyz) / 1)  > 0.01) & (areas / (length(ds_xyz) / 1)  < 0.2));
        
        centroids = cat(1,stats.Centroid);
        bboxes = cat(1,stats.BoundingBox);
    end
    
    function [centroids, bboxes] = detectObjectsHorizontalCam2(color_frame, depth_frame, ptc)
    
            GroundMaxDistance = 1;
            % GroundReferenceVector Reference vector of ground plane
            GroundReferenceVector = [0 0 1];
            % GroundMaxAngularDistance Maximum angular distance of point to reference vector
            GroundMaxAngularDistance = 5;
            pcseg = 0;
            % Detect objects by clustering.
            epsilon = 0.01;
            min_points = 5;

            gridStep = 1 ;
            [ds_ptc, ds_indices] = pcdownsample(ptc,'random',gridStep);
            color_frame = reshape(color_frame, [], 3);
            depth_frame = reshape(depth_frame, [], 1);
            ds_color = color_frame(ds_indices, :);
            ds_depth = depth_frame(ds_indices);  
            non_zero_idx = ds_depth ~= 0;
            ds_color = ds_color(non_zero_idx);
            ds_depth = ds_depth(non_zero_idx);
            rgbd = cat(2, double(ds_color) / 75, double(ds_depth) / 1000);
            %rgbd = reshape(rgbd, [], 4);
            
            %objects = dbscan(rgbd, epsilon, min_points);
            %objects = spectralcluster(xyz_rgb, 40);
            %xyz_rgb = cat(2, ds_xyz, double(ds_ptc.Color) / 255);
     
            
            objects = kmeans(rgbd, 40);
            
            %objects = clusterdata(xyz_rgb, 30);
            %objects = clusterdata(xyz_rgb,'Linkage','ward','SaveMemory','off','maxclust',40);
            %objects = knncluster(ds_xyz, 20);
            %gmfit = fitgmdist(ds_xyz, 20, 'SharedCov', true(1));
            %gmfit = fitgmdist(xyz_rgb, 40, 'Regularize', 0.0000001);
            %objects = cluster(gmfit, xyz_rgb);
            
            
            %notnan = ~isnan(ptc.Location(:,:,3));
            %locs = ptc.Location(:,:,3);
            %locs = locs(notnan);
            %minDistance = 0.02;
            %minPoints = 10;
            
            %xds = datastats(reshape(locs, [], 1));
            %q = quantile(reshape(locs, [], 1),5);
            %far_idx = ptc.Location(:,:,3) < 15;
            %far_ptc = select(ptc, far_idx);
            %pcshow(far_ptc)
            %[objects, numClusters] = pcsegdist(ptc, minDistance, "NumClusterPoints", minPoints);
            %far_objects = zeros(720*1280, 1);
            %far_objects(reshape(far_idx,1, [])) = objects;
            %objects = far_objects;
            
    
            %figure
            %colormap(hsv(numClusters))
            %pcshow(ptc.Location, objects)
            %title('Point Cloud Clusters')
            %}
            unq = length(unique(objects(~isnan(objects))));
            mod_objects = objects;
            if pcseg ~= 1
                mod_objects(mod_objects == 0) = unq + 1;
                mod_objects(isnan(mod_objects)) = 0;
                
                objs = zeros(720 * 1280, 1) ;
                objs(ds_indices(non_zero_idx), :) = mod_objects ;
                objs = reshape(objs, 720, 1280, 1) ;
                %nonzero_pixels2 = objs == 0;
            else
                objs = mod_objects;
            end
            %{
            upsampled_objs = zeros(720, 1280);
            if dwsmpl == 1
                upsampled_objs(ds_indices) = objs;
                objs = upsampled_objs;
            end
            %}
            %pcshow(ds_ptc)
            %{
            ds_xyz = ptc.Location;
            ncolor = double(ptc.Color) / 200.;
            %nxyz = cat(2, nx,ny,nz);
            full_xyz_rgb = cat(3, ds_xyz, ncolor);
            full_xyz_rgb = reshape(full_xyz_rgb, [], 3);
            not_clustered_points = full_xyz_rgb(zero_pixels & nonzero_pixels2, :);
            clustered_points = full_xyz_rgb(ds_indices, :);
            idx = knnsearch(clustered_points, not_clustered_points);
            objs(zero_pixels & nonzero_pixels2) = objs(idx);
            %}
            stats = regionprops(objs, 'Centroid', 'BoundingBox', 'Area');
            areas = cat(1,stats.Area);
            
            [sorted_areas, cluster_indices] = sort(areas, 'descend');
            clusters_show = zeros(720*1280, 3);
            for i=1:min(length(colors_dict), unq-1)
                color = colors_dict(num2str(i));
                clusters_show(objs == cluster_indices(i), 1) = color(1);
                clusters_show(objs == cluster_indices(i), 2) = color(2);
                clusters_show(objs == cluster_indices(i), 3) = color(3);
            end
            imshow(reshape(clusters_show, 720,1280,3));
            clusters_img = reshape(clusters_show, 720,1280,3);
            
            
            stats = stats((areas / (length(ds_indices) / 1)  > 0.01) & (areas / (length(ds_indices) / 1)  < 0.2));
            
            centroids = cat(1,stats.Centroid);
            bboxes = cat(1,stats.BoundingBox);
    end


    




function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;

            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);

            % Shift the bounding box so that its center is at
            % the predicted location.
            predictedCentroid = int32(predictedCentroid) - int32(round(bbox(3:4))) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
end

function [assignments,  unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()

        nTracks = length(tracks);
        nDetections = size(centroids, 1);

        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end

        % Solve the assignment problem.
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
end

 function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);

            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);

            % Replace predicted bounding box with detected
            % bounding box.
            tracks(trackIdx).bbox = bbox;

            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;

            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
    end


function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
end

 function deleteLostTracks()
        if isempty(tracks)
            return;
        end

        invisibleForTooLong = 20;
        ageThreshold = 8;

        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;

        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;

        % Delete lost tracks.
        tracks = tracks(~lostInds);
 end

 function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);

        for i = 1:size(centroids, 1)

            centroid = centroids(i,:);
            bbox = bboxes(i, :);

            % Create a Kalman filter object.
            % configureKalmanFilter(MotionModel,InitialLocation,InitialEstimateError,MotionNoise,MeasurementNoise)
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);

            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);

            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;

            % Increment the next id.
            nextId = nextId + 1;
        end
 end

 function myDisplayTrackingResults(image)
        % Convert the frame and the mask to uint8 RGB.
        image = im2uint8(image);

        minVisibleCount = 8;
        if ~isempty(tracks)

            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);

            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);

                % Get ids.
                ids = int32([reliableTracks(:).id]);

                % Create labels for objects indicating the ones for
                % which we display the predicted rather than the actual
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);

                % Draw the objects on the frame.
                image = insertObjectAnnotation(image, 'rectangle', ...
                    bboxes, labels);
                imwrite(image, "./f/"+string(idxxx)+"t.png");
                idxxx = idxxx+1;
            end
        end

        % Display the mask and the frame.
        obj.videoPlayer.step(image);
 end

function displayTrackingResults()
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;

        minVisibleCount = 8;
        if ~isempty(tracks)

            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);

            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);
                
                %{
                cuboid = pcfitcuboid(pointCloud(thisPointData));
                yaw = cuboid.Orientation(3);
                L = cuboid.Dimensions(1);
                W = cuboid.Dimensions(2);
                H = cuboid.Dimensions(3);
                if abs(yaw) > 45
                    possibles = yaw + [-90;90];
                    [~,toChoose] = min(abs(possibles));
                    yaw = possibles(toChoose);
                    temp = L;
                    L = W;
                    W = temp;
                end
                bboxes(:,i) = [cuboid.Center yaw L W H]';
                %}
                % Get ids.
                ids = int32([reliableTracks(:).id]);

                % Create labels for objects indicating the ones for
                % which we display the predicted rather than the actual
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);

                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);

                % Draw the objects on the mask.
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
            end
        end

        % Display the mask and the frame.
        obj.maskPlayer.step(mask);
        obj.videoPlayer.step(frame);
    end
end