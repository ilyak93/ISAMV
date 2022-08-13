classdef myVideoReader < handle
    properties
        png_files_names
        mat_files_names
        d_files_names
        cur_index
        size
    end
   methods
       function obj = myVideoReader(pth)
          all_files = dir(pth);
          all_files = all_files(3:end);
          png_indices = arrayfun(@(x) contains(x.name, "color"), all_files);
          d_indices = arrayfun(@(x) contains(x.name, "depth"), all_files);
          xyz_indices = arrayfun(@(x) contains(x.name, "mat"), all_files);
          obj.png_files_names = all_files(png_indices);
          obj.d_files_names = all_files(d_indices);
          obj.mat_files_names = all_files(xyz_indices);

          png_names = extractfield(obj.png_files_names,'name');
          filenum = cellfun(@(x) sscanf(x,'%dcolor.png'), png_names);
          % sort them, and get the sorting order
          [~,png_indices] = sort(filenum);
          % use to this sorting order to sort the filenames
          obj.png_files_names = obj.png_files_names(png_indices);   
        
          d_names = extractfield(obj.d_files_names,'name');
          filenum = cellfun(@(x) sscanf(x,'%ddepth.png'), d_names);
          % sort them, and get the sorting order
          [~,d_indices] = sort(filenum);
          % use to this sorting order to sort the filenames
          obj.d_files_names = obj.d_files_names(d_indices);

          mat_names = extractfield(obj.mat_files_names,'name');
          filenum = cellfun(@(x) sscanf(x,'%d.mat'), mat_names);
          % sort them, and get the sorting order
          [~,mat_indices] = sort(filenum);
          % use to this sorting order to sort the filenames
          obj.mat_files_names = obj.mat_files_names(mat_indices);

          obj.cur_index = 25;
          obj.size = length(obj.png_files_names);
       end

      function out = hasFrame(obj)
          if obj.cur_index < obj.size
              out = true;
          else
              out = false;
          end
       
      end

      function [image, depth, xyz] = readFrame(obj)
          if obj.cur_index <= obj.size
              png_file_path = obj.png_files_names(obj.cur_index).folder + "\" + obj.png_files_names(obj.cur_index).name;
              d_file_path = obj.d_files_names(obj.cur_index).folder + "\" + obj.d_files_names(obj.cur_index).name;
              mat_file_path = obj.mat_files_names(obj.cur_index).folder + "\" + obj.mat_files_names(obj.cur_index).name;
              image = imread(png_file_path);
              depth = imread(d_file_path);
              xyz = load(mat_file_path);
              obj.cur_index = obj.cur_index + 1;
          else
              throw(MException('Frames ended'));
          end
       
      end
   end
end