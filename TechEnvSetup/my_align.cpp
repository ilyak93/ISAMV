//
// Created by Ily on 2/24/2022.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/hpp/rs_internal.hpp>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include "dirent-1.23.2/include/dirent.h"

int W = 1280;
int H = 720;

using namespace std;

std::vector <std::string> read_directory( const std::string& path = std::string() )
{
    std::vector <std::string> result;
    dirent* de;
    DIR* dp;
    errno = 0;
    dp = opendir( path.empty() ? "." : path.c_str() );
    if (dp)
    {
        while (true)
        {
            errno = 0;
            de = readdir( dp );
            if (de == NULL) break;
            result.push_back( std::string( de->d_name ) );
        }
        closedir( dp );
        std::sort( result.begin(), result.end() );
    }
    return result;
}


int main() {

    std::ifstream infile("F:/Vista_project2/21_ready/intrinsics_extrinsics.txt");

    std::string line;
    vector<string> depth;
    vector<string> color;
    vector<string> depth_to_color;
    while (getline(infile, line)) {
        vector<string> splitted;
        char *token = strtok((char*)(line.c_str()), " ");
        while (token != NULL) {
            splitted.push_back(string(token));
            token = strtok(NULL, " ");
        }
        if(splitted[0].compare(string("depth")) == 0){
            depth = splitted;
        } else if(splitted[0].compare(string("color")) == 0) {
            color = splitted;
        } else {
            depth_to_color = splitted;
        }
    }

    rs2::software_device dev;

    auto depth_sensor = dev.add_sensor("Depth");
    auto color_sensor = dev.add_sensor("Color");

    rs2_intrinsics depth_intrinsics{W, H, stof(depth[3]), stof(depth[4]),
                                    stof(depth[5]), stof(depth[6]),
                                    RS2_DISTORTION_BROWN_CONRADY,
                                    {stof(depth[7]), stof(depth[8]),
                                     stof(depth[9]), stof(depth[10]),
                                     stof(depth[11])}};
    rs2_intrinsics color_intrinsics{W, H, stof(color[3]), stof(color[4]),
                                    stof(color[5]), stof(color[6]),
                                    RS2_DISTORTION_INVERSE_BROWN_CONRADY,
                                    {stof(color[7]), stof(color[8]),
                                     stof(color[9]), stof(color[10]),
                                     stof(color[11])}};

    auto depth_stream = depth_sensor.add_video_stream(
            {RS2_STREAM_DEPTH, 0, 0, W, H, 30, 2, RS2_FORMAT_Z16,
             depth_intrinsics});
    auto color_stream = color_sensor.add_video_stream(
            {RS2_STREAM_COLOR, 0, 1, W, H, 30, 3, RS2_FORMAT_RGB8,
             color_intrinsics});

    depth_sensor.add_read_only_option(RS2_OPTION_DEPTH_UNITS, 0.001f);
    depth_sensor.add_read_only_option(RS2_OPTION_STEREO_BASELINE, 0.001f);

    depth_stream.register_extrinsics_to(color_stream,
                                        {
                                                {
                                                        stof(depth_to_color[1]),stof(depth_to_color[2]),
                                                        stof(depth_to_color[3]),stof(depth_to_color[4]),
                                                        stof(depth_to_color[5]),stof(depth_to_color[6]),
                                                        stof(depth_to_color[7]),stof(depth_to_color[8]),
                                                        stof(depth_to_color[9])
                                                },
                                                {
                                                        stof(depth_to_color[11]), stof(depth_to_color[12]),
                                                        stof(depth_to_color[12])
                                                }
                                        });

    dev.create_matcher(RS2_MATCHER_DEFAULT);
    rs2::syncer sync;

    depth_sensor.open(depth_stream);
    color_sensor.open(color_stream);

    depth_sensor.start(sync);
    color_sensor.start(sync);

    rs2::align align(RS2_STREAM_DEPTH);

    rs2::frameset fs;
    rs2::frame depth_frame;
    rs2::frame color_frame;

    cv::Mat color_image;
    cv::Mat depth_image;

    int idx = 0;

    //inject the images

    string path("F:/Vista_project2/21_ready/");

    vector<string> files_list = read_directory((path+string("/sync/")).c_str());

    for(int i = 0; i < files_list.size(); i+=2) {

        string color_image_path = path + string("/sync/") + files_list[i];
        string depth_image_path = path + string("/sync/") + files_list[i+1];
        color_image = cv::imread(color_image_path.c_str(), cv::IMREAD_COLOR);
        depth_image = cv::imread(depth_image_path.c_str(),cv::IMREAD_UNCHANGED);
        /*
        cv::imshow("image1", color_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
        cv::imshow("image2", depth_image);
        cv::waitKey(0);
        cv::destroyAllWindows();
        */
        color_sensor.on_video_frame({(void*) color_image.data, // Frame pixels from capture API
                                     [](void*) {}, // Custom deleter (if required)
                                     3 * 1280, 3, // Stride and Bytes-per-pixel
                                     double(idx * 30), RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME,
                                     idx, // Timestamp, Frame# for potential sync services
                                     color_stream});
        depth_sensor.on_video_frame({(void*) depth_image.data, // Frame pixels from capture API
                                     [](void*) {}, // Custom deleter (if required)
                                     2 * 1280, 2, // Stride and Bytes-per-pixel
                                     double(idx * 30), RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME,
                                     idx, // Timestamp, Frame# for potential sync services
                                     depth_stream});

        fs = sync.wait_for_frames();
        if (fs.size() == 2) {

            fs = align.process(fs);
            rs2::frame depth_frame = fs.get_depth_frame();
            rs2::frame color_frame = fs.get_color_frame();

            //cv::Mat aligned_image(720, 1280, CV_16UC1, (void *) (depth_frame.get_data()), 2 * 1280);
            //cv::imwrite(path + "/aligned_rs/" +  files_list[i+1], aligned_image);
            cv::Mat aligned_image2(720, 1280, CV_8UC3, (void *) (color_frame.get_data()), 3 * 1280);
            cv::imwrite(path + "/aligned_rs/" +  files_list[i], aligned_image2);
        }
        idx++;
    }
    
    return 0;
}