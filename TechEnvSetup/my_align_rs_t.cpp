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
#include "shlwapi.h"
#pragma comment(lib, "Shlwapi.lib")

int W = 1280;
int H = 720;

using namespace std;

wstring widen(string text)
{
    locale loc("");
    vector<wchar_t> buffer(text.size());
    use_facet< std::ctype<wchar_t> > (loc).widen(text.data(), text.data() + text.size(), &buffer[0]);
    return wstring(&buffer[0], buffer.size());
}

bool compare(const string& first, const string& second)
{
    wstring lhs = widen(first);
    wstring rhs = widen(second);
    bool ordered = StrCmpLogicalW(lhs.c_str(), rhs.c_str()) < 0;
    return ordered;
}




int main() {

    std::ifstream infile("G:/Vista_project/finish1/intrinsics_extrinsics_rs_thermal.txt");

    std::string line;
    vector<string> thermal;
    vector<string> realsense;
    vector<string> thermal_to_rs;
    while (getline(infile, line)) {
        vector<string> splitted;
        char *token = strtok((char*)(line.c_str()), " ");
        while (token != NULL) {
            splitted.push_back(string(token));
            token = strtok(NULL, " ");
        }
        if(splitted[0].compare(string("thermal")) == 0){
            thermal = splitted;
        } else if(splitted[0].compare(string("realsense")) == 0) {
            realsense = splitted;
        } else if (splitted[0].compare(string("rotation")) == 0){
            thermal_to_rs = splitted;
        }
    }

    rs2::software_device dev;

    auto rs_sensor = dev.add_sensor("rs");
    auto thermal_sensor = dev.add_sensor("thermal");

    rs2_intrinsics rs_intrinsics{W, H, stof(realsense[3]), stof(realsense[4]),
                                    stof(realsense[5]), stof(realsense[6]),
                                 RS2_DISTORTION_INVERSE_BROWN_CONRADY,
                                    {stof(realsense[7]), stof(realsense[8]),
                                     stof(realsense[9]), stof(realsense[10]),
                                     stof(realsense[11])}};
    rs2_intrinsics thermal_intrinsics{W, H, stof(thermal[3]), stof(thermal[4]),
                                    stof(thermal[5]), stof(thermal[6]),
                                    RS2_DISTORTION_BROWN_CONRADY,
                                    {stof(thermal[7]), stof(thermal[8]),
                                     stof(thermal[9]), stof(thermal[10]),
                                     stof(thermal[11])}};

    auto thermal_stream = thermal_sensor.add_video_stream(
            {RS2_STREAM_DEPTH, 0, 0, W, H, 30, 2, RS2_FORMAT_Z16,
             thermal_intrinsics});
    auto rs_stream = rs_sensor.add_video_stream(
            {RS2_STREAM_COLOR, 0, 1, W, H, 30, 3, RS2_FORMAT_RGB8,
             rs_intrinsics});

    thermal_sensor.add_read_only_option(RS2_OPTION_DEPTH_UNITS, 0.001f);
    thermal_sensor.add_read_only_option(RS2_OPTION_STEREO_BASELINE, 0.001f);
    thermal_stream.register_extrinsics_to(rs_stream,
                                        {
        {
            float(stod(thermal_to_rs[1])),float(stod(thermal_to_rs[2])),
            float(stod(thermal_to_rs[3])),float(stod(thermal_to_rs[4])),
            float(stod(thermal_to_rs[5])),float(stod(thermal_to_rs[6])),
            float(stod(thermal_to_rs[7])),float(stod(thermal_to_rs[8])),
            float(stod(thermal_to_rs[9]))
            },
            {
            float(stod(thermal_to_rs[11])), float(stod(thermal_to_rs[12])),
            float(stod(thermal_to_rs[12]))
            }
                                        });

    dev.create_matcher(RS2_MATCHER_DEFAULT);
    rs2::syncer sync;

    thermal_sensor.open(thermal_stream);
    rs_sensor.open(rs_stream);

    thermal_sensor.start(sync);
    rs_sensor.start(sync);

    rs2::align align(RS2_STREAM_COLOR);

    rs2::frameset fs;
    rs2::frame thermal_frame;
    rs2::frame rs_frame;

    cv::Mat thermal_image;
    cv::Mat rs_image;
    for(int jj = 6; jj < 14; ++jj) {
        int idx = 0;

        //inject the images

        struct dirent *ent;
        struct dirent ent_rs;
        struct dirent ent_thermal;
        DIR *rs_dir = opendir("G:/Vista_project/finish1/aligned_rs/");
        DIR *thermal_dir = opendir(("G:/Vista_project/finish1/calibration/"+ to_string(jj)+"/right_resized/").c_str());
        assert(rs_dir != NULL);
        assert(thermal_dir != NULL);

        list<string> thermal_images_pathes;
        list<string> realsense_images_pathes;

        while ((ent = readdir(rs_dir)) != NULL) {
            ent_rs = *ent;
            if (string(ent_rs.d_name).find(string("depth")) != std::string::npos) {
                continue;
            }
            if ((ent = readdir(thermal_dir)) == NULL) break;
            ent_thermal = *ent;
            if (string(ent_rs.d_name).compare("..") == 0 ||
                string(ent_rs.d_name).compare(".") == 0)
                continue;
            string rs_image_path = string("G:/Vista_project/finish1/rs/") + string(ent_rs.d_name);
            string thermal_image_path =
                    "G:/Vista_project/finish1/calibration/"+ to_string(jj)+"/right_resized/" + string(ent_thermal.d_name);
            thermal_images_pathes.push_back(thermal_image_path);
            realsense_images_pathes.push_back(rs_image_path);
        }
        thermal_images_pathes.sort(compare);
        realsense_images_pathes.sort(compare);
        int images_num = thermal_images_pathes.size();
        int l = 0;
        int max_len = -1;
        for (int k = 0; k < images_num && l > max_len; ++k) {
            string rs_image_path = realsense_images_pathes.front();
            realsense_images_pathes.pop_front();
            string thermal_image_path = thermal_images_pathes.front();
            thermal_images_pathes.pop_front();
            rs_image = cv::imread(rs_image_path.c_str(), cv::IMREAD_COLOR);
            thermal_image = cv::imread(thermal_image_path.c_str(), cv::IMREAD_UNCHANGED);

            rs_sensor.on_video_frame({(void *) rs_image.data, // Frame pixels from capture API
                                      [](void *) {}, // Custom deleter (if required)
                                      3 * 1280, 3, // Stride and Bytes-per-pixel
                                      double(idx * 30), RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME,
                                      idx, // Timestamp, Frame# for potential sync services
                                      rs_stream});
            thermal_sensor.on_video_frame({(void *) thermal_image.data, // Frame pixels from capture API
                                           [](void *) {}, // Custom deleter (if required)
                                           2 * 1280, 2, // Stride and Bytes-per-pixel
                                           double(idx * 30), RS2_TIMESTAMP_DOMAIN_SYSTEM_TIME,
                                           idx, // Timestamp, Frame# for potential sync services
                                           thermal_stream});

            fs = sync.wait_for_frames();
            /*
            if (fs.size() == 2) {
                rs2::frame thermal_frame_tmp = fs.get_depth_frame();
                rs2::frame rs_frame_tmp = fs.get_color_frame();
                cv::Mat tmp1(720, 1280, CV_16UC1, (void*) (thermal_frame_tmp.get_data()), 2 * 1280);
                cv::Mat tmp2(720, 1280, CV_8UC3, (void*) (rs_frame_tmp.get_data()), 3 * 1280);

                cv::imshow("image1", tmp2);
                cv::waitKey(0);
                cv::destroyAllWindows();
                cv::imshow("image2", tmp1);
                cv::waitKey(0);
                cv::destroyAllWindows();

            }
            */
            if (fs.size() == 2) {
                fs = align.process(fs);
                rs2::frame thermal_frame = fs.get_depth_frame();
                rs2::frame rs_frame = fs.get_color_frame();
                cv::Mat aligned_image(720, 1280, CV_16UC1, (void *) (thermal_frame.get_data()), 2 * 1280);
                cv::imwrite("G:/Vista_project/finish1/calibration/" + to_string(jj) +"/right_resized_aligned/" + to_string(idx + 1) + ".png",
                            aligned_image);
            }
            idx++;
            l++;
        }
        closedir(thermal_dir);
        closedir(rs_dir);
    }
    return 0;
}
