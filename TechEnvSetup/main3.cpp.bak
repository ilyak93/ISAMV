#include <wic/camerafinder.h>
#include <wic/framegrabber.h>
#include <wic/wic.h>

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#include <iostream>
#include <vector>

#include <chrono>
#include <thread>
#include <mutex>
#include <map>
#include <cmath>
#include <limits>

#include<iostream>
#include<fstream>
#include "stb/stb_image_write.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace this_thread; // sleep_for, sleep_until
using namespace chrono; // nanoseconds, system_clock, seconds

struct Frame {
    Frame(const void *pVoid, double ts, int data_sz, int w, int h, short bpp,
          int sib, long long int local_ts) : ts(ts), data_size(data_sz),
                                             width(w), height(h), bytes_per_pixel(bpp),
                                             stride_in_bytes(sib), loc_ts(local_ts){
        frame_data = new uint8_t[w*h*bytes_per_pixel];
        assert(frame_data != nullptr);
        memcpy(frame_data, pVoid, w*h*bytes_per_pixel);
    }

    //~Frame(){
    //    delete this->frame_data;
    //}

    void* frame_data = NULL;
    double ts = -1;
    int data_size = -1;
    int width = -1;
    int height = -1;
    short bytes_per_pixel = -1;
    int stride_in_bytes = -1;
    long long int loc_ts = -1;

};

vector<int> find_closest_by_ts(vector<long long int> TC_ts, std::vector<Frame> rs_frames) {
    vector<int> closest_to_TC(TC_ts.size());
    for (int i = 0; i < TC_ts.size(); ++i) {
        long long int dist = LLONG_MAX;
        for (int j = 0; j < rs_frames.size(); ++j) {
            int tmp = abs(TC_ts[i] - rs_frames[j].loc_ts);
            if(abs(TC_ts[i] - rs_frames[j].loc_ts) < dist){
                dist = abs(TC_ts[i] - rs_frames[j].loc_ts);
                closest_to_TC[i] = j;
            }
        }
    }

    return closest_to_TC;
}

double avg ( std::vector<int>& v )
{
    double return_value = 0.0;
    int n = v.size();

    for ( int i=0; i < n; i++)
    {
        return_value += v[i];
    }

    return ( return_value / n);
}

class handlerA {
private:
    vector<vector<uint8_t>> &frames_conteiner;
    vector<long long int> &local_ts;
public:
    const void operator() (const vector< uint8_t > &cur_frame)  {
        const std::chrono::time_point<std::chrono::steady_clock> now = high_resolution_clock::now();
        long long int loc_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()).count();
        this->frames_conteiner.push_back(cur_frame);
        this->local_ts.push_back(loc_ts);
    }

    handlerA(vector<vector<uint8_t>> &framesConteiner,
             vector<long long int> &local_timestamps) :
             frames_conteiner(framesConteiner), local_ts(local_timestamps) {}
};

class handlerB {
private:
    vector<vector<uint8_t>> &frames_conteiner;
    vector<long long int> &local_ts;
public:
    const void operator() (const vector< uint8_t > &cur_frame)  {
        const std::chrono::time_point<std::chrono::steady_clock> now = high_resolution_clock::now();
        long long int loc_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()).count();
        this->frames_conteiner.push_back(cur_frame);
        this->local_ts.push_back(loc_ts);
    }

    handlerB(vector<vector<uint8_t>> &framesConteiner,
             vector<long long int> &local_timestamps) :
            frames_conteiner(framesConteiner), local_ts(local_timestamps) {}
};



int main() {


    auto serialNumber = "070A1912";
    auto wic = wic::findAndConnect(serialNumber);


    if (!wic) {
        cerr << "Could not connect WIC: " << serialNumber << endl;
        return 1;
    }

    auto defaultRes = wic->doDefaultWICSettings();
    if (defaultRes.first != wic::ResponseStatus::Ok) {
        cerr << "DoDefaultWICSettings: "
                  << wic::responseStatusToStr(defaultRes.first) << endl;
        return 2;
    }

    auto serialNumber2 = "069A1912";
    auto wic2 = wic::findAndConnect(serialNumber2);

    if (!wic2) {
        cerr << "Could not connect WIC: " << serialNumber2 << endl;
        return 1;
    }

    auto defaultRes2 = wic2->doDefaultWICSettings();
    if (defaultRes2.first != wic::ResponseStatus::Ok) {
        cerr << "DoDefaultWICSettings: "
                  << wic::responseStatusToStr(defaultRes2.first) << endl;
        return 2;
    }



    // enable advanced features
    wic->iKnowWhatImDoing();
    // enable advanced features
    wic2->iKnowWhatImDoing();
    // set advanced radiometry if core supports it


    // set core gain
    auto gain = wic->setGain(wic::GainMode::High);

    // set core gain
    auto gain2 = wic2->setGain(wic::GainMode::High);

    auto grabber = wic->frameGrabber();
    grabber->setup();

    auto grabber2 = wic2->frameGrabber();
    grabber2->setup();

    auto status1  = wic->setFFCMode(wic::FFCModes::Manual);
    auto status2  = wic2->setFFCMode(wic::FFCModes::Manual);

    auto resolution = wic->getResolution();
    if (resolution.first == 0 || resolution.second == 0) {
        cerr << "Invalid resolution, core detection error." << endl;
        return 3;
    }

    auto resolution2 = wic2->getResolution();
    if (resolution2.first == 0 || resolution2.second == 0) {
        cerr << "Invalid resolution, core detection error." << endl;
        return 3;
    }

    // default wic settings = OutputType::RAD14
    // every 2 bytes represent radiometric flux of one pixel
    // buffer is in row major format

    auto HT_frames_b1 = vector<vector< uint8_t >>();
    auto HT_frames_b2 = vector<vector< uint8_t >>();
    auto HT_frames_b1_ts = vector<long long int>();
    auto HT_frames_b2_ts = vector<long long int>();

    auto handler_a = handlerA(HT_frames_b1, HT_frames_b1_ts);
    grabber->bindBufferHandler(handler_a);

    auto handler_b = handlerB(HT_frames_b2, HT_frames_b2_ts);
    grabber2->bindBufferHandler(handler_b);








    rs2::context ctx;

    std::vector<rs2::pipeline> pipelines;

    std::vector<std::string> serials;

    auto devs = ctx.query_devices();
    for (auto&& dev : devs)
        serials.push_back(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));

    std::map<int, std::vector<Frame>> frames;
    std::mutex mutex;
    auto callback = [&](const rs2::frame& frame)
    {
        //std::lock_guard<std::mutex> lock(mutex);
        if (rs2::frameset fs = frame.as<rs2::frameset>()) {
            //rs2::disparity_transform disparity2depth(false);
            //fs = fs.apply_filter(disparity2depth);
            // With callbacks, all synchronized stream will arrive in a single frameset
            const std::chrono::time_point<std::chrono::steady_clock> now = high_resolution_clock::now();
            long long int loc_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now.time_since_epoch()).count();
            for (const rs2::frame f : fs) {
                auto vf = f.as<rs2::video_frame>();
                Frame my_f = Frame(vf.get_data(), vf.get_timestamp(),
                                   vf.get_data_size(), vf.get_width(),
                                   vf.get_height(), vf.get_bytes_per_pixel(),
                                   vf.get_stride_in_bytes(), loc_ts);


                double d = vf.get_timestamp();

                frames[f.get_profile().unique_id()].push_back(my_f);
            }
        } //else {
            // Stream that bypass synchronization (such as IMU) will produce single frames
            //frames_bypass.push_back(frame);
        //}
    };

    std::map<int, std::vector<Frame>> frames2;
    std::mutex mutex2;
    auto callback2 = [&](const rs2::frame& frame)
    {
        //std::lock_guard<std::mutex> lock(mutex);
        if (rs2::frameset fs = frame.as<rs2::frameset>()) {
            //rs2::disparity_transform disparity2depth(false);
            //fs = fs.apply_filter(disparity2depth);
            // With callbacks, all synchronized stream will arrive in a single frameset
            for (const rs2::frame f : fs) {
                auto vf = f.as<rs2::video_frame>();
                const std::chrono::time_point<std::chrono::steady_clock> now = high_resolution_clock::now();
                long long int loc_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        now.time_since_epoch()).count();
                Frame my_f = Frame(vf.get_data(), vf.get_timestamp(),
                                   vf.get_data_size(), vf.get_width(),
                                   vf.get_height(), vf.get_bytes_per_pixel(),
                                   vf.get_stride_in_bytes(), loc_ts);
                frames2[f.get_profile().unique_id()].push_back(my_f);
            }
        } //else {
        // Stream that bypass synchronization (such as IMU) will produce single frames
        //frames_bypass.push_back(frame);
        //}
    };

    vector<vector< uint8_t >> TC_frames1;
    vector<uint64_t> TC_frames1_ts;

    auto TC_callback = [&](const vector< uint8_t > & frame) {
        TC_frames1.push_back(frame);
        const std::chrono::time_point<std::chrono::steady_clock> now = high_resolution_clock::now();
        //auto n =  std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
        //TC_frames1_ts.push_back(n);
    };


    rs2::pipeline pipe(ctx);
    rs2::config cfg;
    cfg.enable_device(serials[0]);
    //cfg.enable_stream(RS2_STREAM_ANY, 1280, 720, RS2_FORMAT_ANY, 30);



    rs2::pipeline_profile profiles = pipe.start(cfg, callback);

    bool start_statusA = grabber->start();
    //cout << "CamA started succefully : " << start_statusA << endl;
    bool start_statusB = grabber2->start();
    //cout << "CamB started succefully : " << start_statusB << std::endl;

    pipelines.emplace_back(pipe);

    rs2::pipeline pipe2(ctx);
    rs2::pipeline_profile profiles2;
    if(serials.size() > 1) {
        rs2::config cfg2;
        //cfg2.enable_stream(RS2_STREAM_ANY, 1280, 720, RS2_FORMAT_ANY, 30);
        cfg2.enable_device(serials[1]);
        profiles2 = pipe2.start(cfg2, callback2);
        pipelines.emplace_back(pipe2);
    }




    // Collect the enabled streams names
    std::map<int, std::string> stream_names;
    std::map<std::string, int> stream_numbers;
    for (auto p : profiles.get_streams()) {
        stream_names[p.unique_id()] = p.stream_name();
        stream_numbers[p.stream_name()] =  p.unique_id();
    }

    std::map<int, std::string> stream_names2;
    std::map<std::string, int> stream_numbers2;
    if(serials.size() > 1){
        for (auto p: profiles2.get_streams()) {
            stream_names2[p.unique_id()] = p.stream_name();
            stream_numbers2[p.stream_name()] = p.unique_id();
        }
    }

    sleep_for(nanoseconds(10000000000));



    bool finish_statusA = grabber->stop();
    //cout << "CamA stoped succefully : " << finish_statusA << endl;
    bool finish_statusB = grabber2->stop();
    //cout << "CamB stoped succefully : " << finish_statusB << endl;

    pipe.stop();
    if(serials.size() > 1)
        pipe2.stop();
    cout << "Cams Stopped: " << endl;
    //std::vector<Frame> depth_frames = frames[0];
    //Frame depth_frame0 = depth_frames[0];
    std::vector<Frame> rgb_frames = frames[stream_numbers["Color"]];
    std::vector<Frame> depth_frames = frames[stream_numbers["Depth"]];

    for (int i = 0; i < rgb_frames.size()-1; ++i) {
        assert(rgb_frames[i].loc_ts <= rgb_frames[i+1].loc_ts);
    }

    for (int i = 0; i < depth_frames.size()-1; ++i) {
        assert(depth_frames[i].loc_ts <= depth_frames[i+1].loc_ts);
    }

    for (int i = 0; i < HT_frames_b1_ts.size()-1; ++i) {
        assert(HT_frames_b1_ts[i] <= HT_frames_b1_ts[i+1]);;
    }

    for (int i = 0; i < HT_frames_b2_ts.size()-1; ++i) {
        assert(HT_frames_b2_ts[i] <= HT_frames_b2_ts[i+1]);
    }


    //std::vector<Frame> rgb_frames2 = frames2[stream_numbers2["Color"]];
    //std::vector<Frame> depth_frames2 = frames2[stream_numbers2["Depth"]];

    int frames_n = rgb_frames.size();
    std::stringstream color_png_file;
    std::stringstream depth_png_file;
    for (int i = 0; i < frames_n; ++i) {
        /*
        Frame cur_color_frame = rgb_frames[i];

        std::stringstream png_file;
        png_file << "./" << "Color_" + to_string(i) + ".png";
        stbi_write_png(png_file.str().c_str(), cur_color_frame.width,
                       cur_color_frame.height, cur_color_frame.bytes_per_pixel,
                       cur_color_frame.frame_data, cur_color_frame.stride_in_bytes);
        */
        /*
        string color_frame_name = "./Color_" + to_string(i);
        ofstream fout;
        fout.open(color_frame_name, ios::binary | ios::out);
        fout.write((char*)cur_color_frame.frame_data,
                   cur_color_frame.data_size);
        fout.close();
        delete cur_color_frame.frame_data;

        Frame cur_depth_frame = depth_frames[i];
        string depth_frame_name = "./Depth_" + to_string(i);
        //auto start = high_resolution_clock::now();
        fout.open(depth_frame_name, ios::binary | ios::out);
        fout.write((char*)cur_depth_frame.frame_data,
                   cur_depth_frame.data_size);
        fout.close();
        delete cur_depth_frame.frame_data;
        */
    }


    int H1_b_size = HT_frames_b1.size();
    for (int i = 0; i < H1_b_size; ++i) {
        /*
        ofstream camA_stream("H1_"+ to_string(i) +".dat", ios::out | ios::binary);
        if(!camA_stream) {
            cout << "Cannot open file!" << endl;
            return 1;
        }
        camA_stream.write((const char *)HT_frames_b1[i].data(),
                          sizeof(const char) * HT_frames_b1[i].size());
        camA_stream.close();
        if(!camA_stream.good()) {
            cout << "Error occurred at writing time!" << endl;
            return 1;
        }
         */
    }

    int H2_b_size = HT_frames_b2.size();
    for (int i = 0; i < H2_b_size; ++i) {
        /*
        ofstream camB_stream("H2_"+ to_string(i) +".dat", ios::out | ios::binary);
        if(!camB_stream) {
            cout << "Cannot open file!" << endl;
            return 1;
        }

        camB_stream.write((const char *)HT_frames_b2[i].data(),
                          sizeof(const char) * HT_frames_b2[i].size());

        camB_stream.close();
        if(!camB_stream.good()) {
            cout << "Error occurred at writing time!" << endl;
            return 1;
        }
        */
    }


    //save pairs for illustration

    int min_size = H1_b_size < H2_b_size ? H1_b_size : H2_b_size;
    int rows = 512;
    int cols = 640;
    vector<int> closest_indices = find_closest_by_ts(HT_frames_b1_ts, rgb_frames);
    for (int i = 0; i < min_size; ++i) {
        uint16_t* tc1_data_p = (uint16_t*)(HT_frames_b1[i].data());
        uint16_t* tc2_data_p = (uint16_t*)HT_frames_b2[i].data();

        cv::Mat tc1_img(rows, cols, CV_16UC1, (void*)tc1_data_p);
        cv::Mat tc1_img8u;
        tc1_img.convertTo(tc1_img8u, CV_8UC1, 1/256.0);

        cv::Mat tc2_img(rows, cols, CV_16UC1, (void*)tc2_data_p);
        cv::Mat tc2_img8u;
        tc2_img.convertTo(tc2_img8u, CV_8UC1, 1/256.0);


        //equalized depth image creation
        uint16_t* dep1 = (uint16_t*)depth_frames[closest_indices[i]].frame_data;
        cv::Mat depth_img(720, 1280, CV_16UC1, (void*)dep1);
        //normalize(depth_img, depth_img, 0, 65535, cv::NORM_MINMAX);
        cv::Mat depth_img8u;
        depth_img.convertTo(depth_img8u, CV_8UC1, 1/256.0);

        cv::Mat eq_depth_img;
        equalizeHist( depth_img8u, eq_depth_img );
        // Apply the colormap:
        //cv::Mat eq_depth_img8u_heatmap;
        //cv::applyColorMap(eq_depth_img, eq_depth_img8u_heatmap, cv::COLORMAP_JET);

        cv::Mat closest_rgb(720, 1280, CV_8UC3,
                            rgb_frames[closest_indices[i]].frame_data);
        cv::Mat closest_gray;
        cv::cvtColor(closest_rgb, closest_gray, cv::COLOR_RGB2GRAY);
        cv::Mat HM;

        //tc1_img8u.resize(719, 0);
        //tc2_img8u.resize(719, 0);
        //tc1_img8u.resize(720, 255);
        //tc2_img8u.resize(720, 255);
        //normalize(tc1_img8u, tc1_img8u, 0, 65535, cv::NORM_MINMAX);
        //normalize(tc2_img8u, tc2_img8u, 0, 65535, cv::NORM_MINMAX);
        hconcat(tc1_img8u, tc2_img8u,HM);
        equalizeHist( HM, HM );
        HM.resize(720, 0);

        cv::Mat TC_e_color;
        hconcat(HM,closest_gray, TC_e_color);
        cv::Mat TC_e_color_e_depth;
        hconcat(TC_e_color, eq_depth_img, TC_e_color_e_depth);

        //cv::imshow("image", dst);
        //cv::waitKey(0);
        //cv::destroyAllWindows();

        vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);
        bool result = false;
        try
        {
            result = imwrite("pair_"+ to_string(i)+".png", TC_e_color_e_depth, compression_params);
        }
        catch (const cv::Exception& ex)
        {
            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        }
        if (!result)
            printf("ERROR: Can't save PNG file.\n");

    }
    cout << "Finished" << endl;

}
