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

#include<iostream>
#include<fstream>
#include "stb/stb_image_write.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;

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

class handlerA
{
private:
    vector<vector< uint8_t >> &frames_conteiner;
public:
    const void operator() (const vector< uint8_t > &cur_frame)  {
        this->frames_conteiner.push_back(cur_frame);
    }

    handlerA(vector<vector<uint8_t>> &framesConteiner) : frames_conteiner(framesConteiner) {}
};

class handlerB {
private:
    vector<vector< uint8_t >> &frames_conteiner;
public:
    const void operator() (const vector< uint8_t > &cur_frame)  {
        this->frames_conteiner.push_back(cur_frame);
    }

    handlerB(vector<vector<uint8_t>> &framesConteiner) : frames_conteiner(framesConteiner) {}
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

    auto handler_a = handlerA(HT_frames_b1);
    grabber->bindBufferHandler(handler_a);

    auto handler_b = handlerB(HT_frames_b2);
    grabber2->bindBufferHandler(handler_b);

    using namespace this_thread; // sleep_for, sleep_until
    using namespace chrono; // nanoseconds, system_clock, seconds
    /*
    struct Frame {
        Frame(const void *pVoid, double d, int i, int w, int h, short bpp,
              int sib) : ts(d), data_size(i),
                         width(w), height(h), bytes_per_pixel(bpp),
                         stride_in_bytes(sib){
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

    };

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
            for (const rs2::frame f : fs) {
                auto vf = f.as<rs2::video_frame>();
                Frame my_f = Frame(vf.get_data(), vf.get_timestamp(),
                                   vf.get_data_size(), vf.get_width(),
                                   vf.get_height(), vf.get_bytes_per_pixel(),
                                   vf.get_stride_in_bytes());
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
                Frame my_f = Frame(vf.get_data(), vf.get_timestamp(),
                                   vf.get_data_size(), vf.get_width(),
                                   vf.get_height(), vf.get_bytes_per_pixel(),
                                   vf.get_stride_in_bytes());
                frames2[f.get_profile().unique_id()].push_back(my_f);
            }
        } //else {
        // Stream that bypass synchronization (such as IMU) will produce single frames
        //frames_bypass.push_back(frame);
        //}
    };


    rs2::pipeline pipe(ctx);
    rs2::config cfg;
    cfg.enable_device(serials[0]);
    rs2::pipeline_profile profiles = pipe.start(cfg, callback);
    pipelines.emplace_back(pipe);

    rs2::pipeline pipe2(ctx);
    rs2::pipeline_profile profiles2;
    if(serials.size() > 1) {
        rs2::config cfg2;
        cfg2.enable_device(serials[1]);
        profiles2 = pipe2.start(cfg2, callback2);
        pipelines.emplace_back(pipe2);
    }
    */

    bool start_statusA = grabber->start();
    //cout << "CamA started succefully : " << start_statusA << endl;
    bool start_statusB = grabber2->start();
    //cout << "CamB started succefully : " << start_statusB << std::endl;
    /*
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
    */
    sleep_for(nanoseconds(10000000000));



    bool finish_statusA = grabber->stop();
    //cout << "CamA stoped succefully : " << finish_statusA << endl;
    bool finish_statusB = grabber2->stop();
    //cout << "CamB stoped succefully : " << finish_statusB << endl;
    /*
    pipe.stop();
    if(serials.size() > 1)
        pipe2.stop();

    //std::vector<Frame> depth_frames = frames[0];
    //Frame depth_frame0 = depth_frames[0];
    std::vector<Frame> rgb_frames = frames[stream_numbers["Color"]];
    std::vector<Frame> depth_frames = frames[stream_numbers["Depth"]];


    std::vector<Frame> rgb_frames2 = frames2[stream_numbers2["Color"]];
    std::vector<Frame> depth_frames2 = frames2[stream_numbers2["Depth"]];

    int frames_n = rgb_frames.size();
    std::stringstream color_png_file;
    std::stringstream depth_png_file;
    for (int i = 0; i < frames_n; ++i) {
        Frame cur_color_frame = rgb_frames[i];
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

    }
     */

    int H1_b_size = HT_frames_b1.size();
    for (int i = 0; i < H1_b_size; ++i) {
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
    }

    int H2_b_size = HT_frames_b2.size();
    for (int i = 0; i < H2_b_size; ++i) {
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
    }

    //save pairs for illustration
    /*
    int min_size = H1_b_size < H2_b_size ? H1_b_size : H2_b_size;
    int rows = 512;
    int cols = 640;
    for (int i = 0; i < min_size; ++i) {
        uint16_t* d1 = (uint16_t*)(HT_frames_b1[i].data());
        uint8_t* dd1 =  new uint8_t[rows*cols];

        uint16_t* d2 = (uint16_t*)HT_frames_b2[i].data();
        uint8_t* dd2 =  new uint8_t[rows*cols];

        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                dd1[i*cols+j] = (uint8_t)round(d1[i*cols+j] / 256);
                dd2[i*cols+j] = (uint8_t)round(d2[i*cols+j] / 256);
            }
        }
        cv::Mat HM;
        cv::Mat H1_cv(rows, cols, CV_8UC1, dd1);
        cv::Mat H2_cv(rows, cols, CV_8UC1, dd2);
        hconcat(H1_cv, H2_cv, HM);
        cv::Mat dst;
        equalizeHist( HM, dst );
        //cv::imshow("image", dst);
        //cv::waitKey(0);
        //cv::destroyAllWindows();



        vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);
        bool result = false;
        try
        {
            result = imwrite("pair_"+ to_string(i)+".png", dst, compression_params);
        }
        catch (const cv::Exception& ex)
        {
            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        }
        if (result)
            printf("Saved PNG file with alpha data.\n");
        else
            printf("ERROR: Can't save PNG file.\n");

        delete dd1;
        delete dd2;

    }
    */
}
