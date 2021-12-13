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
    /*
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

    auto frames_buffer1 = vector<vector< uint8_t >>();
    auto frames_buffer2 = vector<vector< uint8_t >>();

    auto handler_a = handlerA(frames_buffer1);
    grabber->bindBufferHandler(handler_a);

    auto handler_b = handlerB(frames_buffer2);
    grabber2->bindBufferHandler(handler_b);
    */
    using namespace this_thread; // sleep_for, sleep_until
    using namespace chrono; // nanoseconds, system_clock, seconds

    struct Frame {
        Frame(const void *pVoid, double d, int i) : frame_data((void*)(pVoid)), ts(d), data_size(i) {}

        void* frame_data = NULL;
        double ts = 0;
        int data_size = -1;
    };

    //std::map<int, std::vector<Frame>> frames;
    std::map<int, std::vector<rs2::frame>> frames;
    std::mutex mutex;
    auto callback = [&](const rs2::frame& frame)
    {
        //std::lock_guard<std::mutex> lock(mutex);
        if (rs2::frameset fs = frame.as<rs2::frameset>()) {
            rs2::disparity_transform disparity2depth(false);
            //fs = fs.apply_filter(disparity2depth);
            // With callbacks, all synchronized stream will arrive in a single frameset
            for (const rs2::frame f : fs) {
                //Frame my_f = Frame(f.get_data(), f.get_timestamp(), f.get_data_size());
                frames[f.get_profile().unique_id()].push_back(f);
            }
        } //else {
            // Stream that bypass synchronization (such as IMU) will produce single frames
            //frames_bypass.push_back(frame);
        //}
    };


    rs2::pipeline pipe;
    rs2::pipeline_profile profiles = pipe.start(callback);
    /*
    bool start_statusA = grabber->start();
    cout << "CamA started succefully : " << start_statusA << endl;
    bool start_statusB = grabber2->start();
    cout << "CamB started succefully : " << start_statusB << std::endl;
    */
    // Collect the enabled streams names
    std::map<int, std::string> stream_names;
    for (auto p : profiles.get_streams()) {
        stream_names[p.unique_id()] = p.stream_name();
        cout << p.fps() << endl;
    }

    sleep_for(nanoseconds(20000000000));


    /*
    bool finish_statusA = grabber->stop();
    cout << "CamA stoped succefully : " << finish_statusA << endl;
    bool finish_statusB = grabber2->stop();
    cout << "CamB stoped succefully : " << finish_statusB << endl;
    */
    pipe.stop();

    //std::vector<Frame> depth_frames = frames[0];
    //Frame depth_frame0 = depth_frames[0];

    std::vector<rs2::frame> depth_frames = frames[0];

    std::vector<rs2::frame> rgb_frames = frames[3];
    rs2::frame color_frame = rgb_frames[15];

    uint8_t* ptr = (uint8_t*)color_frame.get_data();
    int stride = color_frame.as<rs2::video_frame>().get_stride_in_bytes();

    int i2 = 100, j2 = 100; // fetch pixel 100,100

    cout << "  R= " << int(ptr[i2 * stride + (3*j2)    ]);
    cout << ", G= " << int(ptr[i2 * stride + (3*j2) + 1]);
    cout << ", B= " << int(ptr[i2 * stride + (3*j2) + 2]);
    cout << endl;

    std::stringstream png_file2;
    auto cf = color_frame.as<rs2::video_frame>();

    int crows = cf.get_height(); //720
    int ccols = cf.get_width(); // 1280

    png_file2 << "./" << cf.get_profile().stream_name() << ".png";
    stbi_write_png(png_file2.str().c_str(), cf.get_width(), cf.get_height(),
                   cf.get_bytes_per_pixel(), cf.get_data(), cf.get_stride_in_bytes());

    auto df = depth_frames[15].as<rs2::depth_frame>();

    float val = df.get_distance(0,50);

    int bpp = df.get_bits_per_pixel();
    int bpp2 = df.get_bytes_per_pixel();
    int sib = df.get_stride_in_bytes();
    int rows = df.get_height(); //720
    int cols = df.get_width(); // 1280
    uint16_t* d = (uint16_t*)df.get_data();
    vector<int> values(d, d + (df.get_data_size()/sizeof(uint16_t)));
    cout << "\nMin Element = "
         << *min_element(values.begin(), values.end());

    // Find the max element
    cout << "\nMax Element = "
         << *max_element(values.begin(), values.end());

    cout << "\navg Element = "
         << avg(values);

    cout << endl;


    uint8_t* dd = (uint8_t*)df.get_data();
    int shorts_size = df.get_data_size() / 2;
    int size_oif_short = sizeof(short);
    int val2 = d[50];
    int count_non0 = 0;
    int count_non0_2 = 0;
    int same = 0;
    /*
    short* d3 = new short [rows*sib];

    for(int i = 0; i < rows; i++){
        for(int j = 0 ; j < cols ; j++){
            //count_non0 += df.get_distance(i, j) > 0 ? 1 : 0;
            count_non0_2 +=  d[i * sib + j] > 0 ? 1 : 0;
            float dist = df.get_distance(i, j) * 1000;
            float dist2 = float(d[i * sib + j]);
            //same += df.get_distance(i, j) * 1000 == float(d[i * sib + j]);
            //if (df.get_distance(i, j) * 1000 != d[i * sib + j]){
            //   cout << df.get_distance(i, j) * 1000 << endl;
            //   cout << d[i * sib + j] << endl;
            //}
            short b = d[i * sib + j];
            short b2 = d[i * sib + j] / 256 ;
            //d[i * w + j] = d[i * sib + j] / 256;
            float ss = df.get_distance(i, j) * 1000 / 256;
            float sss = round(df.get_distance(i, j) * 1000 / 256) ;
            //d3[i * sib + j] = df.get_distance(i, j) * 1000 / 256  ;
            //if (df.get_distance(i, j) != d[i * sib + j] && i % 100 == 0){
            //    cout << df.get_distance(i, j) << endl;
            //    cout << d[i * w + j] << endl;
            //}
        }
    }
    */



    auto vf = depth_frames[0].as<rs2::video_frame>();

    std::stringstream png_file;
    png_file << "./" << vf.get_profile().stream_name() << ".png";
    stbi_write_png(png_file.str().c_str(), vf.get_width(), vf.get_height(),
                   vf.get_bytes_per_pixel(), vf.get_data(), vf.get_stride_in_bytes());

    cv::Mat img_in(rows, cols, CV_16UC1, (void*)d);
    cv::Mat img_color;
    // Apply the colormap:
    //cv::applyColorMap(img_in, img_color, cv::COLORMAP_JET);
    // Show the result:
    cv::imshow("colorMap", img_in);

    cv::waitKey(0);

    cv::destroyAllWindows();

    vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
    bool result = false;
    try
    {
        result = imwrite("alpha.png", img_in, compression_params);
    }
    catch (const cv::Exception& ex)
    {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
    }
    if (result)
        printf("Saved PNG file with alpha data.\n");
    else
        printf("ERROR: Can't save PNG file.\n");

    //delete d3;

    /*
    ofstream camA_stream("camA.dat", ios::out | ios::binary);
    if(!camA_stream) {
        cout << "Cannot open file!" << endl;
        return 1;
    }
    camA_stream.write((const char *)frames_buffer1[0].data(),
                      sizeof(const char) * frames_buffer1[0].size());

    camA_stream.close();
    if(!camA_stream.good()) {
        cout << "Error occurred at writing time!" << endl;
        return 1;
    }

    ofstream camB_stream("camB.dat", ios::out | ios::binary);
    if(!camB_stream) {
        cout << "Cannot open file!" << endl;
        return 1;
    }

    camB_stream.write((const char *)frames_buffer2[0].data(),
                      sizeof(const char) * frames_buffer2[0].size());

    camB_stream.close();
    if(!camB_stream.good()) {
        cout << "Error occurred at writing time!" << endl;
        return 1;
    }
    */



}
