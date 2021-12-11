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

using namespace std;

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



    auto serialNumber = "";
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

    auto serialNumber2 = "";
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

    /*
    auto ar = wic->setAR(wic::AR::True);

    if (ar.first == wic::ResponseStatus::Ok) {
        // set core temperature resolution, value depends on core calibration
        wic->setTempRes(wic::ARRes::High);
    }
    // set advanced radiometry if core supports it
    auto ar2 = wic2->setAR(wic::AR::True);
    if (ar2.first == wic::ResponseStatus::Ok) {
        // set core temperature resolution, value depends on core calibration
        wic2->setTempRes(wic::ARRes::High);
    }
    */
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

    using namespace this_thread; // sleep_for, sleep_until
    using namespace chrono; // nanoseconds, system_clock, seconds

    struct Frame {
        Frame(const void *pVoid, double d, int i) : frame_data((void*)(pVoid)), ts(d), data_size(i) {}

        void* frame_data = NULL;
        double ts = 0;
        int data_size = -1;
    };

    std::map<int, std::vector<Frame>> frames;

    std::mutex mutex;
    auto callback = [&](const rs2::frame& frame)
    {
        //std::lock_guard<std::mutex> lock(mutex);
        if (rs2::frameset fs = frame.as<rs2::frameset>()) {
            // With callbacks, all synchronized stream will arrive in a single frameset
            for (const rs2::frame& f : fs) {
                Frame my_f = Frame(f.get_data(), f.get_timestamp(), f.get_data_size());
                frames[f.get_profile().unique_id()].push_back(my_f);
            }
        } //else {
            // Stream that bypass synchronization (such as IMU) will produce single frames
            //frames_bypass.push_back(frame);
        //}
    };


    rs2::pipeline pipe;
    rs2::pipeline_profile profiles = pipe.start(callback);

    bool start_statusA = grabber->start();
    cout << "CamA started succefully : " << start_statusA << endl;
    bool start_statusB = grabber2->start();
    cout << "CamB started succefully : " << start_statusB << std::endl;

    // Collect the enabled streams names
    std::map<int, std::string> stream_names;
    for (auto p : profiles.get_streams()) {
        stream_names[p.unique_id()] = p.stream_name();
        cout << p.fps() << endl;
    }

    sleep_for(nanoseconds(10000000000));



    bool finish_statusA = grabber->stop();
    cout << "CamA stoped succefully : " << finish_statusA << endl;
    bool finish_statusB = grabber2->stop();
    cout << "CamB stoped succefully : " << finish_statusB << endl;

    pipe.stop();

    std::vector<Frame> frame0 = frames[0];

    //cout << sizeof(std::vector<int>) + (sizeof(uint8_t) * frames_buffer1.size()) ;

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




}
