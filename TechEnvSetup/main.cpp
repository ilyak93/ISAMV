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


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <boost/iostreams/code_converter.hpp>
#include <boost/iostreams/device/mapped_file.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/asio.hpp>
#include <boost/thread.hpp>

#include "counting_semaphore.cpp"

using namespace std;
using namespace this_thread; // sleep_for, sleep_until
using namespace chrono; // nanoseconds, system_clock, seconds
using namespace boost::iostreams;
using namespace boost::interprocess;

class RSCallback {
public:
    char** color_mfd_ptr = new char*;
    char** depth_mfd_ptr = new char*;
    vector<long long int>& color_depth_ts;
    volatile int& idx_depth;
    std::atomic<volatile int>& idx_color;
    std::mutex& mux;
    boost::shared_mutex& shared_mux;

public:
    RSCallback(char* cmfd_ptr, char* dmfd_ptr, vector<long long int>& ccdts,
               std::atomic<volatile int>& idxc, int& idxd, std::mutex& mx,
               boost::shared_mutex& smx) :
               idx_color(idxc), idx_depth(idxd),
               color_depth_ts(ccdts), mux(mx), shared_mux(smx) {
        *color_mfd_ptr = cmfd_ptr;
        *depth_mfd_ptr = dmfd_ptr;


    }
    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator () (const rs2::frame &frame) {
        boost::shared_lock<boost::shared_mutex> shared_lock(this->shared_mux);
        std::lock_guard<std::mutex> lock(this->mux);
        if (rs2::frameset fs = frame.as<rs2::frameset>()) {
            const std::chrono::time_point<std::chrono::steady_clock> now =
                    high_resolution_clock::now();
            long long int loc_ts =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            now.time_since_epoch()).count();
            this->color_depth_ts.push_back(loc_ts);
            for (const rs2::frame f: fs) {
                auto vf = f.as<rs2::video_frame>();
                if (vf.get_bytes_per_pixel() == 2) {
                    size_t sz = vf.get_data_size();
                    memcpy((void *) ((uint8_t *) (*depth_mfd_ptr) +
                                    idx_depth * sz), vf.get_data(), sz);
                    idx_depth++;
                } else {
                    size_t sz = vf.get_data_size();
+                    memcpy((void *) ((uint8_t *) (*color_mfd_ptr) +
                                    idx_color * sz), vf.get_data(), sz);
                    idx_color.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }
};


class TC1Callback {
public:
    char** tc_mfd_ptr = new char*;
    vector<long long int> &tc_ts;
    int& idx_tc;
    size_t sz;
    std::mutex &mux;
    boost::shared_mutex &shared_mux;

public:
    TC1Callback(char *tcm_ptr, vector<long long int> &tcts, int& ixtc,
                size_t tc_size, std::mutex &mx, boost::shared_mutex &smx) :
             tc_ts(tcts), idx_tc(ixtc), sz(tc_size),
            mux(mx), shared_mux(smx) {
        *tc_mfd_ptr =  tcm_ptr;
    }

    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()(const vector <uint8_t> &cur_frame) {
        boost::shared_lock<boost::shared_mutex> shared_lock(shared_mux);
        std::lock_guard <std::mutex> lock(mux);
        memcpy((void *) ((uint8_t *) (*tc_mfd_ptr) + idx_tc * sz), cur_frame.data(), sz);
        const std::chrono::time_point <std::chrono::steady_clock> now = high_resolution_clock::now();
        long long int loc_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()).count();
        tc_ts.push_back(loc_ts);
        idx_tc++;
    }
};

class TC2Callback {
public:
    char** tc_mfd_ptr = new char*;
    vector<long long int> &tc_ts;
    int& idx_tc;
    size_t sz;
    std::mutex &mux;
    boost::shared_mutex &shared_mux;

public:
    TC2Callback(char *tcm_ptr, vector<long long int> &tcts, int& ixtc,
                size_t tc_size, std::mutex &mx, boost::shared_mutex &smx) :
            tc_ts(tcts), idx_tc(ixtc), sz(tc_size),
            mux(mx), shared_mux(smx) {
        *tc_mfd_ptr =  tcm_ptr;
    }

    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()(const vector <uint8_t> &cur_frame) {
        boost::shared_lock<boost::shared_mutex> shared_lock(shared_mux);
        std::lock_guard <std::mutex> lock(mux);
        memcpy((void *) ((uint8_t *) (*tc_mfd_ptr) + idx_tc * sz), cur_frame.data(), sz);
        const std::chrono::time_point <std::chrono::steady_clock> now = high_resolution_clock::now();
        long long int loc_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()).count();
        tc_ts.push_back(loc_ts);
        idx_tc++;
    }
};

class SaveCallback {
public:
    mapped_file& color_mapped_fd;
    mapped_file& depth_mapped_fd;
    mapped_file& tc1_mapped_fd;
    mapped_file& tc2_mapped_fd;
    vector<long long int>& color_depth_ts;
    vector<long long int>& tc1_ts;
    vector<long long int>& tc2_ts;
    int idx;
    string save_dir;

public:
    SaveCallback(mapped_file& cmfd, mapped_file& dmfd, mapped_file& tc1fd,
                 mapped_file& tc2fd, vector<long long int>& cdts,
                 vector<long long int>& tc1ts, vector<long long int>& tc2ts,
                 int ix, string sdir) : color_mapped_fd(cmfd), depth_mapped_fd(dmfd),
                           tc1_mapped_fd(tc1fd), color_depth_ts(cdts),
                           tc1_ts(tc1ts), tc2_ts(tc2ts), tc2_mapped_fd(tc2fd),
                           idx(ix), save_dir(sdir) {}


    // This operator overloading enables calling
    // operator function () on objects of increment
    void operator()() {
        color_mapped_fd.close();
        depth_mapped_fd.close();
        tc1_mapped_fd.close();
        tc2_mapped_fd.close();

        ofstream cd_fout;
        string color_depth_ts_name = save_dir + to_string(idx) + "color_depth_ts.bin";
        cd_fout.open(color_depth_ts_name, ios::binary | ios::out);
        cd_fout.write((char *) color_depth_ts.data(),
                      color_depth_ts.size() * sizeof(long long int));
        cd_fout.close();

        ofstream tc1_fout;
        string tc1_ts_name = save_dir + to_string(idx) + "tc1_ts.bin";
        tc1_fout.open(tc1_ts_name, ios::binary | ios::out);
        tc1_fout.write((char *) tc1_ts.data(),
                      tc1_ts.size() * sizeof(long long int));
        tc1_fout.close();

        ofstream tc2_fout;
        string tc2_ts_name = save_dir + to_string(idx) + "tc2_ts.bin";
        tc2_fout.open(tc2_ts_name, ios::binary | ios::out);
        tc2_fout.write((char *) tc2_ts.data(),
                       tc2_ts.size() * sizeof(long long int));
        tc2_fout.close();

        /*
         //this code check if the timesstamps are ordered
        vector<long long int> v_sorted(tc1_ts.size());
        partial_sort_copy(begin(tc1_ts), end(tc1_ts), begin(v_sorted), end(v_sorted));
        vector<long long int> vs(v_sorted);
        std::vector<long long int> diff;
        std::set_difference(v_sorted.begin(), v_sorted.end(), tc1_ts.begin(), tc1_ts.end(),
                            std::inserter(diff, diff.begin()));
        cout << accumulate(diff.begin(), diff.end(), 0) << endl;

        ofstream tc2_fout;
        string tc2_ts_name = save_dir + to_string(idx) + "tc2_ts.bin";
        tc2_fout.open(tc2_ts_name, ios::binary | ios::out);
        tc2_fout.write((char *) tc2_ts.data(),
                       tc2_ts.size() * sizeof(long long int));
        tc2_fout.close();

        vector<long long int> v_sorted2(tc2_ts.size());
        partial_sort_copy(begin(tc2_ts), end(tc2_ts), begin(v_sorted2), end(v_sorted2));
        std::vector<long long int> diff2;
        std::set_difference(v_sorted2.begin(), v_sorted2.end(), tc2_ts.begin(), tc2_ts.end(),
                            std::inserter(diff, diff2.begin()));
        cout << accumulate(diff2.begin(), diff2.end(), 0) << endl;
        */
    }
};

class SaveIntrinsicsExtrinsics {
private:
    rs2::pipeline_profile profiles;

public:
    SaveIntrinsicsExtrinsics(rs2::pipeline_profile prfs) : profiles(prfs){}

    void operator()(){
        auto streams = this->profiles.get_streams();

        // Mistake here
        auto color_stream_idx =  streams[0].stream_type() == RS2_STREAM_COLOR ? 0 : 1;
        auto depth_stream_idx = 1 - color_stream_idx;

        auto color_video_stream = streams[color_stream_idx].as<rs2::video_stream_profile>();
        auto depth_video_stream = streams[depth_stream_idx].as<rs2::video_stream_profile>();

        auto depth_intrinsics = depth_video_stream.get_intrinsics();
        auto depth_to_color_extrinsics = depth_video_stream.get_extrinsics_to(streams[color_stream_idx]);
        auto color_intrinsics = color_video_stream.get_intrinsics();

        vector<double> depth_intrinsics_vec {
                (double)depth_intrinsics.width, (double)depth_intrinsics.height, depth_intrinsics.ppx,
                depth_intrinsics.ppy, depth_intrinsics.fx, depth_intrinsics.fy
        };

        ostringstream depth_intrinsics_oss;

        std::copy(depth_intrinsics_vec.begin(), depth_intrinsics_vec.end()-1,
                  std::ostream_iterator<double>(depth_intrinsics_oss, " "));

        // Now add the last element with no delimiter
        depth_intrinsics_oss << depth_intrinsics_vec.back() << " ";

        vector<float> depth_coefs(begin(depth_intrinsics.coeffs), end(depth_intrinsics.coeffs));

        std::copy(depth_coefs.begin(), depth_coefs.end()-1,
                  std::ostream_iterator<float>(depth_intrinsics_oss, " "));

        // Now add the last element with no delimiter
        depth_intrinsics_oss << depth_coefs.back() << " ";

        string depth_distortion = rs2_distortion_to_string(depth_intrinsics.model);

        depth_intrinsics_oss << depth_distortion;


        vector<double> color_intrinsics_vec{
                (double)color_intrinsics.width, (double)color_intrinsics.height, color_intrinsics.ppx,
                color_intrinsics.ppy, color_intrinsics.fx, color_intrinsics.fy
        };

        ostringstream color_intrinsics_oss;

        std::copy(color_intrinsics_vec.begin(), color_intrinsics_vec.end()-1,
                  std::ostream_iterator<double>(color_intrinsics_oss, " "));

        // Now add the last element with no delimiter
        color_intrinsics_oss << color_intrinsics_vec.back() << " ";

        vector<float> color_coefs( begin(color_intrinsics.coeffs), end(color_intrinsics.coeffs));

        std::copy(color_coefs.begin(), color_coefs.end()-1,
                  std::ostream_iterator<float>(color_intrinsics_oss, " "));

        // Now add the last element with no delimiter
        color_intrinsics_oss << color_coefs.back() << " ";

        string color_distortion = rs2_distortion_to_string(color_intrinsics.model);

        color_intrinsics_oss << color_distortion;

        vector<double> rotation_coefs(begin(depth_to_color_extrinsics.rotation), end(depth_to_color_extrinsics.rotation));

        ostringstream rotation_oss;

        std::copy(rotation_coefs.begin(), rotation_coefs.end()-1,
                  std::ostream_iterator<double>(rotation_oss, " "));

        // Now add the last element with no delimiter
        rotation_oss << rotation_coefs.back();

        vector<double> translation_coefs(begin(depth_to_color_extrinsics.translation), end(depth_to_color_extrinsics.translation));

        ostringstream translation_oss;

        std::copy(translation_coefs.begin(), translation_coefs.end()-1,
                  std::ostream_iterator<double>(translation_oss, " "));

        // Now add the last element with no delimiter
        translation_oss << translation_coefs.back();

        ofstream intrinsics_extrinsics_file;
        intrinsics_extrinsics_file.open("intrinsics_extrinsics.txt");

        intrinsics_extrinsics_file << "depth " + depth_intrinsics_oss.str() << endl;

        intrinsics_extrinsics_file << "color " + color_intrinsics_oss.str() << endl;

        intrinsics_extrinsics_file << "rotation " + rotation_oss.str()  +" translation " + translation_oss.str() << endl;

        intrinsics_extrinsics_file.close();
    }
};


int main() {

    string save_dir = "G:/Vista_project/";

    //Connect first Thermal Cam with default settings

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

    //Connect second Thermal Cam with default settings

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

    //Additional settings done in wic example code

    // enable advanced features
    wic->iKnowWhatImDoing();
    // enable advanced features
    wic2->iKnowWhatImDoing();
    // set advanced radiometry if core supports it

    // set core gain
    auto gain = wic->setGain(wic::GainMode::High);

    // set core gain
    auto gain2 = wic2->setGain(wic::GainMode::High);

    auto grabber1 = wic->frameGrabber();
    grabber1->setup();

    auto grabber2 = wic2->frameGrabber();
    grabber2->setup();


    //Manual mode of camera adjustment

    auto status1 = wic->setFFCMode(wic::FFCModes::Manual);
    auto status2 = wic2->setFFCMode(wic::FFCModes::Manual);

    auto emode = wic::ExternalSyncMode(0x0002); //0x0001
    auto resp1 = wic->setExternalSyncMode(emode);

    auto emode2 = wic::ExternalSyncMode(0x0001); //0x0002
    auto resp2 = wic2->setExternalSyncMode(emode2);

    //Sanity check with cameras resolutions

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

    //No-Zoom in thermal cams
    auto zoom_video_mode_None = wic::VideoModeZoom(0);
    wic->setVideoModeZoom(zoom_video_mode_None);
    wic2->setVideoModeZoom(zoom_video_mode_None);

    //time to record of a partion between ctx-switch to he next memory-block to write

    int time_to_record = 60;

    //cameras fps
    int rs_fps = 30 ;
    int tc_fps = 9 + 3;

    //depth and rgb params
    int rgb_ch = 3;
    int depth_px_sz = 2;
    int tc_px_sz = 2;

    //memory allocations size for single image and for total of images per
    // memory block (time_to_record function)
    size_t total_tc_size = 640LL * 512 * tc_px_sz * tc_fps * time_to_record;
    size_t tc_size = 640 * 512 * 2;

    long long color_size = 720LL * 1280 * rgb_ch * rs_fps * time_to_record;
    long long depth_size = 720LL * 1280 * depth_px_sz * rs_fps * time_to_record;

    //number of partitions which gives:
    // total time of recording = number_of_records * time_to_record
    int number_of_records = 1;

    vector <vector<long long int>> HT1_tss_vec(number_of_records);
    vector <vector<long long int>> HT2_tss_vec(number_of_records);

    vector <vector<long long int>> color_depth_tss(number_of_records);

    char **tc1_mfd_ptrs = (char **) new char *[number_of_records];
    mapped_file * tc1_mapped_fds = (mapped_file * ) new mapped_file[number_of_records];

    char **tc2_mfd_ptrs = (char **) new char *[number_of_records];
    mapped_file * tc2_mapped_fds = (mapped_file * ) new mapped_file[number_of_records];

    char **color_mfd_ptrs = (char **) new char *[number_of_records];
    mapped_file * color_mapped_fds = (mapped_file * ) new mapped_file[number_of_records];

    char **depth_mfd_ptrs = (char **) new char *[number_of_records];
    mapped_file * depth_mapped_fds = (mapped_file * ) new mapped_file[number_of_records];

    for (int l = 0; l < number_of_records; ++l) {

        string tc1_file_path = save_dir + to_string(l) + +"tc1.bin";
        const char *tc1_FileName = tc1_file_path.c_str();
        const size_t tc1_FileSize = total_tc_size;

        mapped_file_params tc1_params(tc1_FileName);
        tc1_params.new_file_size = tc1_FileSize;
        tc1_params.flags = mapped_file_base::readwrite;
        tc1_mapped_fds[l] = mapped_file(tc1_params);
        tc1_mfd_ptrs[l] = tc1_mapped_fds[l].data();

        string tc2_file_path = save_dir + to_string(l) + "tc2.bin";
        const char *tc2_FileName = tc2_file_path.c_str();
        const size_t tc2_FileSize = total_tc_size;

        mapped_file_params tc2_params(tc2_FileName);
        tc2_params.new_file_size = tc2_FileSize;
        tc2_params.flags = mapped_file_base::readwrite;
        tc2_mapped_fds[l] = mapped_file(tc2_params);
        tc2_mfd_ptrs[l] = tc2_mapped_fds[l].data();

        string c_file_path = save_dir + to_string(l) + "color.bin";
        const char *c_FileName = c_file_path.c_str();
        const std::size_t ColorFileSize = color_size;

        mapped_file_params params_c(c_FileName);
        params_c.new_file_size = ColorFileSize;
        params_c.flags = mapped_file_base::readwrite;
        color_mapped_fds[l] = mapped_file(params_c);
        color_mfd_ptrs[l] = color_mapped_fds[l].data();


        string d_file_path = save_dir + to_string(l) + "depth.bin";
        const char *d_FileName = d_file_path.c_str();
        const std::size_t FileSize = depth_size;

        mapped_file_params params_d(d_FileName);
        params_d.new_file_size = FileSize;
        params_d.flags = mapped_file_base::readwrite;
        depth_mapped_fds[l] = mapped_file(params_d);
        depth_mfd_ptrs[l] = depth_mapped_fds[l].data();

    }

    boost::shared_mutex shared_mux;
    std::mutex tc1_mutex;
    int idx_tc1 = 0;

    auto tc1_callback = TC1Callback(tc1_mfd_ptrs[0], HT1_tss_vec[0], idx_tc1,
                                    tc_size, tc1_mutex, shared_mux);


    std::mutex tc2_mutex;
    int idx_tc2 = 0;

    auto tc2_callback = TC2Callback(tc2_mfd_ptrs[0], HT2_tss_vec[0], idx_tc2,
                                    tc_size, tc2_mutex, shared_mux);


    grabber1->bindBufferHandler(tc1_callback);

    grabber2->bindBufferHandler(tc2_callback);

    std::mutex mux;

    rs2::pipeline pipe;
    rs2::config cfg;


    std::atomic<volatile int> idx_color(0);
    int idx_depth = 0;
    auto rs_callback = RSCallback(color_mfd_ptrs[0], depth_mfd_ptrs[0],
                                  color_depth_tss[0], idx_color, idx_depth, mux,
                                  shared_mux);


    boost::asio::thread_pool thread_pool(number_of_records);

    cout << "Recording Started" << endl;
    cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8);
    cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16);
    rs2::pipeline_profile profiles = pipe.start(cfg, rs_callback);

    rs2::device selected_device = profiles.get_device();
    auto depth_sensor = selected_device.first<rs2::depth_sensor>();

    if (depth_sensor.supports(RS2_OPTION_EMITTER_ENABLED)) {
        depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 1.f); // Enable emitter
        //depth_sensor.set_option(RS2_OPTION_EMITTER_ENABLED, 0.f); // Disable emitter
    }
    if (depth_sensor.supports(RS2_OPTION_LASER_POWER)) {
        // Query min and max values:
        auto range = depth_sensor.get_option_range(RS2_OPTION_LASER_POWER);
        depth_sensor.set_option(RS2_OPTION_LASER_POWER, range.max); // Set max power
        //depth_sensor.set_option(RS2_OPTION_LASER_POWER, 0.f); // Disable laser
    }

    bool start_statusA = grabber1->start();
    //cout << "CamA started succefully : " << start_statusA << endl;
    bool start_statusB = grabber2->start();
    //cout << "CamB started succefully : " << start_statusB << std::endl;

    auto save_intrinsics_extrinsics = SaveIntrinsicsExtrinsics(profiles);

    post(thread_pool, save_intrinsics_extrinsics);

    for (int cur_idx = 0; cur_idx < number_of_records; ++cur_idx) {

        while (idx_color.load() < time_to_record * (rs_fps-2)) { //TODO: the last substracted value is a hyper-parameter (a funtion of time_to_record, the bigger the value
            continue;
        }

        if(cur_idx == number_of_records - 1){
            bool finish_statusB = grabber1->stop();
            //cout << "CamB stoped succefully : " << finish_statusB << endl;
            bool finish_statusA = grabber2->stop();
            //cout << "CamA stoped succefully : " << finish_statusA << endl;
            pipe.stop();
        }


        {
            boost::unique_lock<boost::shared_mutex> lock(shared_mux);

            auto start = high_resolution_clock::now();

            auto save_callback = SaveCallback(color_mapped_fds[cur_idx],
                                              depth_mapped_fds[cur_idx],
                                              tc1_mapped_fds[cur_idx],
                                              tc2_mapped_fds[cur_idx],
                                              rs_callback.color_depth_ts,
                                              tc1_callback.tc_ts,
                                              tc2_callback.tc_ts,
                                              cur_idx, save_dir);

            post(thread_pool, save_callback);

            if(cur_idx == number_of_records-1){
                break;
            }

            *tc1_callback.tc_mfd_ptr = tc1_mfd_ptrs[cur_idx+1];
            tc1_callback.tc_ts = HT1_tss_vec[cur_idx+1];
            tc1_callback.idx_tc = 0;

            *tc2_callback.tc_mfd_ptr = tc2_mfd_ptrs[cur_idx+1];
            tc2_callback.tc_ts = HT2_tss_vec[cur_idx+1];
            tc2_callback.idx_tc = 0;

            *rs_callback.color_mfd_ptr = color_mfd_ptrs[cur_idx+1];
            *rs_callback.depth_mfd_ptr = depth_mfd_ptrs[cur_idx+1];
            rs_callback.color_depth_ts = color_depth_tss[cur_idx+1];
            rs_callback.idx_color.store(0);
            rs_callback.idx_depth = 0;

            auto stop = high_resolution_clock::now();

            auto duration = duration_cast<nanoseconds>(stop - start);

            cout << duration.count() << endl;
        }
    }


    thread_pool.join();
    cout << "Finished";
    return 0;
}
