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

using namespace std;
using namespace this_thread; // sleep_for, sleep_until
using namespace chrono; // nanoseconds, system_clock, seconds
using namespace boost::iostreams;
using namespace boost::interprocess;




/*
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
            __int64 tmp = abs(TC_ts[i] - rs_frames[j].loc_ts);
            if(abs(TC_ts[i] - rs_frames[j].loc_ts) < dist){
                dist = abs(TC_ts[i] - rs_frames[j].loc_ts);
                closest_to_TC[i] = j;
            }
        }
    }

    return closest_to_TC;
}
*/
/*
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

*/



int main() {

    string save_dir = "E:/";

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

    //auto HT_frames_b1 = vector<vector< uint8_t >>();
    //auto HT_frames_b2 = vector<vector< uint8_t >>();
    //auto t1 = high_resolution_clock::now();

    size_t total_tc_size = 640 * 512 * 2 * 9 * 300;
    size_t tc_size = 640 * 512 * 2;

    long long color_size = 720LL * 1280 * 3 * 30 * 300;
    long long depth_size = 720LL * 1280 * 2 * 30 * 300;


    //auto HT_frames_b2_ts = vector<long long int>();

    int number_of_records = 4;

    vector<vector<long long int>> HT_tss_vec(number_of_records);

    vector<vector<long long int>> color_depth_tss(number_of_records);

    char** tc1_mfd_ptrs = (char**) new char*[number_of_records];
    mapped_file* tc1_mapped_fds = (mapped_file*) new mapped_file[number_of_records];

    char** tc2_mfd_ptrs = (char**) new char*[number_of_records];
    mapped_file* tc2_mapped_fds = (mapped_file*) new mapped_file[number_of_records];

    char** color_mfd_ptrs = (char**) new char*[number_of_records];
    mapped_file* color_mapped_fds = (mapped_file*) new mapped_file[number_of_records];

    char** depth_mfd_ptrs = (char**) new char*[number_of_records];
    mapped_file* depth_mapped_fds = (mapped_file*) new mapped_file[number_of_records];

    for (int l = 0; l < number_of_records; ++l) {

        string tc1_file_path = save_dir + to_string(l) +  + "tc1.bin";
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

    int cur_idx = 0;

    auto cur_tc1_mfd = tc1_mapped_fds[0];
    auto cur_tc1_mfd_ptr = tc1_mfd_ptrs[0];
    auto cur_HT_frames_b1_ts = HT_tss_vec[0];

    auto cur_tc2_mfd = tc2_mapped_fds[0];
    auto cur_tc2_mfd_ptr = tc2_mfd_ptrs[0];
    //auto cur_HT_frames_b2_ts = HT_tss_vec[0];

    auto cur_color_mfd = color_mapped_fds[0];
    auto cur_color_mfd_ptr = color_mfd_ptrs[0];

    auto cur_depth_mfd = depth_mapped_fds[0];
    auto cur_depth_mfd_ptr = depth_mfd_ptrs[0];

    auto cur_color_depth_ts = color_depth_tss[0];

    boost::shared_mutex shared_mux;
    std::mutex tc1_mutex;
    int idx_tc1 = 0;

    auto tc1_callback = [&](const vector<uint8_t> &cur_frame) {
        boost::shared_lock< boost::shared_mutex> shared_lock(shared_mux);
        std::lock_guard<std::mutex> lock(tc1_mutex);
        memcpy((void *) ((uint8_t *) cur_tc1_mfd_ptr + idx_tc1 * tc_size), cur_frame.data(), tc_size);
        const std::chrono::time_point<std::chrono::steady_clock> now = high_resolution_clock::now();
        long long int loc_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                now.time_since_epoch()).count();
        cur_HT_frames_b1_ts.push_back(loc_ts);
        idx_tc1++;

    };


    std::mutex tc2_mutex;
    int idx_tc2 = 0;
    auto tc2_callback = [&](const vector<uint8_t> &cur_frame) {
        boost::shared_lock< boost::shared_mutex> shared_lock(shared_mux);
        std::lock_guard<std::mutex> lock(tc2_mutex);
        memcpy((void *) ((uint8_t *) cur_tc2_mfd_ptr + idx_tc2 * tc_size), cur_frame.data(), tc_size);
        //const std::chrono::time_point<std::chrono::steady_clock> now = high_resolution_clock::now();
        //long long int loc_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
        //now.time_since_epoch()).count();
        //HT_frames_b2_ts.push_back(loc_ts);
        idx_tc2++;
    };

    //auto handler_a = handlerA(HT_frames_b1, HT_frames_b1_ts);
    grabber->bindBufferHandler(tc1_callback);

    //auto handler_b = handlerB(HT_frames_b2, HT_frames_b2_ts);
    grabber2->bindBufferHandler(tc2_callback);


    //Define file names



    volatile int idx_depth = 0;
    std::atomic<volatile int> idx_color(0);

    //std::map<int, std::vector<Frame>> frames;
    std::mutex mutex;

    auto callback = [&](const rs2::frame &frame) {
        boost::shared_lock< boost::shared_mutex> shared_lock(shared_mux);
        std::lock_guard<std::mutex> lock(mutex);
        if (rs2::frameset fs = frame.as<rs2::frameset>()) {
            //rs2::disparity_transform disparity2depth(false);
            //fs = fs.apply_filter(disparity2depth);
            // With callbacks, all synchronized stream will arrive in a single frameset
            const std::chrono::time_point<std::chrono::steady_clock> now = high_resolution_clock::now();
            long long int loc_ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now.time_since_epoch()).count();
            cur_color_depth_ts.push_back(loc_ts);
            for (const rs2::frame f: fs) {
                auto vf = f.as<rs2::video_frame>();
                /*
                Frame my_f = Frame(vf.get_data(), vf.get_timestamp(),
                                   vf.get_data_size(), vf.get_width(),
                                   vf.get_height(), vf.get_bytes_per_pixel(),
                                   vf.get_stride_in_bytes(), loc_ts);

                frames[f.get_profile().unique_id()].push_back(my_f);
                */
                if (vf.get_bytes_per_pixel() == 2) {
                    size_t sz = vf.get_data_size();
                    memcpy((void *) ((uint8_t *) cur_depth_mfd_ptr + idx_depth * sz), vf.get_data(), sz);
                    idx_depth++;
                } else {
                    size_t sz = vf.get_data_size();
                    memcpy((void *) ((uint8_t *) cur_color_mfd_ptr + idx_color * sz), vf.get_data(), sz);
                    idx_color.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    };


    rs2::pipeline pipe;
    rs2::config cfg;
    //cfg.enable_stream(RS2_STREAM_ANY, 1280, 720, RS2_FORMAT_ANY, 30);
    //auto t2 = high_resolution_clock::now();
    //cout << duration_cast<milliseconds>(t2 - t1).count() << endl;


    boost::asio::thread_pool thread_pool(4);
    rs2::pipeline_profile profiles = pipe.start(callback);

    bool start_statusA = grabber->start();
    //cout << "CamA started succefully : " << start_statusA << endl;
    bool start_statusB = grabber2->start();
    //cout << "CamB started succefully : " << start_statusB << std::endl;
    for(int not_been = 0; not_been < number_of_records; ++not_been) {

        //sleep_for(nanoseconds(300000000000));

        while(idx_color.load() < 1790){
            continue;
        }

        //cout << "Cams Stopped: " << endl;
        //std::vector<Frame> depth_frames = frames[0];
        //Frame depth_frame0 = depth_frames[0];
        {
            boost::unique_lock< boost::shared_mutex> lock(shared_mux);
            auto save_callback = [&]() {
                cur_color_mfd.close();
                rename((save_dir + to_string(not_been) + "color.bin").c_str(),
                       (save_dir + to_string(not_been) + "color_f.bin").c_str());
                cur_depth_mfd.close();
                rename((save_dir + to_string(not_been) + "depth.bin").c_str(),
                       (save_dir + to_string(not_been) + "depth_f.bin").c_str());

                cur_tc1_mfd.close();
                rename((save_dir + to_string(not_been) + "tc1.bin").c_str(),
                       (save_dir + to_string(not_been) + "tc1_f.bin").c_str());
                cur_tc2_mfd.close();
                rename((save_dir + to_string(not_been) + "tc2.bin").c_str(),
                       (save_dir + to_string(not_been) + "tc2_f.bin").c_str());

                ofstream cd_fout;
                string color_depth_ts_name = save_dir + to_string(not_been) + "color_depth_ts.bin";
                cd_fout.open(color_depth_ts_name, ios::binary | ios::out);
                cd_fout.write((char *) cur_color_depth_ts.data(),
                              cur_color_depth_ts.size() * sizeof(long long int));
                cd_fout.close();
                rename((save_dir + to_string(not_been) + "color_depth_ts.bin").c_str(),
                       (save_dir + to_string(not_been) + "color_depth_ts_f.bin").c_str());

                ofstream tc_fout;
                string tc_ts_name = save_dir + to_string(not_been) + "tc_ts.bin";
                tc_fout.open(tc_ts_name, ios::binary | ios::out);
                tc_fout.write((char *) cur_HT_frames_b1_ts.data(),
                              cur_HT_frames_b1_ts.size() * sizeof(long long int));
                tc_fout.close();
                rename((save_dir + to_string(not_been) + "tc_ts.bin").c_str(),
                       (save_dir + to_string(not_been) + "tc_ts_f.bin").c_str());
            };

            post(thread_pool, save_callback);

            cur_tc1_mfd = tc1_mapped_fds[++cur_idx];
            cur_tc1_mfd_ptr = tc1_mfd_ptrs[cur_idx];
            cur_HT_frames_b1_ts = HT_tss_vec[cur_idx];

            cur_tc2_mfd = tc2_mapped_fds[cur_idx];
            cur_tc2_mfd_ptr = tc2_mfd_ptrs[cur_idx];
            //auto cur_HT_frames_b2_ts = HT_tss_vec[0];

            cur_color_mfd = color_mapped_fds[cur_idx];
            cur_color_mfd_ptr = color_mfd_ptrs[cur_idx];

            cur_depth_mfd = depth_mapped_fds[cur_idx];
            cur_depth_mfd_ptr = depth_mfd_ptrs[cur_idx];

            cur_color_depth_ts = color_depth_tss[cur_idx];
        }



        //auto stop = high_resolution_clock::now();

        // Get duration. Substart timepoints to
        // get durarion. To cast it to proper unit
        // use duration cast method
        //auto duration = duration_cast<milliseconds>(stop - start);

        //cout << duration.count() << endl;

        /*
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
        */

        //std::vector<Frame> rgb_frames2 = frames2[stream_numbers2["Color"]];
        //std::vector<Frame> depth_frames2 = frames2[stream_numbers2["Depth"]];

        //cout << "Cams Stopped: " + to_string(idx_depth) << endl;
        //cout << "Cams Stopped: " + to_string(idx_color) << endl;
        //cout << "Cams Stopped: " + to_string(idx_tc1) << endl;
        //cout << "Cams Stopped: " + to_string(idx_tc2) << endl;
        /*
        int size = rgb_frames.size();
        for (int i = 0; i < size; ++i) {
            ofstream fout;
            string color_frame_name = save_dir + "Color_" + to_string(i);
            Frame cur_color_frame = rgb_frames[i];
            fout.open(color_frame_name, ios::binary | ios::out);
            fout.write((char*)cur_color_frame.frame_data,
                       cur_color_frame.data_size);
            fout.close();
            //delete cur_color_frame.frame_data;
            Frame cur_depth_frame = depth_frames[i];
            string depth_frame_name = save_dir + "Depth_" + to_string(i);
            //auto start = high_resolution_clock::now();
            fout.open(depth_frame_name, ios::binary | ios::out);
            fout.write((char*)cur_depth_frame.frame_data,
                       cur_depth_frame.data_size);
            fout.close();
            //delete cur_depth_frame.frame_data;

        }
        */
        //auto start = high_resolution_clock::now();


        // Get ending timepoint


        /*
        size_t H1_b_size = HT_frames_b1.size();
        size_t H2_b_size = HT_frames_b2.size();
        size_t min_size = H1_b_size < H2_b_size ? H1_b_size : H2_b_size;



        for (int i = 0; i < min_size; ++i) {

            ofstream fout;
            Frame cur_color_frame = rgb_frames[i];
            fout.open(color_frame_name, ios::binary | ios::out);
            fout.write((char*)cur_color_frame.frame_data,
                       cur_color_frame.data_size);
            fout.close();
            //delete cur_color_frame.frame_data;

            Frame cur_depth_frame = depth_frames[i];
            string depth_frame_name = save_dir + "Depth_" + to_string(i);
            //auto start = high_resolution_clock::now();
            fout.open(depth_frame_name, ios::binary | ios::out);
            fout.write((char*)cur_depth_frame.frame_data,
                       cur_depth_frame.data_size);
            fout.close();
            //delete cur_depth_frame.frame_data;



            ofstream camA_stream(save_dir + "H1_"+ to_string(i) +".dat",
                                 ios::out | ios::binary);

            camA_stream.write((const char *)HT_frames_b1[i].data(),
                              sizeof(const char) * HT_frames_b1[i].size());
            camA_stream.close();

            ofstream camB_stream(save_dir + "H2_"+ to_string(i) +".dat",
                                 ios::out | ios::binary);
            camB_stream.write((const char *)HT_frames_b2[i].data(),
                              sizeof(const char) * HT_frames_b2[i].size());

            camB_stream.close();

        }
         */






        //save pairs for illustration
        /*
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
                result = imwrite(save_dir + "pair_"+ to_string(i)+".png", TC_e_color_e_depth, compression_params);
            }
            catch (const cv::Exception& ex)
            {
                fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
            }
            if (!result)
                printf("ERROR: Can't save PNG file.\n");

        }
        */
        //cout << "Finished" << endl;
        //auto stop = high_resolution_clock::now();
        //cout << duration_cast<milliseconds>(stop - start).count() << endl;
    }
    bool finish_statusA = grabber->stop();
    //cout << "CamA stoped succefully : " << finish_statusA << endl;
    bool finish_statusB = grabber2->stop();
    //cout << "CamB stoped succefully : " << finish_statusB << endl;

    pipe.stop();
    
    thread_pool.join();

    //cout << duration_cast<milliseconds>(not_been2[1] - not_been2[0]).count() << endl;
    return 0;

}
