#include <rosefusion.h>
#include <DataReader.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <fstream>
#include <pangolin/pangolin.h>
#include <sl/Camera.hpp>
#include <stdio.h>
#include <string.h>
using namespace std;
using namespace sl;



// Handle the CTRL-C keyboard signal
static bool exit_app = false;
#ifdef _WIN32
#include <Windows.h>
void CtrlHandler(DWORD fdwCtrlType) {
    exit_app = (fdwCtrlType == CTRL_C_EVENT);
}
#else
#include <signal.h>
void nix_exit_handler(int s) {
    exit_app = true;
}
#endif

// Set the function to handle the CTRL-C
void SetCtrlHandler() {
#ifdef _WIN32
    SetConsoleCtrlHandler((PHANDLER_ROUTINE) CtrlHandler, TRUE);
#else // unix
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = nix_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
#endif
}


// Sample functions
void print(string msg_prefix, ERROR_CODE err_code = ERROR_CODE::SUCCESS, string msg_suffix = "");
vector< string> split(const string& s, char seperator) {
    vector< string> output;
    string::size_type prev_pos = 0, pos = 0;

    while ((pos = s.find(seperator, pos)) != string::npos) {
        string substring(s.substr(prev_pos, pos - prev_pos));
        output.push_back(substring);
        prev_pos = ++pos;
    }

    output.push_back(s.substr(prev_pos, pos - prev_pos));
    return output;
}

void setStreamParameter(InitParameters& init_p, string& argument) {
    vector< string> configStream = split(argument, ':');
    String ip(configStream.at(0).c_str());
    if (configStream.size() == 2) {
        init_p.input.setFromStream(ip, atoi(configStream.at(1).c_str()));
    } else init_p.input.setFromStream(ip);
}


int main(int argc,char* argv[]){

    std::cout<<"Read configure file\n";
    const std::string camera_file(argv[1]);
    const std::string data_file(argv[2]);
    const std::string controller_file(argv[3]);

    std::cout<<"Init configuration\n";
    printf("%s\n",camera_file.c_str());
    printf("%s\n",data_file.c_str());
    printf("%s\n",controller_file.c_str());

    // 获取ZED数据 
    Camera zed;
    // Set configuration parameters for the ZED
    InitParameters init_parameters;
    // init_parameters.depth_mode = DEPTH_MODE::NONE;
    init_parameters.sdk_verbose = true;
    init_parameters.depth_mode = sl::DEPTH_MODE::NEURAL;
	init_parameters.coordinate_units = sl::UNIT::MILLIMETER;
	init_parameters.depth_minimum_distance = 200;
	init_parameters.depth_maximum_distance = 1000;
	init_parameters.depth_stabilization = 100;
	init_parameters.camera_disable_self_calib = true;
    init_parameters.enable_image_enhancement = true;
    init_parameters.enable_right_side_measure = false;
    init_parameters.camera_resolution = RESOLUTION::VGA;

    string stream_params;
    stream_params = string("192.168.11.150:30000");
    setStreamParameter(init_parameters, stream_params);
    // Open the camera
    auto returned_state = zed.open(init_parameters);
    if (returned_state != ERROR_CODE::SUCCESS) {
        print("Camera Open", returned_state, "Exit program.");
        return EXIT_FAILURE;
    }
    // Print camera information
    auto camera_info = zed.getCameraInformation();
    cout << endl;
    cout << "ZED Model                 : " << camera_info.camera_model << endl;
    cout << "ZED Serial Number         : " << camera_info.serial_number << endl;
    cout << "ZED Camera Firmware       : " << camera_info.camera_configuration.firmware_version << "/" << camera_info.sensors_configuration.firmware_version << endl;
    cout << "ZED Camera Resolution     : " << camera_info.camera_configuration.resolution.width << "x" << camera_info.camera_configuration.resolution.height << endl;
    cout << "ZED Camera FPS            : " << zed.getInitParameters().camera_fps << endl;

    auto zed_info = zed.getCameraInformation();
    int width = zed_info.camera_configuration.resolution.width;
	int height = zed_info.camera_configuration.resolution.height;

    rosefusion::CameraParameters camera_config(camera_file);
    const rosefusion::DataConfiguration data_config(data_file);
    const rosefusion::ControllerConfiguration controller_config(controller_file);

    camera_config.focal_x = zed_info.camera_configuration.calibration_parameters.left_cam.fx;
    camera_config.focal_y = zed_info.camera_configuration.calibration_parameters.left_cam.fy;
    camera_config.principal_x = zed_info.camera_configuration.calibration_parameters.left_cam.cx;
    camera_config.principal_y = zed_info.camera_configuration.calibration_parameters.left_cam.cy;
    camera_config.image_height = camera_info.camera_configuration.resolution.height;
    camera_config.image_width = camera_info.camera_configuration.resolution.width;


    pangolin::View color_cam;
    pangolin::View shaded_cam; 
    pangolin::View depth_cam; 

    pangolin::GlTexture imageTexture;
    pangolin::GlTexture shadTexture;
    pangolin::GlTexture depthTexture;

    if (controller_config.render_surface){

        pangolin::CreateWindowAndBind("Main",1280,720);

        color_cam = pangolin::Display("color_cam")
            .SetAspect((float)camera_config.image_width/(float)camera_config.image_height);
        shaded_cam = pangolin::Display("shaded_cam")
            .SetAspect((float)camera_config.image_width/(float)camera_config.image_height);
        depth_cam = pangolin::Display("depth_cam")
            .SetAspect((float)camera_config.image_width/(float)camera_config.image_height);

        pangolin::Display("window")
            .SetBounds(0.0, 1.0, 0.0, 1.0 )
            .SetLayout(pangolin::LayoutEqual)
            .AddDisplay(shaded_cam)
            .AddDisplay(color_cam)
            .AddDisplay(depth_cam);


    
        imageTexture=pangolin::GlTexture(camera_config.image_width,camera_config.image_height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
        shadTexture=pangolin::GlTexture(camera_config.image_width,camera_config.image_height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
        depthTexture=pangolin::GlTexture(camera_config.image_width,camera_config.image_height,GL_LUMINANCE,false,0,GL_LUMINANCE,GL_UNSIGNED_BYTE);
    }
    cv::Mat shaded_img(camera_config.image_height, camera_config.image_width,CV_8UC3);

    rosefusion::Pipeline pipeline { camera_config, data_config, controller_config };


    // DataReader d_reader(data_config.seq_file,false);
    clock_t time_stt=clock( );

    
    // Create a Mat to store images
    //获取图像
	sl::Mat imageLeft = sl::Mat();
	sl::Mat imageRight = sl::Mat();
	sl::Mat disparity = sl::Mat();
	sl::Mat disparity_show = sl::Mat();
	sl::Mat depth = sl::Mat();
	sl::Mat depth_show = sl::Mat();

    SetCtrlHandler();
    RuntimeParameters runParameters;
    // Setting the depth confidence parameters
    runParameters.enable_depth = true;
    runParameters.enable_fill_mode = false;
    runParameters.confidence_threshold = 10;
    runParameters.texture_confidence_threshold = 100;
    runParameters.remove_saturated_areas = true;
    while(!exit_app){
        returned_state = zed.grab(runParameters);
        if (returned_state == ERROR_CODE::SUCCESS) {
            // Retrieve left image
            // 左图
            zed.retrieveImage(imageLeft, VIEW::LEFT);
            cv::Mat imgleft(height, width, CV_8UC4, imageLeft.getPtr<sl::uchar1>(sl::MEM::CPU));

            // 拆分图像通道
            std::vector<cv::Mat> channels;
            cv::split(imgleft, channels);

            // 创建一个只有前三个通道的图像
            cv::merge(std::vector<cv::Mat>{channels[0], channels[1], channels[2]}, imgleft);


            imgleft.convertTo(imgleft, CV_8UC3);
            // 深度图
            zed.retrieveMeasure(depth, sl::MEASURE::DEPTH);
			cv::Mat imgDep(height, width, CV_32FC1, depth.getPtr<sl::float1>(sl::MEM::CPU));
            cv::patchNaNs(imgDep, 0);
			// imgDep = imgDep * 16;
			// imgDep.convertTo(imgDep, CV_16UC1);
			//获取视差图
			// zed.retrieveMeasure(disparity, sl::MEASURE::DISPARITY);
			// cv::Mat imgDis(height, width, CV_32FC1, disparity.getPtr<sl::float1>(sl::MEM::CPU));
			// imgDis = imgDis * -160;
			// imgDis.convertTo(imgDis, CV_16UC1);

            bool success = pipeline.process_frame(imgDep, imgleft, shaded_img);

            if (!success){
                std::cout << "Frame could not be processed" << std::endl;
            }


            if (controller_config.render_surface){
                glClear(GL_COLOR_BUFFER_BIT);

                color_cam.Activate();
                imageTexture.Upload(imgleft.data,GL_BGR,GL_UNSIGNED_BYTE);
                imageTexture.RenderToViewportFlipY();
                depth_cam.Activate();

                imgDep.convertTo(imgDep,CV_8U,256/5000.0);
                depthTexture.Upload(imgDep.data,GL_LUMINANCE,GL_UNSIGNED_BYTE);
                depthTexture.RenderToViewportFlipY();

                if (success){
                    shaded_cam.Activate();
                    shadTexture.Upload(shaded_img.data,GL_BGR,GL_UNSIGNED_BYTE);
                    shadTexture.RenderToViewportFlipY();
                }
                pangolin::FinishFrame();

            }
        }
    }
    zed.close();

    if (controller_config.save_trajectory){
        pipeline.get_poses();
    }

    if (controller_config.save_scene){
        auto points = pipeline.extract_pointcloud();
        rosefusion::export_ply(data_config.result_path+data_config.seq_name+"_points.ply",points);
    }


    return 0;
}

void print(string msg_prefix, ERROR_CODE err_code, string msg_suffix) {
    cout << "[Sample]";
    if (err_code != ERROR_CODE::SUCCESS)
        cout << "[Error] ";
    else
        cout << " ";
    cout << msg_prefix << " ";
    if (err_code != ERROR_CODE::SUCCESS) {
        cout << " | " << toString(err_code) << " : ";
        cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        cout << " " << msg_suffix;
    cout << endl;
}