#include <rosefusion.h>
#include <DataReader.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <fstream>
#include <pangolin/pangolin.h>

using namespace std;

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


void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);
int main(int argc,char* argv[]){

    std::cout<<"Read configure file\n";
    const std::string camera_file(argv[1]);
    const std::string data_file(argv[2]);
    const std::string controller_file(argv[3]);

    std::cout<<"Init configuration\n";
    printf("%s\n",camera_file.c_str());
    printf("%s\n",data_file.c_str());
    printf("%s\n",controller_file.c_str());

    const rosefusion::CameraParameters camera_config(camera_file);
    const rosefusion::DataConfiguration data_config(data_file);
    const rosefusion::ControllerConfiguration controller_config(controller_file);

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

    // Retrieve paths to images
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesD;
    std::vector<double> vTimestamps;
    // associate文件的路径
    std::string strAssociationFilename = std::string(data_config.association_file);
    cout<<"associate文件路径"<<strAssociationFilename<<endl;
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);
    cout<<"Load done!"<<endl;
    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    SetCtrlHandler();
    // DataReader d_reader(data_config.seq_file,false);
    clock_t time_stt=clock( );
    cv::Mat color_img;
    cv::Mat depth_map;  
    
    for(int ni=0; ni<nImages && !exit_app; ni++){
        // Read image and depthmap from file
        color_img = cv::imread(data_config.data_path+"/"+vstrImageFilenamesRGB[ni],cv::IMREAD_COLOR); //,cv::IMREAD_UNCHANGED);
        depth_map = cv::imread(data_config.data_path+"/"+vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED); //,cv::IMREAD_UNCHANGED);
        // depth_map.convertTo(depth_map,CV_16UC1,1/16.0);
        depth_map.convertTo(depth_map,CV_32FC1,1/5.0);
        if(color_img.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << data_config.data_path+"/"+vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        bool success = pipeline.process_frame(depth_map, color_img,shaded_img);

        if (!success){
            std::cout << "Frame could not be processed" << std::endl;
        }


        if (controller_config.render_surface){
            glClear(GL_COLOR_BUFFER_BIT);

            color_cam.Activate();
            imageTexture.Upload(color_img.data,GL_BGR,GL_UNSIGNED_BYTE);
            imageTexture.RenderToViewportFlipY();
            depth_cam.Activate();

            depth_map.convertTo(depth_map,CV_8U,256/5000.0);
            depthTexture.Upload(depth_map.data,GL_LUMINANCE,GL_UNSIGNED_BYTE);
            depthTexture.RenderToViewportFlipY();

            if (success){
                shaded_cam.Activate();
                shadTexture.Upload(shaded_img.data,GL_BGR,GL_UNSIGNED_BYTE);
                shadTexture.RenderToViewportFlipY();
            }
            pangolin::FinishFrame();

        }
    }


    std::cout <<"time per frame="<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC/nImages<<"ms"<<std::endl;

    if (controller_config.save_trajectory){
        pipeline.get_poses();
    }

    if (controller_config.save_scene){
        auto points = pipeline.extract_pointcloud();
        rosefusion::export_ply(data_config.result_path+data_config.seq_name+"_points.ply",points);
    }


    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        // cout<<s<<endl;
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            string t;
            string sRGB, sD;
            ss >> t;
            // vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}