#include <rosefusion.h>
#include <DataReader.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <fstream>
#include <pangolin/pangolin.h>



int main(int argc,char* argv[]){

    std::cout<<"Read configure file\n";
    const std::string camera_file(argv[1]);
    const std::string data_file(argv[2]);
    const std::string controller_file(argv[3]);

    std::cout<<"Init configuration\n";
    printf("%s\n",camera_file.c_str());
    printf("%s\n",data_file.c_str());
    printf("%s\n",controller_file.c_str());

    const rosefusion::CameraParameters camera_config(camera_file); // 给相机赋值参数
    const rosefusion::DataConfiguration data_config(data_file);    // 定义TSDF参数
    const rosefusion::ControllerConfiguration controller_config(controller_file);  // 运行时的参数

    // 创建三个视图
    pangolin::View color_cam;
    pangolin::View shaded_cam; 
    pangolin::View depth_cam; 

    // 创建glTexture容器用于读取图像
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

        // 创建显示窗口
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

    rosefusion::Pipeline pipeline { camera_config, data_config, controller_config }; // 初始化相机位姿和上面定义的参数

    clock_t time_stt=clock( );
    cv::Mat color_img;
    cv::Mat depth_map;  
    int n_imgs=0;
    
    std::cout<<"Read seq file: "<<data_config.seq_file<<"\n";

    DataReader d_reader(data_config.seq_file,false);

    while( d_reader.hasMore()){


        printf("n:%d\n",n_imgs);

        d_reader.getNextFrame(color_img,depth_map);
        bool success = pipeline.process_frame(depth_map, color_img, shaded_img);

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

        n_imgs++;

    }


    std::cout <<"time per frame="<<1000*(clock()-time_stt)/(double)CLOCKS_PER_SEC/n_imgs<<"ms"<<std::endl;

    if (controller_config.save_trajectory){
        pipeline.get_poses(); // 保存位姿
    }

    if (controller_config.save_scene){
        auto points = pipeline.extract_pointcloud();
        rosefusion::export_ply(data_config.result_path+data_config.seq_name+"_points.ply",points); // 保存三维点
    }


    return 0;
}
