#ifndef ROSEFUSION_DATA_TYPES_H
#define ROSEFUSION_DATA_TYPES_H

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Eigen>
#include <time.h>
#else
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <Eigen/Dense>
#endif



using cv::cuda::GpuMat;

namespace rosefusion {

    struct CameraParameters {
        int image_width {0};
        int image_height {0};
        float focal_x {.0};
        float focal_y {.0};
        float principal_x {.0};
        float principal_y {.0};


        CameraParameters(const std::string &camera_file)
        {   
            cv::FileStorage cameraSetting(camera_file.c_str(),cv::FileStorage::READ);
            cameraSetting["Camera.fx"]>>focal_x;
            cameraSetting["Camera.fy"]>>focal_y;
            cameraSetting["Camera.cx"]>>principal_x;
            cameraSetting["Camera.cy"]>>principal_y;

            cameraSetting["Camera.width"]>>image_width;
            cameraSetting["Camera.height"]>>image_height;

            cameraSetting.release();

        }

    };

    /**
     *
     * \brief Representation of a cloud of three-dimensional points (vertices)
     *
     * This data structure contains
     * (1) the world coordinates of the vertices,
     * (2) the corresponding normals and
     * (3) the corresponding RGB color value
     *
     * - vertices: A 1 x buffer_size opencv matrix with CV_32FC3 values, representing the coordinates
     * - normals: A 1 x buffer_size opencv matrix with CV_32FC3 values, representing the normal direction
     * - color: A 1 x buffer_size opencv matrix with CV_8UC3 values, representing the RGB color
     *
     * Same indices represent the same point
     *
     * The total number of valid points in those buffers is stored in num_points
     *
     */
    struct PointCloud {
        cv::Mat vertices;
        cv::Mat normals;
        cv::Mat color;
        int num_points;
    };





    struct ControllerConfiguration {
        int max_iteration {20};
        std::string PST_path {"~"};
        float scaling_coefficient1 {0.12};
        float scaling_coefficient2 {0.12};
        float init_fitness {0.5};
        float momentum {0.9};
        bool scaling_inherit_directly {false};
        bool save_trajectory {false};
        bool save_scene {false};
        bool render_surface {false};

        ControllerConfiguration(const std::string &config_file){
            cv::FileStorage controllerSetting(config_file.c_str(),cv::FileStorage::READ);
            controllerSetting["PST_path"]>>PST_path;
            max_iteration=controllerSetting["max_iteration"];
            scaling_coefficient1=controllerSetting["scaling_coefficient1"];
            scaling_coefficient2=controllerSetting["scaling_coefficient2"];
            init_fitness=controllerSetting["init_fitness"];
            momentum=controllerSetting["momentum"];

            scaling_inherit_directly=bool(int(controllerSetting["scaling_inherit_directly"]));
            save_trajectory=bool(int(controllerSetting["save_trajectory"]));
            save_scene=bool(int(controllerSetting["save_scene"]));
            render_surface=bool(int(controllerSetting["render_surface"]));
        }
    };


    struct DataConfiguration {

        int3 volume_size { make_int3(812, 512, 812) };

        float voxel_scale { 30.f };

        float3 init_pos { volume_size.x / 2 * voxel_scale, volume_size.x / 2 * voxel_scale ,  volume_size.x / 2 * voxel_scale };

        float truncation_distance { 120.f };

        float depth_cutoff_distance { 8000.f };

        int pointcloud_buffer_size { 3 * 2000000 };

        std::string result_path {"~/"}; // 存储结果的位置
        std::string seq_file {"~/"};    // 读取seq数据
        std::string seq_name {"~/"};    // 没啥用
        std::string data_path {"~/"};   // 目录
        std::string association_file {"~/"};  


        DataConfiguration(const std::string &config_file){

            cv::FileStorage dataSetting(config_file.c_str(),cv::FileStorage::READ);
            std::string temp_str;
            dataSetting["result_path"]>>result_path;

            // 都是用来设置体素网格volume
            voxel_scale=dataSetting["voxel_size"];               // 体素块的尺寸
            truncation_distance=dataSetting["truncated_size"];   // 截断距离
            int voxel_x=dataSetting["voxel_x"];
            int voxel_y=dataSetting["voxel_y"];
            int voxel_z=dataSetting["voxel_z"];
            volume_size=make_int3(voxel_x,voxel_y,voxel_z);      //体素块的分辨率

            float init_x=dataSetting["init_x"];
            float init_y=dataSetting["init_y"];
            float init_z=dataSetting["init_z"];

            float init_pos_x=volume_size.x / 2 * voxel_scale - init_x; // 世界坐标系的原点坐标
            float init_pos_y=volume_size.y / 2 * voxel_scale - init_y;
            float init_pos_z=volume_size.z / 2 * voxel_scale - init_z;
            init_pos=make_float3(init_pos_x,init_pos_y,init_pos_z);

            dataSetting["seq_path"]>>seq_file;
            dataSetting["name"]>>seq_name;
            dataSetting["data_path"]>>data_path;
            dataSetting["association_file"]>>association_file;
            dataSetting.release();
        }
       
    };

    // 这个名字空间中定义了一些在KinectFusion中使用的关于模型的数据类型
    namespace internal {
        /*
         * Contains the internal data representation of one single frame as read by the depth camera
         * Consists of depth, smoothed depth and color pyramids as well as vertex and normal pyramids
         */
        // - 保存了每一帧的输入数据的特性
        struct FrameData {
            GpuMat depth_map;            
            GpuMat color_map;
            GpuMat vertex_map;
            GpuMat normal_map;
            GpuMat shading_buffer;


            explicit FrameData(const int image_height,const int image_width) 
            { 
                depth_map = cv::cuda::createContinuous(image_height, image_width, CV_32FC1);
                color_map = cv::cuda::createContinuous(image_height, image_width, CV_8UC3);
                vertex_map = cv::cuda::createContinuous(image_height, image_width, CV_32FC3);
                normal_map = cv::cuda::createContinuous(image_height, image_width, CV_32FC3);
                shading_buffer = cv::cuda::createContinuous(image_height, image_width, CV_8UC3);
            }


        };


        // 体素：TSDF、权重、颜色、尺寸、体素大小
        struct VolumeData {
            GpuMat tsdf_volume; 
            GpuMat weight_volume; 
            GpuMat color_volume; 
            int3 volume_size;
            float voxel_scale;

            VolumeData(const int3 _volume_size, const float _voxel_scale) :
                    tsdf_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_16SC1)),
                    weight_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_16SC1)),
                    color_volume(cv::cuda::createContinuous(_volume_size.y * _volume_size.z, _volume_size.x, CV_8UC3)),
                    volume_size(_volume_size), voxel_scale(_voxel_scale)
            {
                tsdf_volume.setTo(32767); // 初始化为32767，使用时/32767转为float，相当于存储的是1，但是此时权重为0，所以不影响计算
                weight_volume.setTo(0);
                color_volume.setTo(0);
            }
        };


        struct QuaternionData{
            std::vector<GpuMat> q;
            std::vector<cv::Mat> q_trans;
            int num=20;

            // 初始化粒子，从PST中读取
            QuaternionData(std::vector<int> particle_level, std::string PST_path):
            q(60),q_trans(60)
            {

                // 第一层 0-20，粒子数10240
                for (int i=0;i<num;i++){
                    q_trans[i]=cv::Mat(particle_level[0],6,CV_32FC1);  // row表示粒子数，col表示6个参数，即6D位姿
                    q[i]=cv::cuda::createContinuous(particle_level[0], 6, CV_32FC1);

                    // 初始化的粒子来自于PST(采样模板)，第一行设为全0，1，2为平移，3，4，5为旋转四元数的虚部
                    q_trans[i]=cv::imread(PST_path+"pst_10240_"+std::to_string(i)+".tiff",cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
                    q_trans[i].ptr<float>(0)[0]=0;
                    q_trans[i].ptr<float>(0)[1]=0;
                    q_trans[i].ptr<float>(0)[2]=0;
                    q_trans[i].ptr<float>(0)[3]=0;
                    q_trans[i].ptr<float>(0)[4]=0;
                    q_trans[i].ptr<float>(0)[5]=0;
                    q[i].upload(q_trans[i]);

                }

                // 第二层 20-40，粒子数3072
                for (int i=num;i<num*2;i++){
                    q_trans[i]=cv::Mat(particle_level[1],6,CV_32FC1);
                    q[i]=cv::cuda::createContinuous(particle_level[1], 6, CV_32FC1);

                    q_trans[i]=cv::imread(PST_path+"pst_3072_"+std::to_string(i-20)+".tiff",cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
                    q_trans[i].ptr<float>(0)[0]=0;
                    q_trans[i].ptr<float>(0)[1]=0;
                    q_trans[i].ptr<float>(0)[2]=0;
                    q_trans[i].ptr<float>(0)[3]=0;
                    q_trans[i].ptr<float>(0)[4]=0;
                    q_trans[i].ptr<float>(0)[5]=0;
                    q[i].upload(q_trans[i]);

                }

                // 第三层 40-60，粒子数1024
                for (int i=num*2;i<num*3;i++){
                    q_trans[i]=cv::Mat(particle_level[2],6,CV_32FC1);
                    q[i]=cv::cuda::createContinuous(particle_level[2], 6, CV_32FC1);

                    q_trans[i]=cv::imread(PST_path+"pst_1024_"+std::to_string(i-40)+".tiff",cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
                    q_trans[i].ptr<float>(0)[0]=0;
                    q_trans[i].ptr<float>(0)[1]=0;
                    q_trans[i].ptr<float>(0)[2]=0;
                    q_trans[i].ptr<float>(0)[3]=0;
                    q_trans[i].ptr<float>(0)[4]=0;
                    q_trans[i].ptr<float>(0)[5]=0;
                    q[i].upload(q_trans[i]);
                }
            }

        };

        struct SearchData{
            std::vector<GpuMat> gpu_search_count;
            std::vector<cv::Mat> search_count;
            std::vector<GpuMat> gpu_search_value;
            std::vector<cv::Mat> search_value;

            SearchData(std::vector<int> particle_level):
            gpu_search_count(3),search_count(3),gpu_search_value(3),search_value(3)
            {
                

                search_count[0]=cv::Mat::zeros(particle_level[0],1,CV_32FC1);
                search_count[1]=cv::Mat::zeros(particle_level[1],1,CV_32FC1);
                search_count[2]=cv::Mat::zeros(particle_level[2],1,CV_32FC1);

                gpu_search_count[0]=cv::cuda::createContinuous(particle_level[0], 1, CV_32FC1);
                gpu_search_count[1]=cv::cuda::createContinuous(particle_level[1], 1, CV_32FC1);
                gpu_search_count[2]=cv::cuda::createContinuous(particle_level[2], 1, CV_32FC1);

                search_value[0]=cv::Mat::zeros(particle_level[0],1,CV_32FC1);
                search_value[1]=cv::Mat::zeros(particle_level[1],1,CV_32FC1);
                search_value[2]=cv::Mat::zeros(particle_level[2],1,CV_32FC1);

                gpu_search_value[0]=cv::cuda::createContinuous(particle_level[0], 1, CV_32FC1);
                gpu_search_value[1]=cv::cuda::createContinuous(particle_level[1], 1, CV_32FC1);
                gpu_search_value[2]=cv::cuda::createContinuous(particle_level[2], 1, CV_32FC1);



 
            }

        };

        
        /*
         *
         * \brief Contains the internal pointcloud representation
         *
         * This is only used for exporting the data kept in the internal volumes
         *
         * It holds GPU containers for vertices, normals and vertex colors
         * It also contains host containers for this data and defines the total number of points
         *
         */
        struct CloudData {
            GpuMat vertices;
            GpuMat normals;
            GpuMat color;

            cv::Mat host_vertices;
            cv::Mat host_normals;
            cv::Mat host_color;

            int* point_num;
            int host_point_num;

            explicit CloudData(const int max_number) :
                    vertices{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
                    normals{cv::cuda::createContinuous(1, max_number, CV_32FC3)},
                    color{cv::cuda::createContinuous(1, max_number, CV_8UC3)},
                    host_vertices{}, host_normals{}, host_color{}, point_num{nullptr}, host_point_num{}
            {
                vertices.setTo(0.f);
                normals.setTo(0.f);
                color.setTo(0.f);

                cudaMalloc(&point_num, sizeof(int));
                cudaMemset(point_num, 0, sizeof(int));
            }

            CloudData(const CloudData&) = delete;
            CloudData& operator=(const CloudData& data) = delete;

            void download()
            {
                vertices.download(host_vertices);
                normals.download(host_normals);
                color.download(host_color);

                cudaMemcpy(&host_point_num, point_num, sizeof(int), cudaMemcpyDeviceToHost);
            }
        };


    }
}

#endif 
