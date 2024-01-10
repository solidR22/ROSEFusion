#include <rosefusion.h>
#include <iostream>
#include <fstream>

using cv::cuda::GpuMat;
using Matf61da = Eigen::Matrix<double, 6, 1, Eigen::DontAlign>;

namespace rosefusion {

    Pipeline::Pipeline(const CameraParameters _camera_config,
                       const DataConfiguration _data_config,
                       const ControllerConfiguration _controller_config) :
            //  生成参数对象  
            camera_parameters(_camera_config), data_config(_data_config),
            controller_config(_controller_config),
            // 设置volume
            volume(data_config.volume_size, data_config.voxel_scale),
            frame_data(_camera_config.image_height, _camera_config.image_width),
            particle_leve{10240, 3072, 1024}, PST(particle_leve, _controller_config.PST_path),
            search_data(particle_leve),
            // 初始化数据: 清空位姿,轨迹等
            current_pose{}, poses{}, frame_id{0}, last_model_frame{}, iter_tsdf{_controller_config.init_fitness} {
        // The pose starts in the middle of the cube, offset along z by the initial depth
        // 第一帧的相机位姿设置在 Volume 的中心, 然后在z轴上拉远一点
        current_pose.setIdentity();
        current_pose(0, 3) = data_config.init_pos.x;
        current_pose(1, 3) = data_config.init_pos.y;
        current_pose(2, 3) = data_config.init_pos.z;
    }
    // 每一帧的数据处理都要调用这个函数
    // HERE
    bool Pipeline::process_frame(const cv::Mat_<float> &depth_map, const cv::Mat_<cv::Vec3b> &color_map,
                                 cv::Mat &shaded_img) {
        // STEP 1: Surface measurement
        // 主要工作：
        // 计算顶点和法向

        internal::surface_measurement(
            color_map,                          // 输入的彩色图
            depth_map,                          // 输入的深度图
            frame_data,                         // 保存当前帧的所有数据，顶点和法向量
            camera_parameters,                  // 文件读取的相机参数
            data_config.depth_cutoff_distance); // 文件读取的配置参数
        
        // STEP 2: pose estimation
        // 主要工作：位姿计算
        bool tracking_success{true};
        if (frame_id > 0) {
            tracking_success = internal::pose_estimation(
                    volume,                     // 体素网格
                    PST,                        // 预采样的粒子集，6D位姿列表，索引0-20：10240 *6，20-40：3072 *6，40-60：1024 *6
                    search_data,                // 粒子的搜索数据，在粒子估计中使用
                    current_pose,               // 输入: 上一帧的相机位姿; 输出: 当前帧得到的相机位姿
                    frame_data,                 // 当前帧的彩色图/深度图/顶点图/法向图数据
                    camera_parameters,          // 相机内参
                    controller_config,          // 控制参数
                    particle_leve,              // 粒子等级：[10240, 3072, 1024]
                    &iter_tsdf,                 // 文件参数 init_fitness
                    &previous_frame_success,    // 上一帧是否成功
                    initialize_search_size      // 初始6D位姿的搜索范围
            );

        }
        // 如果 icp 过程不成功, 那么就说明当前失败了
        if (!tracking_success)
            // icp失败之后本次处理退出,但是上一帧推理的得到的平面将会一直保持, 每次新来一帧都会重新icp后一直都在尝试重新icp, 尝试重定位回去
            return false;
        // 记录当前帧的位姿
        poses.push_back(current_pose);


        // STEP 3: Surface reconstruction
        // 进行表面重建的工作, 其实是是将当前帧的观测信息融合到Global Volume
        internal::cuda::surface_reconstruction(frame_data.depth_map, frame_data.color_map, // 读取的深度图和RGB图
                                               volume, camera_parameters,                  // 体素，相机参数
                                               data_config.truncation_distance,            // 截断距离u
                                               current_pose.inverse());                    // 相机外参 -- 其实这里可以加速的, 直接对Eigen::Matrix4f求逆有点耗时间
        

        // Step 4: Surface prediction
        // 在当前帧的位姿上得到对表面的推理结果，渲染表面
        if (controller_config.render_surface) {
            internal::cuda::surface_prediction(volume,  // Global Volume
                                               frame_data.shading_buffer,
                                               camera_parameters, 
                                               data_config.truncation_distance,   // 截断距离
                                               data_config.init_pos,
                                               shaded_img,                        // 输出
                                               current_pose);                     // 当前时刻的相机位姿(注意没有取逆)
        }
        // 帧id++
        ++frame_id;
        return true;
    }

    // 保存计算的位姿
    void Pipeline::get_poses() const {
        Eigen::Matrix4d init_pose = poses[0];
        std::ofstream trajectory;
        trajectory.open(data_config.result_path + data_config.seq_name + ".txt");
        std::cout << data_config.result_path + data_config.seq_name + ".txt" << std::endl;
        int iter_count = 0; // 第几帧
        for (auto pose: poses) {
            Eigen::Matrix4d temp_pose = init_pose.inverse() * pose;
            Eigen::Matrix3d rotation_m = temp_pose.block(0, 0, 3, 3);
            Eigen::Vector3d translation = temp_pose.block(0, 3, 3, 1) / 1000;
            Eigen::Quaterniond q(rotation_m);
            if(!vTimestamps.empty())
                trajectory << std::setprecision(18) << vTimestamps[iter_count] << " " << translation.x() << " " << translation.y() << " " << translation.z() << \
                " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            else
                trajectory << iter_count << " " << translation.x() << " " << translation.y() << " " << translation.z() << \
                " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
            iter_count++;
        }
        trajectory.close();
    }

    // PointCloud包含顶点坐标、法线、颜色、顶点数量
    PointCloud Pipeline::extract_pointcloud() const {
        // 从体素提取出三维点云
        PointCloud cloud_data = internal::cuda::extract_points(volume, data_config.pointcloud_buffer_size);
        return cloud_data;
    }


    void export_ply(const std::string &filename, const PointCloud &point_cloud) {
        std::ofstream file_out{filename};
        if (!file_out.is_open())
            return;

        file_out << "ply" << std::endl;
        file_out << "format ascii 1.0" << std::endl;
        file_out << "element vertex " << point_cloud.num_points << std::endl;
        file_out << "property float x" << std::endl;
        file_out << "property float y" << std::endl;
        file_out << "property float z" << std::endl;
        file_out << "property float nx" << std::endl;
        file_out << "property float ny" << std::endl;
        file_out << "property float nz" << std::endl;
        file_out << "property uchar red" << std::endl;
        file_out << "property uchar green" << std::endl;
        file_out << "property uchar blue" << std::endl;
        file_out << "end_header" << std::endl;

        for (int i = 0; i < point_cloud.num_points; ++i) {
            float3 vertex = point_cloud.vertices.ptr<float3>(0)[i];
            float3 normal = point_cloud.normals.ptr<float3>(0)[i];
            uchar3 color = point_cloud.color.ptr<uchar3>(0)[i];
            file_out << vertex.x << " " << vertex.y << " " << vertex.z << " " << normal.x << " " << normal.y << " "
                     << normal.z << " ";
            file_out << static_cast<int>(color.x) << " " << static_cast<int>(color.y) << " "
                     << static_cast<int>(color.z) << std::endl;
        }
    }


}