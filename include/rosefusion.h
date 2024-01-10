#ifndef ROSEFUSION_H
#define ROSEFUSION_H

#include "data_types.h"
using Matf61da = Eigen::Matrix<double, 6, 1, Eigen::DontAlign>;

namespace rosefusion {

    class Pipeline {
    public:

        Pipeline(const CameraParameters _camera_config,
                 const DataConfiguration _data_config,
                 const ControllerConfiguration _controller_config);

        ~Pipeline() = default;

        /**
         * Invoke this for every frame you want to fuse into the global volume
         * @param depth_map The depth map for the current frame. 
         * ! Must consist of float values representing the depth in mm
         * @param color_map The RGB color map. Must be a matrix (datatype CV_8UC3)
         * @return Whether the frame has been fused successfully. Will only be false if the ICP failed.
         */
        bool process_frame(const cv::Mat_<float>& depth_map, const cv::Mat_<cv::Vec3b>& color_map, cv::Mat& shaded_img);
        /**
         * Retrieve all camera poses computed so far
         * @return A vector for 4x4 camera poses, consisting of rotation and translation
         */
        void get_poses() const;
        /**
         * Extract a point cloud
         * @return A PointCloud representation (see description of PointCloud for more information on the data layout)
         */
        PointCloud extract_pointcloud() const;


    private:
        const CameraParameters camera_parameters;
        const DataConfiguration data_config;
        const ControllerConfiguration controller_config;
        const std::vector<int> particle_leve;

        float iter_tsdf;
        internal::VolumeData volume;        // 体素信息
        internal::QuaternionData PST; 
        internal::SearchData search_data;
        internal::FrameData frame_data;     // 当前帧的所有数据
        Eigen::Matrix4d current_pose;
        std::vector<Eigen::Matrix4d> poses; // 存储所有的位姿
        bool previous_frame_success=false;
        Matf61da initialize_search_size;
        size_t frame_id;
        cv::Mat last_model_frame;
    };

    void export_ply(const std::string& filename, const PointCloud& point_cloud);



    namespace internal {


        void surface_measurement(const cv::Mat_<cv::Vec3b>& color_map,
                                      const cv::Mat_<float>& depth_map,
                                      FrameData& frame_data,
                                      const CameraParameters& camera_params,
                                      const float depth_cutoff);




        bool pose_estimation(const VolumeData& volume,
                             const QuaternionData& quaternions,
                              SearchData& search_data,
                             Eigen::Matrix4d& pose,
                             FrameData& frame_data,
                             const CameraParameters& cam_params,
                             const ControllerConfiguration& controller_config,
                             const std::vector<int> particle_level,
                             float * iter_tsdf,
                             bool * previous_frame_success,
                             Matf61da& initialize_search_size
                            );
        namespace cuda {


            void surface_reconstruction(const cv::cuda::GpuMat& depth_image,
                                        const cv::cuda::GpuMat& color_image,
                                        VolumeData& volume,
                                        const CameraParameters& cam_params,
                                        const float truncation_distance,
                                        const Eigen::Matrix4d& model_view);


            void surface_prediction(const VolumeData& volume,
                                    cv::cuda::GpuMat& shading_buffer,
                                    const CameraParameters& cam_parameters,
                                    const float truncation_distance,
                                    const float3 init_pos,
                                    cv::Mat& shaded_img,
                                    const Eigen::Matrix4d& pose);

            // 从体素提取出三维点云
            PointCloud extract_points(const VolumeData& volume, const int buffer_size);

        }

    }
}
#endif 
