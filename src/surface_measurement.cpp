#include <rosefusion.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Weffc++"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#pragma GCC diagnostic pop

using cv::cuda::GpuMat;

namespace rosefusion {
    namespace internal {

        namespace cuda { 
            /**
             * @brief 计算某层深度图像的顶点图
             * @param[in]  depth_map        某层滤波后的深度图
             * @param[out] vertex_map       计算得到的顶点图
             * @param[in]  depth_cutoff     不考虑的过远的点的距离
             * @param[in]  cam_params       该层图像下的相机内参
             */
            void compute_vertex_map(const GpuMat& depth_map, GpuMat& vertex_map, const float depth_cutoff,
                                    const CameraParameters cam_params);
            /**
             * @brief 根据某层顶点图计算法向图
             * @param[in]  vertex_map       某层顶点图
             * @param[out] normal_map       计算得到的法向图
             */
            void compute_normal_map(const GpuMat& vertex_map, GpuMat& normal_map);
        }
        // 计算输入深度图的顶点\法向金字塔
        void surface_measurement(const cv::Mat_<cv::Vec3b>& color_frame,
                                      const cv::Mat_<float>& depth_frame,
                                      FrameData& frame_data,
                                      const CameraParameters& camera_params,
                                      const float depth_cutoff)
        {

            frame_data.color_map.upload(color_frame);
            frame_data.depth_map.upload(depth_frame);
            cuda::compute_vertex_map(frame_data.depth_map, frame_data.vertex_map,
                                     depth_cutoff, camera_params);
            cuda::compute_normal_map(frame_data.vertex_map, frame_data.normal_map);

        }
    }
}