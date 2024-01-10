#include <rosefusion.h>
// This is the CPU part of the ICP implementation
// Author: Christian Diller, git@christian-diller.de
using Matf31da = Eigen::Matrix<double, 3, 1, Eigen::DontAlign>;
using Matf61da = Eigen::Matrix<double, 6, 1, Eigen::DontAlign>;
// 后缀 rm = Row Major
using Matrix3frm = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;


namespace rosefusion {
    namespace internal {

        namespace cuda {  // Forward declare CUDA functions
            void estimate_step(const Eigen::Matrix3f &rotation_current, const Matf31da &translation_current,
                               const cv::cuda::GpuMat &vertex_map_current, const cv::cuda::GpuMat &normal_map_current,
                               const Eigen::Matrix3f &rotation_previous_inv, const Matf31da &translation_previous,
                               const CameraParameters &cam_params,
                               const cv::cuda::GpuMat &vertex_map_previous, const cv::cuda::GpuMat &normal_map_previous,
                               float distance_threshold,     // ICP 过程中视为外点的距离阈值
                               float angle_threshold,        // ICP 过程中视为外点的角度阈值
                               Eigen::Matrix<double, 6, 6, Eigen::RowMajor> &A, Eigen::Matrix<double, 6, 1> &b);

            bool
            particle_evaluation(const VolumeData &volume, const QuaternionData &quaterinons, SearchData &search_data,
                                const Eigen::Matrix3d &rotation_current, const Matf31da &translation_current,
                                const cv::cuda::GpuMat &vertex_map_current, const cv::cuda::GpuMat &normal_map_current,
                                const Eigen::Matrix3d &rotation_previous_inv, const Matf31da &translation_previous,
                                const CameraParameters &cam_params, const int particle_index, const int particle_size,
                                const Matf61da &search_size, const int resolution_level, const int level_index,
                                Eigen::Matrix<double, 7, 1> &mean_transform, float *tsdf
            );
        }


        void update_search_size(const float tsdf, const float scaling_coefficient, Matf61da &search_size,
                                Eigen::Matrix<double, 7, 1> &mean_transform) {
            // 加上1e-3可能是为了防止分母为0
            double s_tx = fabs(mean_transform(0, 0)) + 1e-3;
            double s_ty = fabs(mean_transform(1, 0)) + 1e-3;
            double s_tz = fabs(mean_transform(2, 0)) + 1e-3;

            double s_qx = fabs(mean_transform(4, 0)) + 1e-3;
            double s_qy = fabs(mean_transform(5, 0)) + 1e-3;
            double s_qz = fabs(mean_transform(6, 0)) + 1e-3;

            // 计算模长，用于归一化
            double trans_norm = sqrt(s_tx * s_tx + s_ty * s_ty + s_tz * s_tz + s_qx * s_qx + s_qy * s_qy + s_qz * s_qz);

            double normal_tx = s_tx / trans_norm;
            double normal_ty = s_ty / trans_norm;
            double normal_tz = s_tz / trans_norm;
            double normal_qx = s_qx / trans_norm;
            double normal_qy = s_qy / trans_norm;
            double normal_qz = s_qz / trans_norm;

            // 这里的1e-3 参考论文Eq.17，用于防止PST退化
            search_size(3, 0) = scaling_coefficient * tsdf * normal_qx + 1e-3;
            search_size(4, 0) = scaling_coefficient * tsdf * normal_qy + 1e-3;
            search_size(5, 0) = scaling_coefficient * tsdf * normal_qz + 1e-3;
            search_size(0, 0) = scaling_coefficient * tsdf * normal_tx + 1e-3;
            search_size(1, 0) = scaling_coefficient * tsdf * normal_ty + 1e-3;
            search_size(2, 0) = scaling_coefficient * tsdf * normal_tz + 1e-3;
        }


        bool pose_estimation(const VolumeData &volume,                              // 体素网格
                             const QuaternionData &quaternions,                     // 预采样的粒子集，6D位姿列表，索引0-20：10240 *6，20-40：3072 *6，40-60：1024 *6
                             SearchData &search_data,
                             Eigen::Matrix4d &pose,                                 // 输入: 上一帧的相机位姿; 输出: 当前帧得到的相机位姿
                             FrameData &frame_data,                                 // 当前帧的彩色图/深度图/顶点图/法向图数据
                             const CameraParameters &cam_params,                    // 相机内参
                             const ControllerConfiguration &controller_config,      // 控制参数
                             const std::vector<int> particle_level,                 // 粒子等级：[10240, 3072, 1024]
                             float *iter_tsdf,
                             bool *previous_frame_success,                          // 上一帧是否成功
                             Matf61da &initialize_search_size                       // 初始的搜索范围
        ) {
            // step 0 数据准备
            // Get initial rotation and translation
            // 其实就是得到的上一帧的相机旋转和平移, 如果是放在迭代过程中看的话, 其实就是在进行第一次迭代之前, 相机的位姿
            Eigen::Matrix3d current_global_rotation = pose.block(0, 0, 3, 3);
            Eigen::Vector3d current_global_translation = pose.block(0, 3, 3, 1);

            // 上一帧相机的旋转, 外参表示, 可以将世界坐标系下的点转换到相机坐标系下
            Eigen::Matrix3d previous_global_rotation_inverse(current_global_rotation.inverse());
            Eigen::Vector3d previous_global_translation = pose.block(0, 3, 3, 1);


            float beta = controller_config.momentum;  // 上一次迭代`search_size`在与本次迭代`search_size`加权时的权重，论文取0.1
            Matf61da previous_search_size;  // 上一次迭代的搜索范围
            Matf61da search_size;  // 搜索范围，即论文中PST的r

            // 这里的搜索范围是每帧开始时，还未进行迭代的搜索范围，即搜索范围初值
            // 如果上一帧成功了,且配置项`scaling_inherit_directly`为true，则继承上一帧的搜索范围
            if (*previous_frame_success && controller_config.scaling_inherit_directly) {
                search_size << initialize_search_size(0, 0),
                        initialize_search_size(1, 0),
                        initialize_search_size(3, 0),
                        initialize_search_size(4, 0),
                        initialize_search_size(5, 0),
                        initialize_search_size(0, 0);

            } else {
                // 如果上一帧没有成功，或者配置项`scaling_inherit_directly`为false，则根据`iter_tsdf`乘配置项的`scaling_coefficient1`参数得到搜索范围
                float lens = controller_config.scaling_coefficient1 * (*iter_tsdf);
                search_size << lens, lens, lens, lens, lens, lens;
            }

            // 长度是20对应，每层的粒子只有20个模板，对应迭代次数为20
            // 分三层，加20、40的偏移量是取其他层的粒子
            int particle_index[20] = {0, 1 + 20, 2 + 40, 3, 4 + 20, 5 + 40, 6 + 0, 7 + 20, 8 + 40,
                                      9 + 0, 10 + 20, 11 + 40, 12 + 0, 13 + 20, 14 + 40,
                                      15 + 0, 16 + 20, 17 + 40, 18 + 0, 19 + 20};
            // level表示采样层级，1层粒子有10240，2层3072，3层1024
            int level[20] = {32, 16, 8, 32, 16, 8, 32, 16, 8, 32, 16, 8, 32, 16, 8, 32, 16, 8, 32, 16};


            int count_particle = 0;
            int level_index = 5;  // level_index是下采样步长范围内的一个值，随机取增加泛化程度
            bool success = true;  // 本次迭代是否成功
            bool previous_success = true;  // 上一次迭代是否成功

            int count = 0;  // 迭代次数
            int count_success = 0;  // 迭代成功数（可以修改为bool，下面只是判断是否为0，成功一次就算成功，）
            float min_tsdf;
            // 旋转四元数的虚部
            double qx;
            double qy;
            double qz;

            // 粒子估计迭代搜索位姿主循环
            while (true) {
                // 存储位姿的变换矩阵，初始化为全0
                Eigen::Matrix<double, 7, 1> mean_transform = Eigen::Matrix<double, 7, 1>::Zero();
                // 到达最大迭代次数 退出
                // ？只有这一种退出条件，论文中还有其他的退出条件，但是实现没看到
                if (count == controller_config.max_iteration) {
                    break;
                }
                // 上一次迭代失败，这里不是又重复执行了吗？
                if (!success) {
                    count_particle = 0;
                }
                // 执行一次粒子估计
                success = cuda::particle_evaluation(
                        volume,                                                         // 体素网格
                        quaternions,                                                    // 预采样粒子集合
                        search_data,                                                    // 搜索数据，存储粒子的tsdf累积误差和搜索顶点数
                        current_global_rotation,                                        // 旋转矩阵3*3
                        current_global_translation,                                     // 平移3*1
                        frame_data.vertex_map,                                          // 顶点图
                        frame_data.normal_map,                                          // 法向图
                        previous_global_rotation_inverse,                               // 上一帧的旋转
                        previous_global_translation,                                    // 上一帧的平移
                        cam_params,                                                     // 相机参数，投影到图像平面用
                        particle_index[count_particle],                                 // 粒子索引
                        particle_level[particle_index[count_particle] / 20],            // 迭代时10240，3072，1024循环，表示粒子数量
                        search_size,                                                    // PST的r，其实就是PST六个参数的缩放系数，通过优化这个缩放系数来逼近最优位姿
                        level[count_particle],                                          // 深度图下采样倍数：32,16,8循环
                        level_index,                                                    // 下采样步长范围内的一个值，随机取增加泛化程度
                        mean_transform,                                                 // 位姿变换矩阵
                        &min_tsdf                                                       // 归一化后的tsdf平均差值
                );
                // 粒子估计失败，使用上次的最优位姿下的tsdf差
                if (count == 0 && !success) {
                    *iter_tsdf = min_tsdf;
                }

                qx = mean_transform(4, 0);
                qy = mean_transform(5, 0);
                qz = mean_transform(6, 0);

                // 搜索成功，更新位姿，用于下一次迭代
                if (success) {
                    if (count_particle < 19) {
                        ++count_particle;

                    }
                    ++count_success;  // 成功次数自增
                    // 平移增量t_12
                    auto camera_translation_incremental = mean_transform.head<3>();

                    double qw = mean_transform(3, 0);
                    // 旋转增量R_12
                    Eigen::Matrix3d camera_rotation_incremental;
                    camera_rotation_incremental << 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz +
                                                                                                              qy * qw),
                            2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
                            2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy);

                    // 迭代位姿
                    current_global_translation = current_global_translation + camera_translation_incremental * 1000;
                    current_global_rotation = camera_rotation_incremental * current_global_rotation;

                }

                // 下采样的时候随机取采样步长范围内的点
                level_index += 5;
                level_index = level_index % level[count_particle];

                // 论文中的PST update
                update_search_size(min_tsdf, controller_config.scaling_coefficient2, search_size, mean_transform);
                // 连续两次迭代成功，加权前后两次的搜索尺度，更新搜索尺度，论文Eq.18
                if (previous_success && success) {
                    search_size(0, 0) = beta * search_size(0, 0) + (1 - beta) * previous_search_size(0, 0);
                    search_size(1, 0) = beta * search_size(1, 0) + (1 - beta) * previous_search_size(1, 0);
                    search_size(2, 0) = beta * search_size(2, 0) + (1 - beta) * previous_search_size(2, 0);
                    search_size(3, 0) = beta * search_size(3, 0) + (1 - beta) * previous_search_size(3, 0);
                    search_size(4, 0) = beta * search_size(4, 0) + (1 - beta) * previous_search_size(4, 0);
                    search_size(5, 0) = beta * search_size(5, 0) + (1 - beta) * previous_search_size(5, 0);
                    previous_search_size << search_size(0, 0), search_size(1, 0), search_size(2, 0),
                            search_size(3, 0), search_size(4, 0), search_size(5, 0);

                } else if (success) {
                    // 上帧没成功，则直接使用本次的搜索尺度
                    previous_search_size << search_size(0, 0), search_size(1, 0), search_size(2, 0),
                            search_size(3, 0), search_size(4, 0), search_size(5, 0);
                }
                // 上一次迭代是否成功
                if (success) {
                    previous_success = true;
                } else {
                    previous_success = false;
                }

                if (count == 0) {
                    // 仅第一次迭代时执行
                    if (success) {
                        // 本帧的第一次迭代成功的话更新下一帧的初始搜索范围为本次的搜索范围
                        initialize_search_size << search_size(0, 0), search_size(1, 0), search_size(2, 0),
                                search_size(3, 0), search_size(4, 0), search_size(5, 0);
                        *previous_frame_success = true;  // 第一次成功就算成功
                    } else {
                        *previous_frame_success = false;
                    }
                }
                ++count;
            }  // 迭代结束

            // 迭代到达最大次数，没有一次粒子估计成功，标志为位姿估计失败
            if (count_success == 0) {
                return false;
            }

            // 位姿估计成功, 更新位姿
            pose.block(0, 0, 3, 3) = current_global_rotation;
            pose.block(0, 3, 3, 1) = current_global_translation;

            return true;
        }
    }
}
