#include "include/common.h"
// Extracts a point cloud from the internal volume
// This is CUDA code; compile with nvcc
// Author: Christian Diller, git@christian-diller.de

namespace rosefusion {
    namespace internal {
        namespace cuda {

            __global__
            void extract_points_kernel(const PtrStep<short> tsdf_volume, const PtrStep<short> weight_volume,const PtrStep<uchar3> color_volume,
                                       const int3 volume_size, const float voxel_scale,
                                       PtrStep<float3> vertices, PtrStep<float3> normals, PtrStep<uchar3> color, // 这三个是输出点云的坐标、法向量、颜色
                                       int *point_num)                                                           // 点的数量
            {
                const int x = blockIdx.x * blockDim.x + threadIdx.x;
                const int y = blockIdx.y * blockDim.y + threadIdx.y;

                // 处理边界，保证x+1,y+1,z+1存在
                if (x >= volume_size.x - 1 || y >= volume_size.y - 1)
                    return;
                // 对一列体素进行处理
                for (int z = 0; z < volume_size.z - 1; ++z) {

                    // 提取tsdf
                    const float tsdf = static_cast<float>(tsdf_volume.ptr(z * volume_size.y + y)[x]) * DIVSHORTMAX;
                    // 如果tsdf接近 1和-1 则不处理
                    if (fabs(tsdf-1)<1e-5 || tsdf <= -0.99f || tsdf >= 0.99f)
                        continue;
                    // 提取(x+1)，(y+1)，(z+1)处的体素
                    short vx = tsdf_volume.ptr((z) * volume_size.y + y)[x + 1];
                    short vy = tsdf_volume.ptr((z) * volume_size.y + y + 1)[x];
                    short vz = tsdf_volume.ptr((z + 1) * volume_size.y + y)[x];

                    // 提取上述三个位置的权重
                    short w_vx = weight_volume.ptr((z) * volume_size.y + y)[x + 1];
                    short w_vy = weight_volume.ptr((z) * volume_size.y + y + 1)[x];
                    short w_vz = weight_volume.ptr((z + 1) * volume_size.y + y)[x];
                    if (w_vx <= 0 || w_vy <= 0 || w_vz <= 0)
                        continue;

                    // 三个位置的tsdf转float
                    const float tsdf_x = static_cast<float>(vx) * DIVSHORTMAX;
                    const float tsdf_y = static_cast<float>(vy) * DIVSHORTMAX;
                    const float tsdf_z = static_cast<float>(vz) * DIVSHORTMAX;

                    // x，y，z三个方向分别计算，如果当前值与该值逆号则为true
                    const bool is_surface_x = ((tsdf > 0) && (tsdf_x < 0)) || ((tsdf < 0) && (tsdf_x > 0));
                    const bool is_surface_y = ((tsdf > 0) && (tsdf_y < 0)) || ((tsdf < 0) && (tsdf_y > 0));
                    const bool is_surface_z = ((tsdf > 0) && (tsdf_z < 0)) || ((tsdf < 0) && (tsdf_z > 0));

                    // 有一个方向逆号，当前点为表面
                    if (is_surface_x || is_surface_y || is_surface_z) {
                        // 计算法线
                        Eigen::Vector3f normal;
                        normal.x() = (tsdf_x - tsdf);
                        normal.y() = (tsdf_y - tsdf);
                        normal.z() = (tsdf_z - tsdf);
                        if (normal.norm() == 0)
                            continue;
                        normal.normalize();

                        int count = 0;
                        if (is_surface_x) count++;
                        if (is_surface_y) count++;
                        if (is_surface_z) count++;
                        int index = atomicAdd(point_num, count); // 将求和结果写回到 address 指针指向的内存地址中，并返回未做计算前的旧值

                        // 初始化表面点的坐标，直接使用体素转为空间坐标
                        Vec3fda position((static_cast<float>(x) + 0.5f) * voxel_scale,
                                         (static_cast<float>(y) + 0.5f) * voxel_scale,
                                         (static_cast<float>(z) + 0.5f) * voxel_scale);
                        // 如果在x方向上认为是表面，插值计算坐标，修改上面初始化的坐标（似乎有问题，三个方向都认为满足的话，则会直接添加三个点）
                        if (is_surface_x) {
                            position.x() = position.x() - (tsdf / (tsdf_x - tsdf)) * voxel_scale;

                            vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};
                            normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                            color.ptr(0)[index] = color_volume.ptr(z * volume_size.y + y)[x];
                            index++;
                        }
                        if (is_surface_y) {
                            position.y() -= (tsdf / (tsdf_y - tsdf)) * voxel_scale;

                            vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};;
                            normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                            color.ptr(0)[index] = color_volume.ptr(z * volume_size.y + y)[x];
                            index++;
                        }
                        if (is_surface_z) {
                            position.z() -= (tsdf / (tsdf_z - tsdf)) * voxel_scale;

                            vertices.ptr(0)[index] = float3{position(0), position(1), position(2)};;
                            normals.ptr(0)[index] = float3{normal(0), normal(1), normal(2)};
                            color.ptr(0)[index] = color_volume.ptr(z * volume_size.y + y)[x];
                            index++;
                        }
                    }
                }
            }

            // 从体素提取三维点云
            PointCloud extract_points(const VolumeData& volume, const int buffer_size)
            {
                CloudData cloud_data { buffer_size }; // 缓存

                dim3 threads(32, 32);
                dim3 blocks((volume.volume_size.x + threads.x - 1) / threads.x,
                            (volume.volume_size.y + threads.y - 1) / threads.y);

                extract_points_kernel<<<blocks, threads>>>(volume.tsdf_volume, volume.weight_volume,volume.color_volume, // （输入）三个体素网格
                        volume.volume_size, volume.voxel_scale,                                                          // （输入）体素网格尺寸和尺度
                        cloud_data.vertices, cloud_data.normals, cloud_data.color,                                       // （输出）顶点，法线，颜色
                        cloud_data.point_num);                                                                           // （输出）点云数量

                cudaDeviceSynchronize();
                cloud_data.download();

                return PointCloud {cloud_data.host_vertices, cloud_data.host_normals,
                                   cloud_data.host_color, cloud_data.host_point_num};
            }
        }
    }
}