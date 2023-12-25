#ifndef DATAREADER_H
#define DATAREADER_H


#include <iostream>
#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>



class DataReader
{
    public:
    DataReader(std::string file, bool flipColors);
    virtual ~DataReader();
    int getFramesNum();
    void getNextFrame(cv::Mat& Color_mat,cv::Mat& Depth_mat); // 每次取一帧
    bool hasMore();

    private:
    std::vector<cv::Mat> v_color; // 初始化时读取所有的数据
    std::vector<cv::Mat> v_depth; // 初始化时读取所有的数据

    void readFile();
    FILE * fp;
    int numFrames;  // 输入
    int height;     // 输入
    int width;      // 输入
    int currentFrame;

};

#endif