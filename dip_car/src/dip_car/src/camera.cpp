#include <stdlib.h>
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

// 相机状态枚举
enum CameraState {
    COMPUTER = 0, // 计算机本地摄像头
    ZED, // ZED双目摄像头
    REALSENSE // Intel RealSense深度摄像头
};

// 当前使用的相机类型
CameraState cameraState = REALSENSE;
using namespace cv;

// 全局变量：用于存储RealSense相机图像
Mat frame_msg;

/**
 * @brief RealSense相机回调函数
 * @param img 接收到的ROS图像消息
 * 
 * 当使用RealSense相机时，该函数被调用以获取图像数据
 */
void rcvCameraCallBack(const sensor_msgs::Image::ConstPtr& img)
{
    // cv_bridge::CvImageConstPtr cv_ptr;
    // // 将ROS图像消息转换为OpenCV格式
    // cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);
    // frame_msg = cv_ptr->image;

    cv_bridge::CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvShare(img, sensor_msgs::image_encodings::BGR8);frame_msg = cv_ptr->image;
}

/**
 * @brief 相机节点主函数
 * @param argc 参数个数
 * @param argv 参数数组
 * @return 程序执行状态
 * 
 * 功能：从指定相机获取图像，发布到ROS话题供其他节点使用
 */
int main(int argc, char **argv)
{
    // ROS节点初始化
    ros::init(argc, argv, "camera");
    ros::NodeHandle n;
    
    // ROS订阅者和发布者声明
    ros::Subscriber camera_sub; // 仅RealSense相机使用
    ros::Publisher camera_pub = n.advertise<sensor_msgs::Image>("camera/image", 1);
    
    // OpenCV视频捕获对象
    VideoCapture capture;

    // 根据相机状态初始化对应的相机
    switch (cameraState) {
    case COMPUTER:
        // 打开计算机本地摄像头
        capture.open(0);
        if (!capture.isOpened())
        {
            ROS_ERROR("计算机摄像头无法打开。");
            return -1;
        }
        ROS_INFO("成功连接到计算机摄像头。");
        break;
    
    case ZED:
        // 打开ZED相机
        capture.open(2);
        if (!capture.isOpened())
        {
            ROS_ERROR("ZED相机无法打开。");
            return -1;
        }
        ROS_INFO("成功连接到ZED相机。");
        break;

    case REALSENSE:
        // 订阅RealSense相机的话题
        camera_sub = n.subscribe("/camera/color/image_raw", 1, rcvCameraCallBack);
        waitKey(1000); // 等待回调函数接收数据
        ROS_INFO("Ready to receive Realsense");
        break;

    default:
        ROS_ERROR("未知的相机状态。");
        return -1;
    }

    Mat frIn; // 临时存储捕获的帧
    
    // 主循环：持续捕获和发布图像
    while (ros::ok())
    {
        switch (cameraState) {
        case COMPUTER:
            capture.read(frIn);
            if (frIn.empty())
            {
                ROS_ERROR("从计算机摄像头未获取到图像。");
                continue;
            }
            break;
        
        case ZED:
            capture.read(frIn);
            if (frIn.empty())
            {
                ROS_ERROR("从ZED相机未获取到图像。");
                return -1;
            }
            // ZED相机输出左右眼图像，这里只取左眼图像
            frIn = frIn(Rect(0, 0, frIn.cols / 2, frIn.rows));
            break;

        case REALSENSE:
            // if (frame_msg.cols == 0)
            // {
            //     ROS_ERROR("Failed to receive RealSense");
            //     ros::spinOnce();
            //     continue;
            // }
            frIn = frame_msg;
            break;

        default:
            break;
        }

        // 如果成功获取图像，则发布到ROS话题
        if (!frIn.empty())
        {
            sensor_msgs::ImagePtr msg = cv_bridge::CvImage(
                std_msgs::Header(), "bgr8", frIn).toImageMsg();
            camera_pub.publish(msg);
        }

        waitKey(10); // 控制发布频率
        ros::spinOnce(); // 处理回调函数
    }

    return 0;
}
