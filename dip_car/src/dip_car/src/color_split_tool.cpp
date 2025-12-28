#include <stdlib.h>
#include <string>
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;

/**
 * @brief 对BGR图像进行直方图均衡化
 * @param src 输入图像
 * @param dst 输出图像
 * 
 * 分别对B、G、R三个通道进行直方图均衡化，然后合并
 */
void equalizeHistBGR(Mat &src, Mat &dst);

/**
 * @brief 手动颜色分割函数
 * @param hsv_input 输入HSV图像
 * @param thresholded_output 输出二值化图像
 * @param window_name 显示窗口名称
 * 
 * 创建滑动条动态调整HSV阈值，实时显示分割效果
 */
void colorSplitManual(const Mat &hsv_input, Mat &thresholded_output, const char* window_name);

/**
 * @brief 图像回调函数
 * @param msg ROS图像消息
 * 
 * 接收ROS图像消息，进行直方图均衡化和颜色分割
 */
void imageCallback(const sensor_msgs::Image::ConstPtr& msg);

/**
 * @brief 主函数
 * @param argc 参数个数
 * @param argv 参数数组
 * @return 程序执行状态
 */
int main(int argc, char **argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "color_split_tool");
    ros::NodeHandle nh;
    
    // 订阅相机图像话题
    ros::Subscriber sub = nh.subscribe("camera/image", 1, imageCallback);

    // ROS事件循环
    ros::spin();
    return 0;
}

void colorSplitManual(const Mat &hsv_input, Mat &thresholded_output, const char* window_name)
{
    // 静态变量：HSV阈值（保持滑动条位置）
    static int hmin = 0, hmax = 255;
    static int smin = 0, smax = 255;
    static int vmin = 0, vmax = 255;
    
    // 创建滑动条窗口和控件
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, 800, 300);  // 设置窗口大小
    
    // 创建HSV各通道的滑动条
    createTrackbar("Hue Min", window_name, &hmin, 255);
    createTrackbar("Hue Max", window_name, &hmax, 255);
    createTrackbar("Sat Min", window_name, &smin, 255);
    createTrackbar("Sat Max", window_name, &smax, 255);
    createTrackbar("Val Min", window_name, &vmin, 255);
    createTrackbar("Val Max", window_name, &vmax, 255);

    // 根据当前阈值进行颜色分割
    inRange(hsv_input, 
            Scalar(hmin, smin, vmin), 
            Scalar(hmax, smax, vmax), 
            thresholded_output);
    
    // 调整显示大小（如果图像太大）
    if (thresholded_output.cols > 640)
        resize(thresholded_output, thresholded_output, Size(960, 540));
    
    // 显示分割结果
    imshow(window_name, thresholded_output);
}

void equalizeHistBGR(Mat &src, Mat &dst)
{
    // 分离BGR三个通道
    std::vector<Mat> channels;
    split(src, channels);

    // 分别对每个通道进行直方图均衡化
    for (int i = 0; i < 3; i++)
        equalizeHist(channels[i], channels[i]);

    // 合并处理后的通道
    merge(channels, dst);
}

void imageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    try {
        // 将ROS图像消息转换为OpenCV格式
        Mat img_raw = cv_bridge::toCvShare(msg, "bgr8")->image;
        
        // 显示原始图像
        imshow("Raw Image", img_raw);

        // 直方图均衡化
        Mat img_equalized = img_raw.clone();
        // equalizeHistBGR(img_raw, img_equalized);
        // imshow("equalized", img_equalized);
        
        // 转换为HSV颜色空间
        Mat img_hsv, img_hsv_split;
        cvtColor(img_equalized, img_hsv, COLOR_BGR2HSV);
        
        // 进行手动颜色分割
        colorSplitManual(img_hsv, img_hsv_split, "HSV Thresholding");

        waitKey(30);  // 刷新显示
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge异常: %s", e.what());
        return;
    }
}
