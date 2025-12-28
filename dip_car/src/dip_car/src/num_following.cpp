#include <stdlib.h>
#include <vector>
#include <string>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include <geometry_msgs/Twist.h>
#include "sensor_msgs/Image.h"
#include <math.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <algorithm>

using namespace cv;
using namespace std;

std::mutex img_mutex;
Mat img_raw;

/**
 * @brief 从文件加载数字模板
 */
bool loadDigitTemplatesFromFiles();

/**
 * @brief 加载数字模板（优先从文件，失败则使用内置模板）
 */
void loadDigitTemplates();

/**
 * @brief 图像回调函数
 */
void imageCallback(const sensor_msgs::Image::ConstPtr& msg);

/**
 * @brief 直接在全图像中检测数字
 */
int detectDigitDirectly(Mat& src, Rect& digitRect, double& matchScore);

/**
 * @brief 基于数字区域计算控制指令
 */
geometry_msgs::Twist calculateControlByDigit(Mat& src, Rect& digitRect, int detectedDigit, double matchScore);

// 模板图像
Mat template1, template2, template3;
bool templates_loaded = false;

// 控制参数
double g_targetArea = 10000.0; // 目标区域面积
const double AREA_TOLERANCE = 0.15; // 面积容差
const double MAX_SPEED = 0.3; // 最大速度
const double MIN_SPEED = 0.1; // 最小速度

// 数字检测参数
const int MIN_DIGIT_AREA = 500;
const int MAX_DIGIT_AREA = 50000;
const double MATCH_THRESHOLD = 0.5;  // 模板匹配阈值(yushehaode)

// 模板文件路径
std::string template_paths[3] = {
    "pic/template_0.png",
    "pic/template_1.png", 
    "pic/template_2.png"
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "num_following_node");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("camera/image", 1, imageCallback);
    ros::Publisher vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    
    loadDigitTemplates();
    ROS_INFO("Number following node started.");
    
    int lost_frames = 0;
    const int MAX_LOST_FRAMES = 10;
    
    while (ros::ok())
    {
        ros::spinOnce();
        
        Mat local_img;
        {
            std::lock_guard<std::mutex> lock(img_mutex);
            if (img_raw.empty())
            {
                ros::Duration(0.05).sleep();
                continue;
            }
            local_img = img_raw.clone();
        }
        
        waitKey(30);
        if (local_img.empty()) continue;
        
        Mat img_show = local_img.clone();
        
        // 直接在全图像中检测数字
        Rect digitRect;
        double matchScore = 0;
        int detectedDigit = detectDigitDirectly(local_img, digitRect, matchScore);
        
        // 计算控制指令
        geometry_msgs::Twist cmd;
        if (detectedDigit >= 0)
        {
            cmd = calculateControlByDigit(local_img, digitRect, detectedDigit, matchScore);
            lost_frames = 0;
        }
        else
        {
            lost_frames++;
            if (lost_frames > MAX_LOST_FRAMES)
            {
                // 长时间未检测到数字，停止
                cmd.linear.x = 0;
                cmd.angular.z = 0;
                // g_targetArea = 0.0; // 重置目标尺寸
            }
            else
            {
                // 短暂丢失，保持上次指令
                cmd.linear.x = 0;
                cmd.angular.z = 0;
            }
        }
        
        // 发布控制指令
        vel_pub.publish(cmd);
        
        // 显示信息
        char statusText[100];
        if (detectedDigit >= 0)
        {
            sprintf(statusText, "Digit: %d | Score: %.2f | Area: %d", 
                   detectedDigit, matchScore, digitRect.area());
            
            // 绘制数字区域
            rectangle(img_show, digitRect, Scalar(0, 255, 0), 3);
            
            // 绘制数字位置
            Point center(digitRect.x + digitRect.width/2, 
                        digitRect.y + digitRect.height/2);
            circle(img_show, center, 5, Scalar(0, 0, 255), -1);
            
            // 绘制图像中心线
            line(img_show, Point(img_show.cols/2, 0), 
                 Point(img_show.cols/2, img_show.rows), 
                 Scalar(255, 0, 255), 2);
        }
        else
            sprintf(statusText, "No digit detected | Lost: %d", lost_frames);
        
        putText(img_show, statusText, Point(10, 30), 
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        
        // 显示控制状态
        char controlText[100];
        sprintf(controlText, "Control: %.2f m/s | Turn: %.2f rad/s", 
               cmd.linear.x, cmd.angular.z);
        putText(img_show, controlText, Point(10, 60), 
               FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        
        imshow("Digit Tracking", img_show);
        
        char key = waitKey(10);
        if (key == 27) break;  // ESC退出
        // else if (key == 'r' || key == 'R')
        // {
        //     g_targetArea = 0.0;
        //     ROS_INFO("目标尺寸已重置");
        // }
        // else if (key == 's' || key == 'S')
        // {
        //     // 保存当前帧用于调试
        //     static int save_count = 0;
        //     string filename = "debug_" + to_string(save_count++) + ".jpg";
        //     imwrite(filename, img_show);
        //     ROS_INFO("已保存调试图像: %s", filename.c_str());
        // }
    }
    
    return 0;
}

bool loadDigitTemplatesFromFiles()
{
    Mat* templates[3] = {&template1, &template2, &template3};
    
    for (int i = 0; i < 3; i++)
    {
        Mat img = imread(template_paths[i], IMREAD_GRAYSCALE);
        
        if (img.empty())
        {
            ROS_ERROR("Cannot load template file: %s", template_paths[i].c_str());
            return false;
        }
        
        // 确保模板是二值图像
        threshold(img, *templates[i], 128, 255, THRESH_BINARY);
        
        ROS_INFO("Loaded %d: %dx%d", i, templates[i]->cols, templates[i]->rows);
    }
    
    templates_loaded = true;
    return true;
}

void loadDigitTemplates()
{
    if (templates_loaded) return;
    
    if (loadDigitTemplatesFromFiles())
    {
        ROS_INFO("Loaded digit templates from files successfully");
        return;
    }
    
    // 如果文件加载失败，使用内置模板
    ROS_WARN("Loading digit templates from files failed, using built-in templates");
    
    template1 = Mat::zeros(100, 60, CV_8UC1);
    rectangle(template1, Point(25, 10), Point(35, 90), Scalar(255), -1);
    
    template2 = Mat::zeros(100, 60, CV_8UC1);
    vector<Point> points2;
    points2.push_back(Point(15, 15));
    points2.push_back(Point(45, 15));
    points2.push_back(Point(45, 40));
    points2.push_back(Point(15, 40));
    points2.push_back(Point(15, 85));
    points2.push_back(Point(45, 85));
    vector<vector<Point>> contours2;
    contours2.push_back(points2);
    drawContours(template2, contours2, -1, Scalar(255), 3);
    
    template3 = Mat::zeros(100, 60, CV_8UC1);
    putText(template3, "3", Point(15, 70), FONT_HERSHEY_SIMPLEX, 3, Scalar(255), 5);
    
    templates_loaded = true;
    ROS_INFO("内置数字模板加载完成");
}

void imageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    try {
        Mat new_img = cv_bridge::toCvShare(msg, "bgr8")->image;
        {
            std::lock_guard<std::mutex> lock(img_mutex);
            img_raw = new_img.clone();
        }
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
}

int detectDigitDirectly(Mat& src, Rect& digitRect, double& matchScore)
{
    if (!templates_loaded) return 0;
    
    Mat gray, equalized;
    cvtColor(src, gray, COLOR_BGR2GRAY); // 转为灰度图
    equalizeHist(gray, equalized); // 直方图均衡化
    
    // 二值化
    Mat binary1, binary2, binary3;
    // adaptiveThreshold(equalized, binary1, 255, ADAPTIVE_THRESH_GAUSSIAN_C, 
    //                  THRESH_BINARY_INV, 11, 2); // 自适应阈值法
    threshold(equalized, binary2, 0, 255, THRESH_BINARY_INV | THRESH_OTSU); // Otsu法
    
    // 形态学操作去除噪声
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    // morphologyEx(binary1, binary1, MORPH_CLOSE, kernel); // 闭运算
    // morphologyEx(binary1, binary1, MORPH_OPEN, kernel); // 开运算
    
    morphologyEx(binary2, binary2, MORPH_CLOSE, kernel);
    morphologyEx(binary2, binary2, MORPH_OPEN, kernel);
    
    // 选择白色像素较多的二值图像
    // Mat binary = (countNonZero(binary1) > countNonZero(binary2)) ? binary1 : binary2;
    Mat binary = binary2;
    
    // 显示二值化结果用于调试
    imshow("Binary Image", binary);
    
    // 查找所有轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    int bestDigit = -1;
    double bestScore = 0;
    Rect bestRect;
    
    // 对每个轮廓进行模板匹配
    vector<Mat> templates = {template1, template2, template3};
    
    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        
        // 面积过滤
        if (area < MIN_DIGIT_AREA || area > MAX_DIGIT_AREA) continue;
        Rect rect = boundingRect(contours[i]); // 外接矩形
        
        // // 宽高比过滤
        // float aspectRatio = (float)rect.width / rect.height;
        // if (aspectRatio < 0.2 || aspectRatio > 1.2) continue;
        
        // 提取候选区域
        Mat candidate = binary(rect);
        
        // 多尺度模板匹配
        for (int scale = 80; scale <= 120; scale += 20)
        {
            Mat resizedCandidate; // 调整大小后的候选区域
            float scale_factor = scale / 100.0f;
            resize(candidate, resizedCandidate, Size(), scale_factor, scale_factor, INTER_LINEAR);
            
            // 与三个模板匹配
            for (int t = 0; t < 3; t++)
            {
                // 确保模板和候选区域尺寸匹配
                Mat resizedTemplate;
                if (resizedCandidate.cols > templates[t].cols || 
                    resizedCandidate.rows > templates[t].rows)
                {
                    // 候选区域太大，缩小模板
                    float scale_w = (float)templates[t].cols / resizedCandidate.cols;
                    float scale_h = (float)templates[t].rows / resizedCandidate.rows;
                    float scale_min = min(scale_w, scale_h);
                    resize(templates[t], resizedTemplate, Size(), scale_min, scale_min);
                }
                else
                {
                    // 候选区域小，放大模板
                    float scale_w = (float)resizedCandidate.cols / templates[t].cols;
                    float scale_h = (float)resizedCandidate.rows / templates[t].rows;
                    float scale_min = min(scale_w, scale_h);
                    resize(templates[t], resizedTemplate, Size(), scale_min, scale_min);
                }
                
                // 调整尺寸使其匹配
                if (resizedCandidate.rows != resizedTemplate.rows || 
                    resizedCandidate.cols != resizedTemplate.cols)
                {
                    resize(resizedCandidate, resizedCandidate, resizedTemplate.size());
                }
                
                // 模板匹配
                Mat result;
                matchTemplate(resizedCandidate, resizedTemplate, result, TM_CCOEFF_NORMED);
                double score = result.at<float>(0, 0);
                
                // if (score > bestScore && score > MATCH_THRESHOLD)
                if (score > bestScore && score > MATCH_THRESHOLD)
                {
                    bestScore = score;
                    bestDigit = t;
                    bestRect = rect;
                }
            }
        }
    }
    
    // 返回结果
    if (bestDigit >= 0)
    {
        digitRect = bestRect;
        matchScore = bestScore;
    }
    
    return bestDigit;
}

geometry_msgs::Twist calculateControlByDigit(Mat& src, Rect& digitRect, int detectedDigit, double matchScore)
{
    geometry_msgs::Twist cmd;
    cmd.linear.x = 0;
    cmd.linear.y = 0;
    cmd.linear.z = 0;
    cmd.angular.x = 0;
    cmd.angular.y = 0;
    cmd.angular.z = 0;
    
    // 距离控制：基于数字区域的面积
    float currentSize = digitRect.area();
    
    // if (g_targetArea == 0.0f)
    // {
    //     // 第一次检测到数字，设置为目标尺寸，让小车保持在这个距离附近
    //     g_targetArea = currentSize;
    //     ROS_INFO("目标尺寸已设定为: %.0f (匹配分数: %.2f)", g_targetArea, matchScore);
    // }
    
    float sizeRatio = currentSize / g_targetArea;
    
    // 横向控制：基于数字在图像中的水平位置
    Point digitCenter(digitRect.x + digitRect.width/2, digitRect.y + digitRect.height/2);
    float horizontalError = (digitCenter.x - src.cols/2) / (float)(src.cols/2);
    
    // // 纵向控制：基于数字在图像中的垂直位置
    // float verticalPosition = digitCenter.y / (float)src.rows;
    
    // 综合控制策略
    if (sizeRatio > (1.0f + AREA_TOLERANCE)) // 数字太大（太近），后退
        cmd.linear.x = -MAX_SPEED * min(1.0f, sizeRatio - 1.0f);
    else if (sizeRatio < (1.0f - AREA_TOLERANCE)) // 数字太小（太远），前进
        cmd.linear.x = MAX_SPEED * min(1.0f, 1.0f - sizeRatio);
    else // 距离合适，保持静止或微调
        cmd.linear.x = 0;
    
    // 转向控制：使数字保持在图像中央
    cmd.angular.z = -horizontalError * 0.5;
    
    // // 如果数字太靠下（太近），增加后退速度
    // if (verticalPosition > 0.7)  // 数字在图像下方70%区域
    // {
    //     cmd.linear.x = min(cmd.linear.x, -MIN_SPEED);  // 确保后退
    // }
    
    // 限制速度范围
    if (cmd.linear.x > 0) cmd.linear.x = min(cmd.linear.x, MAX_SPEED);
    if (cmd.linear.x < 0) cmd.linear.x = max(cmd.linear.x, -MAX_SPEED);
    
    printf("控制决策 | 数字: %d | 尺寸比: %.2f | 水平误差: %.2f | 速度: %.2f | 转向: %.2f\n",
           detectedDigit, sizeRatio, horizontalError, cmd.linear.x, cmd.angular.z);
    
    return cmd;
}
