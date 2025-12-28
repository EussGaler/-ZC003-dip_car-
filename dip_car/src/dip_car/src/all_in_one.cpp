#include <stdlib.h>
#include <vector>
#include <string>
#include <mutex>
#include <math.h>
#include <algorithm>

#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include <geometry_msgs/Twist.h>
#include "sensor_msgs/Image.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

// ================== 全局变量和状态定义 ==================
std::mutex img_mutex;
Mat img_raw;

// 系统状态枚举
enum SystemState {
    OBSTACLE_AVOIDANCE = 0,  // 避障模式
    DIGIT_FOLLOWING         // 数字追踪模式
};

SystemState current_state = OBSTACLE_AVOIDANCE;  // 初始状态为避障

// ================== 避障相关参数和变量 ==================
// HSV阈值参数（锥桶）
int hsv_cone_min[3] = {0, 142, 88};
int hsv_cone_max[3] = {183, 255, 255};

// 避障相关参数
int obstacle_detection_threshold = 13000;  // 锥桶像素阈值
float avoidance_turn_angle = 1.5;          // 绕行转向角度
float avoidance_forward_speed = 0.2;       // 绕行前进速度
float normal_speed = 0.2;                  // 正常行驶速度
int frame_turn_big = 25, frame_turn_small = 52, frame_turn_straight = 10;  // 绕行各阶段帧数
const int frame_align = 50;  // 对准阶段帧数
int lane_keeping_frames = 0;  // 道路保持帧数计数

// ROI参数
int roi_cone_width = 200, roi_cone_height = 120;
int roi_cone_y_start = 330;
Rect roi_cone;

// 避障状态机变量
int avoidance_state = 0;               // 避障状态 (0-4)
int obstacle_detected_frames = 0;      // 连续检测到障碍物的帧数
int avoidance_complete_frames = 0;     // 绕行完成帧数
int lane_keeping_start_time = 0;       // 直线保持开始时间（帧计数）
const int LANE_KEEPING_DURATION = 375; // 直线保持约30秒

// ================== 数字追踪相关参数和变量 ==================
// 模板图像
Mat template1, template2, template3;
bool templates_loaded = false;

// 控制参数
double g_targetArea = 10000.0;               // 目标区域面积
const double AREA_TOLERANCE = 0.15;          // 面积容差
const double MAX_SPEED = 0.3;                // 最大速度
const double MIN_SPEED = 0.1;                // 最小速度

// 数字检测参数
const int MIN_DIGIT_AREA = 500;
const int MAX_DIGIT_AREA = 50000;
const double MATCH_THRESHOLD = 0.5;          // 模板匹配阈值

// 模板文件路径
std::string template_paths[3] = {
    "pic/template_0.png",
    "pic/template_1.png", 
    "pic/template_2.png"
};

// 数字追踪变量
int lost_frames = 0;
const int MAX_LOST_FRAMES = 10;

// ================== 函数声明 ==================
// 图像回调函数
void imageCallback(const sensor_msgs::Image::ConstPtr& msg);

// 避障相关函数
geometry_msgs::Twist obstacleAvoidance(Mat& src, Mat& display_img);
int getConeCenter(Mat& hsv_image, Rect roi, Point& center);

// 数字追踪相关函数
bool loadDigitTemplatesFromFiles();
void loadDigitTemplates();
int detectDigitDirectly(Mat& src, Rect& digitRect, double& matchScore);
geometry_msgs::Twist digitFollowing(Mat& src, Mat& display_img);

// ================== 主函数 ==================
int main(int argc, char **argv)
{
    ros::init(argc, argv, "all_in_one_node");
    ros::NodeHandle nh;
    
    // 订阅相机图像
    ros::Subscriber sub = nh.subscribe("camera/image", 1, imageCallback);
    
    // 发布速度控制指令
    ros::Publisher vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    
    // 加载数字模板
    loadDigitTemplates();
    ROS_INFO("All-in-one node started. Initial state: Obstacle Avoidance");
    
    int frame_count = 0;  // 总帧数计数
    
    while (ros::ok())
    {
        ros::spinOnce();
        
        Mat local_img;
        {
            std::lock_guard<std::mutex> lock(img_mutex);
            if (img_raw.empty()) {
                ros::Duration(0.05).sleep();
                continue;
            }
            local_img = img_raw.clone();
        }
        
        if (local_img.empty()) continue;
        
        frame_count++;
        
        geometry_msgs::Twist cmd;
        Mat display_img = local_img.clone();
        
        // 根据当前状态选择处理模式
        switch (current_state) {
            case OBSTACLE_AVOIDANCE: {
                cmd = obstacleAvoidance(local_img, display_img);
                
                // 检查是否应该切换到数字追踪模式
                if (avoidance_state == 4) {  // 直线保持状态
                    if (lane_keeping_start_time == 0) {
                        lane_keeping_start_time = frame_count;
                        ROS_INFO("Starting straight driving, will switch to digit tracking in 30 seconds");
                    }
                    
                    // 计算已保持的时间（帧数）
                    int frames_elapsed = frame_count - lane_keeping_start_time;
                    float seconds_elapsed = frames_elapsed * 0.08;  // 假设80ms/帧
                    
                    // 显示剩余时间
                    char time_text[100];
                    sprintf(time_text, "Switch to digit tracking: %.1f seconds", 30.0 - seconds_elapsed);
                    putText(display_img, time_text, Point(10, 160), 
                           FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
                    
                    // 30秒后切换状态
                    if (frames_elapsed >= LANE_KEEPING_DURATION) {
                        current_state = DIGIT_FOLLOWING;
                        lane_keeping_start_time = 0;  // 重置计时器
                        ROS_INFO("Switching to digit tracking mode");
                        destroyAllWindows();  // 关闭避障相关窗口
                    }
                }
                
                // 显示当前状态
                string state_text;
                switch (avoidance_state) {
                    case 0: state_text = "Normal Driving - Find 1st"; break;
                    case 1: state_text = "Right Avoidance - 1st"; break;
                    case 2: state_text = "Normal Driving - Find 2nd"; break;
                    case 3: state_text = "Left Avoidance - 2nd"; break;
                    case 4: state_text = "Straight Exit"; break;
                }
                putText(display_img, "State: " + state_text, Point(10, 30), 
                       FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                putText(display_img, "Mode: Obstacle Avoidance", Point(10, 60), 
                       FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                
                // 显示控制指令
                char controlText[100];
                sprintf(controlText, "Control: %.2f m/s | Turn: %.2f rad/s", 
                        cmd.linear.x, cmd.angular.z);
                putText(display_img, controlText, Point(10, 90), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
                
                imshow("All-in-One: Obstacle Avoidance", display_img);
                break;
            }
            
            case DIGIT_FOLLOWING: {
                cmd = digitFollowing(local_img, display_img);
                
                // 显示当前状态
                putText(display_img, "Mode: Digit Following", Point(10, 30), 
                       FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);
                
                // 显示控制指令
                char controlText[100];
                sprintf(controlText, "Control: %.2f m/s | Turn: %.2f rad/s", 
                        cmd.linear.x, cmd.angular.z);
                putText(display_img, controlText, Point(10, 90), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
                
                imshow("All-in-One: Digit Following", display_img);
                break;
            }
        }
        
        // 发布控制指令
        vel_pub.publish(cmd);
        
        // 键盘控制
        char key = waitKey(80);
        if (key == 27) break;  // ESC退出
        else if (key == 'm' || key == 'M') {
            // 手动切换模式
            if (current_state == OBSTACLE_AVOIDANCE) {
                current_state = DIGIT_FOLLOWING;
                ROS_INFO("Manually switching to digit tracking mode");
                destroyAllWindows();
            } else {
                current_state = OBSTACLE_AVOIDANCE;
                ROS_INFO("Manually switching to obstacle avoidance mode");
                destroyAllWindows();
            }
        }
    }
    
    return 0;
}

// ================== 图像回调函数 ==================
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

// ================== 避障相关函数 ==================
geometry_msgs::Twist obstacleAvoidance(Mat& src, Mat& display_img)
{
    geometry_msgs::Twist cmd;
    cmd.linear.x = normal_speed;
    cmd.angular.z = 0;
    
    // 更新图像尺寸和ROI
    int img_width = src.cols;
    int img_height = src.rows;
    roi_cone = Rect(img_width / 2 - roi_cone_width / 2, roi_cone_y_start, 
                   roi_cone_width, roi_cone_height);
    int roi_center_x = roi_cone.x + roi_cone.width / 2;
    
    Mat img_blur, img_hsv, img_hsv_split_cone;
    
    // 图像预处理
    GaussianBlur(src, img_blur, Size(3, 3), 0, 0);
    cvtColor(img_blur, img_hsv, COLOR_BGR2HSV);
    
    // 锥桶颜色分割
    inRange(img_hsv,
            Scalar(hsv_cone_min[0], hsv_cone_min[1], hsv_cone_min[2]),
            Scalar(hsv_cone_max[0], hsv_cone_max[1], hsv_cone_max[2]),
            img_hsv_split_cone);
    
    // 形态学操作
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(img_hsv_split_cone, img_hsv_split_cone, MORPH_OPEN, element);
    morphologyEx(img_hsv_split_cone, img_hsv_split_cone, MORPH_CLOSE, element);
    
    // 在ROI内检测锥桶
    Mat roiImage_cone = img_hsv_split_cone(roi_cone);
    int cone_pixel_count = countNonZero(roiImage_cone);
    
    // 计算锥桶中心
    Point cone_center;
    int cone_detected = getConeCenter(img_hsv_split_cone, roi_cone, cone_center);
    float alignment_error = 0.0;
    
    // 绘制ROI矩形和中心线
    rectangle(display_img, roi_cone, Scalar(0, 255, 0), 2); // 绘制ROI矩形
    line(display_img, Point(roi_center_x, roi_cone.y), 
         Point(roi_center_x, roi_cone.y + roi_cone.height), 
         Scalar(255, 255, 0), 2); // 绘制ROI中心线
    
    // 在图像上显示锥桶像素数
    char pixelText[50];
    sprintf(pixelText, "Cone Pixels: %d", cone_pixel_count);
    putText(display_img, pixelText, Point(roi_cone.x, roi_cone.y - 10),
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    
    if (cone_detected) {
        alignment_error = (cone_center.x - roi_center_x) / (float)(roi_cone.width / 2);
        
        // 在图像上绘制锥桶中心点
        circle(display_img, cone_center, 5, Scalar(255, 0, 0), -1);
        
        // 绘制从ROI中心到锥桶中心的连线
        line(display_img, Point(roi_center_x, roi_cone.y + roi_cone.height/2),
             cone_center, Scalar(0, 255, 255), 2);
        
        // 显示对齐误差
        char alignText[50];
        sprintf(alignText, "Align Error: %.2f", alignment_error);
        putText(display_img, alignText, Point(cone_center.x + 10, cone_center.y - 10),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
        
        // 绘制距离文本
        char distText[50];
        int dist = abs(cone_center.x - roi_center_x);
        sprintf(distText, "Dist: %d", dist);
        putText(display_img, distText, Point(roi_center_x, roi_cone.y + roi_cone.height + 20),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 200, 0), 1);
    }
    
    // 显示锥桶检测结果
    imshow("Cone Detection", img_hsv_split_cone);
    
    // 状态机：避障决策
    switch (avoidance_state) {
        case 0:  // 正常行驶，检测第一个障碍物
            if (cone_pixel_count > obstacle_detection_threshold) {
                obstacle_detected_frames++;
                if (obstacle_detected_frames > 5) {
                    avoidance_complete_frames = 0;
                    avoidance_state = 1;  // 右绕行
                    obstacle_detected_frames = 0;
                    ROS_INFO("First obstacle detected, starting right avoidance");
                    
                    // 绘制检测到第一个障碍物的指示
                    putText(display_img, "First Obstacle Detected!", 
                            Point(img_width/2 - 100, 120), 
                            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
                }
            } else {
                obstacle_detected_frames = 0;
                cmd.linear.x = normal_speed;
                cmd.angular.z = 0;
            }
            break;
            
        case 1:  // 右绕行（绕过第一个障碍物）
            // 绘制绕行指示
            putText(display_img, "Right Avoidance (1st)", Point(10, 120), 
                   FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
            
            if (avoidance_complete_frames <= frame_turn_big) {
                cmd.linear.x = avoidance_forward_speed;
                cmd.angular.z = -avoidance_turn_angle;
                
                // 绘制大右转指示
                putText(display_img, "Big Right Turn", Point(img_width - 200, 120), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            } else if (avoidance_complete_frames <= (frame_turn_big + frame_turn_small)) {
                cmd.linear.x = 0.05;
                cmd.angular.z = avoidance_turn_angle;
                
                // 绘制小左转指示
                putText(display_img, "Small Left Turn", Point(img_width - 200, 120), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            } else if (avoidance_complete_frames <= (frame_turn_big + frame_turn_small + frame_turn_straight)) {
                cmd.linear.x = avoidance_forward_speed;
                cmd.angular.z = 0;
                
                // 绘制直行指示
                putText(display_img, "Straight", Point(img_width - 200, 120), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            } else if (avoidance_complete_frames <= (2 * frame_turn_big + frame_turn_small + frame_turn_straight + 3)) {
                cmd.linear.x = avoidance_forward_speed;
                cmd.angular.z = avoidance_turn_angle;
                
                // 绘制大左转指示
                putText(display_img, "Big Left Turn", Point(img_width - 200, 120), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            } else if (avoidance_complete_frames <= (2 * frame_turn_big + 2 * frame_turn_small + frame_turn_straight)) {
                cmd.linear.x = 0.05;
                cmd.angular.z = -avoidance_turn_angle;
                
                // 绘制小右转指示
                putText(display_img, "Small Right Turn", Point(img_width - 200, 120), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            } else if (avoidance_complete_frames <= (2 * frame_turn_big + 2 * frame_turn_small + frame_turn_straight + frame_align)) {
                // 根据锥桶中心与ROI中心的水平偏差进行转向
                if (abs(cone_center.x - roi_center_x) > 20) {  // 偏差阈值20像素
                    // 比例控制转向
                    float turn_gain = 0.05;  // 转向增益系数
                    cmd.linear.x = 0;  // 原地转向，线速度为0
                    cmd.angular.z = (roi_center_x - cone_center.x) * turn_gain;
                    
                    // 限制最大角速度
                    if (cmd.angular.z > avoidance_turn_angle) cmd.angular.z = 0.1;
                    if (cmd.angular.z < -avoidance_turn_angle) cmd.angular.z = -0.1;
                    
                    // 绘制对准指示
                    char alignMsg[100];
                    sprintf(alignMsg, "Aligning to Cone: %.2f rad/s", cmd.angular.z);
                    putText(display_img, alignMsg, Point(img_width/2 - 150, 150), 
                            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
                    
                    // 绘制目标线
                    line(display_img, Point(roi_center_x, 0), 
                            Point(roi_center_x, img_height), 
                            Scalar(0, 255, 0), 2);
                    line(display_img, Point(cone_center.x, 0), 
                            Point(cone_center.x, img_height), 
                            Scalar(255, 0, 0), 2);
                } else {
                    // 已经对准，准备进入直行阶段
                    cmd.linear.x = 0;
                    cmd.angular.z = 0;
                    putText(display_img, "Aligned to Cone!", Point(img_width/2 - 100, 150), 
                            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
                }
                // 绘制对准阶段指示
                putText(display_img, "Alignment Phase", Point(10, 150), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            } else {
                avoidance_state = 2;  // 进入检测第二个障碍物阶段
                avoidance_complete_frames = 0;
                ROS_INFO("First obstacle cleared, alignment completed");
                
                // 绘制绕行完成指示
                putText(display_img, "First Obstacle Cleared", Point(img_width/2 - 100, 120), 
                       FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            }
            avoidance_complete_frames++;
            break;
            
        case 2:  // 检测第二个障碍物
            if (cone_pixel_count > obstacle_detection_threshold) {
                obstacle_detected_frames++;
                if (obstacle_detected_frames > 5) {
                    avoidance_complete_frames = 0;
                    avoidance_state = 3;  // 左绕行
                    obstacle_detected_frames = 0;
                    ROS_INFO("Second obstacle detected, starting left avoidance");
                    
                    // 绘制检测到第二个障碍物的指示
                    putText(display_img, "Second Obstacle Detected!", 
                            Point(img_width/2 - 100, 120), 
                            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
                }
            } else {
                obstacle_detected_frames = 0;
                cmd.linear.x = normal_speed;
                cmd.angular.z = 0;
            }
            break;
            
        case 3:  // 左绕行（绕过第二个障碍物）
            // 绘制绕行指示
            putText(display_img, "Left Avoidance (2nd)", Point(10, 120), 
                   FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
            
            if (avoidance_complete_frames <= frame_turn_big) {
                cmd.linear.x = avoidance_forward_speed;
                cmd.angular.z = avoidance_turn_angle;
                
                // 绘制大左转指示
                putText(display_img, "Big Left Turn", Point(img_width - 200, 120), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            } else if (avoidance_complete_frames <= (frame_turn_big + frame_turn_small)) {
                cmd.linear.x = 0.05;
                cmd.angular.z = -avoidance_turn_angle;
                
                // 绘制小右转指示
                putText(display_img, "Small Right Turn", Point(img_width - 200, 120), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            } else if (avoidance_complete_frames <= (frame_turn_big + frame_turn_small + frame_turn_straight)) {
                cmd.linear.x = avoidance_forward_speed;
                cmd.angular.z = 0;
                
                // 绘制直行指示
                putText(display_img, "Straight", Point(img_width - 200, 120), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            } else if (avoidance_complete_frames <= (frame_turn_big + 2 * frame_turn_small + frame_turn_straight - 4)) {
                cmd.linear.x = avoidance_forward_speed;
                cmd.angular.z = -avoidance_turn_angle;
                
                // 绘制减速大转弯指示
                putText(display_img, "Slow Big Turn", Point(img_width - 200, 120), 
                       FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            } else {
                avoidance_state = 4;  // 进入道路保持阶段（使用两个ROI进行闭环控制）
                avoidance_complete_frames = 0;
                lane_keeping_frames = 0;  // 重置道路保持帧数
                ROS_INFO("Second obstacle cleared, entering lane keeping phase");
                
                // 绘制绕行完成指示
                putText(display_img, "Second Obstacle Cleared", Point(img_width/2 - 100, 120), 
                       FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            }
            avoidance_complete_frames++;
            break;
            
        case 4:  // 道路保持阶段（使用两个ROI进行闭环控制）
            // 定义左右两侧ROI区域
            Rect roi_left(roi_cone.x - roi_cone_width, roi_cone_y_start, 
                          roi_cone_width, roi_cone_height);
            Rect roi_right(roi_cone.x + roi_cone_width, roi_cone_y_start, 
                           roi_cone_width, roi_cone_height);
            
            // 在图像上绘制左右ROI
            rectangle(display_img, roi_left, Scalar(255, 0, 0), 2);  // 蓝色 - 左ROI
            rectangle(display_img, roi_right, Scalar(0, 0, 255), 2); // 红色 - 右ROI
            
            // 检测左右锥桶
            Mat roi_left_image = img_hsv_split_cone(roi_left);
            Mat roi_right_image = img_hsv_split_cone(roi_right);
            
            int left_pixel_count = countNonZero(roi_left_image);
            int right_pixel_count = countNonZero(roi_right_image);
            
            int left_center_x = -1;
            int right_center_x = -1;
            
            // 计算左侧锥桶中心（如果有）
            if (left_pixel_count > 300) {
                std::vector<std::vector<Point>> left_contours;
                findContours(roi_left_image, left_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                if (!left_contours.empty()) {
                    double max_area = 0;
                    int max_idx = 0;
                    for (size_t i = 0; i < left_contours.size(); i++) {
                        double area = contourArea(left_contours[i]);
                        if (area > max_area) {
                            max_area = area;
                            max_idx = i;
                        }
                    }
                    Moments mu = moments(left_contours[max_idx]);
                    if (mu.m00 != 0) {
                        left_center_x = roi_left.x + static_cast<int>(mu.m10 / mu.m00);
                        int left_center_y = roi_left.y + static_cast<int>(mu.m01 / mu.m00);
                        circle(display_img, Point(left_center_x, left_center_y), 5, Scalar(255, 0, 0), -1);
                        putText(display_img, "L", Point(left_center_x + 5, left_center_y - 5),
                                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
                    }
                }
            }
            
            // 计算右侧锥桶中心（如果有）
            if (right_pixel_count > 300) {
                std::vector<std::vector<Point>> right_contours;
                findContours(roi_right_image, right_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                if (!right_contours.empty()) {
                    double max_area = 0;
                    int max_idx = 0;
                    for (size_t i = 0; i < right_contours.size(); i++) {
                        double area = contourArea(right_contours[i]);
                        if (area > max_area) {
                            max_area = area;
                            max_idx = i;
                        }
                    }
                    Moments mu = moments(right_contours[max_idx]);
                    if (mu.m00 != 0) {
                        right_center_x = roi_right.x + static_cast<int>(mu.m10 / mu.m00);
                        int right_center_y = roi_right.y + static_cast<int>(mu.m01 / mu.m00);
                        circle(display_img, Point(right_center_x, right_center_y), 5, Scalar(0, 0, 255), -1);
                        putText(display_img, "R", Point(right_center_x + 5, right_center_y - 5),
                                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
                    }
                }
            }
            
            // 显示左右锥桶像素数
            char left_text[50], right_text[50];
            sprintf(left_text, "Left: %d", left_pixel_count);
            sprintf(right_text, "Right: %d", right_pixel_count);
            putText(display_img, left_text, Point(roi_left.x, roi_left.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
            putText(display_img, right_text, Point(roi_right.x, roi_right.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
            
            // 控制逻辑：计算道路中心线并控制小车
            float center_error = 0.0;
            
            if (left_center_x != -1 && right_center_x != -1) {
                // 情况1：两个锥桶都检测到，计算道路中心
                int lane_center = (left_center_x + right_center_x) / 2;
                center_error = (lane_center - img_width / 2) / (float)(img_width / 2);
                
                // 在图像上绘制道路中心线
                line(display_img, Point(lane_center, roi_cone_y_start - 50),
                     Point(lane_center, roi_cone_y_start + roi_cone_height + 50),
                     Scalar(0, 255, 255), 2);
                
                // 绘制理想中心线（图像中线）
                line(display_img, Point(img_width / 2, roi_cone_y_start - 50),
                     Point(img_width / 2, roi_cone_y_start + roi_cone_height + 50),
                     Scalar(255, 255, 255), 1);
                
                putText(display_img, "Both cones detected", Point(10, 150),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
                
            } else if (left_center_x != -1) {
                // 情况2：只检测到左侧锥桶，保持安全距离
                int desired_left_distance = roi_cone_width * 1.2; // 期望与左侧锥桶的距离
                int actual_distance = img_width / 2 - left_center_x;
                center_error = (desired_left_distance - actual_distance) / (float)desired_left_distance;
                
                putText(display_img, "Left cone only", Point(10, 150),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
                
            } else if (right_center_x != -1) {
                // 情况3：只检测到右侧锥桶，保持安全距离
                int desired_right_distance = roi_cone_width * 1.2; // 期望与右侧锥桶的距离
                int actual_distance = right_center_x - img_width / 2;
                center_error = (actual_distance - desired_right_distance) / (float)desired_right_distance;
                
                putText(display_img, "Right cone only", Point(10, 150),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
                
            } else {
                // 情况4：没有检测到锥桶，维持当前方向或小幅搜索
                center_error = 0.0;
                
                putText(display_img, "No cones detected", Point(10, 150),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
            }
            
            // 根据误差调整速度和转向
            if (fabs(center_error) < 0.1) {
                // 误差很小，直行
                cmd.linear.x = normal_speed;
                cmd.angular.z = -center_error * 0.8; // 轻微调整
            } else if (fabs(center_error) < 0.3) {
                // 中等误差，减速并转向
                cmd.linear.x = normal_speed * 0.8;
                cmd.angular.z = -center_error * 1.2;
            } else {
                // 大误差，大幅减速并转向
                cmd.linear.x = normal_speed * 0.5;
                cmd.angular.z = -center_error * 1.5;
            }
            
            // 显示控制信息
            char control_text[50];
            sprintf(control_text, "Error: %.2f, Speed: %.2f, Turn: %.2f",
                    center_error, cmd.linear.x, cmd.angular.z);
            putText(display_img, control_text, Point(10, 180),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
            
            // 增加道路保持帧数计数
            lane_keeping_frames++;
            
            // 检查是否应该切换到数字追踪模式
            if (lane_keeping_start_time == 0) {
                lane_keeping_start_time = lane_keeping_frames;
                ROS_INFO("Starting lane keeping, will switch to digit tracking in 30 seconds");
            }
            
            // 计算已保持的时间（帧数）
            int frames_elapsed = lane_keeping_frames - lane_keeping_start_time;
            float seconds_elapsed = frames_elapsed * 0.08;  // 假设80ms/帧
            
            // 显示剩余时间
            char time_text[100];
            sprintf(time_text, "Switch to digit tracking: %.1f seconds", 30.0 - seconds_elapsed);
            putText(display_img, time_text, Point(10, 210), 
                   FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            
            // 30秒后切换状态
            if (frames_elapsed >= LANE_KEEPING_DURATION) {
                current_state = DIGIT_FOLLOWING;
                lane_keeping_start_time = 0;  // 重置计时器
                lane_keeping_frames = 0;      // 重置帧数
                ROS_INFO("Switching to digit tracking mode");
                destroyAllWindows();  // 关闭避障相关窗口
            }
            break;
    }
    
    return cmd;
}

int getConeCenter(Mat& hsv_image, Rect roi, Point& center)
{
    Mat roiImage = hsv_image(roi);
    vector<vector<Point>> contours;
    findContours(roiImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (!contours.empty()) {
        // 找到最大轮廓
        int max_contour_idx = 0;
        double max_area = 0;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                max_contour_idx = i;
            }
        }
        
        // 计算轮廓中心
        Moments mu = moments(contours[max_contour_idx]);
        if (mu.m00 != 0) {
            int cx = static_cast<int>(mu.m10 / mu.m00);
            int cy = static_cast<int>(mu.m01 / mu.m00);
            center.x = roi.x + cx;
            center.y = roi.y + cy;
            return 1;
        }
    }
    
    return 0;
}

// ================== 数字追踪相关函数 ==================
geometry_msgs::Twist digitFollowing(Mat& src, Mat& display_img)
{
    geometry_msgs::Twist cmd;
    cmd.linear.x = 0;
    cmd.angular.z = 0;
    
    // 检测数字
    Rect digitRect;
    double matchScore = 0;
    int detectedDigit = detectDigitDirectly(src, digitRect, matchScore);
    
    if (detectedDigit >= 0) {
        // 基于数字区域计算控制指令
        float currentSize = digitRect.area();
        float sizeRatio = currentSize / g_targetArea;
        
        // 横向控制：基于数字在图像中的水平位置
        Point digitCenter(digitRect.x + digitRect.width/2, digitRect.y + digitRect.height/2);
        float horizontalError = (digitCenter.x - src.cols/2) / (float)(src.cols/2);
        
        // 距离控制
        if (sizeRatio > (1.0f + AREA_TOLERANCE)) {
            cmd.linear.x = -MAX_SPEED * min(1.0f, sizeRatio - 1.0f);
        } else if (sizeRatio < (1.0f - AREA_TOLERANCE)) {
            cmd.linear.x = MAX_SPEED * min(1.0f, 1.0f - sizeRatio);
        } else {
            cmd.linear.x = 0;
        }
        
        // 转向控制
        cmd.angular.z = -horizontalError * 0.5;
        
        // 限制速度范围
        if (cmd.linear.x > 0) cmd.linear.x = min(cmd.linear.x, MAX_SPEED);
        if (cmd.linear.x < 0) cmd.linear.x = max(cmd.linear.x, -MAX_SPEED);
        
        lost_frames = 0;
        
        // 在图像上显示数字信息
        char infoText[100];
        sprintf(infoText, "Digit: %d | Match: %.2f | Area: %d", 
                detectedDigit, matchScore, digitRect.area());
        putText(display_img, infoText, Point(10, 60), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
        
        // 绘制数字区域和中心
        rectangle(display_img, digitRect, Scalar(0, 255, 0), 3);
        circle(display_img, digitCenter, 5, Scalar(0, 0, 255), -1);
        
        // 绘制图像中心线
        line(display_img, Point(display_img.cols/2, 0), 
             Point(display_img.cols/2, display_img.rows), 
             Scalar(255, 0, 255), 2);
        
        // 绘制数字位置到图像中心的连线
        line(display_img, Point(display_img.cols/2, digitCenter.y), 
             digitCenter, Scalar(255, 255, 0), 2);
        
        // 显示距离信息
        char distanceText[100];
        int horizontalDist = digitCenter.x - display_img.cols/2;
        sprintf(distanceText, "Distance from center: %d", abs(horizontalDist));
        putText(display_img, distanceText, Point(digitRect.x, digitRect.y - 10), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
        
        // 显示控制信息
        char controlInfo[100];
        sprintf(controlInfo, "Size Ratio: %.2f | Horz Error: %.2f", 
                sizeRatio, horizontalError);
        putText(display_img, controlInfo, Point(10, 120), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
        
        printf("Control Decision | Digit: %d | Size Ratio: %.2f | Horizontal Error: %.2f | Speed: %.2f | Turn: %.2f\n",
               detectedDigit, sizeRatio, horizontalError, cmd.linear.x, cmd.angular.z);
    } else {
        lost_frames++;
        if (lost_frames > MAX_LOST_FRAMES) {
            cmd.linear.x = 0;
            cmd.angular.z = 0;
        }
        
        char infoText[100];
        sprintf(infoText, "No digit detected | Lost frames: %d", lost_frames);
        putText(display_img, infoText, Point(10, 60), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
        
        // 绘制图像中心线
        line(display_img, Point(display_img.cols/2, 0), 
             Point(display_img.cols/2, display_img.rows), 
             Scalar(255, 0, 255), 2);
    }
    
    return cmd;
}

// ================== 数字模板相关函数 ==================
bool loadDigitTemplatesFromFiles()
{
    Mat* templates[3] = {&template1, &template2, &template3};
    
    for (int i = 0; i < 3; i++) {
        Mat img = imread(template_paths[i], IMREAD_GRAYSCALE);
        
        if (img.empty()) {
            ROS_ERROR("Cannot load template file: %s", template_paths[i].c_str());
            return false;
        }
        
        threshold(img, *templates[i], 128, 255, THRESH_BINARY);
        ROS_INFO("Loaded template %d: %dx%d", i, templates[i]->cols, templates[i]->rows);
    }
    
    templates_loaded = true;
    return true;
}

void loadDigitTemplates()
{
    if (templates_loaded) return;
    
    if (loadDigitTemplatesFromFiles()) {
        ROS_INFO("Digit templates loaded from files successfully");
        return;
    }
    
    ROS_WARN("Failed to load templates from files, using built-in templates");
    
    // 创建内置模板（简化版数字0,1,2）
    template1 = Mat::zeros(100, 60, CV_8UC1);  // 数字0
    rectangle(template1, Point(25, 10), Point(35, 90), Scalar(255), -1);
    
    template2 = Mat::zeros(100, 60, CV_8UC1);  // 数字1
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
    
    template3 = Mat::zeros(100, 60, CV_8UC1);  // 数字2
    putText(template3, "2", Point(15, 70), FONT_HERSHEY_SIMPLEX, 3, Scalar(255), 5);
    
    templates_loaded = true;
    ROS_INFO("Built-in digit templates loaded");
}

int detectDigitDirectly(Mat& src, Rect& digitRect, double& matchScore)
{
    if (!templates_loaded) return -1;
    
    Mat gray, equalized;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, equalized);
    
    // 仅使用Otsu二值化
    Mat binary;
    threshold(equalized, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU); // Otsu法
    
    // 形态学操作去除噪声
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);
    morphologyEx(binary, binary, MORPH_OPEN, kernel);
    
    // 显示二值化结果
    imshow("Binary Image", binary);
    
    // 查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    int bestDigit = -1;
    double bestScore = 0;
    Rect bestRect;
    
    vector<Mat> templates = {template1, template2, template3};
    
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        
        if (area < MIN_DIGIT_AREA || area > MAX_DIGIT_AREA) continue;
        Rect rect = boundingRect(contours[i]);
        
        Mat candidate = binary(rect);
        
        // 多尺度模板匹配
        for (int scale = 80; scale <= 120; scale += 20) {
            Mat resizedCandidate;
            float scale_factor = scale / 100.0f;
            resize(candidate, resizedCandidate, Size(), scale_factor, scale_factor, INTER_LINEAR);
            
            for (int t = 0; t < 3; t++) {
                Mat resizedTemplate;
                if (resizedCandidate.cols > templates[t].cols || 
                    resizedCandidate.rows > templates[t].rows) {
                    float scale_w = (float)templates[t].cols / resizedCandidate.cols;
                    float scale_h = (float)templates[t].rows / resizedCandidate.rows;
                    float scale_min = min(scale_w, scale_h);
                    resize(templates[t], resizedTemplate, Size(), scale_min, scale_min);
                } else {
                    float scale_w = (float)resizedCandidate.cols / templates[t].cols;
                    float scale_h = (float)resizedCandidate.rows / templates[t].rows;
                    float scale_min = min(scale_w, scale_h);
                    resize(templates[t], resizedTemplate, Size(), scale_min, scale_min);
                }
                
                if (resizedCandidate.rows != resizedTemplate.rows || 
                    resizedCandidate.cols != resizedTemplate.cols) {
                    resize(resizedCandidate, resizedCandidate, resizedTemplate.size());
                }
                
                Mat result;
                matchTemplate(resizedCandidate, resizedTemplate, result, TM_CCOEFF_NORMED);
                double score = result.at<float>(0, 0);
                
                if (score > bestScore && score > MATCH_THRESHOLD) {
                    bestScore = score;
                    bestDigit = t;
                    bestRect = rect;
                }
            }
        }
    }
    
    if (bestDigit >= 0) {
        digitRect = bestRect;
        matchScore = bestScore;
    }
    
    return bestDigit;
}
