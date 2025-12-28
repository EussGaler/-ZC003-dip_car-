#include <stdlib.h>
#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include <geometry_msgs/Twist.h>
#include "sensor_msgs/Image.h"
#include <math.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;

std::mutex img_mutex;
Mat img_raw; // 接收的图像

/**
 * @brief 图像回调函数
 * @param msg ROS图像消息
 */
void imageCallback(const sensor_msgs::Image::ConstPtr& msg)
{
    try {
        Mat new_img = cv_bridge::toCvShare(msg, "bgr8")->image;
        {
            std::lock_guard<std::mutex> lock(img_mutex);
            img_raw = new_img.clone();
        }
        ROS_DEBUG("Receiving image from camera");
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
}

// HSV阈值参数
int hsv_cone_min[3] = {0, 142, 88};
int hsv_cone_max[3] = {183, 255, 255};

// 尺寸相关参数
int img_width = 672, img_height = 376; // 图像尺寸
int roi_cone_width = 200, roi_cone_height = 120; // ROI矩形尺寸
int roi_cone_y_start = 330; // ROI矩形垂直开始位置

// 创建水平居中的ROI矩形
Rect roi_cone(img_width / 2 - roi_cone_width / 2, roi_cone_y_start, roi_cone_width, roi_cone_height);

// 避障相关参数
int obstacle_detection_threshold = 13000; // 锥桶像素阈值，用于判断前方是否有障碍物
float avoidance_turn_angle = 1.5; // 绕行时的转向角度
float avoidance_forward_speed = 0.2; // 绕行时前进速度
float normal_speed = 0.2; // 正常行驶速度
int frame_turn_big = 25, frame_turn_small = 52, frame_turn_straight = 10; // 绕行各阶段所需帧数

Mat img_roi; // 用于显示roi区域
Mat img_show; // 处理后的图像
int cone_center_x = 0; // 锥桶中心x坐标
int roi_center_x = 0; // ROI区域中心x坐标

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_processor");
    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("camera/image", 1, imageCallback);
    ros::Publisher vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);

    int avoidance_state = 0; // 避障状态
    int obstacle_detected_frames = 0; // 连续检测到障碍物的帧数
    int avoidance_complete_frames = 0; // 绕行完成后的帧数
    
    // 初始化ROI中心坐标
    roi_center_x = roi_cone.x + roi_cone.width / 2;
    
    while (ros::ok())
    {
        ros::spinOnce();
        Mat local_img;
        {
            std::lock_guard<std::mutex> lock(img_mutex);
            if (img_raw.empty()) {
                ROS_WARN("Waiting for image...");
                ros::Duration(0.1).sleep();
                continue;
            }
            local_img = img_raw.clone();
        }

        waitKey(80);
        if (local_img.empty())
        {
            ROS_WARN("Waiting for image...");
            continue;
        }
        
        // 更新图像尺寸
        img_width = local_img.cols;
        img_height = local_img.rows;
        
        // 更新ROI位置和中心坐标
        roi_cone = Rect(img_width / 2 - roi_cone_width / 2, roi_cone_y_start, roi_cone_width, roi_cone_height);
        roi_center_x = roi_cone.x + roi_cone.width / 2;
        
        Mat img_show = local_img.clone();

        // 高斯模糊
        Mat img_blur;
        GaussianBlur(local_img, img_blur, Size(3, 3), 0, 0);

        // HSV转换
        Mat img_hsv;
        cvtColor(img_blur, img_hsv, COLOR_BGR2HSV);
        
        // 在HSV图像中检测锥桶颜色
        Mat img_hsv_split_cone;
        inRange(img_hsv,
            Scalar(hsv_cone_min[0], hsv_cone_min[1], hsv_cone_min[2]),
            Scalar(hsv_cone_max[0], hsv_cone_max[1], hsv_cone_max[2]),
            img_hsv_split_cone);
        
        // 形态学操作去除噪声
        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(img_hsv_split_cone, img_hsv_split_cone, MORPH_OPEN, element);
        morphologyEx(img_hsv_split_cone, img_hsv_split_cone, MORPH_CLOSE, element);
        
        // 在ROI区域内检测锥桶
        rectangle(img_show, roi_cone, Scalar(0, 255, 0), 2); // 绘制ROI矩形
        // 在ROI中心绘制一条垂直线，用于观察对齐情况
        line(img_show, Point(roi_center_x, roi_cone.y), 
                     Point(roi_center_x, roi_cone.y + roi_cone.height), 
                     Scalar(255, 255, 0), 2);
        
        Mat roiImage_cone = img_hsv_split_cone(roi_cone); // 只包含ROI区域的部分
        int cone_pixel_count = countNonZero(roiImage_cone); // 统计ROI区域中白色像素的数量
        
        // 计算锥桶中心位置
        cone_center_x = -1;
        if (cone_pixel_count > 500) // 有足够像素点才计算中心
        {
            // 查找轮廓（只检测最外层轮廓）
            std::vector<std::vector<Point>> contours;
            findContours(roiImage_cone, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            if (!contours.empty())
            {
                // 找到最大轮廓
                int max_contour_idx = 0;
                double max_area = 0;
                for (size_t i = 0; i < contours.size(); i++) // 遍历所有轮廓
                {
                    double area = contourArea(contours[i]);
                    if (area > max_area)
                    {
                        max_area = area;
                        max_contour_idx = i;
                    }
                }
                
                // 计算轮廓中心
                Moments mu = moments(contours[max_contour_idx]);
                if (mu.m00 != 0) // 零阶矩（轮廓的面积）不为0
                {
                    // 计算轮廓质心
                    int cx = static_cast<int>(mu.m10 / mu.m00); // m10为关于x轴的一阶矩
                    int cy = static_cast<int>(mu.m01 / mu.m00); // m01为关于y轴的一阶矩
                    cone_center_x = roi_cone.x + cx; // 转换为全局坐标

                    // 在图像上绘制中心点
                    circle(img_show, Point(cone_center_x, roi_cone.y + cy), 5, Scalar(255, 0, 0), -1);
                    // 绘制从ROI中心到锥桶中心的连线
                    line(img_show, Point(roi_center_x, roi_cone.y + roi_cone.height/2),
                                 Point(cone_center_x, roi_cone.y + cy),
                                 Scalar(0, 255, 255), 2);
                    drawContours(img_show(roi_cone), contours, max_contour_idx, Scalar(0, 0, 255), 2); // 绘制轮廓
                }
            }
        }
        
        // 显示锥桶检测结果
        imshow("Cone Detection", img_hsv_split_cone);
        
        // 控制逻辑
        static geometry_msgs::Twist msg;
        msg.linear.x = normal_speed;
        msg.angular.z = 0;

        // 状态机：避障决策
        float alignment_error = 0.0;
        if (cone_center_x != -1) {
            alignment_error = (cone_center_x - roi_center_x) / (float)(roi_cone.width / 2);
            printf("State: %d, Cone pixels: %d, Cone center X: %d, ROI center X: %d, Alignment error: %.2f\n", 
                   avoidance_state, cone_pixel_count, cone_center_x, roi_center_x, alignment_error);
        } else {
            printf("State: %d, Cone pixels: %d, No cone center detected\n", 
                   avoidance_state, cone_pixel_count);
        }

        switch (avoidance_state)
        {
        case 0:  // 正常行驶，检测第一个障碍物
            if (cone_pixel_count > obstacle_detection_threshold)
            {
                // 锥桶像素足够多且锥桶中心与ROI中心对齐，表示小车正对锥桶
                obstacle_detected_frames++;
                if (obstacle_detected_frames > 5) // 连续5帧检测到障碍物
                {
                    // 第一个障碍物：向右绕行
                    avoidance_complete_frames = 0;
                    avoidance_state = 1; // 右绕行
                    printf("First obstacle detected and aligned, turning RIGHT\n");
                    obstacle_detected_frames = 0;
                }
            } 
            else 
            {
                obstacle_detected_frames = 0;
                
                // // 如果检测到锥桶但未对齐，进行对齐控制
                // if (cone_center_x != -1 && cone_pixel_count > 1000)
                // {
                //     // 计算对齐误差并控制转向
                //     alignment_error = (cone_center_x - roi_center_x) / (float)(roi_cone.width / 2);
                //     msg.angular.z = -alignment_error * 0.6; // 比例控制，使锥桶中心对准ROI中心
                    
                //     // 当对齐误差较大时，降低前进速度
                //     if (fabs(alignment_error) > 0.3) {
                //         msg.linear.x = normal_speed * 0.6;
                //     }
                    
                //     printf("Aligning to cone: error=%.2f, angular.z=%.2f\n", alignment_error, msg.angular.z);
                // }
                // else
                {
                    // 未检测到锥桶或锥桶像素太少，直行
                    msg.linear.x = normal_speed;
                    msg.angular.z = 0;
                }
            }
            break;
            
        case 1: // 右绕行（绕过第一个障碍物）
            if (avoidance_complete_frames <= frame_turn_big)
            {
                msg.linear.x = avoidance_forward_speed;
                msg.angular.z = -avoidance_turn_angle; // 大右转
            }
            else if (avoidance_complete_frames <= (frame_turn_big + frame_turn_small))
            {
                msg.linear.x = 0.05;
                msg.angular.z = avoidance_turn_angle; // 小左转回正
            }
            else if (avoidance_complete_frames <= (frame_turn_big + frame_turn_small + frame_turn_straight))
            {
                msg.linear.x = avoidance_forward_speed; // 直行
                msg.angular.z = 0;
            }
            else if (avoidance_complete_frames <= (2 * frame_turn_big + frame_turn_small + frame_turn_straight + 3))
            {
                msg.linear.x = avoidance_forward_speed;
                msg.angular.z = avoidance_turn_angle; // 大左转
            }
            else if (avoidance_complete_frames <= (2 * frame_turn_big + 2 * frame_turn_small + frame_turn_straight))
            {
                msg.linear.x = 0.05;
                msg.angular.z = -avoidance_turn_angle; // 小右转回正
            }
            else
            {
                avoidance_state = 2;  // 进入检测第二个障碍物阶段
                printf("First obstacle cleared\n\n");
            }
            
            avoidance_complete_frames ++;
            break;

        case 2:  // 检测第二个障碍物
            if (cone_pixel_count > obstacle_detection_threshold)
            {
                // 锥桶像素足够多且锥桶中心与ROI中心对齐，表示小车正对第二个锥桶
                obstacle_detected_frames++;
                if (obstacle_detected_frames > 5) // 连续5帧检测到障碍物
                {
                    // 第二个障碍物：向左绕行
                    avoidance_complete_frames = 0;
                    avoidance_state = 3; // 左绕行
                    printf("Second obstacle detected and aligned, turning LEFT\n");
                    obstacle_detected_frames = 0;
                }
            } 
            else 
            {
                obstacle_detected_frames = 0;
                
                // // 如果检测到锥桶但未对齐，进行对齐控制
                // if (cone_center_x != -1 && cone_pixel_count > 1000)
                // {
                //     // 计算对齐误差并控制转向
                //     alignment_error = (cone_center_x - roi_center_x) / (float)(roi_cone.width / 2);
                //     msg.angular.z = -alignment_error * 0.8; // 比例控制，使锥桶中心对准ROI中心
                    
                //     // 当对齐误差较大时，降低前进速度
                //     if (fabs(alignment_error) > 0.3) {
                //         msg.linear.x = normal_speed * 0.6;
                //     }
                    
                //     printf("Aligning to second cone: error=%.2f, angular.z=%.2f\n", alignment_error, msg.angular.z);
                // }
                // else
                // {
                    // 未检测到锥桶或锥桶像素太少，直行
                    msg.linear.x = normal_speed;
                    msg.angular.z = 0;
                // }
            }
            break;
            
        case 3: // 左绕行（绕过第二个障碍物）
            if (avoidance_complete_frames <= frame_turn_big)
            {
                msg.linear.x = avoidance_forward_speed;
                msg.angular.z = avoidance_turn_angle; // 大左转
            }
            else if (avoidance_complete_frames <= (frame_turn_big + frame_turn_small))
            {
                msg.linear.x = 0.05;
                msg.angular.z = -avoidance_turn_angle; // 小右转回正
            }
            else if (avoidance_complete_frames <= (frame_turn_big + frame_turn_small + frame_turn_straight))
            {
                msg.linear.x = avoidance_forward_speed; // 直行
                msg.angular.z = 0;
            }
            else if (avoidance_complete_frames <= (frame_turn_big + 2 * frame_turn_small + frame_turn_straight - 2))
            {
                // 微调时间和速度使得正好90度转弯
                msg.linear.x = avoidance_forward_speed; // 减速大转弯
                msg.angular.z = -avoidance_turn_angle;
            }
            else
            {
                avoidance_state = 4;  // 进入直线驶出阶段
                printf("First obstacle cleared, returning to path\n\n");
                msg.linear.x = avoidance_forward_speed;
                msg.angular.z = 0;
            }
            
            avoidance_complete_frames ++;
            break;

        case 4: // 闭环控制：沿道路中心线行驶
        {
            // 定义左右两侧ROI区域
            Rect roi_left(roi_cone.x - roi_cone_width, roi_cone_y_start, 
                          roi_cone_width, roi_cone_height);
            Rect roi_right(roi_cone.x + roi_cone_width, roi_cone_y_start, 
                           roi_cone_width, roi_cone_height);
            
            // 在图像上绘制左右ROI
            rectangle(img_show, roi_left, Scalar(255, 0, 0), 2);  // 蓝色 - 左ROI
            rectangle(img_show, roi_right, Scalar(0, 0, 255), 2); // 红色 - 右ROI
            
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
                        circle(img_show, Point(left_center_x, left_center_y), 5, Scalar(255, 0, 0), -1);
                        putText(img_show, "L", Point(left_center_x + 5, left_center_y - 5),
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
                        circle(img_show, Point(right_center_x, right_center_y), 5, Scalar(0, 0, 255), -1);
                        putText(img_show, "R", Point(right_center_x + 5, right_center_y - 5),
                                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
                    }
                }
            }
            
            // 显示左右锥桶像素数
            char left_text[50], right_text[50];
            sprintf(left_text, "Left: %d", left_pixel_count);
            sprintf(right_text, "Right: %d", right_pixel_count);
            putText(img_show, left_text, Point(roi_left.x, roi_left.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
            putText(img_show, right_text, Point(roi_right.x, roi_right.y - 10),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
            
            // 控制逻辑：计算道路中心线并控制小车
            float center_error = 0.0;
            
            if (left_center_x != -1 && right_center_x != -1) {
                // 情况1：两个锥桶都检测到，计算道路中心
                int lane_center = (left_center_x + right_center_x) / 2;
                center_error = (lane_center - img_width / 2) / (float)(img_width / 2);
                
                // 在图像上绘制道路中心线
                line(img_show, Point(lane_center, roi_cone_y_start - 50),
                     Point(lane_center, roi_cone_y_start + roi_cone_height + 50),
                     Scalar(0, 255, 255), 2);
                
                // 绘制理想中心线（图像中线）
                line(img_show, Point(img_width / 2, roi_cone_y_start - 50),
                     Point(img_width / 2, roi_cone_y_start + roi_cone_height + 50),
                     Scalar(255, 255, 255), 1);
                
                putText(img_show, "Both cones detected", Point(10, 110),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
                
            } else if (left_center_x != -1) {
                // 情况2：只检测到左侧锥桶，保持安全距离
                int desired_left_distance = roi_cone_width * 1.2; // 期望与左侧锥桶的距离
                int actual_distance = img_width / 2 - left_center_x;
                center_error = (desired_left_distance - actual_distance) / (float)desired_left_distance;
                
                putText(img_show, "Left cone only", Point(10, 110),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
                
            } else if (right_center_x != -1) {
                // 情况3：只检测到右侧锥桶，保持安全距离
                int desired_right_distance = roi_cone_width * 1.2; // 期望与右侧锥桶的距离
                int actual_distance = right_center_x - img_width / 2;
                center_error = (actual_distance - desired_right_distance) / (float)desired_right_distance;
                
                putText(img_show, "Right cone only", Point(10, 110),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
                
            } else {
                // 情况4：没有检测到锥桶，维持当前方向或小幅搜索
                center_error = 0.0;
                
                putText(img_show, "No cones detected", Point(10, 110),
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
            }
            
            // 根据误差调整速度和转向
            if (fabs(center_error) < 0.1) {
                // 误差很小，直行
                msg.linear.x = normal_speed;
                msg.angular.z = -center_error * 0.8; // 轻微调整
            } else if (fabs(center_error) < 0.3) {
                // 中等误差，减速并转向
                msg.linear.x = normal_speed * 0.8;
                msg.angular.z = -center_error * 1.2;
            } else {
                // 大误差，大幅减速并转向
                msg.linear.x = normal_speed * 0.5;
                msg.angular.z = -center_error * 1.5;
            }
            
            // 显示控制信息
            char control_text[50];
            sprintf(control_text, "Error: %.2f, Speed: %.2f, Turn: %.2f",
                    center_error, msg.linear.x, msg.angular.z);
            putText(img_show, control_text, Point(10, 135),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
            
            // // 检查是否应该结束任务（可选）
            // static int lane_keeping_frames = 0;
            // lane_keeping_frames++;
            
            // // 例如：保持30秒后停止
            // if (lane_keeping_frames > 375) { // 80ms/帧，30秒≈375帧
            //     msg.linear.x = 0;
            //     msg.angular.z = 0;
            //     putText(img_show, "Mission Completed!", Point(img_width/2-100, img_height/2),
            //             FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
            // }
            
            break;
        }
        }

        // 发布速度指令
        vel_pub.publish(msg);

        // 显示当前状态
        std::string state_text;
        switch (avoidance_state)
        {
            case 0: state_text = "Normal Driving - Find 1st"; break;
            case 1: state_text = "Right Avoidance - 1st"; break;
            case 2: state_text = "Normal Driving - Find 2nd"; break;
            case 3: state_text = "Left Avoidance - 2nd"; break;
            case 4: state_text = "Straight Exit"; break;
        }
        putText(img_show, "State: " + state_text, Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        
        // 显示对齐信息
        if (cone_center_x != -1) {
            char align_text[50];
            sprintf(align_text, "Align error: %.2f", alignment_error);
            putText(img_show, align_text, Point(10, 60), 
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 0), 2);
            
            // 显示ROI中心与锥桶中心的距离
            char dist_text[50];
            sprintf(dist_text, "Dist to center: %d", abs(cone_center_x - roi_center_x));
            putText(img_show, dist_text, Point(10, 85), 
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 200, 0), 2);
        }
                
        imshow("Navigation View", img_show);
        
        char key = waitKey(1);
        if (key == 27) break; // ESC退出
    }
    
    return 0;
}