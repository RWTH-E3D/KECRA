#include "ur5e_gripper_control/dual_ur5e_gripper.h"
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sstream>
#include <vector>
#include <algorithm>
#include <regex>

class DynamicGraspControl : public rclcpp::Node
{
public:
    DynamicGraspControl(std::shared_ptr<DualUR5eGripper> dual_ur5e_node)
        : Node("dynamic_grasp_control"), dual_ur5e_node_(dual_ur5e_node)
    {
        frame_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "dynamic_frames", 10,
            std::bind(&DynamicGraspControl::frameCallback, this, std::placeholders::_1));

        action_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "action_sequence", 10,
            std::bind(&DynamicGraspControl::actionCallback, this, std::placeholders::_1));

        left_pose_publisher_ = this->create_publisher<std_msgs::msg::String>("left_arm_poses", 10);
        right_pose_publisher_ = this->create_publisher<std_msgs::msg::String>("right_arm_poses", 10);

        pose_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/pose_sequence", 10,
            std::bind(&DynamicGraspControl::poseCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Dynamic grasp control node initialized.");
    }

private:
    void frameCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        // 解析动态帧列表
        std::vector<std::string> frame_list;
        std::istringstream iss(msg->data);
        std::string frame;
        while (std::getline(iss, frame, ','))
        {
            frame.erase(frame.begin(), std::find_if(frame.begin(), frame.end(), [](unsigned char ch)
                                                    { return !std::isspace(ch); }));
            frame.erase(std::find_if(frame.rbegin(), frame.rend(), [](unsigned char ch)
                                     { return !std::isspace(ch); })
                            .base(),
                        frame.end());

            if (!frame.empty())
            {
                frame_list.push_back(frame);
            }
        }

        if (frame_list.empty())
        {
            // RCLCPP_WARN(this->get_logger(), "Received empty frame list. Waiting for new frames...");
            return;
        }

        // 存储左右臂位姿
        std::vector<std::pair<std::string, std::vector<double>>> left_cube_pose_list, right_cube_pose_list;

        for (const auto &frame : frame_list)
        {
            std::vector<double> cube_pose;
            try
            {
                dual_ur5e_node_->get_cube_pose("world", frame, cube_pose);

                double y = cube_pose[1]; // 根据Y轴位置判断分配给左臂或右臂
                if (y > 0)
                {
                    dual_ur5e_node_->get_cube_pose("left_base_link", frame, cube_pose);
                    adjust_grasp_pose(cube_pose);
                    left_cube_pose_list.emplace_back(frame, cube_pose);
                }
                else
                {
                    dual_ur5e_node_->get_cube_pose("right_base_link", frame, cube_pose);
                    adjust_grasp_pose(cube_pose);
                    right_cube_pose_list.emplace_back(frame, cube_pose);
                }
            }
            catch (const std::exception &e)
            {
                RCLCPP_ERROR(this->get_logger(), "Error processing frame '%s': %s", frame.c_str(), e.what());
            }
        }

        publish_poses(left_cube_pose_list, true);   // 发布左手臂位姿
        publish_poses(right_cube_pose_list, false); // 发布右手臂位姿
    }

    void actionCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        if (!msg || msg->data.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received an empty or invalid action sequence.");
            return;
        }

        std::string action_sequence = trim(msg->data); // 去除整体字符串多余空格
        std::string buffer;                            // 用于存储当前解析的完整动作
        int parenthesis_count = 0;                     // 用于括号配对计数

        for (size_t i = 0; i < action_sequence.size(); ++i)
        {
            char c = action_sequence[i];

            // 累积字符到缓冲区
            buffer += c;

            // 更新括号计数
            if (c == '(')
            {
                parenthesis_count++;
            }
            else if (c == ')')
            {
                parenthesis_count--;
            }

            // 判断是否到达动作结尾（括号闭合且遇到逗号或最后一个字符）
            if (parenthesis_count == 0 && (c == ',' || i == action_sequence.size() - 1))
            {
                // 如果是最后一个字符但不是逗号，确保动作完整
                if (c != ',')
                {
                    buffer = trim(buffer); // 去除当前缓冲区的多余字符
                }

                // 去除尾部的逗号
                if (!buffer.empty() && buffer.back() == ',')
                {
                    buffer.pop_back();
                }

                // 确保没有额外空格
                buffer = trim(buffer);

                // 处理动作
                if (!buffer.empty())
                {
                    try
                    {
                        processAction(buffer); // 调用动作处理函数
                    }
                    catch (const std::exception &e)
                    {
                        RCLCPP_ERROR(this->get_logger(), "Error processing action '%s': %s", buffer.c_str(), e.what());
                    }
                }

                // 清空缓冲区，准备解析下一个动作
                buffer.clear();
            }
        }

        // 检查未完成的动作
        if (!buffer.empty() || parenthesis_count != 0)
        {
            RCLCPP_ERROR(this->get_logger(), "Incomplete or mismatched action detected: '%s'", buffer.c_str());
        }
    }

    double evaluateExpression(const std::string &expr)
    {
        try
        {
            double result = 0.0;
            size_t start = 0, end = 0;

            // 简单解析 "+" 运算符的表达式
            while ((end = expr.find('+', start)) != std::string::npos)
            {
                result += std::stod(trim(expr.substr(start, end - start)));
                start = end + 1;
            }
            result += std::stod(trim(expr.substr(start))); // 最后一个值
            return result;
        }
        catch (const std::exception &e)
        {
            throw std::invalid_argument("Failed to evaluate expression: " + expr);
        }
    }

    std::vector<double> parseMoveCommand(const std::string &action)
    {
        size_t start = action.find('(') + 1;
        size_t end = action.find(')');
        if (start == std::string::npos || end == std::string::npos || start >= end)
        {
            throw std::invalid_argument("Invalid move command format: " + action);
        }

        std::string params = action.substr(start, end - start);
        std::istringstream iss(params);
        std::vector<double> position;
        std::string value;

        while (std::getline(iss, value, ','))
        {
            try
            {
                position.push_back(evaluateExpression(trim(value))); // 解析表达式
            }
            catch (const std::exception &e)
            {
                throw std::invalid_argument("Invalid parameter in move command: " + value + " (" + e.what() + ")");
            }
        }

        if (position.size() != 3)
        {
            throw std::invalid_argument("Move command must have exactly 3 parameters: " + action);
        }

        // 添加默认姿态
        position.push_back(0.0);  // Roll
        position.push_back(M_PI); // Pitch
        position.push_back(0.0);  // Yaw

        return position;
    }

    void processAction(const std::string &action)
    {
        RCLCPP_INFO(this->get_logger(), "Processing action: '%s'", action.c_str());
        std::vector<double> do_nothing;

        try
        {
            if (action == "closeLeftGripper")
            {
                RCLCPP_INFO(this->get_logger(), "Closing left gripper.");
                dual_ur5e_node_->left_grasp(0.38);
                rclcpp::sleep_for(std::chrono::seconds(1));
            }
            else if (action == "closeRightGripper")
            {
                RCLCPP_INFO(this->get_logger(), "Closing right gripper.");
                dual_ur5e_node_->right_grasp(0.38);
                rclcpp::sleep_for(std::chrono::seconds(1));
            }
            else if (action == "releaseLeftGripper")
            {
                RCLCPP_INFO(this->get_logger(), "Releasing left gripper.");
                dual_ur5e_node_->left_grasp(0.0);
                rclcpp::sleep_for(std::chrono::seconds(1));
            }
            else if (action == "releaseRightGripper")
            {
                RCLCPP_INFO(this->get_logger(), "Releasing right gripper.");
                dual_ur5e_node_->right_grasp(0.0);
                rclcpp::sleep_for(std::chrono::seconds(1));
            }
            else if (action == "leftArmReset")
            {
                RCLCPP_INFO(this->get_logger(), "Left Arm Reset.");
                dual_ur5e_node_->go_to_ready_position(true);
                rclcpp::sleep_for(std::chrono::seconds(1));
            }
            else if (action == "rightArmReset")
            {
                RCLCPP_INFO(this->get_logger(), "Right Arm Reset.");
                dual_ur5e_node_->go_to_ready_position(false);
                rclcpp::sleep_for(std::chrono::seconds(1));
            }
            else if (action.rfind("leftArmMove", 0) == 0)
            { // 确保以 "leftArmMove" 开头
                auto position = parseMoveCommand(action);
                // std::vector<double> position = {0.688558, 0, 0.170252, 0.0, 3.14, 0.0};
                // rclcpp::sleep_for(std::chrono::seconds(1));
                RCLCPP_INFO(this->get_logger(), "Executing leftArmMove to position: [%.3f, %.3f, %.3f]",
                            position[0], position[1], position[2]);
                dual_ur5e_node_->plan_and_execute(position, do_nothing);
                rclcpp::sleep_for(std::chrono::seconds(1));
            }
            else if (action.rfind("rightArmMove", 0) == 0)
            { // 确保以 "rightArmMove" 开头
                auto position = parseMoveCommand(action);
                RCLCPP_INFO(this->get_logger(), "Executing rightArmMove to position: [%.3f, %.3f, %.3f]",
                            position[0], position[1], position[2]);
                dual_ur5e_node_->plan_and_execute(do_nothing, position);
                rclcpp::sleep_for(std::chrono::seconds(1));
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Unknown action: '%s'", action.c_str());
            }
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error processing action '%s': %s", action.c_str(), e.what());
        }
    }

    std::string trim(const std::string &str)
    {
        size_t start = str.find_first_not_of(" \t\r\n");
        size_t end = str.find_last_not_of(" \t\r\n");
        return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
    }

    void adjust_grasp_pose(std::vector<double> &pose)
    {
        pose[0] -= 0.025; // 调整x用于抓取
        pose[1] += 0.0; // 调整y
        pose[2] += 0.14;  // 调整z用于抓取
        pose[3] = 0.0;    // Roll
        pose[4] = M_PI;   // Pitch
        pose[5] = 0.0;    // Yaw
    }

    // void poseCallback(const std_msgs::msg::String::SharedPtr msg)
    // {
    //     std::vector<double> do_nothing;

    //     if (!msg || msg->data.empty()){
    //         return;
    //     }

    //     try
    //     {
    //         std::vector<double> pose;
    //         std::string json = msg->data;
    //         json.erase(std::remove(json.begin(), json.end(), '['), json.end());
    //         json.erase(std::remove(json.begin(), json.end(), ']'), json.end());
    //         std::istringstream iss(json);
    //         std::string token;
    //         while (std::getline(iss, token, ','))
    //         {
    //             pose.push_back(std::stod(token));
    //         }

    //         if (pose.size() != 8)
    //         {
    //             RCLCPP_ERROR(this->get_logger(), "Invalid pose size: expected 8, got %zu", pose.size());
    //             return;
    //         }

    //         int arm = static_cast<int>(pose[6]);
    //         int gripper = static_cast<int>(pose[7]);

    //         std::vector<double> target_pose = {
    //             pose[0], pose[1], pose[2],
    //             pose[3], pose[4], pose[5]};

    //         if (arm == 1)
    //         {
    //             RCLCPP_INFO(this->get_logger(), "Left arm executing pose.");
    //             dual_ur5e_node_->plan_and_execute(target_pose, do_nothing); // 左臂动作
    //             if (gripper == 1)
    //                 dual_ur5e_node_->left_grasp(0.38);
    //             else
    //                 dual_ur5e_node_->left_grasp(0.0);
    //         }
    //         else
    //         {
    //             RCLCPP_INFO(this->get_logger(), "Right arm executing pose.");
    //             dual_ur5e_node_->plan_and_execute(do_nothing, target_pose); // 右臂动作
    //             if (gripper == 1)
    //                 dual_ur5e_node_->right_grasp(0.38);
    //             else
    //                 dual_ur5e_node_->right_grasp(0.0);
    //         }

    //         rclcpp::sleep_for(std::chrono::milliseconds(500));
    //     }
    //     catch (const std::exception &e)
    //     {
    //         RCLCPP_ERROR(this->get_logger(), "Pose parsing error: %s", e.what());
    //     }
    // }

    // void executePose(const std::vector<double> &pose)
    // {
    //     std::vector<double> do_nothing;
    //     int arm = static_cast<int>(pose[6]);
    //     int gripper = static_cast<int>(pose[7]);

    //     std::vector<double> xyzrpy = {pose.begin(), pose.begin() + 6};

    //     if (arm == 1) // ---- Left arm ----
    //     {
    //         RCLCPP_INFO(this->get_logger(), "Left arm executing pose");
    //         dual_ur5e_node_->plan_and_execute(xyzrpy, do_nothing);
    //         dual_ur5e_node_->left_grasp(gripper ? 0.38 : 0.0);
    //     }
    //     else // ---- Right arm ---
    //     {
    //         RCLCPP_INFO(this->get_logger(), "Right arm executing pose");
    //         dual_ur5e_node_->plan_and_execute(do_nothing, xyzrpy);
    //         dual_ur5e_node_->right_grasp(gripper ? 0.38 : 0.0);
    //     }

    //     rclcpp::sleep_for(std::chrono::milliseconds(500));
    // }

    void executePose(const std::vector<double> &pose)
    {
        std::vector<double> nothing;
        const int arm = static_cast<int>(pose[6]); // 1 = left, 0 = right
        const int gripper = static_cast<int>(pose[7]);

        // xyzrpy = pose[0..5]
        std::vector<double> xyzrpy(pose.begin(), pose.begin() + 6);

        const bool xyz_is_zero = (pose[0] == 0.0 && pose[1] == 0.0 && pose[2] == 0.0);

        if (xyz_is_zero)
        {
            if (arm == 1)
            {
                RCLCPP_INFO(this->get_logger(), "Left arm reset");
                dual_ur5e_node_->go_to_ready_position(true); // 左臂
            }
            else
            {
                RCLCPP_INFO(this->get_logger(), "Right arm reset");
                dual_ur5e_node_->go_to_ready_position(false); // 右臂
            }
        }
        else
        {

            if (arm == 1)
            {
                RCLCPP_INFO(this->get_logger(), "Left arm executing pose");
                dual_ur5e_node_->plan_and_execute(xyzrpy, nothing);
                dual_ur5e_node_->left_grasp(gripper ? 0.38 : 0.0);
            }
            else
            {
                RCLCPP_INFO(this->get_logger(), "Right arm executing pose");
                dual_ur5e_node_->plan_and_execute(nothing, xyzrpy);
                dual_ur5e_node_->right_grasp(gripper ? 0.38 : 0.0);
            }
            rclcpp::sleep_for(std::chrono::milliseconds(500));
        }
    }

    void poseCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        if (!msg || msg->data.empty())
            return;

        try
        {
            std::string raw = trim(msg->data);

            if (!raw.empty() && raw.front() == '[')
                raw.erase(0, 1);
            if (!raw.empty() && raw.back() == ']')
                raw.pop_back();

            if (raw.find("],") == std::string::npos)
            {
                /* ---------- 只有一条 pose ---------- */
                std::istringstream iss(raw);
                std::string tok;
                std::vector<double> pose;
                while (std::getline(iss, tok, ','))
                    pose.push_back(std::stod(trim(tok)));

                if (pose.size() != 8)
                    throw std::runtime_error("pose length != 8");

                executePose(pose);
            }
            else
            {
                /* ---------- 多条 pose ---------- */
                std::regex delim(R"(\]\s*,\s*\[)");
                std::sregex_token_iterator it(raw.begin(), raw.end(), delim, -1), end;

                for (; it != end; ++it)
                {
                    std::string item = it->str();
                    item.erase(std::remove(item.begin(), item.end(), '['), item.end());
                    item.erase(std::remove(item.begin(), item.end(), ']'), item.end());

                    std::istringstream iss(item);
                    std::string tok;
                    std::vector<double> pose;
                    while (std::getline(iss, tok, ','))
                        pose.push_back(std::stod(trim(tok)));

                    if (pose.size() != 8)
                    {
                        RCLCPP_ERROR(this->get_logger(),
                                     "Invalid pose size: %zu (skip)", pose.size());
                        continue;
                    }
                    executePose(pose);
                }
            }
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Pose parsing error: %s", e.what());
        }
    }

    void publish_poses(const std::vector<std::pair<std::string, std::vector<double>>> &cube_pose_list, bool is_left)
    {
        if (cube_pose_list.empty())
        {
            // RCLCPP_WARN(this->get_logger(), "No poses to publish for %s arm.", is_left ? "left" : "right");
            return;
        }

        std_msgs::msg::String msg;
        std::ostringstream oss;

        oss << (is_left ? "Left arm poses:\n" : "Right arm poses:\n");
        for (const auto &[frame_name, pose] : cube_pose_list)
        {
            oss << "Frame: " << frame_name << ", Position: [" << pose[0] << ", " << pose[1] << ", " << pose[2]
                << "], Orientation: [" << pose[3] << ", " << pose[4] << ", " << pose[5] << "]\n";
        }

        msg.data = oss.str();
        if (is_left)
        {
            left_pose_publisher_->publish(msg);
        }
        else
        {
            right_pose_publisher_->publish(msg);
        }

        // RCLCPP_INFO(this->get_logger(), "Published %zu poses for %s arm.", cube_pose_list.size(), is_left ? "left" : "right");
    }

    std::shared_ptr<DualUR5eGripper> dual_ur5e_node_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr frame_subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr left_pose_publisher_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr right_pose_publisher_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr action_subscription_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr pose_subscription_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);

    auto dual_ur5e_node_ = std::make_shared<DualUR5eGripper>(node_options);
    dual_ur5e_node_->init();
    auto grasp_control_node_ = std::make_shared<DynamicGraspControl>(dual_ur5e_node_);

    // Run both nodes in a MultiThreadedExecutor
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(dual_ur5e_node_);
    executor.add_node(grasp_control_node_);
    executor.spin();

    // auto node = std::make_shared<DynamicGraspControl>(dual_ur5e_node_);
    // rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}
