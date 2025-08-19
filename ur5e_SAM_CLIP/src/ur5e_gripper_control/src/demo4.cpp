#include "ur5e_gripper_control/dual_ur5e_gripper.h"

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sstream>

class DynamicFrameSubscriber : public rclcpp::Node {
public:
    DynamicFrameSubscriber() : Node("dynamic_frame_subscriber") {
        frame_subscription_ = this->create_subscription<std_msgs::msg::String>(
            "dynamic_frames", 10, 
            std::bind(&DynamicFrameSubscriber::frameCallback, this, std::placeholders::_1)
        );
        RCLCPP_INFO(this->get_logger(), "Subscribed to 'dynamic_frames' topic.");
    }

    const std::vector<std::string>& getFrames() const {
        return frame_list_;
    }

private:
    void frameCallback(const std_msgs::msg::String::SharedPtr msg) {
        frame_list_.clear();

        // Split the input string by spaces to extract individual frame names
        std::istringstream iss(msg->data);
        std::string frame;
        while (std::getline(iss, frame, ' ')) { 
            if (!frame.empty()) { // Ignore empty frames
                frame_list_.push_back(frame);
            }
        }

        if (!frame_list_.empty()) {
            RCLCPP_INFO(this->get_logger(), "Updated frames: [%s]", msg->data.c_str());
        } else {
            RCLCPP_WARN(this->get_logger(), "Received empty frame list.");
        }
    }

    std::vector<std::string> frame_list_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr frame_subscription_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);

    auto dual_ur5e_node = std::make_shared<DualUR5eGripper>(node_options);
    auto frame_node = std::make_shared<DynamicFrameSubscriber>();
    dual_ur5e_node->init();

    // Run both nodes in a MultiThreadedExecutor
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(frame_node);
    executor.add_node(dual_ur5e_node);
    std::thread([&executor]() { executor.spin(); }).detach();

    // Wait for frames to be updated
    RCLCPP_INFO(rclcpp::get_logger("demo4"), "Waiting for dynamic frames...");
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::vector<std::string> to_frame_list = frame_node->getFrames();
    if (to_frame_list.empty()) {
        RCLCPP_ERROR(rclcpp::get_logger("demo4"), "No frames received. Exiting...");
        rclcpp::shutdown();
        return -1;
    }

    std::vector<std::vector<double>> left_cube_pose_list, right_cube_pose_list;
    std::vector<std::vector<double>> left_target_pose_list, right_target_pose_list;

    dual_ur5e_node->get_target_pose_list(left_target_pose_list, right_target_pose_list);

    std::string from_frame_left = "left_base_link";
    std::string from_frame_right = "right_base_link";

    for (const auto& frame : to_frame_list) {
        std::vector<double> cube_pose;
        try {
            dual_ur5e_node->get_cube_pose("world", frame, cube_pose);

            double y = cube_pose[1]; // cube position y relative to world frame
            if (y > 0) {
                dual_ur5e_node->get_cube_pose(from_frame_left, frame, cube_pose);
                cube_pose[0] -= 0.025; // modify x for grasp
                cube_pose[1] = 0.0;
                cube_pose[2] += 0.14; // modify z for grasp
                cube_pose[3] = 0.0; // roll
                cube_pose[4] = M_PI; // pitch
                cube_pose[5] = 0.0; // yaw
                left_cube_pose_list.push_back(cube_pose);
            } else {
                dual_ur5e_node->get_cube_pose(from_frame_right, frame, cube_pose);
                cube_pose[0] -= 0.025; // modify x for grasp
                cube_pose[1] = 0.0;
                cube_pose[2] += 0.14; // modify z for grasp
                cube_pose[3] = 0.0; // roll
                cube_pose[4] = M_PI; // pitch
                cube_pose[5] = 0.0; // yaw
                right_cube_pose_list.push_back(cube_pose);
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(rclcpp::get_logger("demo4"), "Error getting cube pose for frame '%s': %s", frame.c_str(), e.what());
        }
    }

    std::vector<double> do_nothing;

    // Use the maximum size of left and right cube lists
    size_t max_size = std::max(left_cube_pose_list.size(), right_cube_pose_list.size());
    bool left_grasped = false;  
    bool right_grasped = false;

    for (size_t i = 0; i < max_size; ++i) {
        std::vector<double> left_pose = (i < left_cube_pose_list.size()) ? left_cube_pose_list[i] : do_nothing;
        std::vector<double> right_pose = (i < right_cube_pose_list.size()) ? right_cube_pose_list[i] : do_nothing;

        if (!left_pose.empty() || !right_pose.empty()) {
            dual_ur5e_node->plan_and_execute(left_pose, right_pose);
        }

        if (i < left_cube_pose_list.size()) {
            dual_ur5e_node->left_grasp(0.38); // Grasp with left gripper
            left_grasped = true; 
        }
        if (i < right_cube_pose_list.size()) {
            dual_ur5e_node->right_grasp(0.38); // Grasp with right gripper
            right_grasped = true; 
        }
        rclcpp::sleep_for(std::chrono::seconds(1));

        if (left_grasped && i < left_target_pose_list.size()) {
            dual_ur5e_node->plan_and_execute(left_target_pose_list[i], do_nothing);
            dual_ur5e_node->left_grasp(0.0); // Release left gripper
            rclcpp::sleep_for(std::chrono::seconds(1));
            dual_ur5e_node->go_to_ready_position(true); // Go to ready position
            left_grasped = false; 
        }

        if (right_grasped && i < right_target_pose_list.size()) {
            dual_ur5e_node->plan_and_execute(do_nothing, right_target_pose_list[i]);
            dual_ur5e_node->right_grasp(0.0); // Release right gripper
            rclcpp::sleep_for(std::chrono::seconds(1));
            dual_ur5e_node->go_to_ready_position(false); // Go to ready position
            right_grasped = false;
        }
    }

    // dual_ur5e_node->go_to_ready_position(false);
    rclcpp::shutdown();
    return 0;
}



