#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <std_msgs/msg/string.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <regex>

// 获取动态帧列表
std::vector<std::string> get_dynamic_frame_list(tf2_ros::Buffer &tf_buffer, const std::vector<std::string> &prefixes) {
    std::string all_frames_yaml = tf_buffer.allFramesAsYAML();
    YAML::Node frames = YAML::Load(all_frames_yaml);

    std::vector<std::string> frame_list;
    for (YAML::const_iterator it = frames.begin(); it != frames.end(); ++it) {
        std::string frame_name = it->first.as<std::string>();
        for (const auto &prefix : prefixes) {
            // 使用正则表达式判断 frame_name 是否以 prefix 开头，后跟数字
            std::regex pattern("^" + prefix + "\\s*\\d+$");
            if (std::regex_match(frame_name, pattern)) { // 满足 "prefix+数字" 格式
                frame_list.push_back(frame_name);
                break;
            }
        }
    }

    std::sort(frame_list.begin(), frame_list.end());
    return frame_list;
}

class DynamicFramePublisher : public rclcpp::Node {
public:
    DynamicFramePublisher() : Node("dynamic_frame_publisher") {
        // 创建 tf2 Buffer 和 Listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 创建 Publisher
        frame_list_pub_ = this->create_publisher<std_msgs::msg::String>("dynamic_frames", 10);

        // 创建订阅者订阅 class_names
        class_names_sub_ = this->create_subscription<std_msgs::msg::String>(
            "class_names", 10,
            std::bind(&DynamicFramePublisher::class_names_callback, this, std::placeholders::_1));

        // 定时器，定期发布帧列表
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(500), // 500ms 更新一次
            std::bind(&DynamicFramePublisher::publish_frame_list, this)
        );
    }

private:
    void class_names_callback(const std_msgs::msg::String::SharedPtr msg) {
        class_names_prefixes_.clear();
        std::istringstream iss(msg->data);
        std::string prefix;
        while (std::getline(iss, prefix, ',')) {
            // 仅去除前后空格，保留中间空格
            prefix.erase(prefix.begin(), std::find_if(prefix.begin(), prefix.end(), [](unsigned char ch) { return !std::isspace(ch); }));
            prefix.erase(std::find_if(prefix.rbegin(), prefix.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), prefix.end());
            if (!prefix.empty()) {
                class_names_prefixes_.push_back(prefix);
            }
        }

        // 打印解析后的前缀
        // RCLCPP_INFO(this->get_logger(), "Updated class_names prefixes:");
        // for (const auto &prefix : class_names_prefixes_) {
        //     RCLCPP_INFO(this->get_logger(), "Prefix: '%s'", prefix.c_str());
        // }
    }



    void publish_frame_list() {
        // 动态获取帧列表
        std::vector<std::string> frame_list = get_dynamic_frame_list(*tf_buffer_, class_names_prefixes_);

        // 将帧列表拼接为逗号分隔的字符串
        std::ostringstream oss;
        for (size_t i = 0; i < frame_list.size(); ++i) {
            oss << frame_list[i];
            if (i != frame_list.size() - 1) { // 如果不是最后一个元素，添加逗号
                oss << ", ";
            }
        }

        // 创建消息并发布
        auto msg = std_msgs::msg::String();
        msg.data = oss.str();
        frame_list_pub_->publish(msg);

        // RCLCPP_INFO(this->get_logger(), "Published frame list: %s", msg.data.c_str());
    }



    // 成员变量
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr frame_list_pub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr class_names_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<std::string> class_names_prefixes_; // 存储解析后的前缀
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DynamicFramePublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
