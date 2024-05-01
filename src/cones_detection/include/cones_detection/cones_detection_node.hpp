// Copyright 2024 BartlomiejGasyna
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CONES_DETECTION__CONES_DETECTION_NODE_HPP_
#define CONES_DETECTION__CONES_DETECTION_NODE_HPP_

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include "cones_detection/cones_detection.hpp"
#include "cones_interfaces/msg/cones.hpp" 

namespace cones_detection
{
using ConesDetectionPtr = std::unique_ptr<cones_detection::ConesDetection>;

class CONES_DETECTION_PUBLIC ConesDetectionNode : public rclcpp::Node
{
public:
  explicit ConesDetectionNode(const rclcpp::NodeOptions & options);

private:
  ConesDetectionPtr cones_detection_{nullptr};

  bool build_engine{false};
  bool show_image{false};
  std::string onnx_path{"model.onnx"};
  std::string engine_path{"engine.engine"};
  std::string svo_path{""};

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  void inference();

  rclcpp::Publisher<cones_interfaces::msg::Cones>::SharedPtr cones_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

  Yolo detector;
};
}  // namespace cones_detection

#endif  // CONES_DETECTION__CONES_DETECTION_NODE_HPP_
