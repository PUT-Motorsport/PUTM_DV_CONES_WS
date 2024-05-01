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

#include "cones_detection/cones_detection_node.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <filesystem>
#include "std_msgs/msg/header.hpp"

#define NMS_THRESH 0.4
#define CONF_THRESH 0.3



void print(std::string msg_prefix, sl::ERROR_CODE err_code, std::string msg_suffix) {
    std::cout << "[Sample] ";
    if (err_code != sl::ERROR_CODE::SUCCESS)
        std::cout << "[Error] ";
    std::cout << msg_prefix << " ";
    if (err_code != sl::ERROR_CODE::SUCCESS) {
        std::cout << " | " << toString(err_code) << " : ";
        std::cout << toVerbose(err_code);
    }
    if (!msg_suffix.empty())
        std::cout << " " << msg_suffix;
    std::cout << std::endl;
}

cv::Rect get_rect(BBox box) {
    return cv::Rect(round(box.x1), round(box.y1), round(box.x2 - box.x1), round(box.y2 - box.y1));
}

std::vector<sl::uint2> cvt(const BBox &bbox_in) {
    std::vector<sl::uint2> bbox_out(4);
    bbox_out[0] = sl::uint2(bbox_in.x1, bbox_in.y1);
    bbox_out[1] = sl::uint2(bbox_in.x2, bbox_in.y1);
    bbox_out[2] = sl::uint2(bbox_in.x2, bbox_in.y2);
    bbox_out[3] = sl::uint2(bbox_in.x1, bbox_in.y2);
    return bbox_out;
}

namespace cones_detection
{
auto custom_qos = rclcpp::QoS(rclcpp::KeepLast(1), rmw_qos_profile_sensor_data);

ConesDetectionNode::ConesDetectionNode(const rclcpp::NodeOptions & options)
:  Node("cones_detection", options)
{
  cones_detection_ = std::make_unique<cones_detection::ConesDetection>();

  build_engine = this->declare_parameter("build_engine", false);
  show_image = this->declare_parameter("show_image", false);
  onnx_path = this->declare_parameter("model_path", "model.onnx");
  engine_path = this->declare_parameter("engine_path", "engine.engine");
  svo_path = this->declare_parameter("svo_path", "");

  cones_pub_ = this->create_publisher<cones_interfaces::msg::Cones>("output_cones", custom_qos);
  
  if (show_image){
  image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("output_image2", custom_qos);
  }

  setenv("CUDA_MODULE_LOADING", "LAZY", 1);


  if (!std::filesystem::exists(engine_path))
  {
    build_engine = true;
  }

  if (build_engine)
  {
    OptimDim dyn_dim_profile;
    Yolo::build_engine(onnx_path, engine_path, dyn_dim_profile);
    std::cout << "Build finished" << std::endl;
  }

  inference();
}

void ConesDetectionNode::inference()
{
  sl::Camera zed;
  sl::InitParameters init_parameters;
  init_parameters.sdk_verbose = false;
  init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
  init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP; // OpenGL's coordinate system is right_handed
  
  if (!svo_path.empty())
  {
    init_parameters.input.setFromSVOFile(svo_path.c_str());
  }

  // Open the camera
  auto returned_state = zed.open(init_parameters);
  if (returned_state != sl::ERROR_CODE::SUCCESS) {
      print("Camera Open", returned_state, "Exit program.");
      // return EXIT_FAILURE;
  }


  zed.enablePositionalTracking();
  // Custom OD
  sl::ObjectDetectionParameters detection_parameters;
  detection_parameters.enable_tracking = false;
  detection_parameters.enable_segmentation = false; // designed to give person pixel mask
  detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::CUSTOM_BOX_OBJECTS;
  returned_state = zed.enableObjectDetection(detection_parameters);

  if (returned_state != sl::ERROR_CODE::SUCCESS) {
      print("enableObjectDetection", returned_state, "\nExit program.");
      zed.close();
      // return EXIT_FAILURE;
  }

  auto camera_config = zed.getCameraInformation().camera_configuration;
  sl::Resolution pc_resolution(std::min((int) camera_config.resolution.width, 720), std::min((int) camera_config.resolution.height, 404));
  auto camera_info = zed.getCameraInformation(pc_resolution).camera_configuration;


  if (detector.init(engine_path)) {
    std::cerr << "Detector init failed" << std::endl;
  }
  else {
    std::cout << "Detector init success" << std::endl;
  }


  auto display_resolution = zed.getCameraInformation().camera_configuration.resolution;
  sl::Mat left_sl, point_cloud;
  cv::Mat left_cv;
  sl::ObjectDetectionRuntimeParameters objectTracker_parameters_rt;
  sl::Objects objects;
  sl::Pose cam_w_pose;
  cam_w_pose.pose_data.setIdentity();
  

  while (rclcpp::ok())
  {
    if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
            auto start_time = std::chrono::high_resolution_clock::now();
            // Get image for inference
            zed.retrieveImage(left_sl, sl::VIEW::LEFT);

            // Running inference
            auto detections = detector.run(left_sl, display_resolution.height, display_resolution.width, CONF_THRESH);
            // Get image for display
            left_cv = slMat2cvMat(left_sl);

            std::unordered_map<std::string, int> labels;

            // Preparing for ZED SDK ingesting
            std::vector<sl::CustomBoxObjectData> objects_in;
            for (auto &it : detections) {
                sl::CustomBoxObjectData tmp;
                // Fill the detections into the correct format
                tmp.unique_object_id = sl::generate_unique_id();
                tmp.probability = it.prob;
                tmp.label = (int) it.label;
                tmp.bounding_box_2d = cvt(it.box);
                tmp.is_grounded = ((int) it.label == 0); // Only the first class (person) is grounded, that is moving on the floor plane
                                                         // others are tracked in full 3D space                
                objects_in.push_back(tmp);

                labels[(std::string)tmp.unique_object_id] = tmp.label;

            }
            // Send the custom detected boxes to the ZED
            zed.ingestCustomBoxObjects(objects_in);

            // Displaying 'raw' objects
            if (show_image){
            for (size_t j = 0; j < detections.size(); j++) {
                cv::Rect r = get_rect(detections[j].box);
                cv::rectangle(left_cv, r, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
                cv::putText(left_cv, std::to_string((int) detections[j].label), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            std_msgs::msg::Header header;
            header.stamp = this->get_clock()->now();
            header.frame_id = "map";
            sensor_msgs::msg::Image::SharedPtr msg_out = cv_bridge::CvImage(header, "bgr8", left_cv).toImageMsg();
            image_pub_->publish(*msg_out);
            }

            // Retrieve the tracked objects, with 2D and 3D attributes


            zed.retrieveObjects(objects, objectTracker_parameters_rt);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<float, std::milli> (end_time - start_time);
 
            std::cout << duration.count()<< "ms" << std::endl;



            cones_interfaces::msg::Cones cones_;
            cones_.header.stamp = this->get_clock()->now();
            cones_.header.frame_id = "map";


            for (auto object : objects.object_list){
              sl::float3 position = object.position;
              int label = labels[(std::string)object.unique_object_id];

              geometry_msgs::msg::Pose tmp;

              tmp.position.x = position.x / 1000;
              tmp.position.y = position.y / 1000;
              tmp.position.z = position.z / 1000;
              tmp.orientation.w = 0.707;
              tmp.orientation.y = 0.707;

              if (label == 0)
              cones_.yellow_cones.push_back(tmp);
              if (label == 1)
              cones_.blue_cones.push_back(tmp);
            }
            cones_pub_->publish(cones_);
            }
  }
}

}  // namespace cones_detection

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(cones_detection::ConesDetectionNode)
