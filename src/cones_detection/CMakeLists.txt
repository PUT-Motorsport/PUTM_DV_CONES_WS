cmake_minimum_required(VERSION 3.5)
project(cones_detection)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies(REQUIRED
  ${${PROJECT_NAME}_BUILD_DEPENDS}
  ${${PROJECT_NAME}_BUILDTOOL_DEPENDS}
)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cones_interfaces REQUIRED)
find_package(ZED 4 REQUIRED)

# tensorrt
include_directories(/usr/include/)
link_directories(/usr/lib/)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)
# ZED
link_directories(${ZED_LIBRARY_DIR})
link_libraries(${ZED_LIBRARIES})
include_directories(${ZED_INCLUDE_DIRS})

add_definitions(-O3 -D_MWAITXINTRIN_H_INCLUDED -Wno-deprecated-declarations)

IF(NOT WIN32)
     SET(SPECIAL_OS_LIBS "pthread" "X11")
    add_definitions(-Wno-write-strings)
ENDIF()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-O3 -D_MWAITXINTRIN_H_INCLUDED -Wno-deprecated-declarations)


link_directories(${CUDA_LIBRARY_DIRS})

link_libraries(${CUDA_LIBRARIES} ${OpenCV_LIBS} nvinfer nvinfer_plugin nvparsers nvonnxparser ${SPECIAL_OS_LIBS})

# add_compile_options(-Wall -Wextra -pedantic -Werror)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

if (LINK_SHARED_ZED)
    SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})
else()
    SET(ZED_LIBS ${ZED_STATIC_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_LIBRARY})
endif()

set(CONES_DETECTION_LIB_SRC
  src/cones_detection.cpp
)

set(CONES_DETECTION_LIB_HEADERS
  include/cones_detection/cones_detection.hpp
  include/cones_detection/visibility_control.hpp
)

ament_auto_add_library(${PROJECT_NAME} SHARED
  ${CONES_DETECTION_LIB_SRC}
  ${CONES_DETECTION_LIB_HEADERS}
)

set(CONES_DETECTION_NODE_SRC
  src/cones_detection_node.cpp
)

set(CONES_DETECTION_NODE_HEADERS
  include/cones_detection/cones_detection_node.hpp
)

ament_auto_add_library(${PROJECT_NAME}_node SHARED
  ${CONES_DETECTION_NODE_SRC}
  ${CONES_DETECTION_NODE_HEADERS}
)
rclcpp_components_register_node(${PROJECT_NAME}_node
  PLUGIN "cones_detection::ConesDetectionNode"
  EXECUTABLE ${PROJECT_NAME}_node_exe
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_ros_isolated_gtest(test_cones_detection
    test/test_cones_detection.cpp
  )
  ament_target_dependencies(test_cones_detection ${TEST_SOURCES})
  target_link_libraries(test_cones_detection ${PROJECT_NAME})
endif()

ament_auto_package(INSTALL_TO_SHARE
  launch
  config
)
