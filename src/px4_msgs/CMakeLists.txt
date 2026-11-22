cmake_minimum_required(VERSION 3.5)

project(px4_msgs)


# CMake 3.22/Make can emit unquoted custom-command paths containing
# parentheses. rosidl_generator_py creates an ament_cmake_python target, and
# /bin/sh then fails with: Syntax error: "(" unexpected. Fail early with a
# clear diagnostic instead. Build this workspace from a simple Linux path
# such as ~/px4_nmpc.
foreach(_px4_msgs_path "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
  if("${_px4_msgs_path}" MATCHES "[()]")
    message(FATAL_ERROR
      "px4_msgs cannot be built from a path containing '(' or ')': "
      "${_px4_msgs_path}. Copy the workspace to a simple path such as "
      "$ENV{HOME}/px4_nmpc and rebuild from a clean build directory.")
  endif()
endforeach()

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ##############################################################################
# Generate ROS messages, ROS2 interfaces and IDL files #
# ##############################################################################

# get all msg files
set(MSGS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/msg")
file(GLOB PX4_MSGS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${MSGS_DIR}/*.msg")

# get all srv files
set(SRVS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/srv")
file(GLOB PX4_SRVS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${SRVS_DIR}/*.srv")

# Generate introspection typesupport for C and C++ and IDL files
rosidl_generate_interfaces(${PROJECT_NAME}
	${PX4_MSGS}
	${PX4_SRVS}
	DEPENDENCIES builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
