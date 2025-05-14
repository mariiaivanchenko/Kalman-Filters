cmake_minimum_required(VERSION 3.26)

set(CMAKE_CXX_STANDARD 17)
set(PROJECT_NAME EKF-draft)
project(${PROJECT_NAME} C CXX)

set(CMAKE_PREFIX_PATH "/usr/local/" ${CMAKE_PREFIX_PATH})

# Eigen
find_package(Eigen3 REQUIRED)

add_executable(${PROJECT_NAME} main.cpp EKF.cpp EKF.h)
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
