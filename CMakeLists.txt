# Name: Akash Shridhar Shetty
# Date: February 2025
# File: CMakeLists.txt
#
# Purpose:
# Build system for Content-Based Image Retrieval project
# Compiles two programs: extract_features and query

cmake_minimum_required(VERSION 3.10)

# Project name and version
project(ContentBasedImageRetrieval VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add compiler warnings for better code quality
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Source files for utility functions
set(UTILS_SOURCES
    src/utils.cpp
    src/features.cpp
    src/distance.cpp
)

# ========================================
# Program 1: extract_features
# ========================================
add_executable(extract_features
    src/main_extract_features.cpp
    ${UTILS_SOURCES}
)

target_link_libraries(extract_features
    ${OpenCV_LIBS}
    stdc++fs  # For filesystem support on some systems
)

# ========================================
# Program 2: query
# ========================================
add_executable(query
    src/main_query.cpp
    ${UTILS_SOURCES}
)

target_link_libraries(query
    ${OpenCV_LIBS}
    stdc++fs  # For filesystem support on some systems
)

# ========================================
# Installation (optional)
# ========================================
install(TARGETS extract_features query
    RUNTIME DESTINATION bin
)

# ========================================
# Print configuration info
# ========================================
message(STATUS "========================================")
message(STATUS "Content-Based Image Retrieval")
message(STATUS "========================================")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "========================================")