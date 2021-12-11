cmake_minimum_required(VERSION 3.9)

project(data VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WIC_DIR "${CMAKE_CURRENT_LIST_DIR}/../../lib/cmake/WIC/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../lib/cmake/WIC/")

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Make project require C++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

# Simple non robust way to find the librealsense library
if(WIN32)
	if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
		set(LIBRARY_DIR "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\lib\\x64")          # TODO: Update this variable to correct path - folder where realsense2.lib is found
		set(DLL_DIR "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\bin\\x64")              # TODO: Update this variable to correct path - folder where realsense2.dll is found
	else()
		set(LIBRARY_DIR "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\lib\\x86")          # TODO: Update this variable to correct path - folder where realsense2.lib is found
		set(DLL_DIR "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\bin\\x86")              # TODO: Update this variable to correct path - folder where realsense2.dll is found
	endif()
	set(PROJECT_BINARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/\$\(Configuration\)")                # TODO: Update this variable to correct path - folder to which your project will compile
	set(ADDITIONAL_INCLUDE_DIRS "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\include")   # TODO: Update this variable to correct path - folder where librealsense2 folder is found
endif()


find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)
if(NOT REALSENSE2_FOUND)
	SET(REALSENSE2_FOUND "realsense2")
	message(WARN "Failed to find_library(realsense2)")
endif()

find_package(WIC REQUIRED)
find_package(Pleora REQUIRED)

add_executable(data main.cpp)

target_include_directories(data PUBLIC ${Pleora_INCLUDE_DIR})

target_link_libraries(data PRIVATE WIC::WIC ${Pleora_LIBRARIES} ${REALSENSE2_FOUND})

include_directories(data ${ADDITIONAL_INCLUDE_DIRS})

set_target_properties(data PROPERTIES FOLDER Examples)