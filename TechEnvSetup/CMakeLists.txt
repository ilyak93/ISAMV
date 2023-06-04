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
		set(LIBRARY_DIR "D:\\Program Files (x86)\\Intel RealSense SDK 2.0\\lib\\x64")          # TODO: Update this variable to correct path - folder where realsense2.lib is found
		set(DLL_DIR "D:\\Program Files (x86)\\Intel RealSense SDK 2.0\\bin\\x64")              # TODO: Update this variable to correct path - folder where realsense2.dll is found
	else()
		set(LIBRARY_DIR "D:\\Program Files (x86)\\Intel RealSense SDK 2.0\\lib\\x86")          # TODO: Update this variable to correct path - folder where realsense2.lib is found
		set(DLL_DIR "D:\\Program Files (x86)\\Intel RealSense SDK 2.0\\bin\\x86")              # TODO: Update this variable to correct path - folder where realsense2.dll is found
	endif()
	set(PROJECT_BINARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/\$\(Configuration\)")                # TODO: Update this variable to correct path - folder to which your project will compile
	set(ADDITIONAL_INCLUDE_DIRS "D:\\Program Files (x86)\\Intel RealSense SDK 2.0\\include")   # TODO: Update this variable to correct path - folder where librealsense2 folder is found
endif()


find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)
if(NOT REALSENSE2_FOUND)
	SET(REALSENSE2_FOUND "realsense2")
	message(WARN "Failed to find_library(realsense2)")
endif()

#-------------------------------------------------------------------------------------


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was Config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
	set(${_var} "${_file}")
	if(NOT EXISTS "${_file}")
		message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
	endif()
endmacro()

macro(check_required_components _NAME)
	foreach(comp ${${_NAME}_FIND_COMPONENTS})
		if(NOT ${_NAME}_${comp}_FOUND)
			if(${_NAME}_FIND_REQUIRED_${comp})
				set(${_NAME}_FOUND FALSE)
			endif()
		endif()
	endforeach()
endmacro()

####################################################################################

include("${CMAKE_CURRENT_LIST_DIR}/../../lib/cmake/WIC/WICTargets.cmake")

include(CMakeFindDependencyMacro)

find_dependency(Pleora)
find_dependency(TBB 2020.3)
find_dependency(Boost 1.70.0 COMPONENTS container)

check_required_components(WIC)


#-------------------------------------------------------------------------------------
find_package(Boost REQUIRED iostreams)
find_package(Pleora REQUIRED)
find_package(OpenCV REQUIRED)
#set(Pleora_INCLUDE_DIR E:/Program Files/Pleora Technologies Inc/eBUS SDK/Libraries)
#add_subdirectory("F:/WIC_SDK/examples/wic-module/liblmdb")

add_executable(data my_align.cpp) #main2.cpp stb.cpp multicam_rs.cpp main3.cpp open_check.cpp

#set(OPENCV_INCLUDE_DIR "C:\\OCV\\build\\install\\include\\")

#set(OPENCV_DLL_DIR "F:/OCV/opencv/build/x64/vc15/bin")

#set(LMDB_SAFE_DIR "F:/WIC_SDK/lmdb/lmdb-safe")
#set(LMDB_DIR "F:/WIC_SDK/lmdb/liblmdb")

target_include_directories(data PUBLIC ${Pleora_INCLUDE_DIR} ${OPENCV_INCLUDE_DIR} ${LMDB_SAFE_DIR})

set(LMDB_LIB_DIR "E:/WIC_SDK/lmdb/liblmdb")

target_link_libraries(data PRIVATE WIC::WIC ${Pleora_LIBRARIES} ${REALSENSE2_FOUND} ${OpenCV_LIBS})

include_directories(data ${ADDITIONAL_INCLUDE_DIRS})

set_target_properties(data PROPERTIES FOLDER Examples)



