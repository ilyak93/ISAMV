cmake_minimum_required(VERSION 3.9)

project(wic-module VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WIC_DIR "${CMAKE_CURRENT_LIST_DIR}/../../lib/cmake/WIC/")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../lib/cmake/WIC/")

find_package(WIC REQUIRED)
find_package(Pleora REQUIRED)

add_executable(wic-module
	main.cpp
	)

target_include_directories(wic-module
	PUBLIC
	    ${Pleora_INCLUDE_DIR}
	)

target_link_libraries(wic-module
	PRIVATE
		WIC::WIC
		${Pleora_LIBRARIES}
	)
