cmake_minimum_required(VERSION 3.0)

get_filename_component(name ${CMAKE_CURRENT_SOURCE_DIR} NAME)

include(fib-addon/build_tools/cmake-scripts/get_env.cmake)

set(WORK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
build("${CMAKE_CURRENT_SOURCE_DIR}" "${WORK_ROOT}" "${name}")
