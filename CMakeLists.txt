cmake_minimum_required(VERSION 2.6)

file(GLOB_RECURSE src_list "src/*.cc" "vendor/fastText/src/*.c*")
include_directories(
    "include"
    "vendor/fastText"
    "vendor/json/include"
)

include(fib-addon/common.cmake)
