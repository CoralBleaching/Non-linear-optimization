cmake_minimum_required (VERSION 3.12)

set (cmake_cxx_standard 20)

set (project_name "non-linear-optimization")

project (${project_name})

add_executable (${project_name} main.cpp)

find_library (linalgdeb linear_algebra_debug)

target_link_libraries (${project_name} linalgdeb)