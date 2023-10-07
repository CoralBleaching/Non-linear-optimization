CMAKE_MINIMUM_REQUIRED (VERSION 3.12)

SET (CMAKE_CXX_STANDARD 17)

SET (CMAKE_DEBUG_POSTFIX d)

SET (PROJECT_NAME "non-linear-optimization")

PROJECT (${PROJECT_NAME})


# add_library (linalg linear_algebra.cpp)
# set_target_properties (linalg 
#     PROPERTIES 
#     debug_postfix ${cmake_debug_postfix}
#     LINKER_LANGUAGE CXX    
# )

ADD_EXECUTABLE (${PROJECT_NAME} main.cpp)
SET_TARGET_PROPERTIES (${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

# find_library (linalg linear_algebra)

# target_link_libraries (${project_name} linalg)