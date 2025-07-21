cmake_minimum_required(VERSION 3.2.0)
project(curi_udp)

###################################Add library###########################################
add_library(CURI_UDP_INTERFACE INTERFACE)
target_include_directories(CURI_UDP_INTERFACE INTERFACE ${PROJECT_SOURCE_DIR})
link_libraries(CURI_UDP_INTERFACE)

message("MOTOR_DATA_OBJ ${PROJECT_SOURCE_DIR}")
add_library(CURI_UDP_OBJ OBJECT
    ${PROJECT_SOURCE_DIR}/curi_udp.h
    ${PROJECT_SOURCE_DIR}/curi_udp.c
)

###################################Generate Executable###################################
#Generate Executable if this CMakeLists isn't a subdirectory 
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_executable(local
        $<TARGET_OBJECTS:CURI_UDP_OBJ> 
        ${CMAKE_SOURCE_DIR}/local.c
    )
    add_executable(remote
        $<TARGET_OBJECTS:CURI_UDP_OBJ> 
        ${CMAKE_SOURCE_DIR}/remote.c
    )
endif()