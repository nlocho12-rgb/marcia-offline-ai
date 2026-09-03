cmake_minimum_required(VERSION 3.22.1)
project(marcia_native CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(llama.cpp)

add_library(
    marcia_engine
    SHARED
    native_bridge.cpp
)

find_library(log-lib log)

target_link_libraries(
    marcia_engine
    llama
    ${log-lib}
)
