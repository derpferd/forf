cmake_minimum_required(VERSION 3.13)

# set the project name
project(cforf_project)

set(CMAKE_C_STANDARD 11)

# add the executable
add_executable(cforfrunner src/cforfrunner.c src/forf.c src/rand.c src/libcforf.c)

add_library(${PROJECT_NAME} SHARED src/libcforf src/forf src/rand)
#add_subdirectory(src)