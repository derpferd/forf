cmake_minimum_required(VERSION 3.13)

# set the project name
project(cforf_project)

set(CMAKE_C_STANDARD 11)

# add the executable
add_executable(cforf_project cforfrunner.c forf.c)