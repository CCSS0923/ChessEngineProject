cmake_minimum_required(VERSION 3.18)
project(ChessEngineProject CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Torch
set(Torch_DIR "${CMAKE_SOURCE_DIR}/extern/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# 엔진 라이브러리
add_subdirectory(engine)

# UCI 실행 파일
add_executable(chess_uci
    engine/app/main_uci.cpp
)

add_custom_command(TARGET chess_uci POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/extern/libtorch/lib
        $<TARGET_FILE_DIR:chess_uci>
)

target_include_directories(chess_uci PUBLIC
    ${CMAKE_SOURCE_DIR}/engine/include
)

target_link_libraries(chess_uci PUBLIC
    chess_engine
    ${TORCH_LIBRARIES}
)
