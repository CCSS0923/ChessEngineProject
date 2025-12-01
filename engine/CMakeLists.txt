add_library(chess_engine STATIC
    src/board.cpp
    src/Engine.cpp
    src/move.cpp
    src/movegen.cpp
    src/search.cpp
    src/eval.cpp
    src/NNWrapper.cpp
    src/NNBoardTensor.cpp
)

target_include_directories(chess_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/engine
)

target_link_libraries(chess_engine PUBLIC
    ${TORCH_LIBRARIES}
)
