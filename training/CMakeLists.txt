# ============================
# Training Library
# ============================
add_library(training_lib
    src/model.cpp
    src/dataset.cpp
)

# Include paths
target_include_directories(training_lib PUBLIC
    include                                 # training/include
    ${CMAKE_SOURCE_DIR}/extern/json/include # nlohmann/json.hpp
)

# Torch link
target_link_libraries(training_lib PUBLIC ${TORCH_LIBRARIES})


# ============================
# Training Executable (train_main)
# ============================
add_executable(chess_train src/train_main.cpp)
target_link_libraries(chess_train PRIVATE training_lib)


# ============================
# Torch DLL 자동 복사 (훈련 실행파일용)
# 최상위 CMakeLists.txt 내 TORCH_INSTALL_PREFIX 사용
# ============================
add_custom_command(TARGET chess_train POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${TORCH_INSTALL_PREFIX}/bin"
            $<TARGET_FILE_DIR:chess_train>

    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${TORCH_INSTALL_PREFIX}/lib"
            $<TARGET_FILE_DIR:chess_train>
)
