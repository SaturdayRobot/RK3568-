# CMake 最低版本 3.16，确保 target_link_libraries 等现代特性可用
cmake_minimum_required(VERSION 3.16)

set(APP_NAME rk3568_inspection_terminal)
project(${APP_NAME} VERSION 1.0.0 LANGUAGES CXX)

# 平台检查：本项目仅构建于 Linux（目标 RK3568/aarch64），macOS 等不在支持范围
if(NOT UNIX OR APPLE)
    message(FATAL_ERROR "This project is Linux-only (targeting RK3568/aarch64).")
endif()

# C++17：结构化绑定、if constexpr 等特性简化代码，且 RK3568 交叉编译工具链均支持
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # 禁用 GNU 扩展，提高可移植性

# 默认 Release 构建以获得最高性能；Debug 模式通过 -DCMAKE_BUILD_TYPE=Debug 切换
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 可选功能开关：SQLite 本地存储 与 MQTT 云端上传，可按需裁剪
option(ENABLE_SQLITE "Enable SQLite local storage" ON)
option(ENABLE_MQTT "Enable Mosquitto MQTT uploader" ON)

# pthread 是基础依赖，找不到则终止
find_package(Threads REQUIRED)

# --- 第三方库路径定义 ---
# 所有预编译库统一放在 lib/ 下，方便管理交叉编译依赖
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(THIRD_PARTY_ROOT ${PROJECT_ROOT}/lib)
set(DEBIAN11_EXTRA_PATH ${THIRD_PARTY_ROOT}/debian11-extra)

# FFmpeg：RTSP 拉流与 H.264 软编解码（硬件编解码由 MPP 负责）
set(FFMPEG_PATH ${THIRD_PARTY_ROOT}/ffmpeg-debian11 CACHE PATH "Target FFmpeg installation")
# Rockchip 三件套：MPP 硬编解码 / RGA 2D 加速 / RKNN NPU 推理
set(MPP_PATH ${THIRD_PARTY_ROOT}/mpp)
set(RGA_PATH ${THIRD_PARTY_ROOT}/rga)
set(RKNN_PATH ${THIRD_PARTY_ROOT}/rknn)
# OpenCV：仅用 core + imgproc 两个模块（cv::Mat 基础 + 图像处理）
set(OPENCV_PATH ${THIRD_PARTY_ROOT}/opencv-debian11 CACHE PATH "Target OpenCV installation")

set(FFMPEG_LIB_DIR ${FFMPEG_PATH}/lib)
set(MPP_LIB_DIR ${MPP_PATH}/lib)
set(RGA_LIB_DIR ${RGA_PATH}/lib)
set(RKNN_LIB_DIR ${RKNN_PATH}/lib)
set(OPENCV_LIB_DIR ${OPENCV_PATH}/lib/aarch64-linux-gnu)

set(LOCAL_LIBRARY_DIRS
    ${FFMPEG_LIB_DIR}
    ${MPP_LIB_DIR}
    ${RGA_LIB_DIR}
    ${RKNN_LIB_DIR}
    ${OPENCV_LIB_DIR}
)

set(LOCAL_INCLUDE_DIRS
    ${PROJECT_ROOT}/include
    ${FFMPEG_PATH}/include
    ${MPP_PATH}/include
    ${RGA_PATH}/include
    ${RKNN_PATH}/include
    ${OPENCV_PATH}/include
)

# 仅将实际存在的头文件目录加入最终列表，避免无效路径干扰编译
foreach(inc_dir IN LISTS LOCAL_INCLUDE_DIRS)
    if(EXISTS ${inc_dir})
        list(APPEND APP_INCLUDE_DIRS ${inc_dir})
    endif()
endforeach()

# 对缺失的库目录仅警告而非终止——部分依赖是条件可选的
foreach(lib_dir IN LISTS LOCAL_LIBRARY_DIRS)
    if(NOT EXISTS ${lib_dir})
        message(WARNING "Local library directory missing: ${lib_dir}")
    endif()
endforeach()

# --- OpenCV 精简链接策略 ---
# 仅需 cv::Mat 和 imgproc（resize/cvtColor 等），不引入 contrib/features2d 等大模块
# 避免 .so 体积膨胀和符号冲突
set(OpenCV_VERSION "4.5.1")
set(OpenCV_INCLUDE_DIRS "${OPENCV_PATH}/include/opencv4")
set(OpenCV_LIBS
    "${OPENCV_LIB_DIR}/libopencv_core.so"
    "${OPENCV_LIB_DIR}/libopencv_imgproc.so"
)
foreach(opencv_lib IN LISTS OpenCV_LIBS)
    if(NOT EXISTS "${opencv_lib}")
        message(FATAL_ERROR "Required OpenCV library is missing: ${opencv_lib}")
    endif()
endforeach()

# Rockchip 核心运行时库（硬性要求，缺失则功能不可用）
set(RKNN_RUNTIME_LIB ${RKNN_LIB_DIR}/librknnrt.so)       # NPU 推理运行时
set(MPP_RUNTIME_LIB ${MPP_LIB_DIR}/librockchip_mpp.so)   # 硬件编解码
set(RGA_RUNTIME_LIB ${RGA_LIB_DIR}/librga.so)            # 2D 图像加速

# FFmpeg 必选库：编解码 + 封装/解封装 + 工具 + 缩放 + 音频重采样
set(FFMPEG_REQUIRED_LIBS
    ${FFMPEG_LIB_DIR}/libavcodec.so
    ${FFMPEG_LIB_DIR}/libavformat.so
    ${FFMPEG_LIB_DIR}/libavutil.so
    ${FFMPEG_LIB_DIR}/libswscale.so
    ${FFMPEG_LIB_DIR}/libswresample.so
)
foreach(ff_lib IN LISTS FFMPEG_REQUIRED_LIBS)
    if(NOT EXISTS ${ff_lib})
        message(FATAL_ERROR "Required FFmpeg library is missing: ${ff_lib}")
    endif()
endforeach()

# --- 源文件（按模块分组）---
add_executable(${APP_NAME}
    ${PROJECT_ROOT}/src/main.cpp
    ${PROJECT_ROOT}/src/app/app_initializer.cpp
    ${PROJECT_ROOT}/src/app/inference_result_dispatcher.cpp
    ${PROJECT_ROOT}/src/config/ini_config.cpp
    ${PROJECT_ROOT}/src/data_collection/stream_loader.cpp
    ${PROJECT_ROOT}/src/data_processing/mpp_decoder.cpp
    ${PROJECT_ROOT}/src/data_processing/mpp_encoder.cpp
    ${PROJECT_ROOT}/src/data_processing/postprocess.cpp
    ${PROJECT_ROOT}/src/monitoring/board_status_collector.cpp
    ${PROJECT_ROOT}/src/network/data_upload/mqtt_uploader.cpp
    ${PROJECT_ROOT}/src/pipeline/frame_hub.cpp
    ${PROJECT_ROOT}/src/pipeline/inference_service.cpp
    ${PROJECT_ROOT}/src/pipeline/mosaic_stream_pipeline.cpp
    ${PROJECT_ROOT}/src/pipeline/encoded_media_service.cpp
    ${PROJECT_ROOT}/src/pipeline/rga_preprocessor.cpp
    ${PROJECT_ROOT}/src/pipeline/v4l2_camera_pipeline.cpp
    ${PROJECT_ROOT}/src/pipeline/visible_rtsp_pipeline.cpp
    ${PROJECT_ROOT}/src/storage/sqlite_store.cpp
    ${PROJECT_ROOT}/src/storage/event_recorder.cpp
    ${PROJECT_ROOT}/src/utils/thread_runtime.cpp
)

target_include_directories(${APP_NAME} PRIVATE
    ${APP_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# 编译警告：Wall + Wextra 全面检测，但关闭 unused-parameter（回调函数签名常见）
target_compile_options(${APP_NAME} PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# Debug 构建时定义 DEBUG 宏，用于条件编译调试日志和断言
target_compile_definitions(${APP_NAME} PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
)

# --- 链接库汇总 ---
set(APP_LINK_LIBS
    Threads::Threads
    ${FFMPEG_REQUIRED_LIBS}
    ${OpenCV_LIBS}
    dl   # dlopen/dlsym 动态加载，用于运行时加载模型
)

# Rockchip 三件套条件链接：存在则加入，缺失仅警告
if(EXISTS ${MPP_RUNTIME_LIB})
    list(APPEND APP_LINK_LIBS ${MPP_RUNTIME_LIB})
else()
    message(WARNING "MPP library not found: ${MPP_RUNTIME_LIB}")
endif()
if(EXISTS ${RKNN_RUNTIME_LIB})
    list(APPEND APP_LINK_LIBS ${RKNN_RUNTIME_LIB})
else()
    message(WARNING "RKNN runtime library not found: ${RKNN_RUNTIME_LIB}")
endif()
if(EXISTS ${RGA_RUNTIME_LIB})
    list(APPEND APP_LINK_LIBS ${RGA_RUNTIME_LIB})
else()
    message(WARNING "RGA library not found: ${RGA_RUNTIME_LIB}")
endif()

# SQLite 条件链接：header 来自 debian11-extra，.so 从 SDK rootfs 获取
if(ENABLE_SQLITE)
    set(SQLITE3_HEADER "${DEBIAN11_EXTRA_PATH}/usr/include/sqlite3.h")
    set(SQLITE3_RUNTIME "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/libsqlite3.so.0")
    get_filename_component(SQLITE3_RUNTIME_REAL "${SQLITE3_RUNTIME}" REALPATH)
    if(EXISTS "${SQLITE3_HEADER}" AND EXISTS "${SQLITE3_RUNTIME}")
        target_compile_definitions(${APP_NAME} PRIVATE HAVE_SQLITE3)
        list(APPEND APP_LINK_LIBS ${SQLITE3_RUNTIME})
        target_include_directories(${APP_NAME} PRIVATE "${DEBIAN11_EXTRA_PATH}/usr/include")
    else()
        message(WARNING "SQLite3 not found, local storage disabled")
    endif()
endif()

# Mosquitto MQTT 条件链接
if(ENABLE_MQTT)
    set(MOSQUITTO_LIB "${DEBIAN11_EXTRA_PATH}/usr/lib/aarch64-linux-gnu/libmosquitto.so.1")
    get_filename_component(MOSQUITTO_RUNTIME_REAL "${MOSQUITTO_LIB}" REALPATH)
    if(EXISTS "${MOSQUITTO_LIB}" AND EXISTS "${DEBIAN11_EXTRA_PATH}/usr/include/mosquitto.h")
        target_compile_definitions(${APP_NAME} PRIVATE HAVE_MOSQUITTO)
        list(APPEND APP_LINK_LIBS ${MOSQUITTO_LIB})
        target_include_directories(${APP_NAME} PRIVATE "${DEBIAN11_EXTRA_PATH}/usr/include")
    else()
        message(WARNING "Mosquitto not found, MQTT upload disabled")
    endif()
endif()

# systemd sd_notify 集成：service Type=notify 需要，找不到则降级为 Type=simple
find_library(SYSTEMD_LIB systemd)
if(SYSTEMD_LIB)
    list(APPEND APP_LINK_LIBS ${SYSTEMD_LIB})
    message(STATUS "systemd found: ${SYSTEMD_LIB}")
else()
    message(WARNING "libsystemd not found, sd_notify disabled (service Type=simple fallback)")
endif()

target_link_libraries(${APP_NAME} PRIVATE ${APP_LINK_LIBS})

# RPATH：构建阶段指向本地 lib 目录；安装后使用 $ORIGIN 相对路径，实现可重定位部署
set_target_properties(${APP_NAME} PROPERTIES
    BUILD_RPATH "${LOCAL_LIBRARY_DIRS}"
    INSTALL_RPATH "$ORIGIN/../lib"
)

# --- 安装规则 ---
include(GNUInstallDirs)

install(TARGETS ${APP_NAME} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY ${PROJECT_ROOT}/config/ DESTINATION config)
install(DIRECTORY ${PROJECT_ROOT}/model/ DESTINATION model
        FILES_MATCHING PATTERN "*.rknn" PATTERN "*.txt")
install(FILES
    ${PROJECT_ROOT}/systemd/rk_video_ai.service
    ${PROJECT_ROOT}/systemd/rkaiq_3A.service
    DESTINATION systemd)
install(FILES ${PROJECT_ROOT}/config/mosquitto.conf DESTINATION config)
install(PROGRAMS
    ${PROJECT_ROOT}/scripts/install_board_services.sh
    ${PROJECT_ROOT}/scripts/clean_board_install.sh
    ${PROJECT_ROOT}/scripts/start_board_services.sh
    ${PROJECT_ROOT}/scripts/board_hardware_status.sh
    DESTINATION scripts)
install(DIRECTORY ${PROJECT_ROOT}/docs/ DESTINATION docs)
install(FILES ${PROJECT_ROOT}/README.md DESTINATION .)
install(FILES ${PROJECT_ROOT}/windows/start_camera_publisher.ps1 DESTINATION windows)

# 安装运行时 .so 文件（含符号链接），实现自包含部署
file(GLOB FFMPEG_RUNTIME_FILES "${FFMPEG_LIB_DIR}/*.so*")
file(GLOB OPENCV_RUNTIME_FILES
    "${OPENCV_LIB_DIR}/libopencv_core.so*"
    "${OPENCV_LIB_DIR}/libopencv_imgproc.so*"
)
install(FILES
    ${FFMPEG_RUNTIME_FILES}
    ${OPENCV_RUNTIME_FILES}
    ${RKNN_RUNTIME_LIB}
    ${RGA_RUNTIME_LIB}
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
# MPP SO 名是 librockchip_mpp.so.1，但交叉编译时开发包名为 librockchip_mpp.so，
# 安装时重命名为 ld.so 期望的 SONAME
install(FILES ${MPP_RUNTIME_LIB}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RENAME librockchip_mpp.so.1)

# CTest 集成
include(CTest)
if(BUILD_TESTING AND EXISTS "${PROJECT_ROOT}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()

# 构建摘要
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Target: ${APP_NAME}")
