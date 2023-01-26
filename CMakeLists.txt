cmake_minimum_required(VERSION 3.19)

# set(CUDA_VERSION "cuda-12.0")
# set(CMAKE_CUDA_COMPILER "/usr/local/${CUDA_VERSION}/bin/nvcc")
# set(CUDAToolkit_ROOT "/usr/local/${CUDA_VERSION}/bin")

list(APPEND CMAKE_CUDA_FLAGS "-arch=sm_86")
list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

set(CMAKE_CXX_STANDARD 17)

project(nerf-lab LANGUAGES C CXX CUDA)

# option(USE_PREBUILT_TCNN "" ON)
# message(STATUS ${tcnn_lib})
# if(${USE_PREBUILT_TCNN})
#     set(tcnn_build_dir ${CMAKE_SOURCE_DIR}/dependencies/tiny-cuda-nn/build)
#     set(tcnn_lib ${tcnn_build_dir}/libtiny-cuda-nn.a)

#     if(NOT EXISTS ${tcnn_lib})
#         execute_process(COMMAND mkdir -p ${tcnn_build_dir})
#         execute_process(COMMAND cmake ..
#                                 -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER} 
#                                 -DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}
#                         WORKING_DIRECTORY ${tcnn_build_dir})
#         execute_process(COMMAND make -j8 WORKING_DIRECTORY ${tcnn_build_dir})
#     endif()
# else()

set(TCNN_CUDA_ARCHITECTURES 86)
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)

add_subdirectory(dependencies/tiny-cuda-nn)
# set(tcnn_lib tiny-cuda-nn)

# endif()

# add_compile_definitions(-DTCNN_MIN_GPU_ARCH=86)

include_directories(
    "include"
    "dependencies"
    "dependencies/imgui"
    "dependencies/tiny-cuda-nn/include"
    "dependencies/tiny-cuda-nn/dependencies"
    "dependencies/tiny-cuda-nn/dependencies/cutlass/include"
)

find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(OpenGL REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

set(gui_src src/gui/app.cpp
            src/gui/main_camera.cpp
            src/gui/arcball.cpp
            src/gui/shader.cpp
            src/gui/texture.cpp)

set(backend_src src/data/colmap_dataset.cpp
                src/data/image_handle.cpp
                src/pipeline/data_manager.cu
                src/cuda/cuda_texture.cu
                src/models/nerf.cu)

set(dependencies_src dependencies/loguru/loguru.cpp
                     dependencies/imgui/imgui.cpp
                     dependencies/imgui/backends/imgui_impl_glfw.cpp
                     dependencies/imgui/backends/imgui_impl_opengl3.cpp
                     dependencies/imgui/imgui_draw.cpp
                     dependencies/imgui/imgui_tables.cpp
                     dependencies/imgui/imgui_widgets.cpp
                     dependencies/ImGuizmo/ImGuizmo.cpp)

set(gui_libs GLEW
             glfw
             OpenGL::GL)

set(common_libs Boost::filesystem
                cuda
                cudart
                dl
                Eigen3::Eigen
                fmt
                ${OpenCV_LIBS}
                pthread
                tiny-cuda-nn)

set(app_src src/nerf-lab.cpp ${backend_src} ${gui_src})
set(cli_src src/cli.cu ${backend_src})

add_executable(nerf-lab ${app_src} ${dependencies_src})
target_link_libraries(nerf-lab ${gui_libs} ${common_libs})
target_compile_options(nerf-lab PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_compile_definitions(nerf-lab PUBLIC -DTCNN_MIN_GPU_ARCH=86)

add_executable(nerf-lab-cli ${cli_src} dependencies/loguru/loguru.cpp)
target_link_libraries(nerf-lab-cli ${common_libs})
target_compile_options(nerf-lab-cli PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_compile_definitions(nerf-lab-cli PUBLIC -DTCNN_MIN_GPU_ARCH=86)

add_subdirectory(tests)