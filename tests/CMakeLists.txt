include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

macro(add_test test_name)
    add_executable(${test_name} ${ARGN})
    target_link_libraries(${test_name} GTest::gtest_main)
    # target_compile_definitions(${test_name} PUBLIC -DTCNN_MIN_GPU_ARCH=52)
    target_link_libraries(${test_name} tiny-cuda-nn fmt)
    gtest_discover_tests(${test_name})
endmacro(add_test)

enable_testing()
include(GoogleTest)

add_test(cuda_reduce_test cuda_reduce_test.cu)
add_test(volume_renderer_test volume_renderer_test.cu)
add_test(qat_threadblock_layer_test qat_threadblock_layer_test.cu)