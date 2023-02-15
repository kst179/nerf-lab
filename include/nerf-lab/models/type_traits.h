#pragma once

#include <tiny-cuda-nn/common.h>

#include <mma.h>

namespace nerf {

typedef nvcuda::wmma::experimental::precision::b1 uint1b_t;
typedef nvcuda::wmma::experimental::precision::s4 int4b_t;
typedef nvcuda::wmma::experimental::precision::u4 uint4b_t;

template <typename T> struct helper_traits;
template<> struct helper_traits< uint1b_t > { static const uint32_t block_height = 8;
                                              static const uint32_t block_width  = 128;
                                              static const uint32_t elements_per_storage_unit = 32;
                                              typedef int32_t accumulator_type;
                                              typedef uint32_t storage_type; };

template<> struct helper_traits<  int4b_t > { static const uint32_t block_height = 8;
                                              static const uint32_t block_width  = 32;
                                              static const uint32_t elements_per_storage_unit = 8;
                                              typedef int32_t accumulator_type; 
                                              typedef int32_t storage_type; };

template<> struct helper_traits< uint4b_t > { static const uint32_t block_height = 8;
                                              static const uint32_t block_width  = 32;
                                              static const uint32_t elements_per_storage_unit = 8;
                                              typedef int32_t accumulator_type; 
                                              typedef uint32_t storage_type; };

template<> struct helper_traits<   int8_t > { static const uint32_t block_height = 16;
                                              static const uint32_t block_width  = 16;
                                              static const uint32_t elements_per_storage_unit = 1;
                                              typedef int32_t accumulator_type; 
                                              typedef  int8_t storage_type; };

template<> struct helper_traits<  uint8_t > { static const uint32_t block_height = 16;
                                              static const uint32_t block_width  = 16;
                                              static const uint32_t elements_per_storage_unit = 1;
                                              typedef int32_t accumulator_type; 
                                              typedef uint8_t storage_type; };
                                             
template<> struct helper_traits<   __half > { static const uint32_t block_height = 16;
                                              static const uint32_t block_width  = 16;
                                              static const uint32_t elements_per_storage_unit = 1;
                                              typedef  __half accumulator_type;
                                              typedef  __half storage_type; };

template <typename T> using     storage_t = typename helper_traits<T>::storage_type;
template <typename T> using accumulator_t = typename helper_traits<T>::accumulator_type;

}