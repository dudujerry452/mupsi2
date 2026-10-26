#pragma once

#include "core/basic/types.h"


#include "core/scene/scene.h"
#include "core/basic/types.h"
#include "core/utils/cuda_util.h"
#include <driver_types.h>

namespace mps::kernels {

  template<typename T>
  MPH T* upload(T* src, u32 count) {
    T* t = nullptr; 
    u32 byte_size = count * sizeof(T); 
    cudaCheck(cudaMalloc(&t, byte_size));
    cudaCheck(cudaMemcpy(t, src, byte_size, cudaMemcpyHostToDevice)); 
    return t; 
  }
  template<typename T> 
  MPH void destory(T* item) {
    cudaCheck(cudaFree(item)); 
  }
  template<typename T> 
  MPH T* download(T* src, u32 count) {
    T* ret = new T[count]; 
    u32 byte_size = count * sizeof(T); 
    cudaCheck(cudaMemcpy(ret, src, byte_size, cudaMemcpyDeviceToHost)); 
    return ret; 
  }

  MPH core::SceneDevice uploadScene(const core::SceneDevice& scene); 
  
  template<typename T>
  MPH FrameBuffer<T> uploadFrameBuffer(const FrameBuffer<T>& fb) {
    FrameBuffer<T> ret = fb; 
    auto dev = upload(fb.pixels, fb.pixelnum()); 
    ret.pixels = dev; 
    return ret; 
  }

  MPH void destoryScene(core::SceneDevice& scene); 
  
  template<typename T>
  MPH void destoryFrameBuffer(FrameBuffer<T>& fb) {
    destory(fb.pixels); 
    fb.pixels = nullptr;  
  }

  template<typename T> 
  MPH FrameBuffer<T> downloadFrameBuffer(const FrameBuffer<T>& fb) {
    FrameBuffer<T> ret = fb; 
    ret.pixels = download(fb.pixels, fb.pixelnum()); 
    return ret; 
  }

}