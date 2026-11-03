#pragma once 

namespace mps {

  template<typename T>
  void safe_free_arr(T** ptr) {
    if(*ptr != nullptr) {
      delete[] *ptr; 
      *ptr = nullptr; 
    }
  }

  template<typename T>
  void safe_free_sig(T** ptr) {
    if(*ptr != nullptr) {
      delete *ptr; 
      *ptr = nullptr; 
    }
  }

}