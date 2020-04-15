
/*
* File:   cuda_utils.h
* Author: Emilio Vicari, Michele Amoretti
*/

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <iostream>
#include "common.h"

using namespace std;

namespace dci
{
  namespace CUDAUtils
  {

    /*
    * Prints CUDA device info to cout
    */
    void printDeviceInfo()
    {
      const int kb = 1024;
      const int mb = kb * kb;

      cout << "NBody.GPU" << endl << "=========" << endl << endl;

      cout << "CUDA version:   v" << CUDART_VERSION << endl;
      //cout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl;

      int devCount;
      cudaGetDeviceCount(&devCount);
      wcout << "CUDA Devices: " << endl << endl;

      for(int i = 0; i < devCount; ++i)
      {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        cout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
        cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
        cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
        cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
        cout << "  Block registers: " << props.regsPerBlock << endl << endl;
        cout << "  Num blocks = " << props.multiProcessorCount << endl;
        cout << "  Warp size:         " << props.warpSize << endl;
        cout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
        cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << endl;
        cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << endl;
        cout << endl;
      }
    }

  }
}

#endif /* CUDA_UTILS_H */
