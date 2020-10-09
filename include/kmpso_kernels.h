
/*
* File:   kmpso_kernels.h
* Authors: Emilio Vicari, Michele Amoretti
*/

#ifndef KMPSO_KERNELS_H
#define KMPSO_KERNELS_H

#include <curand_kernel.h>

__global__ void setup_kernel(curandState * state, unsigned long seed)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init ( seed+id, id, 0, &state[id] );
  // SC    curand_init ( seed, id, 0, &state[id] );
}

__device__ float generate(curandState* globalState, int ind)
{
  curandState localState = globalState[ind];
  float random = curand_uniform( &localState );
  globalState[ind] = localState;
  return random;
}

__global__ void compute(double* V,double* X,double* P,int* seed,double* bp,double* sigma,int xmin,int xmax,int S,int D, double c,curandState_t *state)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < S*D)
  {
    if(seed[i/D]!=-1) // if belongs to a cluster
    {
      V[i] = V[i] +  (generate(state,i)*c) * ( P[i] - X[i] );
      V[i] = V[i] + (generate(state,i)*c) * ( bp[seed[i/D]*D+(i%D)] - X[i] );
      V[i] = 0.73*V[i];
      if(V[i]>2*(sqrt(sigma[seed[i/D]]))) V[i]=2*(sqrt(sigma[seed[i/D]]));
      if(V[i]<-2*(sqrt(sigma[seed[i/D]]))) V[i]=-2*(sqrt(sigma[seed[i/D]]));
      X[i] = X[i] + V[i];
    }
    else
    {
      V[i] = V[i] + (generate(state,i)*c) * ( P[i] - X[i] ); // cognition only
      V[i] = 0.73*V[i];
      X[i] = X[i] + V[i];
    }
    if ( X[i] < xmin )
    {
      X[i] = xmin; V[i] = 0;
    }
    if ( X[i] > xmax )
    {
      X[i] = xmax; V[i] = 0;
    }
  }
  else return;
}

#endif /* KMPSO_KERNELS_H */
