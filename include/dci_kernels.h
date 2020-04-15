
/*
* File:   dci_kernels.h
* Author: e.vicari
*
*/

#ifndef DCI_KERNELS_H
#define DCI_KERNELS_H

#include <math.h>
#include "common.h"
#include "register_utils.h"


typedef struct __SystemParameters
{
  unsigned int N; unsigned int NA; unsigned int M;
  unsigned int L; unsigned int CB; unsigned int C;
} SystemParameters;

__constant__ SystemParameters SP;

/*
* Computes entropy, cluster index and/or statistical index for a cluster batch (one thread = one cluster)
*
* S  -> cluster size in number of registers
* N  -> sample size in bits
* NA -> number of agents
* M  -> number of samples
* L  -> number of maximum possible sample values
* C  -> number of clusters in current batch
* R  -> cluster size in number of variables
* system_entropies contains joint entropy at index N
*/
template<unsigned int S, bool HistogramOnly> __global__ void histo_kernel(
  register_t* system,
  register_t* clusters,
  register_t* histogram,
  unsigned int* frequencies,
  register_t** next)
  {
    //unsigned int md = blockIdx.x & (unsigned int)1;
    //unsigned int i = (blockIdx.x - md)*blockDim.x + threadIdx.x * 2 + md;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= SP.C) return;

    unsigned int free_block = 0, pos, prev;
    register unsigned int temp_freq = 0;
    bool eq;
    register_t cur_sample[S];
    register_t prev_sample[S];

    unsigned int block_root = blockIdx.x * blockDim.x * SP.L;

    register_t* histo_root = histogram + block_root * S;
    unsigned int* freq_root = frequencies + block_root;
    register_t** next_root = next + block_root;

    register_t cur_cluster[S];
    dci::RegisterUtils::assign<S>(cur_cluster, clusters + i * S);
    unsigned int tmp_pos;

    // cycle all samples
    for (unsigned int j = 0; j != SP.M; ++j)
    {

      // get current sample
      dci::RegisterUtils::assignFromMask<S>(cur_sample, system + j * S, cur_cluster);

      // check if this is a repeated sample
      if (j == 0 || !dci::RegisterUtils::equal<S>(cur_sample, prev_sample))
      {

        // get position from hash
        pos = HistogramOnly ? dci::RegisterUtils::murmur3_32<S>((char*)cur_sample, SP.L) : dci::RegisterUtils::basic_hash<S>(cur_sample, SP.L);

        // get position in low-level array
        tmp_pos = pos * blockDim.x + threadIdx.x;

        // check if pos is free
        if ((temp_freq = freq_root[tmp_pos]) == 0)
        {
          dci::RegisterUtils::assign<S>(histo_root + tmp_pos * S, cur_sample);
          if (free_block == pos) while(freq_root[(++free_block) * blockDim.x + threadIdx.x]); // move free block index to next free block
        }
        else
        {

          // find sample in chain or get to end of chain
          while (!(eq = dci::RegisterUtils::equal<S>(cur_sample, histo_root + tmp_pos * S)) && next_root[tmp_pos] != 0)
          {
            pos = (unsigned int)(next_root[tmp_pos] - histo_root) / (blockDim.x * S);
            tmp_pos = pos * blockDim.x + threadIdx.x;
          }

          // sample was not found in chain -> create new at first free position
          if (!eq)
          {
            prev = pos;
            pos = free_block;
            unsigned int tmp_prev = prev * blockDim.x + threadIdx.x;
            tmp_pos = pos * blockDim.x + threadIdx.x;
            next_root[tmp_prev] = histo_root + tmp_pos * S;
            dci::RegisterUtils::assign<S>(next_root[tmp_prev], cur_sample);
            while(freq_root[(++free_block) * blockDim.x + threadIdx.x]); // move free block index to next free block
          }

          // read temp freq
          temp_freq = freq_root[tmp_pos];

        }

        // store prev sample
        dci::RegisterUtils::assign<S>(prev_sample, cur_sample);

      }

      // update frequency
      freq_root[tmp_pos] = ++temp_freq;

    }

  }

/*
* Computes entropy, cluster index and/or statistical index for a cluster batch (one thread = one cluster)
*
* S  -> cluster size in number of registers
* N  -> sample size in bits
* NA -> number of agents
* M  -> number of samples
* L  -> number of maximum possible sample values
* C  -> number of clusters in current batch
* R  -> cluster size in number of variables
* system_entropies contains joint entropy at index N
*/
template<unsigned int S> __global__ void entropy_kernel(unsigned int* frequencies,
  float* entropies, unsigned int* count)
  {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= SP.C) return;

    unsigned int block_root = blockIdx.x * blockDim.x * SP.L;
    unsigned int* freq_root = frequencies + block_root;

    // store temp entropy in double-precision var
    double temp = 0.0f, den = (double)SP.M, freq;
    unsigned int tmp_f = threadIdx.x;
    if (count) *count = 0;

    // compute entropy
    for (unsigned int f = 0; f != SP.L; ++f)
    {
      // add to entropy only if frequency is non-zero
      if (freq_root[tmp_f])
      {
        if (count) (*count)++;
        freq = (double)freq_root[tmp_f] / den;
        temp -= freq * log(freq);
      }

      // update temp index
      tmp_f += blockDim.x;
    }

    // now store entropy in float output
    entropies[i] = (float)temp;
  }

/*
* Computes entropy, cluster index and/or statistical index for a cluster batch (one thread = one cluster)
*
* S  -> cluster size in number of registers
* N  -> sample size in bits
* NA -> number of agents
* M  -> number of samples
* L  -> number of maximum possible sample values
* C  -> number of clusters in current batch
* R  -> cluster size in number of variables
* system_entropies contains joint entropy at index N
*/
template<unsigned int S, bool ComputeTc, bool ComputeZI, bool ComputeSI, bool ComputeSI_2, bool HistogramOnly, bool ComputeMutualInformation> __global__ void cluster_kernel(
  float* system_entropies,
  float* hsystem_stats,
  register_t* clusters,
  unsigned int* cluster_sizes,
  float* entropies, float* output)
  {
    //  float gval=0;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= SP.C) return;

    register_t* cur_cluster = clusters + i * S;

    // store entropy to system entropies and exit
    if (HistogramOnly)
    {
      bool ent_set = false;
      for (unsigned int v = 0; v != SP.N; ++v)
      if (dci::RegisterUtils::getBitAtPos(cur_cluster, v))
      {
        system_entropies[v] = ent_set ? 0.0f : entropies[i]; // entropy stored only in first agent variable
        ent_set = true;
      }
    }
    else
    {
      // check if whole system thread (store joint entropy and exit)
      if (cluster_sizes[i] == SP.NA)
      {
        system_entropies[SP.N] = entropies[i];
        output[i] = 0.0f;
        return;
      }

      // check if empty cluster
      if (cluster_sizes[i] == 0)
      {
        output[i] = 0.0f;
        return;
      }

      // check if empty comp. cluster
      unsigned int comp_i = i % 2 ? (i-1) : (i+1);
      if (cluster_sizes[comp_i] == 0)
      {
        output[i] = 0.0f;
        return;
      }

      // compute integration
      float integration = -entropies[i];

      for (unsigned int v = 0; v != SP.N; ++v)
      if (dci::RegisterUtils::getBitAtPos(cur_cluster, v))
      integration += system_entropies[v];

      // wait for all threads in block - complementary cluster is needed
      __syncthreads();

      // compute cluster index
      if (ComputeMutualInformation)
      {
        float mutual_information = entropies[i] + entropies[comp_i] - system_entropies[SP.N];
        output[i] = mutual_information != 0.0f ? (integration / mutual_information) : 0.0f;
      }
      else
      output[i] = integration;

      // check which index has to be computed
      if (ComputeTc)
      {
        output[i] = (float)(((double)output[i] - (double)hsystem_stats[2 * cluster_sizes[i] - 2]) / (double)hsystem_stats[2 * cluster_sizes[i] - 1]);
        if (output[i] != output[i]) output[i] = 0.0f;
      }

      if (ComputeZI)  // ZI= 2nI -g / std(2g)
      {
        //cardinality of the alphabet of a single random variable:
        unsigned int cardinality = powf(2,SP.N/SP.NA); // SP.N/SP.NA=number of bit per random variable -> cardinality is 2^(num. of bit)
        //unsigned int cardinality=3;
        // exemple: if X is encoded with 2 bits then I can have: 00,01,10,11 -> 2^2 = 4 possible values. If I have 3 bits then 2^3 = 8 values.
        // The DoF formula is: [Prod_i(card_i)-1] - [sum_i(card_i-1)] where prod and sum goes from 1 to number of elements in the cluster.
        //assuming all the RV to have same cardinality it becomes:
        float gval= powf(cardinality,cluster_sizes[i]) -1 - cluster_sizes[i] * (cardinality -1); // DoF
        output[i]=(float)((2*SP.M*integration - gval) / sqrtf(2*gval));
      }

      if (ComputeSI)  // SI = 2nI/g
      {
        //cardinality of the alphabet of a single random variable:
        unsigned int cardinality = powf(2,SP.N/SP.NA); // SP.N/SP.NA=number of bit per random variable -> cardinality is 2^(num. of bit)
        //unsigned int cardinality=3;
        // exemple: if X is encoded with 2 bits then I can have: 00,01,10,11 -> 2^2 = 4 possible values. If I have 3 bits then 2^3 = 8 values.
        // The DoF formula is: [Prod_i(card_i)-1] - [sum_i(card_i-1)] where prod and sum goes from 1 to number of elements in the cluster.
        //assuming all the RV to have same cardinality it becomes:
        float gval= powf(cardinality,cluster_sizes[i]) -1 - cluster_sizes[i] * (cardinality -1); // DoF

        output[i]=(float)((2*SP.M*integration) /gval);
      }

      if (ComputeSI_2) // SI2= I/Imax
      {
        float min_entropy = 100;
        float sum_entropies=0;
        for (unsigned int v = 0; v != SP.N; ++v)
        {
          if (dci::RegisterUtils::getBitAtPos(cur_cluster, v))
          {
            // search the minimum entropy in the cluster
            min_entropy = ((min_entropy < system_entropies[v]) ? min_entropy : system_entropies[v]);
            sum_entropies = sum_entropies +  system_entropies[v]; // sum of the entropies in the cluster
          }
        }
        output[i]=  integration/(sum_entropies - min_entropy);
      }
    }
  }

/*
* Cardinality
* Computes entropy, cluster index and/or statistical index for a cluster batch (one thread = one cluster)
*
* S  -> cluster size in number of registers
* N  -> sample size in bits
* NA -> number of agents
* M  -> number of samples
* L  -> number of maximum possible sample values
* C  -> number of clusters in current batch
* R  -> cluster size in number of variables
* system_entropies contains joint entropy at index N
*/
template<unsigned int S, bool ComputeZI,bool ComputeSI,bool ComputeSI_2, bool HistogramOnly, bool ComputeMutualInformation> __global__ void cluster_kernel_card(
  float* system_entropies,
  float* hsystem_stats,
  register_t* clusters,
  unsigned int* cluster_sizes,
  float* entropies, float* output, unsigned int* cardinalities)
  {
    //  float gval=0;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= SP.C) return;

    register_t* cur_cluster = clusters + i * S;

    // store entropy to system entropies and exit
    if (HistogramOnly)
    {
      bool ent_set = false;
      for (unsigned int v = 0; v != SP.N; ++v)
      if (dci::RegisterUtils::getBitAtPos(cur_cluster, v))
      {
        system_entropies[v] = ent_set ? 0.0f : entropies[i]; // entropy stored only in first agent variable
        ent_set = true;
      }
    }
    else
    {
      // check if whole system thread (store joint entropy and exit)
      if (cluster_sizes[i] == SP.NA)
      {
        system_entropies[SP.N] = entropies[i];
        output[i] = 0.0f;
        return;
      }

      // check if empty cluster
      if (cluster_sizes[i] == 0)
      {
        output[i] = 0.0f;
        return;
      }

      // check if empty comp. cluster
      unsigned int comp_i = i % 2 ? (i-1) : (i+1);
      if (cluster_sizes[comp_i] == 0)
      {
        output[i] = 0.0f;
        return;
      }

      // compute integration
      float integration = -entropies[i];

      for (unsigned int v = 0; v != SP.N; ++v)
      if (dci::RegisterUtils::getBitAtPos(cur_cluster, v))
      integration += system_entropies[v];

      // wait for all threads in block - complementary cluster is needed
      __syncthreads();

      // compute cluster index
      if (ComputeMutualInformation)
      {
        float mutual_information = entropies[i] + entropies[comp_i] - system_entropies[SP.N];
        output[i] = mutual_information != 0.0f ? (integration / mutual_information) : 0.0f;
      }
      else
      output[i] = integration;

      // check which index has to be computed
      if (ComputeZI)  // ZI= 2nI -g / std(2g)
      {
        //cardinality of the alphabet of a single random variable:
        //unsigned int cardinality = powf(2,SP.N/SP.NA); // SP.N/SP.NA=number of bit per random variable -> cardinality is 2^(num. of bit)
        //unsigned int cardinality=cardinalities[1];
        unsigned int prod_card = 1;
        unsigned int sum_card = 0;
        unsigned int num_var = 0;

        for (unsigned int v = 0; v != SP.N; ++v)
        {
          if (dci::RegisterUtils::getBitAtPos(cur_cluster, v) && system_entropies[v]!=0.0f)
          {
            // entropy stored only in first agent variable
            prod_card=prod_card*cardinalities[num_var];
            sum_card=sum_card+cardinalities[num_var]-1;
          }
          if(system_entropies[v]!=0.0f)
          num_var++;
        }

        prod_card=prod_card-1;

        // exemple: if X is encoded with 2 bits then I can have: 00,01,10,11 -> 2^2 = 4 possible values. If I have 3 bits then 2^3 = 8 values.
        // The DoF formula is: [Prod_i(card_i)-1] - [sum_i(card_i-1)] where prod and sum goes from 1 to number of elements in the cluster.
        //assuming all the RV to have same cardinality it becomes:
        //float gval= powf(cardinality,cluster_sizes[i]) -1 - cluster_sizes[i] * (cardinality -1); // DoF

        unsigned int gval= prod_card-sum_card; // DoF

        output[i]=(float)((2*SP.M*integration - gval) / sqrtf(2*gval));
      }

      if (ComputeSI)  // SI = 2nI/g
      {
        unsigned int prod_card=1;
        unsigned int sum_card=0;

        unsigned int num_var=0;

        for (unsigned int v = 0; v != SP.N; ++v)
        if (dci::RegisterUtils::getBitAtPos(cur_cluster, v) && system_entropies[v]!=0.0f)
        {
          // entropy stored only in first agent variable

          prod_card=prod_card*cardinalities[num_var];
          sum_card=sum_card+cardinalities[num_var]-1;

          num_var++;
        }

        prod_card=prod_card-1;

        //float gval= powf(2,cluster_sizes[i]) -1 - cluster_sizes[i]; // DoF

        //cardinality of the alphabet of a single random variable:
        //unsigned int cardinality = powf(2,SP.N/SP.NA); // SP.N/SP.NA=number of bit per random variable -> cardinality is 2^(num. of bit)
        //unsigned int cardinality=3;
        // exemple: if X is encoded with 2 bits then I can have: 00,01,10,11 -> 2^2 = 4 possible values. If I have 3 bits then 2^3 = 8 values.
        // The DoF formula is: [Prod_i(card_i)-1] - [sum_i(card_i-1)] where prod and sum goes from 1 to number of elements in the cluster.
        //assuming all the RV to have same cardinality it becomes:
        //float gval= powf(cardinality,cluster_sizes[i]) -1 - cluster_sizes[i] * (cardinality -1); // DoF

        unsigned int gval= prod_card-sum_card;
        output[i]=(float)((2*SP.M*integration) /gval);
      }

      if (ComputeSI_2) // SI2= I/Imax
      {
        float min_entropy = 100;
        float sum_entropies=0;

        for (unsigned int v = 0; v != SP.N; ++v)
        {
          if (dci::RegisterUtils::getBitAtPos(cur_cluster, v))
          {
            // search the minimum entropy in the cluster
            min_entropy = ((system_entropies[v]!=0.0f && system_entropies[v]<min_entropy) ? system_entropies[v] : min_entropy );
            sum_entropies = sum_entropies +  system_entropies[v]; // sum of the entropies in the cluster
          }
        }
        output[i]=  integration/(sum_entropies - min_entropy);
      }
    }
  }

/*
* Computes entropy, cluster index and/or statistical index for a cluster batch (one thread = one cluster)
*
* S  -> cluster size in number of registers
* N  -> sample size in bits
* NA -> number of agents
* M  -> number of samples
* L  -> number of maximum possible sample values
* C  -> number of clusters in current batch
* R  -> cluster size in number of variables
* system_entropies contains joint entropy at index N
*/
template<unsigned int S, bool ComputeZI,bool ComputeSI,bool ComputeSI_2, bool HistogramOnly, bool ComputeMutualInformation> __global__ void cluster_kernel_agent_pool(
  float* system_entropies,
  float* hsystem_stats,
  register_t* clusters,
  unsigned int* cluster_sizes,
  float* entropies, float* output, register_t* agent_pool)
  {
    //  float gval=0;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= SP.C) return;

    register_t* cur_cluster = clusters + i * S;

    // store entropy to system entropies and exit
    if (HistogramOnly)
    {
      bool ent_set = false;
      for (unsigned int v = 0; v != SP.N; ++v)
      if (dci::RegisterUtils::getBitAtPos(cur_cluster, v))
      {
        system_entropies[v] = ent_set ? 0.0f : entropies[i]; // entropy stored only in first agent variable
        ent_set = true;
      }
    }
    else
    {
      // check if whole system thread (store joint entropy and exit)
      if (cluster_sizes[i] == SP.NA)
      {
        system_entropies[SP.N] = entropies[i];
        output[i] = 0.0f;
        return;
      }

      // check if empty cluster
      if (cluster_sizes[i] == 0)
      {
        output[i] = 0.0f;
        return;
      }

      // check if empty comp. cluster
      unsigned int comp_i = i % 2 ? (i-1) : (i+1);
      if (cluster_sizes[comp_i] == 0)
      {
        output[i] = 0.0f;
        return;
      }

      // compute integration
      float integration = -entropies[i];

      for (unsigned int v = 0; v != SP.N; ++v)
      if (dci::RegisterUtils::getBitAtPos(cur_cluster, v))
      integration += system_entropies[v];

      // wait for all threads in block - complementary cluster is needed
      __syncthreads();

      // compute cluster index
      if (ComputeMutualInformation)
      {
        float mutual_information = entropies[i] + entropies[comp_i] - system_entropies[SP.N];
        output[i] = mutual_information != 0.0f ? (integration / mutual_information) : 0.0f;
      }
      else
      output[i] = integration;

      // check which index has to be computed
      if (ComputeZI)  // ZI= 2nI -g / std(2g)
      {
        //cardinality of the alphabet of a single random variable:
        //unsigned int cardinality = powf(2,SP.N/SP.NA); // SP.N/SP.NA=number of bit per random variable -> cardinality is 2^(num. of bit)
        unsigned int cardinality=3;

        //LAURA
        /* unsigned int dim_Ts=1;
        unsigned int s_dim_Tj=0;

        for (unsigned int v = 0; v != SP.N; ++v)
        if (dci::RegisterUtils::GetBitAtPos(cur_cluster, v))
        {
        unsigned int card_var=1;
        unsigned int vv=v;

        //entropy stored only in first agent variable
        while(system_entropies[vv]==0.0f)
        {
        card_var++;
        vv++;
      }

      if(card_var!=1)
      {
      dim_Ts=dim_Ts*card_var;
      s_dim_Tj=s_dim_Tj+card_var-1;
      v=vv-1;
    }
  }

  dim_Ts=dim_Ts-1;
  float gval=(float) dim_Ts-s_dim_Tj;

  */

      // exemple: if X is encoded with 2 bits then I can have: 00,01,10,11 -> 2^2 = 4 possible values. If I have 3 bits then 2^3 = 8 values.
      // The DoF formula is: [Prod_i(card_i)-1] - [sum_i(card_i-1)] where prod and sum goes from 1 to number of elements in the cluster.
      //assuming all the RV to have same cardinality it becomes:
      float gval= powf(cardinality,cluster_sizes[i]) -1 - cluster_sizes[i] * (cardinality -1); // DoF
      output[i]=(float)((2*SP.M*integration - gval) / sqrtf(2*gval));
    }

    if (ComputeSI)  // SI = 2nI/g
    {
      //float gval= powf(2,cluster_sizes[i]) -1 - cluster_sizes[i]; // DoF

      //cardinality of the alphabet of a single random variable:
      //unsigned int cardinality = powf(2,SP.N/SP.NA); // SP.N/SP.NA=number of bit per random variable -> cardinality is 2^(num. of bit)
      unsigned int cardinality=3;
      // exemple: if X is encoded with 2 bits then I can have: 00,01,10,11 -> 2^2 = 4 possible values. If I have 3 bits then 2^3 = 8 values.
      // The DoF formula is: [Prod_i(card_i)-1] - [sum_i(card_i-1)] where prod and sum goes from 1 to number of elements in the cluster.
      //assuming all the RV to have same cardinality it becomes:
      float gval= powf(cardinality,cluster_sizes[i]) -1 - cluster_sizes[i] * (cardinality -1); // DoF

      output[i]=(float)((2*SP.M*integration) /gval);
    }

    if (ComputeSI_2) // SI2= I/Imax
    {
      float min_entropy = 100;
      //float min_entropy = FLT_MAX;

      float sum_entropies=0;

      for (unsigned int v = 0; v != SP.N; ++v)
      {
        if (dci::RegisterUtils::getBitAtPos(cur_cluster, v))
        {
          // search the minimum entropy in the cluster
          min_entropy = ((min_entropy < system_entropies[v]) ? min_entropy : system_entropies[v]);
          sum_entropies = sum_entropies +  system_entropies[v]; // sum of the entropies in the cluster
        }
      }
      output[i]=  integration/(sum_entropies - min_entropy);
    }
  }
}

#endif /* DCI_KERNELS_H */
