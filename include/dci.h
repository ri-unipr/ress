//
//  dci.h
//
//  Created by Emilio Vicari on 1/3/2016.
//  Updated by Michele Amoretti on 29/3/2020.
//  Copyright © 2020 University of Parma. All rights reserved.
//


/* choose index mod. If you don't specify the index by command line --zi, --si, --si2 the default is zi*/

#ifndef DCI_H
#define DCI_H

#include <iostream>
#include <fstream>
#include <queue>
#include <stack>
#include <unordered_map>
#include <utility>
#include <string>
#include <vector>
#include <random>
#include <stdlib.h>
#include <time.h>
#include "common.h"
#include "command_line.h"
#include "file_utils.h"
#include "register_utils.h"
#include "cuda_utils.h"
#include "stat_utils.h"
#include "cluster_descriptor.h"
#include "dci_kernels.hu"
#include <cuda.h>
#include <cuda_profiler_api.h>

// prints to cout only if debug messages are enabled
#define verbose_cout if(!dci::Application::conf->verbose) {} else std::cout

// prints to cout only if not silent
#define not_silent_cout if(dci::Application::conf->silent) {} else std::cout

// error codes
#define SUCCESS 0
#define OUT_OF_MEMORY 1

using namespace std;

namespace dci
{

  class Application
  {

  private:

    /*
    * Application-scope parameters
    */
    unsigned int
    N,                              // sample size (i.e.: number of single-bit variables in the system)
    NA,                             // number of agents in the system
    NO,                             // number of agents in the original system (in case of sieving)
    NBO,                            // number of bits/sample in the original system (in case of sieving)
    M,                              // number of samples
    K,                              // number of results to keep
    KI,                             // number of results to keep (internally)
    CB,                             // number of clusters in batch
    HB,                             // size of histogram
    S,                              // size of sample in registers (used for clusters)
    SO,                             // size of starting agent cluster in registers (used for sieving)
    SBO,                            // size of starting agent mask in registers (used for sieving)
    SA,                             // size of agent cluster in registers (used for external calls only - internal agent clusters are S-sized)
    num_blocks,                     // number of parallel execution blocks
    num_threads,   	                // number of threads per block
    sample_size_bytes,       		// size of sample in bytes (used for clusters)
    agent_sample_size_bytes,        // size of agent sample in bytes (used for clusters)
    agent_pool_size_bytes,    		// size of agent pool in bytes
    system_size_bytes;       		// size of system in bytes

    unsigned long allocated_device_bytes;   // total number bytes of global memory allocated on device

    bool implicit_agents;                   // if true, all the agents are boolean and are not explicitly defined

    bool has_mi_mask;                       // if true, mutual information is computed only relative to a subset of agents

    bool compute_mutual_information;        // if true, mutual information is computed; if false, only integration is used

    bool is_initialized;                    // if true, Init() was called (i.e. device memory is allocated)

    unsigned long long L;                   // number of possible system states (2^N)

    dci::RunInfo* conf;                     // stores a pointer to current configuration

    vector<string> agent_names;             // stores agent names (if specified)
    vector<string> starting_agent_names;             // stores starting agent names (if specified, for sieving)

    //LAURA
    vector<vector<pair<unsigned int, register_t*> > > card; // histogram (agent cardinalities)


    /*
    * Host memory buffers
    */
    register_t* original_system_data;
    register_t* system_data;
    register_t* original_agent_pool;
    register_t* agent_pool;
    register_t* starting_agent_pool;
    register_t* mutual_information_mask;
    float* system_entropies;
    float* hsystem_stats;
    unsigned int* cluster_sizes;
    unsigned int* frequencies;
    register_t* hsystem_data;
    register_t* histogram;
    //LAURA
    unsigned int* cardinalities;

    /*
    * Device memory buffers
    */
    register_t* dev_system;
    float* dev_system_entropies;
    float* dev_hsystem_stats;
    register_t* dev_clusters;
    register_t* dev_histogram;
    unsigned int* dev_frequencies;
    unsigned int* dev_cluster_size;
    register_t** dev_next;
    float* dev_entropies;
    float* dev_output;
    unsigned int* dev_count;
    //LAURA
    unsigned int* dev_cardinalities;
    register_t* dev_agent_pool;

    /*
    * Temp structures for constant memory
    */
    SystemParameters params;

    /*
    * Dynamic container that holds tuning information
    */
    unordered_map<string, pair<unsigned int, clock_t> > tuning_map;

    /*
    * Function declarations
    */
    template<bool ComputeZi,bool ComputeSI, bool ComputeSI_2, bool HistogramOnly> inline unsigned int callKernel(const unsigned int& C, const register_t* clusters, const unsigned int* cluster_sizes, float* output);
    //LAURA
    template<bool ComputeZi,bool ComputeSI, bool ComputeSI_2, bool HistogramOnly> inline unsigned int callKernelCard(const unsigned int& C, const register_t* clusters, const unsigned int* cluster_sizes, float* output, unsigned int* cardinalities);
    template<bool ComputeZi,bool ComputeSI, bool ComputeSI_2, bool HistogramOnly> inline unsigned int callKernelPool(const unsigned int& C, const register_t* clusters, const unsigned int* cluster_sizes, float* output, register_t* agent_pool);

    template<bool IsFullComputation> int computeHomogeneousSystemStatistics();
    unique_ptr<vector<dci::ClusterDescriptor> > computeSystemStatistics(ostream& out);
    void collectAgentStatistics();
    void printSystemDataToStream(ostream& out, register_t* data);
    void printSystemStatsToStream(ostream& out, const float* stats);
    void generateHomogeneousSystem();
    void tuneFunction(const string& function_name, const clock_t& execution_time);
    void printTuningInfo();
    void reallocateHistogramMemory(const unsigned int& new_HB);
    void debugCluster(register_t* cluster, register_t* dev_histogram, unsigned int* dev_frequencies);
    inline void printCluster(const register_t* cluster);
    inline void printClusterValue(const register_t* cluster, const register_t* value);
    inline void printProgress(clock_t start, const unsigned long long& n_clusters);
    inline void printProgress(clock_t start, const unsigned long long& n_clusters, const unsigned long long& total_clusters);
    void printAgentCluster(const dci::ClusterDescriptor& cluster, ostream& out, bool tab_format);
    void checkInsertClusterInResults(priority_queue<dci::ClusterDescriptor>& q, const register_t* cluster, const float& ind_index);
    template<bool ascending, bool realloc=false> void performSieving(const vector<register_t*>& input_clusters, const vector<register_t*>& input_agent_clusters, const vector<float>& input_ind, vector<register_t*>& output, vector<float>& output_ind, const unsigned int& max_output_count);
    template<bool ascending> unique_ptr<vector<dci::ClusterDescriptor> > selectSuperClusters(const unique_ptr<vector<dci::ClusterDescriptor> >& results);
    void writeSystemToFileAfterSieving(const unique_ptr<vector<dci::ClusterDescriptor> >& super_clusters, const string& output_path);
    string getAgentName(unsigned int a);
    inline void printBitmask(const register_t* mask, const unsigned int n, ostream& out);
    register_t* getOriginalAgentMaskFromCluster(const register_t* cluster);
    register_t* getCurrentAgentMaskFromCluster(const register_t* cluster);
    void pushFullClusterInfo(const register_t* cluster_orig, vector<string>& temp_agent_names, vector<string>& temp_agent_rep, vector<unsigned int>& temp_agent_sizes, vector<vector<unsigned int> >& temp_agent_comp);
    void pushMissingAgentInfo(const unsigned int& a, vector<string>& temp_agent_names, vector<string>& temp_agent_rep, vector<unsigned int>& temp_agent_sizes, vector<vector<unsigned int> >& temp_agent_comp);
    string getAgentName(unsigned int a, const vector<string>& ref_agent_names);
    string getOriginalAgentClusterName(const dci::ClusterDescriptor& cluster);
    unique_ptr<vector<dci::ClusterDescriptor> > getTranslatedClusterDescriptors(const vector<register_t*>& agent_clusters);
    unique_ptr<vector<dci::ClusterDescriptor> > getTranslatedClusterDescriptors(const vector<register_t*>& agent_clusters, const vector<float>& zi_values);

    // default constructor is hidden
    Application() { }

  public:

    // constructors
    Application(dci::RunInfo& configuration);

    // destructor
    ~Application();

    // application initialization - must be called before any calculation begins
    void Init();

    // application entry point for exhaustive computation
    int Run();

    // test
    void Test();

    // computes statistical indexes for all clusters in list, and stores them in output vector
    void ComputeIndex(const vector<register_t*>& clusters, vector<float>& output_vec);

    // performs sieving algorithm on a set of input clusters given their Tc values
    template<bool ascending> void ApplySievingAlgorithm(const vector<register_t*>& input_clusters, const vector<float>& input_ind, vector<register_t*>& output, vector<float>& output_ind, const unsigned int& max_output_count);

    // selects super clusters from a list of output clusters, using system configuration values
    template<bool ascending> void SelectSuperClustersAndSave(const vector<register_t*>& results, const vector<float>& zi_values);

    // getters
    inline unsigned int getAgentSizeInBytes();
    inline unsigned int getNumberOfAgents();

    // utilities
    float elapsedTimeMilliseconds(clock_t start, clock_t stop);

  };

  /************************************************************
  * dci::Application implementation
  ************************************************************/

  /*
  * Returns agent size in bytes (for external calls)
  */
  inline unsigned int Application::getAgentSizeInBytes() { return this->agent_sample_size_bytes; }

  /*
  * Returns number of agents
  */
  inline unsigned int Application::getNumberOfAgents() { return this->NA; }

  /*
  * Computes statistical indexes for all clusters in list, and stores them in output vector (must be pre-allocated)
  */
  void Application::ComputeIndex(const vector<register_t*>& clusters_vec, vector<float>& output_vec)
  {

    unsigned int total_clusters = clusters_vec.size() * 2; // must include complementary clusters in count
    unsigned int tempCB = total_clusters > CB ? CB : total_clusters;
    register_t* clusters = (register_t*)malloc(tempCB * sample_size_bytes);
    unsigned int* cluster_sizes = (unsigned int*)malloc(tempCB * sizeof(unsigned int));
    register_t* cur_cluster = clusters;
    float* output = (float*)malloc(tempCB * sizeof(float));

    // store only one cluster
    dci::RegisterUtils::SetAllBits<1>(clusters, N, S);
    cluster_sizes[0] = NA; // just one NA-sized cluster

    // invoke kernel for whole system
    callKernel<false, false,false,false>(1, clusters, cluster_sizes, output);

    // this variable holds number of clusters in batch
    unsigned int C = 0;
    unsigned int offset = 0;

    // reset last reg mask for cluster generation
    dci::ClusterUtils::resetLastRegMask(N);

    // cycle all clusters
    for (unsigned int i = 0; i != clusters_vec.size(); ++i)
    {

      // add cluster to list
      dci::ClusterUtils::clusterBitmaskFromAgentCluster(cur_cluster, clusters_vec[i], N, S, NA, agent_pool);

      // get complementary cluster and store it in the next memory block
      dci::ClusterUtils::getComplementaryClusterMask(cur_cluster + S, cur_cluster, N);

      // get cluster size
      unsigned int temp_r = dci::RegisterUtils::GetNumberOf<1>(clusters_vec[i], NA);

      // 2 clusters (original and complementary)
      cluster_sizes[C++] = temp_r;
      cluster_sizes[C++] = NA - temp_r;

      // if we have stored enough clusters for a batch, or if we have reached the end of current group
      if (C == tempCB || i == (clusters_vec.size() - 1))
      {
        // invoke kernel for current batch
        if(conf->zi_index)
        {
          //callKernel<true,false,false, false>(C, clusters, cluster_sizes, output);
          callKernelCard<true,false,false, false>(C, clusters, cluster_sizes, output, cardinalities);
        }
        else if(conf->strength_index)
        {
          //callKernel<false,true, false,false>(C, clusters, cluster_sizes, output);
          callKernelCard<true,false,false, false>(C, clusters, cluster_sizes, output, cardinalities);
        }
        else if(conf->strength2_index)
        {
          //callKernel<false,false,true, false>(C, clusters, cluster_sizes, output);
          callKernelCard<true,false,false, false>(C, clusters, cluster_sizes, output, cardinalities);
        }
        if ( conf->zi_index == false && conf->strength_index == false && conf->strength2_index == false)
        {
          conf->zi_index=true; // pick zi as default

          //callKernel<true,false,false, false>(C, clusters, cluster_sizes, output);
          callKernelCard<true,false,false, false>(C, clusters, cluster_sizes, output, cardinalities);
        }


        // store output
        for (unsigned int j = 0; j != C / 2; ++j)
        output_vec[j + offset] = output[2 * j];

        // update offset
        offset = i + 1;

        // reset current cluster pointer
        cur_cluster = clusters;

        // reset number of clusters
        C = 0;

      }
      else
      // move to next block
      cur_cluster += S * 2;

    }

    free(clusters);
    free(cluster_sizes);
    free(output);


  }

  /*
  * Just a test for arbitrary cluster index computation
  */
  void Application::Test()
  {

    // allocate memory for clusters
    vector<register_t*> clusters(2);

    // allocate memory for cluster indexes
    vector<float> output(2);

    // create agent list for clusters
    vector<unsigned int> cluster1 = { 13, 14 };
    vector<unsigned int> cluster2 = { 6, 13, 14 };

    // allocate cluster bitmasks
    clusters[0] = (register_t*)malloc(SA * sizeof(register_t));
    clusters[1] = (register_t*)malloc(SA * sizeof(register_t));

    // set bitmasks from agent list
    dci::ClusterUtils::setClusterFromPosArray(clusters[0], cluster1, NA);
    dci::ClusterUtils::setClusterFromPosArray(clusters[1], cluster2, NA);

    // perform computation
    ComputeIndex (clusters, output);

    // print clusters and results
    dci::ClusterUtils::println(cout, clusters[0], NA);
    dci::ClusterUtils::println(cout, clusters[1], NA);
    cout << output[0] << endl;
    cout << output[1] << endl;

    // free memory
    free(clusters[0]);
    free(clusters[1]);

  }

  /*
  * Constructor
  */
  Application::Application(dci::RunInfo& configuration)
  {

    // store start/end time
    clock_t start = clock(), prev;
    prev = start;

    // store configuration for global use
    conf = &configuration;

    // adjust flags
    if (conf->silent) conf->verbose = false;

    // collect system parameters
    dci::FileUtils::CollectSystemParameters(conf->input_file_name, N, M, S, L, NA, SA, NO, SO, NBO, SBO, has_mi_mask);

    // check sample size
    if (N > BITS_PER_REG * 32)
    {
      cerr << "Samples must be no more than " << (BITS_PER_REG * 32) << " bits in size. Aborting.\n";
      exit(1);
    }

    // compute additional parameters
    implicit_agents = (NA == 0);
    compute_mutual_information = (conf->metric == "rel");
    if (implicit_agents) NA = N;
    sample_size_bytes = S * BYTES_PER_REG;
    agent_sample_size_bytes = SA * BYTES_PER_REG; // only for external calls
    agent_pool_size_bytes = sample_size_bytes * NA;
    system_size_bytes = sample_size_bytes * M;
    K = conf->res;
    KI = conf->sieving ? (conf->sieving_max > L ? L : conf->sieving_max) : K;
    num_threads = conf->num_threads;
    num_blocks = conf->num_blocks;
    CB = num_threads * num_blocks;
    HB = M;
    allocated_device_bytes =
    sizeof (unsigned int) +                     // sample values count for whole system
    system_size_bytes +                         // system data
    sizeof(float) * (
      3 * N +                                 // system entropies, hsystem stats
      1 +                                     // joint entropy
      2 * CB) +                                 // cluster entropies, elaboration output
      CB * (
        sizeof(unsigned int) +                     // cluster sizes
        sample_size_bytes +                     // clusters
        HB * (
          sample_size_bytes +                 // histogram value
          sizeof(unsigned int) +                 // histogram frequency
          sizeof(register_t*)                    // histogram next hop
        )
      );

      // set constant memory temp struct
      params.N = N;
      params.NA = NA;
      params.M = M;
      params.CB = CB;
      params.C = 0;
      params.L = HB;

      // TODO check for available device memory

      // print device info if necessary
      if (conf->show_device_stats) dci::CUDAUtils::PrintDeviceInfo();

      //size_t mem_free = 0, mem_total = 0;
      //cudaMemGetInfo(&mem_free, &mem_total);

      not_silent_cout <<"Parallel DCI (C++ and CUDA)" << endl;
      not_silent_cout << "Sample size                 " << N << " (" << NA << " agents)\n";
      not_silent_cout << "Samples                     " << M << "\n";
      not_silent_cout << "Clusters                    " << L << "\n";
      //not_silent_cout << "Device memory (free/total)  " << (mem_free >> 20) << " MB / " << (mem_total >> 20) << " MB\n";

      // check system dimensions
      if (NA > num_threads * num_blocks)
      {
        cerr << "Error: number of agents must not exceed " << (num_blocks * num_threads) << " (nb * nt). Aborting.\n";
      }

      verbose_cout << "Loading system data\n";

      // allocate memory for system data
      system_data = (register_t*)malloc(system_size_bytes);
      agent_pool = (register_t*)malloc(agent_pool_size_bytes);
      system_entropies = (float*)malloc(N * sizeof(float));
      hsystem_stats = (float*)malloc(2 * NA * sizeof(float));
      cluster_sizes = (unsigned int*)malloc(CB * sizeof(unsigned int));
      frequencies = (unsigned int*)malloc(CB * HB * sizeof(unsigned int));
      histogram = (register_t*)malloc(CB * HB * sample_size_bytes);
      if (has_mi_mask) mutual_information_mask = (register_t*)malloc(SA * BYTES_PER_REG);
      if (NO) starting_agent_pool = (register_t*)malloc(NO * SO * BYTES_PER_REG);
      if (NBO)
      {
        original_agent_pool = (register_t*)malloc(NO * SBO * BYTES_PER_REG);
        original_system_data = (register_t*)malloc(M * SBO * BYTES_PER_REG);
      }
      //LAURA
      cardinalities = (unsigned int*)malloc(NA * sizeof(unsigned int));

      prev = clock();
      // load system data
      dci::FileUtils::LoadSystemData(conf->input_file_name, N, M, S, SA, SO, SBO, NA, NO, NBO, system_data,
        original_system_data, agent_pool, original_agent_pool, starting_agent_pool, implicit_agents, agent_names, starting_agent_names,
        has_mi_mask, mutual_information_mask);
        tuneFunction("LoadSystemData", clock() - prev); prev = clock();

        // adjust mutual information mask
        if (has_mi_mask)
        {
          register_t* new_mask = (register_t*)malloc(sample_size_bytes);
          memset((void*)new_mask, 0, sample_size_bytes);
          dci::ClusterUtils::clusterBitmaskFromAgentCluster(new_mask, mutual_information_mask, N, S, NA, agent_pool);
          free(mutual_information_mask);
          mutual_information_mask = new_mask;
        }
        else
        {
          mutual_information_mask = (register_t*)malloc(sample_size_bytes);
          memset((void*)mutual_information_mask, 0, sample_size_bytes);
          dci::RegisterUtils::SetAllBits<true>(mutual_information_mask, N, S);
        }

        dci::ClusterUtils::setMutualInformationMask(mutual_information_mask);

        // waiting for initialization
        this->is_initialized = false;

        tuneFunction("Application ctor", clock() - start);

      }

      /*
      * Destructor
      */
      Application::~Application()
      {

        // free data
        free(system_data);
        free(hsystem_stats);
        free(system_entropies);
        free(agent_pool);
        free(cluster_sizes);
        if (NO) free(starting_agent_pool);
        if (NBO) { free(original_agent_pool); free(original_system_data); }

        // check if device memory was allocated
        if (!this->is_initialized) return;

        // free allocated data
        HANDLE_ERROR( cudaFree(dev_system) );
        HANDLE_ERROR( cudaFree(dev_system_entropies) );
        HANDLE_ERROR( cudaFree(dev_hsystem_stats) );
        HANDLE_ERROR( cudaFree(dev_clusters) );
        HANDLE_ERROR( cudaFree(dev_histogram) );
        HANDLE_ERROR( cudaFree(dev_frequencies) );
        HANDLE_ERROR( cudaFree(dev_cluster_size) );
        HANDLE_ERROR( cudaFree(dev_next) );
        HANDLE_ERROR( cudaFree(dev_entropies) );
        HANDLE_ERROR( cudaFree(dev_output) );
        HANDLE_ERROR( cudaFree(dev_count) );
        //LAURA
        HANDLE_ERROR( cudaFree(dev_cardinalities) );
        HANDLE_ERROR( cudaFree(dev_agent_pool) );

        //LAURA
        free(cardinalities);
        free(histogram);
        free(frequencies);

      }

      /*
      * Initialization function
      * - device memory allocation
      * - single agent entropy computation
      * -
      */
      void Application::Init()
      {

        // store start/end time
        clock_t start = clock(), prev;
        prev = start;

        // Allocate device memory to hold necessary data
        verbose_cout << "Allocating memory on device, " <<
        (allocated_device_bytes >> 20)
        << " MB total\n";

        HANDLE_ERROR( cudaMalloc( (void**)&dev_system, system_size_bytes ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_system_entropies, (N + 1) * sizeof(float) ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_hsystem_stats, 2 * NA * sizeof(float) ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_clusters, CB * sample_size_bytes ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_histogram, CB * HB * sample_size_bytes ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_frequencies, CB * HB * sizeof(unsigned int) ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_cluster_size, CB * sizeof(unsigned int) ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_next, CB * HB * sizeof(register_t*) ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_entropies, CB * sizeof(float) ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_output, CB * sizeof(float) ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_count, sizeof(unsigned int) ) );
        //LAURA
        HANDLE_ERROR( cudaMalloc( (void**)&dev_cardinalities, NA * sizeof(unsigned int) ) );
        HANDLE_ERROR( cudaMalloc( (void**)&dev_agent_pool, agent_pool_size_bytes ) );


        // now copy system data to device
        HANDLE_ERROR( cudaMemcpy( dev_system, system_data, system_size_bytes, cudaMemcpyHostToDevice ) );

        // compute single agent entropies and frequencies
        for (unsigned int i = 0; i != NA; ++i) cluster_sizes[i] = 1;
        callKernel<false,false, false,true>(NA, agent_pool, cluster_sizes, system_entropies);
        tuneFunction("Agent entropies", clock() - prev); prev = clock();

        //LAURA
        HANDLE_ERROR( cudaMemcpy( frequencies, dev_frequencies, CB * HB * sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
        HANDLE_ERROR( cudaMemcpy( histogram, dev_histogram, CB * HB * sample_size_bytes, cudaMemcpyDeviceToHost ) );


        //inizializzazione cardinalit�
        for (unsigned int a = 0; a != NA; ++a)
        cardinalities[a]=0;

        //calcolo cardinalit�
        for (unsigned int a = 0; a != NA; ++a)
        {
          unsigned int block_idx = a / conf->num_threads;
          unsigned int thread_idx = a % conf->num_threads;
          unsigned int tmp_idx = block_idx * conf->num_threads * HB + thread_idx;
          unsigned int p = 0; // accumulated value
          for (unsigned int h = 0; p!= M && h != HB; ++h)
          {

            if (frequencies[tmp_idx])
            {
              p += frequencies[tmp_idx];
              //card[a].push_back(pair<unsigned int, register_t*>(p, histogram + tmp_idx * S));
              cardinalities[a]++;

            }
            tmp_idx += conf->num_threads;
          }
        }

        //LAURA
        verbose_cout<<"Cardinalities:\n";
        for (unsigned int a = 0; a != NA; ++a)
        //verbose_cout<<card[a].size()<<"\n";
        verbose_cout<<cardinalities[a]<<"\n";


        //        // check if homogeneous system stats have to be loaded from file or generated
        //        if (conf->hs_input_file_name != "")
        //        {
        //            verbose_cout << "Loading homogeneous system stats from file: " << conf->hs_input_file_name << '\n';
        //            dci::FileUtils::LoadSystemStats(conf->hs_input_file_name, NA, hsystem_stats);
        //        }
        //        else // generate it
        //        {
        //
        //            // copy system entropy data to host
        //            HANDLE_ERROR( cudaMemcpy( system_entropies, dev_system_entropies, N * sizeof(float), cudaMemcpyDeviceToHost ) );
        //            HANDLE_ERROR( cudaMemcpy( frequencies, dev_frequencies, CB * HB * sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
        //            HANDLE_ERROR( cudaMemcpy( histogram, dev_histogram, CB * HB * sample_size_bytes, cudaMemcpyDeviceToHost ) );
        //
        //            // allocate memory for homogeneous system
        //            hsystem_data = (register_t*)malloc(system_size_bytes);
        //
        //            if (conf->hs_data_input_file_name != "")
        //                dci::FileUtils::LoadRawSystemData(conf->hs_data_input_file_name, hsystem_data, N, M, S);
        //            else
        //            {
        //                verbose_cout << "Generating homogeneous system from uSystem stats with seed = " << conf->rand_seed << '\n';
        //                generateHomogeneousSystem();
        //                tuneFunction("HSystem generation", clock() - prev); prev = clock();
        //            }
        //
        //            // save h. system to file if specified
        //            if (conf->hs_data_output_file_name != "")
        //                dci::FileUtils::SaveSystemData(conf->hs_data_output_file_name, hsystem_data, N, M, S);
        //
        //            verbose_cout << "Computing homogeneous system statistics\n";
        //
        //            // copy h. system data to device
        //            HANDLE_ERROR( cudaMemcpy( dev_system, hsystem_data, system_size_bytes, cudaMemcpyHostToDevice ) );
        //
        //            free(hsystem_data);

        //            // compute single agent entropies
        //            callKernel<false, true>(NA, agent_pool, cluster_sizes, system_entropies);
        //            tuneFunction("Hsystem entropies", clock() - prev); prev = clock();
        //
        //            // compute h. system statistics
        //            if (conf->hs_count == 0) computeHomogeneousSystemStatistics<true>();
        //            else computeHomogeneousSystemStatistics<false>();
        //
        //            // copy system data back to device
        //            HANDLE_ERROR( cudaMemcpy( dev_system, system_data, system_size_bytes, cudaMemcpyHostToDevice ) );
        //            HANDLE_ERROR( cudaMemcpy( dev_system_entropies, system_entropies, N * sizeof(float), cudaMemcpyHostToDevice ) );

        //        }
        //
        //        // free unused memory
        //        free(histogram);
        //        free(frequencies);

        //        // check if homogeneous system has to be exported
        //        if (conf->hs_output_file_name != "")
        //        {
        //            ofstream hout(conf->hs_output_file_name);
        //            printSystemStatsToStream(hout, hsystem_stats);
        //            verbose_cout << "Exported homogeneous system stats to file: " << conf->hs_output_file_name << '\n';
        //        }
        //
        //        // copy h. system data to device
        //        HANDLE_ERROR( cudaMemcpy( dev_hsystem_stats, hsystem_stats, 2 * NA * sizeof(float), cudaMemcpyHostToDevice ) );
        //        //free(system_data);

        tuneFunction("Application init", clock() - start);

      }

      /*
      * Application entry point
      *
      * - loads system data
      * - loads the homogeneous system, or creates it if not specified
      * - starts the CUDA computation and prints the results
      */
      int Application::Run()
      {

        verbose_cout << "Computing system statistics\n";

        unique_ptr<vector<dci::ClusterDescriptor> > results = computeSystemStatistics(cout);

        // check if sieving output file is specified
        if (conf->sieving_out.length() > 0)
        {

          // TODO check exit conditions

          // decide which agents to keep
          unique_ptr<vector<dci::ClusterDescriptor> > super_clusters = selectSuperClusters<true>(results);

          for (unsigned int i = 0; i != super_clusters->size(); ++i)
          {
            printAgentCluster(super_clusters->at(i), cout, false);
            cout << " --> " << (super_clusters->at(i)).getIndex() << endl;
          }

          // print output file
          writeSystemToFileAfterSieving(super_clusters, conf->sieving_out);

        }

        // check if results file is specified
        if (conf->output_file_name.length() > 0)
        {

          // open file
          ofstream out(conf->output_file_name);

          // write agent names
          for (unsigned int a = 0; a != NA; ++a)
          out << getAgentName(a) << '\t';

          // write end of header line
          out << "ZI" << (NO ? "\tComp" : "") << endl;

          // print all clusters
          for (unsigned int i = 0; i != results->size(); ++i)
          {

            // print cluster
            printAgentCluster(results->at(results->size() - i - 1), out, true);

            // print Tc and newline
            out << results->at(results->size() - i - 1).getIndex();

            // print original agent cluster if necessary
            if (NO)
            out << '\t' << getOriginalAgentClusterName(results->at(results->size() - i - 1));

            // end of line
            out << endl;

          }

        }

        verbose_cout << "Freeing device memory\n";

        printTuningInfo();

        // success
        return SUCCESS;

      }

      /*
      * Computes elapsed time between two clock readings in milliseconds
      */
      float Application::elapsedTimeMilliseconds(clock_t start, clock_t stop)
      {
        return (float)(stop - start) / (float)CLOCKS_PER_SEC * 1000.0f;
      }

      /*
      * Prints progress (clusters/s and remaining time)
      */
      inline void Application::printProgress(clock_t start, const unsigned long long& n_clusters)
      {
        double elapsed = elapsedTimeMilliseconds(start, clock());
        double freq = (double)(n_clusters) * 1000.0f / elapsed;
        double perc = (double)n_clusters / (double)L;
        double eta = (elapsed / perc - elapsed) / 1000.0f;
        if (eta < 0.0f) eta = 0.0f;
        perc *= 100.0f;
        if (!conf->silent) (cout << (perc > 100.0f ? 100 : (int)perc) << "% (" << freq << " clusters/s), " << eta << " s left                      \r").flush();
      }

      /*
      * Prints progress (clusters/s and remaining time, with sampling ratio)
      */
      inline void Application::printProgress(clock_t start, const unsigned long long& n_clusters, const unsigned long long& total_clusters)
      {
        double elapsed = elapsedTimeMilliseconds(start, clock());
        double freq = (double)(n_clusters) * 1000.0f / elapsed;
        double perc = (double)n_clusters / ((double)total_clusters);
        double eta = (elapsed / perc - elapsed) / 1000.0f;
        if (eta < 0.0f) eta = 0.0f;
        perc *= 100.0f;
        if (!conf->silent) (cout << (perc > 100.0f ? 100 : (int)perc) << "% (" << freq << " clusters/s), " << eta << " s left                      \r").flush();
      }

      //    /*
      //     * Computes h. system statistics
      //     */
      //    template<bool IsFullComputation> int Application::computeHomogeneousSystemStatistics()
      //    {
      //
      //        not_silent_cout << "Computing homogeneous system statistics, please wait...\n";
      //        unsigned long long n_clusters = 1;
      //        // case r=N: compute H(U)
      //        register_t* clusters = (register_t*)malloc(CB * sample_size_bytes);
      //        unsigned int* cluster_sizes = (unsigned int*)malloc(CB * sizeof(unsigned int));
      //        register_t* cur_cluster = clusters;
      //        float* output = (float*)malloc(CB * sizeof(float));
      //        bool has_next;
      //        vector<dci::StatUtils::RunningStat> rs(NA);
      //        unsigned long long to_subtract;
      //        unsigned long long cc = 0;
      //        unsigned long long total_clusters = 0;
      //        mt19937 rng(conf->rand_seed);
      //
      //        clock_t start = clock();
      //
      //        // store total number of clusters according to computation type
      //        if (!IsFullComputation) total_clusters = conf->hs_count * (NA - 2);
      //
      //        // store only one cluster
      //        dci::RegisterUtils::SetAllBits<1>(clusters, N, S);
      //        cluster_sizes[0] = NA; // just one NA-sized cluster
      //        unsigned int old_HB = HB, new_HB;
      //
      //        // invoke kernel for whole system
      //        new_HB = callKernel<false, false>(1, clusters, cluster_sizes, output);
      //
      //        // reallocate histogram memory on device
      //        reallocateHistogramMemory(new_HB);
      //
      //        verbose_cout << "hSystem's jointEntropy H(S,U-S) = H(U) = " << output[0] << endl;
      //
      //        // this variable holds number of clusters in batch
      //        unsigned int C = 0;
      //
      //        // cycle all cluster sizes up to N/2, or up to N if a mutual information mask is specified
      //        auto limit = has_mi_mask ? (NA-1) : (NA/2);
      //        for (int r = 1; r <= limit; r++) // cluster size
      //        {
      //
      //            // initialize r-sized cluster mask generator
      //            dci::ClusterUtils::initializeClusterMaskGenerator(NA, r, S, agent_pool, N, mutual_information_mask);
      //
      //            // get number of clusters to use according to percentage
      //            if (!IsFullComputation)
      //                cc = 0;
      //
      //            // cycle all r-sized clusters
      //            while (
      //                (IsFullComputation && dci::ClusterUtils::getNextClusterMask(cur_cluster, has_next)) ||
      //                (!IsFullComputation && (cc++) < conf->hs_count))
      //            {
      //
      //                // get random cluster if necessary
      //                if (!IsFullComputation) dci::ClusterUtils::getNextRandomClusterMask(cur_cluster, rng);
      //
      //                // get complementary cluster and store it in the next memory block
      //                dci::ClusterUtils::getComplementaryClusterMask(cur_cluster + S, cur_cluster, N);
      //
      //                // 2 clusters (original and complementary)
      //                cluster_sizes[C++] = r;
      //                cluster_sizes[C++] = dci::RegisterUtils::GetNumberOf<1>(cur_cluster + S, N) ? (NA - r) : 0;
      //                // if (r==6)
      //                // {
      //                //     printCluster(cur_cluster);
      //                //     printCluster(cur_cluster + S);
      //                // }
      //
      //                // if we have stored enough clusters for a batch, or if we have reached the end of current group
      //                if (C == CB || (IsFullComputation && r == limit && !has_next) || (!IsFullComputation && r == limit && cc == conf->hs_count))
      //                {
      //
      //                    // invoke kernel for current batch
      //                    callKernel<false, false>(C, clusters, cluster_sizes, output);
      //
      //                    // reset extra cluster count
      //                    to_subtract = 0;
      //
      //                    // elaborate results
      //                    for (unsigned int c = 0; c != C / 2; ++c)
      //                    {
      //                        if (cluster_sizes[2*c] != 1) rs[cluster_sizes[2*c]].Push(output[2*c]); // 1-sized clusters are not relevant
      //                        if (!has_mi_mask && cluster_sizes[2*c+1] != NA / 2) rs[cluster_sizes[2*c+1]].Push(output[2*c+1]); else to_subtract++; // in case of an even number of variables, avoid counting N/2-sized clusters twice
      //
      //                        /*if (n_clusters == 1)
      //                        {
      //                            debugCluster(clusters + 2*c * S, dev_histogram + 2*c*HB*S, dev_frequencies + 2*c*HB);
      //                        }
      //                        if (n_clusters == 1)
      //                        {
      //                            debugCluster(clusters + (2*c+1) * S, dev_histogram + (2*c+1) * S * HB, dev_frequencies + (2*c+1)*HB);
      //                        }*/
      //
      //                    }
      //
      //                    // reset current cluster pointer
      //                    cur_cluster = clusters;
      //
      //                    // update number of processed clusters
      //                    n_clusters += (C - to_subtract);
      //
      //                    // print progress
      //                    if (IsFullComputation) printProgress(start, n_clusters);
      //                    else printProgress(start, n_clusters, total_clusters);
      //
      //                    // reset number of clusters
      //                    C = 0;
      //
      //                }
      //                else
      //                    cur_cluster += S * 2; // move two blocks forward
      //
      //            }
      //
      //        }
      //
      //        // cycle all cluster sizes from 2 to N - 1
      //        for (int r = 2; r <= NA - 1; r++) // cluster size
      //        {
      //
      //            hsystem_stats[2*r-2] = rs[r].Mean();
      //            verbose_cout << "\n<Ch> for |S| = " << r << " is " << rs[r].Mean() << endl;
      //            hsystem_stats[2*r-1] = rs[r].StandardDeviation();
      //            verbose_cout << "\nsigma(Ch) for |S| = " << r << " is " << rs[r].StandardDeviation() << endl;
      //
      //        }
      //
      //        free(clusters);
      //        free(cluster_sizes);
      //        free(output);
      //
      //        not_silent_cout << endl;
      //
      //        // reallocate histogram memory on device
      //        reallocateHistogramMemory(old_HB);
      //
      //        return 0;
      //
      //    }

      /*
      * Inserts given cluster in result stack if conditions are met
      */
      void Application::checkInsertClusterInResults(priority_queue<dci::ClusterDescriptor>& q, const register_t* cluster, const float& ind_index)
      {


        bool add = false;

        if (q.size() == 0)
        {
          add = true;
        }
        else
        {
          float worst = ((dci::ClusterDescriptor)q.top()).getIndex();
          add = q.size() < KI || ind_index > worst;
        }
        if (add)
        {
          if (q.size() == KI)
          q.pop();
          dci::ClusterDescriptor clusterDescriptor(cluster, N);
          clusterDescriptor.setIndex(ind_index);
          q.push(clusterDescriptor);
        }
      }

      /*
      * Computes system statistics and prints results to given output stream
      */
      unique_ptr<vector<dci::ClusterDescriptor> > Application::computeSystemStatistics(ostream& out)
      {

        //LAURA
        verbose_cout << "COMPUTE INDEX\n";

        //LAURA
        ////card.resize(NA);

        /* 	//inizializzazione cardinalit�
        for (unsigned int a = 0; a != NA; ++a)
        cardinalities[a]=0;

        //calcolo cardinalit�
        for (unsigned int a = 0; a != NA; ++a)
        {
        unsigned int block_idx = a / conf->num_threads;
        unsigned int thread_idx = a % conf->num_threads;
        unsigned int tmp_idx = block_idx * conf->num_threads * HB + thread_idx;
        unsigned int p = 0; // accumulated value
        for (unsigned int h = 0; p!= M && h != HB; ++h)
        {

        if (frequencies[tmp_idx])
        {
        p += frequencies[tmp_idx];
        //card[a].push_back(pair<unsigned int, register_t*>(p, histogram + tmp_idx * S));
        cardinalities[a]++;

      }
      tmp_idx += conf->num_threads;
    }
  }

  //LAURA
  verbose_cout<<"Cardinalities:\n";
  for (unsigned int a = 0; a != NA; ++a)
  //verbose_cout<<card[a].size()<<"\n";
  verbose_cout<<cardinalities[a]<<"\n"; */

  not_silent_cout << "Computing system statistics, please wait...\n";
  unsigned long long n_clusters = 1;
  // case r=N: compute H(U)
  register_t* clusters = (register_t*)malloc(CB * sample_size_bytes);
  unsigned int* cluster_sizes = (unsigned int*)malloc(CB * sizeof(unsigned int));
  register_t* cur_cluster = clusters;
  float* output = (float*)malloc(CB * sizeof(float));
  bool has_next;
  priority_queue<dci::ClusterDescriptor> q;
  unsigned long long to_subtract;

  //LAURA
  verbose_cout <<"sample_size_bytes: " << sample_size_bytes<<"\n";
  verbose_cout <<"CB: " << CB<<"\n";
  verbose_cout <<"NA: " << NA<<"\n";
  verbose_cout <<"N: " << N<<"\n";
  verbose_cout <<"S: " << S<<"\n";


  clock_t start = clock();

  //unsigned int clusterToCheck = 127;


  // store only one cluster
  dci::RegisterUtils::SetAllBits<1>(clusters, N, S);
  cluster_sizes[0] = NA; // just one NA-sized cluster
  unsigned int old_HB = HB, new_HB;


  // invoke kernel for whole system
  new_HB = callKernel<false,false, false,false>(1, clusters, cluster_sizes, output);

  //LAURA
  //for(int k=0; k<CB; k++)
  //if(clusters[k]>65535)
  //verbose_cout <<"clusters[k]: " << clusters[k]<<"\n";

  // reallocate histogram memory on device
  reallocateHistogramMemory(new_HB);

  verbose_cout << "uSystem's jointEntropy H(S,U-S) = H(U) = " << output[0] << endl;

  // this variable holds number of clusters in batch
  unsigned int C = 0;

  // cycle all cluster sizes up to N/2, or up to N if a mutual information mask is specified
  auto limit = has_mi_mask ? (NA-1) : (NA/2);
  for (int r = 1; r <= limit; r++) // cluster size
  {

    // initialize r-sized cluster mask generator
    dci::ClusterUtils::initializeClusterMaskGenerator(NA, r, S, agent_pool, N, mutual_information_mask);

    // cycle all r-sized clusters
    while (dci::ClusterUtils::getNextClusterMask(cur_cluster, has_next))
    {
      //LAURA
      //if(*cur_cluster>(unsigned int)65535)
      //verbose_cout<<"cur_cluster: "<<*cur_cluster<<"\n";

      // get complementary cluster and store it in the next memory block
      dci::ClusterUtils::getComplementaryClusterMask(cur_cluster + S, cur_cluster, N);

      // 2 clusters (original and complementary)
      cluster_sizes[C++] = r;
      cluster_sizes[C++] = dci::RegisterUtils::GetNumberOf<1>(cur_cluster + S, N) ? (NA - r) : 0;

      // if we have stored enough clusters for a batch, or if we have reached the end of current group
      if (C == CB || (r == limit && !has_next))
      {

        // invoke kernel for current batch
        if(conf->zi_index)
        {
          //callKernel<true,false,false, false>(C, clusters, cluster_sizes, output);
          callKernelCard<true,false,false, false>(C, clusters, cluster_sizes, output, cardinalities);
          //callKernelPool<true,false,false, false>(C, clusters, cluster_sizes, output, agent_pool);

        }
        else if(conf->strength_index)
        {
          //callKernel<false,true, false,false>(C, clusters, cluster_sizes, output);
          callKernelCard<false,true, false,false>(C, clusters, cluster_sizes, output, cardinalities);
          //callKernelPool<true,false,false, false>(C, clusters, cluster_sizes, output, agent_pool);

        }
        else if(conf->strength2_index)
        {
          //callKernel<false,false,true, false>(C, clusters, cluster_sizes, output);
          callKernelCard<false,false,true, false>(C, clusters, cluster_sizes, output, cardinalities);
          //callKernelPool<true,false,false, false>(C, clusters, cluster_sizes, output, agent_pool);

        }
        if ( conf->zi_index == false && conf->strength_index == false && conf->strength2_index == false)
        {
          conf->zi_index=true; // pick zi as default

          //callKernel<true,false,false, false>(C, clusters, cluster_sizes, output);
          callKernelCard<true,false,false, false>(C, clusters, cluster_sizes, output, cardinalities);
          //callKernelPool<true,false,false, false>(C, clusters, cluster_sizes, output, agent_pool);

        }

        // reset extra cluster count
        to_subtract = 0;

        // elaborate results
        for (unsigned int c = 0; c != C / 2; ++c)
        {
          if (cluster_sizes[2 * c] != 1) // 1-sized clusters are not relevant
          checkInsertClusterInResults(q, clusters + 2 * c * S, output[2 * c]);
          if (!has_mi_mask && cluster_sizes[2 * c + 1] != NA / 2) // in case of an even number of variables, avoid adding N/2-sized clusters twice; in case of systems with NA = 3, avoid adding 1-sized clusters
          checkInsertClusterInResults(q, clusters + (2 * c + 1) * S, output[2 * c + 1]);
          else to_subtract++;
        }

        // reset current cluster pointer
        cur_cluster = clusters;

        // update number of processed clusters
        n_clusters += (C - to_subtract);

        // print progress
        printProgress(start, n_clusters);

        // reset number of clusters
        C = 0;

      }
      else
      cur_cluster += S * 2; // move two blocks forward

    }

  }

  free(clusters);
  free(cluster_sizes);
  free(output);

  not_silent_cout << endl;

  if (conf->sieving)
  {
    not_silent_cout << "Performing sieving...";
    vector<register_t*> input_clusters(q.size());
    vector<register_t*> input_agent_clusters(q.size());
    vector<float> input_ind(q.size());
    unsigned int i = 0;
    while (q.size())
    {
      input_clusters[i] = ((dci::ClusterDescriptor)q.top()).cloneClusterMask();
      input_agent_clusters[i] = getOriginalAgentMaskFromCluster(input_clusters[i]);
      input_ind[i] = ((dci::ClusterDescriptor)q.top()).getIndex();
      i++;
      q.pop();
    }
    vector<register_t*> output;
    vector<float> output_ind;
    performSieving<true>(input_clusters, input_agent_clusters, input_ind, output, output_ind, K);

    // store output
    for (unsigned int j = 0; j != output.size(); ++j)
    {
      dci::ClusterDescriptor clusterDescriptor(output[j], N);
      clusterDescriptor.setIndex(output_ind[j]);
      q.push(clusterDescriptor);
    }

    // free memory - takes care of output too
    for (unsigned int j = 0; j != input_clusters.size(); ++j)
    {
      delete[] input_clusters[j];
      free(input_agent_clusters[j]);
    }

    not_silent_cout << "done\n";
  }

  verbose_cout << "q.size = " << q.size() << endl;
  int qsize = q.size();
  unique_ptr<vector<dci::ClusterDescriptor> > result_vector(new vector<dci::ClusterDescriptor>());
  result_vector->reserve(qsize);
  for (int j=1; j <= qsize; j++)
  {
    printAgentCluster((dci::ClusterDescriptor)q.top(), out, false);
    out << " " << ((dci::ClusterDescriptor)q.top()).getIndex() << endl;
    result_vector->push_back((dci::ClusterDescriptor)q.top());
    q.pop();
  }

  // reallocate histogram memory on device
  reallocateHistogramMemory(old_HB);

  return result_vector;

}

/*
* Calls kernel according to sample size, performing copy operations for input and output data
*/
template<bool ComputeZi, bool ComputeSI,bool ComputeSI_2, bool HistogramOnly>
inline unsigned int Application::callKernel(const unsigned int& C, const register_t* clusters, const unsigned int* cluster_sizes, float* output)
{

  // reset return value
  unsigned int ret = 0;

  // copy data to device
  HANDLE_ERROR( cudaMemcpy( dev_clusters, clusters, sample_size_bytes * C, cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( dev_cluster_size, cluster_sizes, sizeof(unsigned int) * C, cudaMemcpyHostToDevice ) );
  params.C = C;
  HANDLE_ERROR( cudaMemcpyToSymbol( SP, (void*)&params, sizeof(SystemParameters) ) );

  // reset device data
  HANDLE_ERROR( cudaMemset( (void*)dev_entropies, 0, sizeof(float) * C ) );
  HANDLE_ERROR( cudaMemset( (void*)dev_frequencies, 0, sizeof(unsigned int) * HB * CB ) );
  HANDLE_ERROR( cudaMemset( (void*)dev_next, 0, sizeof(register_t*) * HB * CB ) );

  // call kernel according to sample size
  switch (S)
  {
    case 1 :
    histo_kernel< 1, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 1><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel< 1, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel< 1, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 2 :
    histo_kernel< 2, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 2><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel< 2, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel< 2, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 3 :
    histo_kernel< 3, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 3><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel< 3, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel< 3, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 4 :
    histo_kernel< 4, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 4><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel< 4, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel< 4, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 5 :
    histo_kernel< 5, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 5><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel< 5, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel< 5, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 6 :
    histo_kernel< 6, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 6><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel< 6, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel< 6, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 7 :
    histo_kernel< 7, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 7><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel< 7, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel< 7, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 8 :
    histo_kernel< 8, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 8><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel< 8, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel< 8, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 9 :
    histo_kernel< 9, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 9><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel< 9, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel< 9, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 10:
    histo_kernel<10, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<10><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<10, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<10, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 11:
    histo_kernel<11, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<11><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<11, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<11, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 12:
    histo_kernel<12, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<12><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<12, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<12, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 13:
    histo_kernel<13, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<13><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<13, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<13, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 14:
    histo_kernel<14, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<14><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<14, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<14, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 15:
    histo_kernel<15, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<15><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<15, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<15, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 16:
    histo_kernel<16, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<16><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<16, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<16, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 17:
    histo_kernel<17, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<17><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<17, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<17, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 18:
    histo_kernel<18, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<18><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<18, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<18, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 19:
    histo_kernel<19, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<19><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<19, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<19, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 20:
    histo_kernel<20, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<20><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<20, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<20, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 21:
    histo_kernel<21, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<21><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<21, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<21, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 22:
    histo_kernel<22, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<22><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<22, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<22, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 23:
    histo_kernel<23, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<23><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<23, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<23, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 24:
    histo_kernel<24, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<24><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<24, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<24, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 25:
    histo_kernel<25, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<25><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<25, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<25, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 26:
    histo_kernel<26, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<26><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<26, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<26, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 27:
    histo_kernel<27, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<27><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<27, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<27, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 28:
    histo_kernel<28, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<28><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<28, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<28, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 29:
    histo_kernel<29, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<29><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<29, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<29, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 30:
    histo_kernel<30, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<30><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<30, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<30, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 31:
    histo_kernel<31, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<31><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<31, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<31, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
    case 32:
    histo_kernel<32, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<32><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel<32, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    else
    cluster_kernel<32, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output);
    break;
  }

  // copy results from device
  if (!HistogramOnly) HANDLE_ERROR( cudaMemcpy( output, dev_output, sizeof(float) * C, cudaMemcpyDeviceToHost ) );

  // copy count from device if necessary
  if (C == 1) HANDLE_ERROR( cudaMemcpy( (void*)&ret, dev_count, sizeof(unsigned int), cudaMemcpyDeviceToHost ) );

  // return number of different sample values
  return ret;

}


/*
* Cardinalities
* Calls kernel according to sample size, performing copy operations for input and output data
*/
template<bool ComputeZi, bool ComputeSI,bool ComputeSI_2, bool HistogramOnly>
inline unsigned int Application::callKernelCard(const unsigned int& C, const register_t* clusters, const unsigned int* cluster_sizes, float* output, unsigned int* cardinalities)
{

  // reset return value
  unsigned int ret = 0;

  // copy data to device
  HANDLE_ERROR( cudaMemcpy( dev_clusters, clusters, sample_size_bytes * C, cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( dev_cluster_size, cluster_sizes, sizeof(unsigned int) * C, cudaMemcpyHostToDevice ) );
  params.C = C;
  HANDLE_ERROR( cudaMemcpyToSymbol( SP, (void*)&params, sizeof(SystemParameters) ) );

  //LAURA
  HANDLE_ERROR( cudaMemcpy( dev_cardinalities, cardinalities, sizeof(unsigned int) * NA, cudaMemcpyHostToDevice ) );

  // reset device data
  HANDLE_ERROR( cudaMemset( (void*)dev_entropies, 0, sizeof(float) * C ) );
  HANDLE_ERROR( cudaMemset( (void*)dev_frequencies, 0, sizeof(unsigned int) * HB * CB ) );
  HANDLE_ERROR( cudaMemset( (void*)dev_next, 0, sizeof(register_t*) * HB * CB ) );

  // call kernel according to sample size
  switch (S)
  {
    case 1 :
    histo_kernel< 1, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 1><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card< 1, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card< 1, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 2 :
    histo_kernel< 2, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 2><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card< 2, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card< 2, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 3 :
    histo_kernel< 3, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 3><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card< 3, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card< 3, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 4 :
    histo_kernel< 4, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 4><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card< 4, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card< 4, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 5 :
    histo_kernel< 5, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 5><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card< 5, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card< 5, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 6 :
    histo_kernel< 6, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 6><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card< 6, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card< 6, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 7 :
    histo_kernel< 7, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 7><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card< 7, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card< 7, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 8 :
    histo_kernel< 8, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 8><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card< 8, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card< 8, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 9 :
    histo_kernel< 9, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 9><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card< 9, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card< 9, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 10:
    histo_kernel<10, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<10><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<10, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<10, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 11:
    histo_kernel<11, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<11><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<11, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<11, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 12:
    histo_kernel<12, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<12><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<12, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<12, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 13:
    histo_kernel<13, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<13><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<13, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<13, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 14:
    histo_kernel<14, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<14><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<14, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<14, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 15:
    histo_kernel<15, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<15><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<15, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<15, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 16:
    histo_kernel<16, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<16><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<16, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<16, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 17:
    histo_kernel<17, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<17><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<17, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<17, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 18:
    histo_kernel<18, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<18><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<18, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<18, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 19:
    histo_kernel<19, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<19><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<19, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<19, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 20:
    histo_kernel<20, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<20><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<20, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<20, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 21:
    histo_kernel<21, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<21><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<21, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<21, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 22:
    histo_kernel<22, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<22><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<22, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<22, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 23:
    histo_kernel<23, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<23><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<23, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<23, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 24:
    histo_kernel<24, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<24><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<24, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<24, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 25:
    histo_kernel<25, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<25><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<25, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<25, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 26:
    histo_kernel<26, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<26><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<26, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<26, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 27:
    histo_kernel<27, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<27><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<27, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<27, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 28:
    histo_kernel<28, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<28><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<28, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<28, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 29:
    histo_kernel<29, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<29><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<29, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<29, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 30:
    histo_kernel<30, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<30><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<30, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<30, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 31:
    histo_kernel<31, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<31><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<31, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<31, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
    case 32:
    histo_kernel<32, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<32><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_card<32, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    else
    cluster_kernel_card<32, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_cardinalities);
    break;
  }

  // copy results from device
  if (!HistogramOnly) HANDLE_ERROR( cudaMemcpy( output, dev_output, sizeof(float) * C, cudaMemcpyDeviceToHost ) );

  // copy count from device if necessary
  if (C == 1) HANDLE_ERROR( cudaMemcpy( (void*)&ret, dev_count, sizeof(unsigned int), cudaMemcpyDeviceToHost ) );

  // return number of different sample values
  return ret;

}


/*
* Cardinalities
* Calls kernel according to sample size, performing copy operations for input and output data
*/
template<bool ComputeZi, bool ComputeSI,bool ComputeSI_2, bool HistogramOnly>
inline unsigned int Application::callKernelPool(const unsigned int& C, const register_t* clusters, const unsigned int* cluster_sizes, float* output, register_t* agent_pool)
{

  // reset return value
  unsigned int ret = 0;

  // copy data to device
  HANDLE_ERROR( cudaMemcpy( dev_clusters, clusters, sample_size_bytes * C, cudaMemcpyHostToDevice ) );
  HANDLE_ERROR( cudaMemcpy( dev_cluster_size, cluster_sizes, sizeof(unsigned int) * C, cudaMemcpyHostToDevice ) );
  params.C = C;
  HANDLE_ERROR( cudaMemcpyToSymbol( SP, (void*)&params, sizeof(SystemParameters) ) );

  //LAURA
  HANDLE_ERROR( cudaMemcpy( dev_agent_pool, agent_pool, agent_pool_size_bytes, cudaMemcpyHostToDevice ) );

  // reset device data
  HANDLE_ERROR( cudaMemset( (void*)dev_entropies, 0, sizeof(float) * C ) );
  HANDLE_ERROR( cudaMemset( (void*)dev_frequencies, 0, sizeof(unsigned int) * HB * CB ) );
  HANDLE_ERROR( cudaMemset( (void*)dev_next, 0, sizeof(register_t*) * HB * CB ) );

  // call kernel according to sample size
  switch (S)
  {
    case 1 :
    histo_kernel< 1, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 1><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool< 1, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool< 1, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 2 :
    histo_kernel< 2, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 2><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool< 2, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool< 2, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 3 :
    histo_kernel< 3, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 3><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool< 3, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool< 3, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 4 :
    histo_kernel< 4, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 4><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool< 4, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool< 4, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 5 :
    histo_kernel< 5, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 5><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool< 5, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool< 5, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 6 :
    histo_kernel< 6, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 6><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool< 6, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool< 6, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 7 :
    histo_kernel< 7, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 7><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool< 7, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool< 7, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 8 :
    histo_kernel< 8, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 8><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool< 8, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool< 8, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 9 :
    histo_kernel< 9, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel< 9><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool< 9, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool< 9, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 10:
    histo_kernel<10, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<10><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<10, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<10, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 11:
    histo_kernel<11, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<11><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<11, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<11, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 12:
    histo_kernel<12, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<12><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<12, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<12, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 13:
    histo_kernel<13, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<13><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<13, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<13, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 14:
    histo_kernel<14, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<14><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<14, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<14, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 15:
    histo_kernel<15, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<15><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<15, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<15, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 16:
    histo_kernel<16, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<16><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<16, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<16, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 17:
    histo_kernel<17, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<17><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<17, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<17, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 18:
    histo_kernel<18, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<18><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<18, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<18, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 19:
    histo_kernel<19, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<19><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<19, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<19, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 20:
    histo_kernel<20, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<20><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<20, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<20, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 21:
    histo_kernel<21, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<21><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<21, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<21, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 22:
    histo_kernel<22, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<22><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<22, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<22, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 23:
    histo_kernel<23, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<23><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<23, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<23, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 24:
    histo_kernel<24, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<24><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<24, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<24, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 25:
    histo_kernel<25, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<25><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<25, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<25, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 26:
    histo_kernel<26, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<26><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<26, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<26, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 27:
    histo_kernel<27, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<27><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<27, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<27, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 28:
    histo_kernel<28, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<28><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<28, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<28, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 29:
    histo_kernel<29, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<29><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<29, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<29, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 30:
    histo_kernel<30, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<30><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<30, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<30, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 31:
    histo_kernel<31, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<31><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<31, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<31, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
    case 32:
    histo_kernel<32, HistogramOnly><<<num_blocks, num_threads>>>(dev_system, dev_clusters, dev_histogram, dev_frequencies, dev_next);
    entropy_kernel<32><<<num_blocks, num_threads>>>(dev_frequencies, dev_entropies, C == 1 ? dev_count : 0);
    if (compute_mutual_information)
    cluster_kernel_agent_pool<32, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, true><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    else
    cluster_kernel_agent_pool<32, ComputeZi, ComputeSI, ComputeSI_2, HistogramOnly, false><<<num_blocks, num_threads>>>(dev_system_entropies, dev_hsystem_stats, dev_clusters, dev_cluster_size, dev_entropies, dev_output, dev_agent_pool);
    break;
  }

  // copy results from device
  if (!HistogramOnly) HANDLE_ERROR( cudaMemcpy( output, dev_output, sizeof(float) * C, cudaMemcpyDeviceToHost ) );

  // copy count from device if necessary
  if (C == 1) HANDLE_ERROR( cudaMemcpy( (void*)&ret, dev_count, sizeof(unsigned int), cudaMemcpyDeviceToHost ) );

  // return number of different sample values
  return ret;

}



/*
* Prints all data to given output stream
*/
void Application::printSystemDataToStream(ostream& out, register_t* data)
{
  for (int row = 0; row != M; ++row)
  {
    for (int col = 0; col != N; ++col)
    out << dci::RegisterUtils::GetBitAtPos(data, col);
    out << '\n';
    data += S;
  }
}

/*
* Prints all stats to given output stream
*/
void Application::printSystemStatsToStream(ostream& out, const float* stats)
{
  for (int col = 0; col != N; ++col)
  out << stats[2 * col] << endl << stats[2 * col + 1] << endl;
}

/*
* Generates a homogeneous system according to the statistical distribution
* of the starting system, using given random seed
*/
//    void Application::generateHomogeneousSystem()
//    {
//
//        vector<vector<pair<unsigned int, register_t*> > > values(NA); // packed histogram
//
//        // fill packed histogram
//        for (unsigned int a = 0; a != NA; ++a)
//        {
//            unsigned int block_idx = a / conf->num_threads;
//            unsigned int thread_idx = a % conf->num_threads;
//            unsigned int tmp_idx = block_idx * conf->num_threads * HB + thread_idx;
//            unsigned int p = 0; // accumulated value
//            for (unsigned int h = 0; p!= M && h != HB; ++h)
//            {
//
//                if (frequencies[tmp_idx])
//                {
//                    p += frequencies[tmp_idx];
//                    values[a].push_back(pair<unsigned int, register_t*>(p, histogram + tmp_idx * S));
//                }
//                tmp_idx += conf->num_threads;
//            }
//        }
//
//        mt19937 rng(conf->rand_seed); // set random seed
//        uniform_int_distribution<unsigned int> gen(0, M - 1); // setup random number generator
//
//        // for each sample
//        for (unsigned int row = 0; row != M; ++row)
//        {
//
//            register_t* sample = hsystem_data + row * S;
//
//            // reset sample
//            dci::RegisterUtils::SetAllBits<0>(sample, N, S);
//
//            // for each agent
//            for (unsigned int a = 0; a != NA; ++a)
//            {
//
//                // get random number between 0 and M - 1
//                unsigned int val = gen(rng);
//
//                // cycle all histogram values
//                for (auto pp : values[a])
//                    if (val < pp.first)
//                    {
//                        dci::RegisterUtils::reg_or(sample, pp.second, S);
//                        break;
//                    }
//
//            }
//
//        }
//
//    }

/*
*
*/
inline void Application::printCluster(const register_t* cluster)
{
  for (unsigned int i = 0; i!= N; ++i)
  if (dci::RegisterUtils::GetBitAtPos(cluster, i))
  cout << '[' << i << ']';
}

/*
*
*/
inline void Application::printClusterValue(const register_t* cluster, const register_t* value)
{
  for (unsigned int i = 0; i!= N; ++i)
  if (dci::RegisterUtils::GetBitAtPos(cluster, i))
  cout << dci::RegisterUtils::GetBitAtPos(value, i);
  else
  cout << 'x';
}

/*
* Prints single cluster debug info
*/
void Application::debugCluster(register_t* cluster, register_t* dev_histogram, unsigned int* dev_frequencies/*, float* dev_entropies, float* dev_comp_entropies, float* dev_system_entropies, float* dev_hsystem_stats*/)
{
  printCluster(cluster); cout << endl;
  register_t* histo = (register_t*)malloc(HB * sample_size_bytes);
  unsigned int* freq = (unsigned int*)malloc(HB * sizeof(unsigned int));
  /*float entropy, compEntropy;
  float* sysent = (float*)malloc((N + 1) * sizeof(float));
  float* hsys = (float*)malloc(2 * N * sizeof(float));
  HANDLE_ERROR( cudaMemcpy( &entropy, dev_entropies, sizeof(float), cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy( &compEntropy, dev_comp_entropies, sizeof(float), cudaMemcpyDeviceToHost ) );

  HANDLE_ERROR( cudaMemcpy( sysent, dev_system_entropies, (N + 1) * sizeof(float), cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy( hsys, dev_hsystem_stats, 2 * N * sizeof(float), cudaMemcpyDeviceToHost ) );*/
  HANDLE_ERROR( cudaMemcpy( histo, dev_histogram, HB * sample_size_bytes, cudaMemcpyDeviceToHost ) );
  HANDLE_ERROR( cudaMemcpy( freq, dev_frequencies, HB * sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
  float ent = 0.0f;
  for (unsigned int i = 0; i != HB; ++i)
  if (freq[i])
  {
    printClusterValue(cluster, histo + i * S);
    float f = (float)freq[i] / (float)M;
    f = -f*log2f(f); ent += f;
    cout << " -> " << freq[i] << ' ' << f << endl;
  }
  //cout << "ent=" << ent << " (on dev=" << entropy << ")" << endl;
  /*ent = -ent;
  int r = 0;
  for (unsigned int i = 0; i != N; ++i)
  if (dci::RegisterUtils::GetBitAtPos(cluster, i))
  {
  ent += sysent[i];
  ++r;
}
cout << "I=" << ent << endl;
float mi = entropy + compEntropy - sysent[N];
cout << "MI=" << mi << endl;
cout << "C=" << (ent / mi) << endl;
cout <<"|C|=" << hsys[2*r-2] << ", sigma=" << hsys[2*r-1] << endl;
cout << "TC=" << (((ent / mi) - hsys[2*r-2]) / hsys[2*r-1]) << endl;*/
free(histo);
free(freq);
//free(sysent);
//free(hsys);
}

/*
* Tuning function
*/
void Application::tuneFunction(const string& function_name, const clock_t& execution_time)
{
  if (!conf->tune) return;
  if (tuning_map.count(function_name) == 0)
  {
    tuning_map[function_name].first = 1;
    tuning_map[function_name].second = execution_time > 0 ? execution_time : 0;
  }
  else
  {
    tuning_map[function_name].first++;
    tuning_map[function_name].second += execution_time > 0 ? execution_time : 0;
  }
}

/*
* Prints tuning info collected during execution
*/
void Application::printTuningInfo()
{
  if (!conf->tune) return;
  for (auto f : tuning_map)
  cout << f.first << ": " << f.second.first << " calls, "
  << ((float)f.second.second / (float)CLOCKS_PER_SEC * 1000.0f) << "ms total" << endl;
}

/*
* Prints an agent cluster to out
*/
void Application::printAgentCluster(const dci::ClusterDescriptor& cluster, ostream& out, bool tab_format)
{

  // allocate temp memory
  register_t* temp = getCurrentAgentMaskFromCluster(cluster.getClusterMask());

  // cycle all agents
  for (unsigned int a = 0; a != NA; ++a)
  {

    // Check number of 1s
    bool agent_check = dci::RegisterUtils::GetBitAtPos(temp, a);

    // check output format
    if (tab_format)
    out << agent_check << '\t';
    else if (agent_check)
    out << '[' << getAgentName(a) << ']';

  }

  // free temp memory
  free(temp);

}

/*
* Prints original agent cluster name to out
*/
string Application::getOriginalAgentClusterName(const dci::ClusterDescriptor& cluster)
{

  // allocate temp memory
  register_t* temp = getOriginalAgentMaskFromCluster(cluster.getClusterMask());
  string name = "";

  // cycle all agents that compose the cluster
  for (unsigned int a = 0; a != NO; ++a)
  if (dci::RegisterUtils::GetBitAtPos(temp, a))
  {

    // update name, size and composition
    if (name.length()) name += '+';
    name += getAgentName(a, starting_agent_names);

  }

  // free temp memory
  free(temp);

  // return agent name
  return name;

}

/*
* Performs sieving algorithm on given array of clusters. Input vectors must be ordered by tc (ascending or descending), output vector will have the same ordering
*/
template<bool ascending, bool realloc> void Application::performSieving(const vector<register_t*>& input_clusters, const vector<register_t*>& input_agent_clusters, const vector<float>& input_ind, vector<register_t*>& output, vector<float>& output_ind, const unsigned int& max_output_count)
{

  // get parameters
  unsigned int input_size = input_clusters.size();
  unsigned int final_size = input_size;
  vector<bool> removed(input_size, false);

  // skip last element of sequence
  for (unsigned int i = 0; i != input_size - 1; ++i)
  {
    unsigned int real_i = ascending ? (input_size - i - 1) : i;
    if (!removed[real_i]) // ignore removed elements
    for (unsigned int j = 0; j != real_i; ++j)
    {
      if (!removed[j] && dci::RegisterUtils::are_subsets(input_agent_clusters[real_i], input_agent_clusters[j], NO ? SO : SA))
      {
        removed[j] = true;
        final_size--;
      }
    }
  }

  // adjust final size
  if (final_size > max_output_count) final_size = max_output_count;

  // set start index and direction
  unsigned int start = ascending ? (input_size - 1) : 0;
  unsigned int last = ascending ? 0 : (input_size - 1);

  // resize output vectors
  output.resize(final_size, 0);
  output_ind.resize(final_size, 0.0f);

  // add elements to output
  for (unsigned int i = 0; i != final_size; ++i)
  {

    // get to next element to add
    while (removed[start] && start != last) start = ascending ? (start - 1) : (start + 1);

    // add element if present
    if (!removed[start])
    {
      output_ind[ascending ? (final_size - i - 1) : i] = input_ind[start];
      if (realloc)
      {
        output[ascending ? (final_size - i - 1) : i] = (register_t*)malloc((NO ? SO : SA) * sizeof(register_t));
        memcpy(output[ascending ? (final_size - i - 1) : i], input_clusters[start], (NO ? SO : SA) * sizeof(register_t));
      }
      else
      output[ascending ? (final_size - i - 1) : i] = input_clusters[start];
    }

    // move to next element
    if (start != last) start = ascending ? (start - 1) : (start + 1);

  }

}

/*
* Selects super clusters from a list of output clusters, using system configuration values
*/
template<bool ascending> void Application::SelectSuperClustersAndSave(const vector<register_t*>& results, const vector<float>& zi_values)
{

  // check input size
  if (!results.size()) return;

  // retrieve translated clusters
  auto descriptors = getTranslatedClusterDescriptors(results, zi_values);

  // select super clusters
  auto super_clusters = selectSuperClusters<ascending>(descriptors);

  // write new system to file
  writeSystemToFileAfterSieving(super_clusters, conf->sieving_out);

}


/*
* Performs sieving on an arbitrary set of clusters, given their Tc values. Automatically translates packed clusters in unpacked ones.
*/
template<bool ascending> void Application::ApplySievingAlgorithm(const vector<register_t*>& input_clusters, const vector<float>& input_ind, vector<register_t*>& output, vector<float>& output_ind, const unsigned int& max_output_count)
{

  if (!input_clusters.size()) return;

  vector<register_t*> translated_clusters(input_clusters.size());
  vector<register_t*> input_agent_clusters(input_clusters.size());

  // reset last reg mask for cluster generation
  dci::ClusterUtils::resetLastRegMask(N);

  for (unsigned int i = 0; i != input_clusters.size(); ++i)
  {

    // add cluster to list
    translated_clusters[i] = (register_t*)malloc(S * sizeof(register_t));
    dci::ClusterUtils::clusterBitmaskFromAgentCluster(translated_clusters[i], input_clusters[i], N, S, NA, agent_pool);

    // set cluster mask according to original agents
    input_agent_clusters[i] = getOriginalAgentMaskFromCluster(translated_clusters[i]);

  }

  performSieving<ascending, true>(translated_clusters, input_agent_clusters, input_ind, output, output_ind, max_output_count);

  // free memory
  for (unsigned int j = 0; j != input_clusters.size(); ++j)
  {
    free(translated_clusters[j]);
    free(input_agent_clusters[j]);
  }

}

/*
* Selects super-clusters from results according to selected sieving mode
*/
template<bool ascending> unique_ptr<vector<dci::ClusterDescriptor> > Application::selectSuperClusters(const unique_ptr<vector<dci::ClusterDescriptor> >& results)
{

  // create result vector
  unique_ptr<vector<dci::ClusterDescriptor> > clusters(new vector<dci::ClusterDescriptor>());
  unsigned int n_res = results->size();

  // according to sieving mode
  switch (conf->sieving_mode)
  {

    // fixed number of clusters
    case 4:

    // manual input
    cout << endl << "Insert number of top clusters to keep and press enter: ";
    cin >> conf->sieving_keep_top;

    case 1:

    // add top clusters to results
    for (unsigned int i = 0; i != conf->sieving_keep_top && i != n_res; ++i)
    if ((results->at(ascending ? (n_res - i - 1) : i)).getIndex() > 0.0f)
    clusters->push_back(results->at(ascending ? (n_res - i - 1) : i));

    break;

    // differential: keep top clusters + search for differentials among sets of adjacent clusters
    case 2:

    // add top clusters to results
    for (unsigned int i = 0; i != conf->sieving_keep_top && i != n_res; ++i)
    if ((results->at(ascending ? (n_res - i - 1) : i)).getIndex() > 0.0f)
    clusters->push_back(results->at(ascending ? (n_res - i - 1) : i));

    // starting from first excluded cluster, compute differences
    for (unsigned int i = conf->sieving_keep_top; i < n_res - 1; ++i)
    {
      bool found = false;
      for (unsigned int j = i + 1; !found && j <= i + conf->sieving_diff_num && j < n_res; ++j)
      {
        float zi1 = (results->at(ascending ? (n_res - i - 1) : i)).getIndex();
        float zi2 = (results->at(ascending ? (n_res - j - 1) : j)).getIndex();
        if (zi1 > 0.0f && (zi2 <= 0.0f || 100.0f * (zi1 - zi2) / zi2 >= conf->sieving_diff))
        {
          clusters->push_back(results->at(ascending ? (n_res - i - 1) : i));
          found = true;
        }
      }
      if (!found) break;
    }

    break;

    // all clusters above mean value in result set
    case 3:

    // stores mean value
    float mean = 0.0f;

    // first compute mean value
    for (unsigned int i = 0; i != n_res; ++i)
    mean += (results->at(i)).getIndex();

    // normalize it
    mean /= (float)n_res;

    // only with mean > 0
    if (mean > 0.0f)
    {

      // cycle all results
      for (unsigned int i = 0; i != n_res; ++i)
      if (mean < (results->at(i)).getIndex())
      clusters->push_back(results->at(i));

    }

    break;

  }

  // return result set
  return clusters;

}

/*
* Allocates and sets original agent mask from cluster
*/
register_t* Application::getOriginalAgentMaskFromCluster(const register_t* cluster)
{

  // allocate output
  register_t* output = (register_t*)malloc((NO ? SO : SA) * BYTES_PER_REG);
  register_t* temp_and = (register_t*)malloc(S * BYTES_PER_REG);

  // reset registers
  dci::RegisterUtils::SetAllBits<0>(output, NO ? NO : NA, NO ? SO : SA);
  dci::RegisterUtils::SetAllBits<0>(temp_and, N, S);

  // build mask from starting agents
  for (unsigned int a = 0; a != NA; ++a)
  {

    // bitwise and
    dci::RegisterUtils::reg_and(temp_and, cluster, agent_pool + S * a, S);

    // count 1s
    if (dci::RegisterUtils::GetNumberOf<1>(temp_and, N) > 0)
    {

      // check if an original system is defined
      if (NO)
      dci::RegisterUtils::reg_or(output, starting_agent_pool + a * SO, SO);
      else
      dci::RegisterUtils::SetBitAtPos(output, a, 1);

    }

  }

  // free temp data
  free(temp_and);

  // return allocated data
  return output;

}

/*
* Allocates and sets current agent mask from cluster
*/
register_t* Application::getCurrentAgentMaskFromCluster(const register_t* cluster)
{

  // allocate output
  register_t* output = (register_t*)malloc(SA * BYTES_PER_REG);
  register_t* temp_and = (register_t*)malloc(S * BYTES_PER_REG);

  // reset registers
  dci::RegisterUtils::SetAllBits<0>(output, NA, SA);
  dci::RegisterUtils::SetAllBits<0>(temp_and, N, S);

  // build mask from starting agents
  for (unsigned int a = 0; a != NA; ++a)
  {

    // bitwise and
    dci::RegisterUtils::reg_and(temp_and, cluster, agent_pool + S * a, S);

    // count 1s
    if (dci::RegisterUtils::GetNumberOf<1>(temp_and, N) > 0)
    dci::RegisterUtils::SetBitAtPos(output, a, 1);

  }

  // free temp data
  free(temp_and);

  // return allocated data
  return output;

}

/*
* Pushes cluster info to the back of each array
*/
void Application::pushFullClusterInfo(const register_t* cluster_orig, vector<string>& temp_agent_names, vector<string>& temp_agent_rep, vector<unsigned int>& temp_agent_sizes, vector<vector<unsigned int> >& temp_agent_comp)
{

  // variable declarations
  string name = "";
  unsigned int size = 0;
  unsigned int ref_na = NO ? NO : NA;
  unsigned int ref_n = NBO ? NBO : N;
  unsigned int ref_s = NBO ? SBO : S;
  register_t* ref_agent_pool = NO ? original_agent_pool : agent_pool;
  temp_agent_comp.emplace_back();
  vector<unsigned int>* comp = &(temp_agent_comp[temp_agent_comp.size() - 1]);
  string rep(ref_na, '0');

  // cycle all agents that compose the cluster
  for (unsigned int a = 0; a != ref_na; ++a)
  if (dci::RegisterUtils::GetBitAtPos(cluster_orig, a))
  {

    // update name, size and composition
    if (name.length()) name += '+';
    name += getAgentName(a, NO ? starting_agent_names : agent_names);
    size += dci::RegisterUtils::GetNumberOf<1>(ref_agent_pool + a * ref_s, ref_n);
    comp->push_back(a);
    rep[a] = '1';

  }

  // push back remaining info
  temp_agent_names.push_back(name);
  temp_agent_sizes.push_back(size);
  temp_agent_rep.push_back(rep);

}

/*
* Pushes missing agent info to the back of each array
*/
void Application::pushMissingAgentInfo(const unsigned int& a, vector<string>& temp_agent_names, vector<string>& temp_agent_rep, vector<unsigned int>& temp_agent_sizes, vector<vector<unsigned int> >& temp_agent_comp)
{

  // variable declarations
  unsigned int ref_na = NO ? NO : NA;
  unsigned int ref_n = NBO ? NBO : N;
  unsigned int ref_s = NBO ? SBO : S;
  register_t* ref_agent_pool = NO ? original_agent_pool : agent_pool;
  string rep(ref_na, '0');
  rep[a] = '1';

  // push back all info
  temp_agent_comp.emplace_back(1, a);
  temp_agent_names.push_back(getAgentName(a, NO ? starting_agent_names : agent_names));
  temp_agent_sizes.push_back(dci::RegisterUtils::GetNumberOf<1>(ref_agent_pool + a * ref_s, ref_n));
  temp_agent_rep.push_back(rep);

}

/*
* Translates agent clusters into full clusters and stores them in descriptors
*/
unique_ptr<vector<dci::ClusterDescriptor> > Application::getTranslatedClusterDescriptors(const vector<register_t*>& agent_clusters)
{

  // create vector of descriptors
  unique_ptr<vector<dci::ClusterDescriptor> > descriptors(new vector<dci::ClusterDescriptor>());

  // reset last reg mask for cluster generation
  dci::ClusterUtils::resetLastRegMask(N);

  for (unsigned int i = 0; i != agent_clusters.size(); ++i)
  {

    // add cluster to list
    register_t* translated_cluster = (register_t*)malloc(S * sizeof(register_t));
    dci::ClusterUtils::clusterBitmaskFromAgentCluster(translated_cluster, agent_clusters[i], N, S, NA, agent_pool);

    // add descriptor
    descriptors->emplace_back(translated_cluster, N);

    // free temp memory
    free(translated_cluster);

  }

  return descriptors;

}

/*
* Translates agent clusters into full clusters and stores them in descriptors, adding given Tc values
*/
unique_ptr<vector<dci::ClusterDescriptor> > Application::getTranslatedClusterDescriptors(const vector<register_t*>& agent_clusters, const vector<float>& zi_values)
{

  // create vector of descriptors
  unique_ptr<vector<dci::ClusterDescriptor> > descriptors(new vector<dci::ClusterDescriptor>());

  // reset last reg mask for cluster generation
  dci::ClusterUtils::resetLastRegMask(N);

  for (unsigned int i = 0; i != agent_clusters.size(); ++i)
  {

    // add cluster to list
    register_t* translated_cluster = (register_t*)malloc(S * sizeof(register_t));
    dci::ClusterUtils::clusterBitmaskFromAgentCluster(translated_cluster, agent_clusters[i], N, S, NA, agent_pool);

    // add descriptor
    descriptors->emplace_back(translated_cluster, N, zi_values[i]);

    // free temp memory
    free(translated_cluster);

  }

  return descriptors;

}

/*
* Writes new system data to file after sieving
*/
void Application::writeSystemToFileAfterSieving(const unique_ptr<vector<dci::ClusterDescriptor> >& super_clusters, const string& output_path)
{

  // declare registers for agent computations
  unsigned int num_super_clusters = super_clusters->size();
  vector<string> temp_agent_names;
  vector<string> temp_agent_rep;
  vector<unsigned int> temp_agent_sizes;
  vector<vector<unsigned int> > temp_agent_comp;
  unsigned int new_n = 0;
  unsigned int new_na = 0;
  register_t* ref_system_data = NBO ? original_system_data : system_data;
  register_t* ref_agent_pool = NBO ? original_agent_pool : agent_pool;
  unsigned int ref_NA = NO ? NO : NA;
  unsigned int ref_SA = NO ? SO : SA;
  unsigned int ref_N = NBO ? NBO : N;
  unsigned int ref_S = NBO ? SBO : S;
  register_t* temp_or = (register_t*)malloc(ref_SA * BYTES_PER_REG);

  // reset temp mask
  dci::RegisterUtils::SetAllBits<0>(temp_or, NA, SA);

  // cycle new super clusters
  for (unsigned int i = 0; i != num_super_clusters; ++i)
  {

    // retrieve super cluster mask
    register_t* cur_mask = getOriginalAgentMaskFromCluster((super_clusters->at(i)).getClusterMask());

    // push cluster info
    pushFullClusterInfo(cur_mask, temp_agent_names, temp_agent_rep, temp_agent_sizes, temp_agent_comp);

    // update global mask with agent
    dci::RegisterUtils::reg_or(temp_or, cur_mask, ref_SA);

    // update total number of variables and agents in new system
    new_n += temp_agent_sizes.back();
    new_na++;

    // free memory
    free(cur_mask);

  }

  // now we have to add missing agents
  unsigned int missing_agents = dci::RegisterUtils::GetNumberOf<0>(temp_or, ref_NA);

  // if there are missing agents, resize new agent pool
  if (missing_agents)
  {

    // now create new agents (size 1, one for each missing agent)
    for (unsigned int i = 0; i != ref_NA; ++i)
    if (!dci::RegisterUtils::GetBitAtPos(temp_or, i))
    {

      // push back agent info
      pushMissingAgentInfo(i, temp_agent_names, temp_agent_rep, temp_agent_sizes, temp_agent_comp);

      // update total number of variables and agents in new system
      new_n += temp_agent_sizes.back();
      new_na++;

    }

  }

  // open file
  ofstream out(output_path);

  // write start of first row
  out << "%% ";

  // now cycle all the agents of the new system, print agent names
  for (unsigned int i = 0; i != temp_agent_names.size(); ++i)
  out << temp_agent_names[i] << ' ';

  // write middle section of first row
  out << "%%";

  // write starting system agent names
  if (NO)
  for (unsigned int i = 0; i!= NO; ++i)
  out << ' ' << starting_agent_names[i];
  else
  for (unsigned int i = 0; i!= NA; ++i)
  out << ' ' << getAgentName(i);

  // write second separator
  out << " %%";

  if (NBO)
  for (unsigned int i = 0; i!= NO; ++i)
  { out << ' '; printBitmask(original_agent_pool + i * SBO, NBO, out); }
  else
  for (unsigned int i = 0; i!= NA; ++i)
  { out << ' '; printBitmask(agent_pool + i * S, N, out); }

  // end of line
  out << '\n';

  // set starting position
  unsigned int pos = 0;

  // write new system agent bitmasks
  for (unsigned int i = 0; i != new_na; ++i)
  {

    // write bitmask
    for (unsigned int j = 0; j != new_n; ++j)
    out << ((j >= pos && j < pos + temp_agent_sizes[i]) ? '1' : '0');

    // advance starting pos
    pos += temp_agent_sizes[i];

    // write separator
    out << " %% " << temp_agent_rep[i] << endl;

  }

  // separator
  out << "%%\n";

  // now write actual system data
  for (unsigned int i = 0; i != M; ++i)
  {

    // retrieve current sample
    register_t* cur_sample = ref_system_data + ref_S * i;

    // iterate new agents
    for (unsigned int j = 0; j != new_na; ++j)
    {

      // iterate all sub-agents
      for (unsigned int k = 0; k != temp_agent_comp[j].size(); ++k)
      for (unsigned int l = 0; l != ref_N; ++l)
      if (dci::RegisterUtils::GetBitAtPos(ref_agent_pool + ref_S * temp_agent_comp[j][k], l))
      out << (dci::RegisterUtils::GetBitAtPos(cur_sample, l) ? '1' : '0');

    }

    // separator
    out << " %% ";

    // if original system was defined
    if (NBO)
    printBitmask(original_system_data + SBO * i, NBO, out);
    else
    printBitmask(system_data + S * i, N, out);

    // newline
    out << '\n';

  }

  // free memory
  free(temp_or);

}

/*
* Returns a-th agent name if present in given list, the string 'a' otherwise
*/
string Application::getAgentName(unsigned int a, const vector<string>& ref_agent_names)
{

  // check agent names
  if (ref_agent_names.size() > a) return ref_agent_names[a];
  return to_string(a);

}

/*
* Returns a-th agent name if present, the string 'a' otherwise
*/
string Application::getAgentName(unsigned int a)
{

  // check agent names
  if (agent_names.size() > a) return agent_names[a];
  return to_string(a);

}

/*
* Prints a bitmask to out
*/
inline void Application::printBitmask(const register_t* mask, const unsigned int n, ostream& out)
{
  for (unsigned int i = 0; i != n; ++i)
  out << (dci::RegisterUtils::GetBitAtPos(mask, i) ? '1' : '0');
}

/*
* Reallocates device memory used for histogram computations according to maximum histogram size
*/
void Application::reallocateHistogramMemory(const unsigned int& new_HB)
{

  // free allocated data first
  HANDLE_ERROR( cudaFree(dev_histogram) );
  HANDLE_ERROR( cudaFree(dev_frequencies) );
  HANDLE_ERROR( cudaFree(dev_next) );

  // set new size
  HB = new_HB;
  params.L = HB;

  // allocate data according to new size
  HANDLE_ERROR( cudaMalloc( (void**)&dev_histogram, CB * HB * sample_size_bytes ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_frequencies, CB * HB * sizeof(unsigned int) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&dev_next, CB * HB * sizeof(register_t*) ) );

}

}

#endif /* DCI_H */
