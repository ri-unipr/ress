
/*
* File:   file_utils.h
* Author: Emilio Vicari, Michele Amoretti
*/

#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include "register_utils.h"
#include "common.h"
#include "cluster_descriptor.h"

using namespace std;

namespace dci
{
  namespace FileUtils
  {

    /*
    * Computes file size in bytes using C system functions
    */
    long fileSize(const char* filename);

    /*
    * Counts number of characters in string
    */
    unsigned int count(const string& str, const char& c);

    /*
    * Collects system parameters examining first sample and file size
    */
    void collectSystemParameters(const string& filename, unsigned int& N, unsigned int& M, unsigned int& S, unsigned long long& L,
      unsigned int& NA, unsigned int& SA, unsigned int& NO, unsigned int& SO, unsigned int& NBO, unsigned int& SBO, bool& has_mi_mask);

    /*
    * Load system data from file
    */
    void loadSystemData(const string& filename, const unsigned int& N, const unsigned int& M,
      const unsigned int& S, const unsigned int& SA, const unsigned int& SO, const unsigned int& SBO,
      const unsigned int& NA, const unsigned int& NO, const unsigned int& NBO,
      register_t* system_data, register_t* original_system_data,
      register_t* agent_pool, register_t* original_agent_pool, register_t* starting_agent_pool,
      const bool& implicit_agents, vector<string>& agent_names, vector<string>& starting_agent_names,
      const bool& has_mi_mask, register_t* mutual_information_mask);

    /*
    * Load system stats from file
    */
    void loadSystemStats(const string& filename, const unsigned int& N, float* system_stats);

    /*
    * Save system data to file
    */
    void saveSystemData(const string& filename, const register_t* system_data, const unsigned int& N, const unsigned int& M, const unsigned int& S);

    /*
    * Load raw system data to file (N, M and S must be known beforehand)
    */
    void loadRawSystemData(const string& filename, register_t* system_data, const unsigned int& N, const unsigned int& M, const unsigned int& S);

  }

}
#endif /* FILE_UTILS_H */
