
/*
* File:   cluster_utils.h
* Author: Emilio Vicari, Michele Amoretti
*/

#ifndef CLUSTER_UTILS_H
#define CLUSTER_UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdlib.h>
#include "register_utils.h"

using namespace std;

namespace dci
{

  namespace ClusterUtils
  {

    /*
    * Fills binary bitmask from its packed representation (i.e. agent cluster)
    */
    void clusterBitmaskFromAgentCluster(register_t* cluster, const register_t* agent_cluster, const unsigned int& N, const unsigned int& S, const unsigned int& NA, register_t* agent_pool);

    /*
    * Fills binary bitmask from its text string representation
    */
    void clusterBitmaskFromTextString(const string& bitmask_str, register_t* bitmask_reg, const unsigned int& N);

    /*
    * Sets current mutual information mask
    */
    void setMutualInformationMask(register_t* mask);

    /*
    * Resets last reg mask according to current length
    */
    void resetLastRegMask(const unsigned int& n);

    /*
    * Fills binary bitmask from its string representation
    */
    void clusterBitmaskFromString(const string& bitmask_str, register_t* bitmask_reg);

    /*
    * Initializes cluster mask permutation generator
    */
    void initializeClusterMaskGenerator(const unsigned int& n, const unsigned int& r, const unsigned int& s, register_t* agent_pool, const unsigned int& nv, register_t* mutual_information_mask);

    /*
    * Gets next cluster mask in sequence. Returns true if more permutations are available
    */
    bool getNextClusterMask(register_t* clusterMask, bool& hasNext);

    string getBitmask();

    /*
    * Gets next random cluster mask (using random permutation)
    */
    template<class RandomNumberGenerator> void getNextRandomClusterMask(register_t* clusterMask, RandomNumberGenerator rng)
    {
      shuffle(getBitmask().begin(), getBitmask().end(), rng);
      clusterBitmaskFromString(getBitmask(), clusterMask);
    }

    /*
    * Gets complementary cluster mask for given mask
    */
    void getComplementaryClusterMask(register_t* dest, const register_t* source, const unsigned int& n);

    void print(ostream& out, register_t* cluster, const unsigned int& N);

    void println(ostream& out, register_t* cluster, const unsigned int& N);

    void setClusterFromPosArray(register_t* cluster, const vector<unsigned int>& positions, const unsigned int& N);

  }

}

#endif /* CLUSTER_UTILS_H */
