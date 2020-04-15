
/*
* File:   cluster_utils.cu
* Author: Emilio Vicari, Michele Amoretti
*/

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdlib.h>
#include "register_utils.h"
#include "cluster_utils.h"

using namespace std;

namespace dci
{

  namespace ClusterUtils
  {

    /*
    * Temporary string to hold cluster mask permutations
    */
    string bitmask;
    bool has_next;
    unsigned int current_n;
    unsigned int current_s;
    register_t last_reg_mask;
    register_t* cur_agent_pool;
    register_t* mi_mask;

    /*
    * Fills binary bitmask from its text string representation
    */
    void clusterBitmaskFromTextString(const string& bitmask_str, register_t* bitmask_reg, const unsigned int& N)
    {
      unsigned int old_n = current_n;
      current_n = N;
      resetLastRegMask(N);
      register_t* cur_reg = bitmask_reg;
      register_t one = 1;
      for (unsigned int i = 0; i != N; ++i)
      {
        // set current bit in bitmask according to mask value
        if (bitmask_str[i] - '0')
        *cur_reg |= one;
        else
        *cur_reg &= ~one;

        // check if we have to advance to next register in sequence
        if (i != N - 1 && i % BITS_PER_REG == BITS_PER_REG - 1)
        {
          one = 1;
          cur_reg++;
        }
        else
        one <<= 1;
      }
      *cur_reg &= last_reg_mask; // reset extra trailing bits to zero
      current_n = old_n;
    }

    /*
    * Sets current mutual information mask
    */
    void setMutualInformationMask(register_t* mask)
    {
      mi_mask = mask;
    }

    /*
    * Resets last reg mask according to current length
    */
    void resetLastRegMask(const unsigned int& n)
    {
      last_reg_mask = ~((register_t)0);
      unsigned int start = n % BITS_PER_REG;
      if (!start) return;  // BUG FIX
      for (unsigned int i = start; i != BITS_PER_REG; ++i) dci::RegisterUtils::setBitAtPos(&last_reg_mask, i, 0);
    }


    /*
    * Fills binary bitmask from its packed representation (i.e. agent cluster)
    */
    void clusterBitmaskFromAgentCluster(register_t* cluster, const register_t* agent_cluster, const unsigned int& N, const unsigned int& S, const unsigned int& NA, register_t* agent_pool)
    {
      dci::RegisterUtils::setAllBits<0>(cluster, N, S);
      for (unsigned int i = 0; i != NA; ++i)
      if (dci::RegisterUtils::getBitAtPos(agent_cluster, i))
      dci::RegisterUtils::reg_or(cluster, agent_pool + i * S, S);
    }

    /*
    * Fills binary bitmask from its string representation
    */
    void clusterBitmaskFromString(const string& bitmask_str, register_t* bitmask_reg)
    {
      dci::RegisterUtils::setAllBits<0>(bitmask_reg, current_s * BITS_PER_REG, current_s);
      for (unsigned int i = 0; i != current_n; ++i)
      if (bitmask_str[i])
      dci::RegisterUtils::reg_or(bitmask_reg, cur_agent_pool + i * current_s, current_s);
    }

    /*
    * Gets next cluster mask in sequence. Returns true if more permutations are available
    */
    bool getNextClusterMask(register_t* clusterMask, bool& hasNext)
    {
      bool ret_val = has_next;
      clusterBitmaskFromString(bitmask, clusterMask);
      has_next = prev_permutation(bitmask.begin(), bitmask.end());
      hasNext = has_next;
      return ret_val;
    }

    /*
    * Initializes cluster mask permutation generator
    */
    void initializeClusterMaskGenerator(const unsigned int& n, const unsigned int& r, const unsigned int& s, register_t* agent_pool, const unsigned int& nv, register_t* mutual_information_mask)
    {
      bitmask.resize(0);
      bitmask.resize(r, 1);
      if (n > r) bitmask.resize(n, 0);
      has_next = true;
      current_n = n;
      current_s = s;
      cur_agent_pool = agent_pool;
      resetLastRegMask(nv);
    }

    string getBitmask() { return bitmask; }

    /*
    * Gets complementary cluster mask for given mask
    */
    void getComplementaryClusterMask(register_t* dest, const register_t* source, const unsigned int& n)
    {
      for (unsigned int i = 0; i != dci::RegisterUtils::getNumberOfRegsFromNumberOfBits(n); ++i)
      dest[i] = ~source[i]; // bitwise invert
      // apply mutual information mask
      dci::RegisterUtils::reg_and(dest, dest, mi_mask, dci::RegisterUtils::getNumberOfRegsFromNumberOfBits(n));
      dest[dci::RegisterUtils::getNumberOfRegsFromNumberOfBits(n) - 1] &= last_reg_mask; // reset extra trailing bits to zero
    }

    void print(ostream& out, register_t* cluster, const unsigned int& N)
    {
      for (unsigned int i = 0; i != N; ++i)
      out << dci::RegisterUtils::getBitAtPos(cluster, i);
    }

    void println(ostream& out, register_t* cluster, const unsigned int& N)
    {
      print(out, cluster, N);
      out << '\n';
    }

    void setClusterFromPosArray(register_t* cluster, const vector<unsigned int>& positions, const unsigned int& N)
    {
      dci::RegisterUtils::setAllBits<0>(cluster, N, dci::RegisterUtils::getNumberOfRegsFromNumberOfBits(N));
      for (unsigned int i = 0; i != positions.size(); ++i)
      dci::RegisterUtils::setBitAtPos(cluster, positions[i], 1);
    }

  }

}
