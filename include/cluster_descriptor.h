
/*
* File:   cluster_descriptor.h
* Author: Emilio Vicari, Michele Amoretti
*/

#ifndef CLUSTER_DESCRIPTOR_H
#define CLUSTER_DESCRIPTOR_H

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdlib.h>
#include "register_utils.h"

using namespace std;

namespace dci
{

  /*
  * This class holds information about a cluster (mask and Tc)
  */
  class ClusterDescriptor
  {
  public:

    ClusterDescriptor(const register_t* clusterMask, const unsigned int& N, const float& IND);

    ClusterDescriptor(const register_t* clusterMask, const unsigned int& N);

    ClusterDescriptor(const ClusterDescriptor& C);

    ClusterDescriptor(ClusterDescriptor& C);

    bool operator<(const ClusterDescriptor& other) const;

    const register_t* getClusterMask() const;

    register_t* cloneClusterMask() const;

    void setIndex(float ind);

    float getIndex() const;

    virtual ~ClusterDescriptor();

    ClusterDescriptor& operator=(const ClusterDescriptor& other);

    friend ostream& operator<<(ostream& out, const ClusterDescriptor& c)
    {
      for (int i = 0; i < c.m_n; i++)
      out << "[" << i << "]";
      return out;
    }

  private:

    unsigned int m_n;
    register_t* m_clusterMask;
    float m_ind;

    void init(const register_t* clusterMask, const unsigned int& N, const float& IND);

  };

}

#endif /* CLUSTER_DESCRIPTOR_H */
