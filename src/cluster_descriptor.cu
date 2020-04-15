
/*
* File:   cluster_descriptor.cu
* Author: Emilio Vicari, Michele Amoretti
*
*/

#include "cluster_descriptor.h"

using namespace std;
using namespace dci;


ClusterDescriptor::ClusterDescriptor(const register_t* clusterMask, const unsigned int& N, const float& IND)
{
  init(clusterMask, N, IND);
}

ClusterDescriptor::ClusterDescriptor(const register_t* clusterMask, const unsigned int& N)
{
  init(clusterMask, N, 0);
}

ClusterDescriptor::ClusterDescriptor(const ClusterDescriptor& C)
{
  init(C.m_clusterMask, C.m_n, C.m_ind);
}

ClusterDescriptor::ClusterDescriptor(ClusterDescriptor& C)
{
  init(C.m_clusterMask, C.m_n, C.m_ind);
}

bool ClusterDescriptor::operator<(const ClusterDescriptor& other) const
{
  return m_ind > other.m_ind;
}

const register_t* ClusterDescriptor::getClusterMask() const
{
  return m_clusterMask;
}

register_t* ClusterDescriptor::cloneClusterMask() const
{
  unsigned int n_regs = dci::RegisterUtils::getNumberOfRegsFromNumberOfBits(m_n);
  register_t* mask = new register_t[n_regs];
  memcpy(mask, m_clusterMask, n_regs * BYTES_PER_REG);
  return mask;
}

void ClusterDescriptor::setIndex(float ind)
{
  m_ind = ind;
}

float ClusterDescriptor::getIndex() const
{
  return m_ind;
}

ClusterDescriptor::~ClusterDescriptor()
{
  if (m_clusterMask)
  {
    delete[] m_clusterMask;
    m_clusterMask = 0;
  }
}

ClusterDescriptor& ClusterDescriptor::operator=(const ClusterDescriptor& other)
{
  if (this != &other)
  {
    delete[] m_clusterMask;
    init(other.m_clusterMask, other.m_n, other.m_ind);
  }
  return *this;
}

void ClusterDescriptor::init(const register_t* clusterMask, const unsigned int& N, const float& IND)
{
  m_n = N;
  unsigned int n_regs = dci::RegisterUtils::getNumberOfRegsFromNumberOfBits(N);
  m_clusterMask = new register_t[n_regs];
  if (m_clusterMask) memcpy(m_clusterMask, clusterMask, n_regs * BYTES_PER_REG);
  m_ind = IND;
}
