/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   cluster_descriptor.h
 * Author: e.vicari
 *
 * Created on 2 marzo 2016, 10.52
 */
 
  /* choose index mod*/

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
    
        ClusterDescriptor(const register_t* clusterMask, const unsigned int& N, const float& IND)
        {
            init(clusterMask, N, IND);
        }
        
        ClusterDescriptor(const register_t* clusterMask, const unsigned int& N)
        {
            init(clusterMask, N, 0);
        }
        
        ClusterDescriptor(const ClusterDescriptor& C)
        {
            init(C.m_clusterMask, C.m_n, C.m_ind);
        }

        ClusterDescriptor(ClusterDescriptor& C)
        {
            init(C.m_clusterMask, C.m_n, C.m_ind);
        }

        bool operator<(const ClusterDescriptor& other) const
        {
            return m_ind > other.m_ind;
        }
        
        friend ostream& operator<<(ostream& out, const ClusterDescriptor& c)
        {
        	for (int i = 0; i < c.m_n; i++)
        	    out << "[" << i << "]";
            return out;
        }
        
        const register_t* getClusterMask() const
        {
            return m_clusterMask;
        }
        
        register_t* cloneClusterMask() const
        {
            unsigned int n_regs = dci::RegisterUtils::GetNumberOfRegsFromNumberOfBits(m_n);
            register_t* mask = new register_t[n_regs];
            memcpy(mask, m_clusterMask, n_regs * BYTES_PER_REG);
            return mask;
        }
        
        void setIndex(float ind) 
        {
        	m_ind = ind;
        }
        
        float getIndex() const
        {
        	return m_ind;
        }
        
        virtual ~ClusterDescriptor()
        {
            if (m_clusterMask) 
            {
                delete[] m_clusterMask;
                m_clusterMask = 0;
            }
        }
        
        ClusterDescriptor& operator=(const ClusterDescriptor& other)
        {
            if (this != &other)
            {
                delete[] m_clusterMask;
                init(other.m_clusterMask, other.m_n, other.m_ind);
            }
            return *this;
        }

    private:
        unsigned int m_n;
        register_t* m_clusterMask;
        float m_ind;
        
        void init(const register_t* clusterMask, const unsigned int& N, const float& IND)
        {
        	m_n = N;
        	unsigned int n_regs = dci::RegisterUtils::GetNumberOfRegsFromNumberOfBits(N);
            m_clusterMask = new register_t[n_regs];
            if (m_clusterMask) memcpy(m_clusterMask, clusterMask, n_regs * BYTES_PER_REG);
            m_ind = IND;
        }    
    };
    
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
         * Fills binary bitmask from its packed representation (i.e. agent cluster)
         */
        void clusterBitmaskFromAgentCluster(register_t* cluster, const register_t* agent_cluster, 
            const unsigned int& N, const unsigned int& S, 
            const unsigned int& NA, register_t* agent_pool)
        {
            dci::RegisterUtils::SetAllBits<0>(cluster, N, S);
            for (unsigned int i = 0; i != NA; ++i)
                if (dci::RegisterUtils::GetBitAtPos(agent_cluster, i)) 
                    dci::RegisterUtils::reg_or(cluster, agent_pool + i * S, S);
        }
		
        /*
         * Fills binary bitmask from its string representation
         */
        void clusterBitmaskFromString(const string& bitmask_str, register_t* bitmask_reg)
        {
            dci::RegisterUtils::SetAllBits<0>(bitmask_reg, current_s * BITS_PER_REG, current_s);
            for (unsigned int i = 0; i != current_n; ++i)
                if (bitmask_str[i]) 
                    dci::RegisterUtils::reg_or(bitmask_reg, cur_agent_pool + i * current_s, current_s);
        }
		
		/*
		 * Resets last reg mask according to current length
		 */
		inline void resetLastRegMask(const unsigned int& n)
		{
			last_reg_mask = ~((register_t)0);
            unsigned int start = n % BITS_PER_REG;
            if (!start) return;  // BUG FIX
            for (unsigned int i = start; i != BITS_PER_REG; ++i) dci::RegisterUtils::SetBitAtPos(&last_reg_mask, i, 0);
		}
        
		/*
		 * Sets current mutual information mask
		 */
		inline void setMutualInformationMask(register_t* mask)
		{
			mi_mask = mask;
		}
        
        /*
         * Fills binary bitmask from its text string representation
         */
        inline void clusterBitmaskFromTextString(const string& bitmask_str, register_t* bitmask_reg, const unsigned int& N)
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
         * Initializes cluster mask permutation generator
         */
        void initializeClusterMaskGenerator(const unsigned int& n, const unsigned int& r, const unsigned int& s, register_t* agent_pool,
            const unsigned int& nv, register_t* mutual_information_mask)
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
         * Gets next random cluster mask (using random permutation)
         */
        template<class RandomNumberGenerator> void getNextRandomClusterMask(register_t* clusterMask, RandomNumberGenerator rng)
        {
            shuffle(bitmask.begin(), bitmask.end(), rng);
            clusterBitmaskFromString(bitmask, clusterMask);
        }
        
        /*
         * Gets complementary cluster mask for given mask
         */
        void getComplementaryClusterMask(register_t* dest, const register_t* source, const unsigned int& n)
        {
            for (unsigned int i = 0; i != dci::RegisterUtils::GetNumberOfRegsFromNumberOfBits(n); ++i)
                dest[i] = ~source[i]; // bitwise invert
            // apply mutual information mask
            dci::RegisterUtils::reg_and(dest, dest, mi_mask, dci::RegisterUtils::GetNumberOfRegsFromNumberOfBits(n));
            dest[dci::RegisterUtils::GetNumberOfRegsFromNumberOfBits(n) - 1] &= last_reg_mask; // reset extra trailing bits to zero
        }
		
		void print(ostream& out, register_t* cluster, const unsigned int& N)
		{
			for (unsigned int i = 0; i != N; ++i)
				out << dci::RegisterUtils::GetBitAtPos(cluster, i);
		}
		
		inline void println(ostream& out, register_t* cluster, const unsigned int& N)
		{
			print(out, cluster, N);
			out << '\n';
		}
        
        void setClusterFromPosArray(register_t* cluster, const vector<unsigned int>& positions, const unsigned int& N)
        {
            dci::RegisterUtils::SetAllBits<0>(cluster, N, dci::RegisterUtils::GetNumberOfRegsFromNumberOfBits(N));
            for (unsigned int i = 0; i != positions.size(); ++i)
                dci::RegisterUtils::SetBitAtPos(cluster, positions[i], 1);
        }
        
    }

}

#endif /* CLUSTER_DESCRIPTOR_H */

