
/*
* File:   register_utils.h
* Author: Emilio Vicari, Michele Amoretti
*/

#ifndef REGISTER_UTILS_H
#define REGISTER_UTILS_H

#include "common.h"

#define m3_c1 0xcc9e2d51
#define m3_c2 0x1b873593
#define m3_r1 15
#define m3_r2 13
#define m3_m 5
#define m3_n 0xe6546b64
#define m3_seed 0x422a0b8f
#define ROT32(x, y) ((x << y) | (x >> (32 - y))) // avoid effort
#define REG_ALL_1 (~((register_t)0))
#define REG_ALL_0 ((register_t)0)

namespace dci
{

  namespace RegisterUtils
  {

    /*
    * Returns number of registers needed to store a given number of bits
    */
    __host__ __device__ inline unsigned int getNumberOfRegsFromNumberOfBits(unsigned int number_of_bits)
    {
      return number_of_bits / BITS_PER_REG + (number_of_bits % BITS_PER_REG ? 1 : 0);
    }

    /*
    * Returns number of bytes needed to store a given number of bits (considering whole registers)
    */
    __host__ __device__ inline unsigned int getNumberOfBytesFromNumberOfBits(unsigned int number_of_bits)
    {
      return getNumberOfRegsFromNumberOfBits(number_of_bits) * BYTES_PER_REG;
    }

    /*
    * Returns absolute bit position in a hypothetical bit array of N-sized samples
    */
    __host__ __device__ inline unsigned int getBitPos(unsigned int N, unsigned int row, unsigned int col)
    {
      return row * N + col;
    }

    /*
    * Returns bit value at given position (absolute)
    */
    __host__ __device__ inline bool getBitAtPos(const register_t* data, unsigned int pos)
    {
      return (data[pos / BITS_PER_REG] >> (pos % BITS_PER_REG)) & SELECTION_BITMASK;
    }

    /*
    * Returns bit value at given position (row, col)
    */
    __host__ __device__ inline bool getBitAtPos(const register_t* data, unsigned int N, unsigned int row, unsigned int col)
    {
      return getBitAtPos(data, getBitPos(N, row, col));
    }

    /*
    * Sets bit value at given position (absolute)
    */
    __host__ __device__ inline void setBitAtPos(register_t* data, unsigned int pos, bool value)
    {
      if (value) data[pos / BITS_PER_REG] |= (SELECTION_BITMASK << (pos % BITS_PER_REG));
      else data[pos / BITS_PER_REG] &= ~(SELECTION_BITMASK << (pos % BITS_PER_REG));
    }

    /*
    * Sets bit value at given position (row, col)
    */
    __host__ __device__ inline void setBitAtPos(register_t* data, unsigned int N, unsigned int row, unsigned int col, bool value)
    {
      setBitAtPos(data, getBitPos(N, row, col), value);
    }

    /*
    * Packs data from a sample in source into dest, using given cluster mask
    */
    __host__ __device__ inline void packClusterData(const register_t* source, register_t* dest, const register_t* cluster_mask, const unsigned int& N, const unsigned int& row)
    {
      unsigned int start_pos = getBitPos(N, row, 0);
      unsigned int dest_pos = 0;
      for (unsigned int mask_pos = 0; mask_pos != N; ++mask_pos)
      if (getBitAtPos(cluster_mask, mask_pos))
      setBitAtPos(dest, dest_pos++, getBitAtPos(source, start_pos + mask_pos));
    }

    /*
    * Returns mask used to reset trailing bits in last register
    */
    inline register_t getLastRegisterResetMask(const unsigned int& N)
    {
      register_t mask = REG_ALL_1;
      for (unsigned int i = N; i != getNumberOfBytesFromNumberOfBits(N) * 8; ++i)
      setBitAtPos(&mask, i % BITS_PER_REG, 0);
      return mask;
    }

    /*
    * Sets all bits from 0 to N-1 to value, the trailing extra bits to zero
    */
    template<bool value> inline void setAllBits(register_t* dest, const unsigned int& N, const unsigned int& S)
    {
      for (unsigned int i = 0; i != S; ++i)
      dest[i] = value ? REG_ALL_1 : REG_ALL_0;
      if (value) dest[S - 1] &= getLastRegisterResetMask(N);
    }

    /*
    * Counts number of zeroes or ones in a mask
    */
    template<bool value> inline unsigned int getNumberOf(const register_t* data, const unsigned int& N)
    {
      unsigned int ret = 0;
      for (unsigned int i = 0; i != N; ++i)
      if (getBitAtPos(data, i) == value) ret++;
      return ret;
    }

    /*
    * Basic hash with template specializations
    */
    template<unsigned int R> __host__ __device__ inline unsigned int basic_hash(const register_t* key, const unsigned int& L)
    {
      register_t tmp = key[0];
      for (unsigned int i = 1; i != R; ++i)
      tmp += key[i];
      return tmp % L;
    }

    template<> __host__ __device__ inline unsigned int basic_hash<1>(const register_t* key, const unsigned int& L) { return key[0] % L; }
    template<> __host__ __device__ inline unsigned int basic_hash<2>(const register_t* key, const unsigned int& L) { return (key[0] + key[1]) % L; }
    template<> __host__ __device__ inline unsigned int basic_hash<3>(const register_t* key, const unsigned int& L) { return (key[0] + key[1] + key[2]) % L; }
    template<> __host__ __device__ inline unsigned int basic_hash<4>(const register_t* key, const unsigned int& L) { return (key[0] + key[1] + key[2] + key[3]) % L; }

    /*
    * murmur3 hash
    */
    template<unsigned int R> __host__ __device__ inline unsigned int murmur3_32(const char *key, const unsigned int& L) {
      //return *((const unsigned int*)key) % L;
      unsigned int hash = m3_seed;
      unsigned int len = R * BYTES_PER_REG;

      const int nblocks = len / 4;
      const unsigned int *blocks = (const unsigned int *) key;
      int i;
      unsigned int k;
      for (i = 0; i < nblocks; i++) {
        k = blocks[i];
        k *= m3_c1;
        k = ROT32(k, m3_r1);
        k *= m3_c2;

        hash ^= k;
        hash = ROT32(hash, m3_r2) * m3_m + m3_n;
      }

      const unsigned char *tail = (const unsigned char *) (key + nblocks * 4);
      unsigned int k1 = 0;

      switch (len & 3) {
        case 3:
        k1 ^= tail[2] << 16;
        case 2:
        k1 ^= tail[1] << 8;
        case 1:
        k1 ^= tail[0];

        k1 *= m3_c1;
        k1 = ROT32(k1, m3_r1);
        k1 *= m3_c2;
        hash ^= k1;
      }

      hash ^= len;
      hash ^= (hash >> 16);
      hash *= 0x85ebca6b;
      hash ^= (hash >> 13);
      hash *= 0xc2b2ae35;
      hash ^= (hash >> 16);

      return hash % L;
    }

    /*
    * FNV-1a hash
    */
    template<unsigned int R> __host__ __device__ inline unsigned int fnv1a(const unsigned char* a, const unsigned int& L)
    {
      unsigned long hash = 0xcbf29ce484222325;
      for (unsigned int i = 0; i != R * BYTES_PER_REG; ++i)
      {
        hash ^= (unsigned long)a[i];
        hash *= 0x100000001b3;
      }
      return hash % L;
    }

    /*
    * Assign source to dest, using mask
    */
    template<unsigned int R> __host__ __device__ inline void assignFromMask(register_t* dest, const register_t* source, const register_t* mask)
    {
      for (unsigned int i = 0; i != R; ++i)
      dest[i] = source[i] & mask[i];
    }

    /*
    * Assign source1 & source2 to dest
    */
    __host__ __device__ inline void reg_and(register_t* dest, const register_t* source1, const register_t* source2, const unsigned int& R)
    {
      for (unsigned int i = 0; i != R; ++i)
      dest[i] = source1[i] & source2[i];
    }

    /*
    * Assign dest | source to dest
    */
    __host__ __device__ inline void reg_or(register_t* dest, const register_t* source, const unsigned int& R)
    {
      for (unsigned int i = 0; i != R; ++i)
      dest[i] |= source[i];
    }

    /*
    * Returns true if <subset> is a subset of <source>
    */
    __host__ __device__ inline bool contains(const register_t* source, const register_t* subset, const unsigned int& R)
    {
      bool res = true;
      for (unsigned int i = 0; i != R; ++i)
      res = res && (source[i] & subset[i]) == subset[i];
      return res;
    }

    /*
    * Returns true if <source1> is a subset of <source2>, or vice versa
    */
    __host__ __device__ inline bool are_subsets(const register_t* source1, const register_t* source2, const unsigned int& R)
    {
      return contains(source1, source2, R) || contains(source2, source1, R);
    }

    /*
    * Assign source to dest
    */
    inline void assign(register_t* dest, const register_t* source, const unsigned int& R)
    {
      for (unsigned int i = 0; i != R; ++i)
      dest[i] = source[i];
    }

    /*
    * Assign source to dest (template)
    */
    template<unsigned int R> __host__ __device__ inline void assign(register_t* dest, const register_t* source)
    {
      for (unsigned int i = 0; i != R; ++i)
      dest[i] = source[i];
    }

    /*
    * Compare a and b
    */
    inline bool equal(const register_t* a, const register_t* b, const unsigned int& R)
    {
      for (unsigned int i = 0; i != R; ++i)
      if (a[i] != b[i]) return false;
      return true;
    }

    /*
    * Compare a and b (template)
    */
    template<unsigned int R> __host__ __device__ inline bool equal(const register_t* a, const register_t* b)
    {
      for (unsigned int i = 0; i != R; ++i)
      if (a[i] != b[i]) return false;
      return true;
    }

    /*
    * Template specializations for R = 1
    */
    template<> __host__ __device__ inline void assignFromMask<1>(register_t* dest, const register_t* source, const register_t* mask)
    { *dest = *source & *mask; }
    template<> __host__ __device__ inline void assign<1>(register_t* dest, const register_t* source)
    { *dest = *source; }
    template<> __host__ __device__ inline bool equal<1>(const register_t* a, const register_t* b)
    { return *a == *b; }

    /*
    * Template specializations for R = 2
    */
    template<> __host__ __device__ inline void assignFromMask<2>(register_t* dest, const register_t* source, const register_t* mask)
    { dest[0] = source[0] & mask[0]; dest[1] = source[1] & mask[1]; }
    template<> __host__ __device__ inline void assign<2>(register_t* dest, const register_t* source)
    { dest[0] = source[0]; dest[1] = source[1]; }
    template<> __host__ __device__ inline bool equal<2>(const register_t* a, const register_t* b)
    { return a[0] == b[0] && a[1] == b[1]; }

    /*
    * Template specializations for R = 3
    */
    template<> __host__ __device__ inline void assignFromMask<3>(register_t* dest, const register_t* source, const register_t* mask)
    { dest[0] = source[0] & mask[0]; dest[1] = source[1] & mask[1]; dest[2] = source[2] & mask[2]; }
    template<> __host__ __device__ inline void assign<3>(register_t* dest, const register_t* source)
    { dest[0] = source[0]; dest[1] = source[1]; dest[2] = source[2]; }
    template<> __host__ __device__ inline bool equal<3>(const register_t* a, const register_t* b)
    { return a[0] == b[0] && a[1] == b[1] && a[2] == b[2]; }

    /*
    * Template specializations for R = 4
    */
    template<> __host__ __device__ inline void assignFromMask<4>(register_t* dest, const register_t* source, const register_t* mask)
    { dest[0] = source[0] & mask[0]; dest[1] = source[1] & mask[1]; dest[2] = source[2] & mask[2]; dest[3] = source[3] & mask[3]; }
    template<> __host__ __device__ inline void assign<4>(register_t* dest, const register_t* source)
    { dest[0] = source[0]; dest[1] = source[1]; dest[2] = source[2]; dest[3] = source[3]; }
    template<> __host__ __device__ inline bool equal<4>(const register_t* a, const register_t* b)
    { return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]; }

  }
}

#endif /* REGISTER_UTILS_H */
