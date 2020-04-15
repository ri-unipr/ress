
/*
* File:   stat_utils.h
* Author: Emilio Vicari, Michele Amoretti
*/

#ifndef STAT_UTILS_H
#define STAT_UTILS_H

#include <cmath>

using namespace std;

namespace dci
{
  namespace StatUtils
  {

    /*
    * This class is used to compute statistics
    * over a running sequence of float values
    */
    class RunningStat
    {
    public:

      RunningStat() : m_n(0) {}

      void clear()
      {
        m_n = 0;
      }

      void push(float x)
      {
        m_n++;

        // See Knuth TAOCP vol 2, 3rd edition, page 232
        if (m_n == 1)
        {
          m_oldM = m_newM = x;
          m_oldS = 0.0;
        }
        else
        {
          m_newM = m_oldM + (x - m_oldM)/m_n;
          m_newS = m_oldS + (x - m_oldM)*(x - m_newM);

          // set up for next iteration
          m_oldM = m_newM;
          m_oldS = m_newS;
        }
      }

      int numDataValues() const
      {
        return m_n;
      }

      float mean() const
      {
        return (m_n > 0) ? m_newM : 0.0;
      }

      float variance() const
      {
        return ( (m_n > 1) ? m_newS/(m_n - 1) : 0.0 );
      }

      float standardDeviation() const
      {
        return sqrt( variance() );
      }

    private:

      int m_n;
      float m_oldM, m_newM, m_oldS, m_newS;
    };

    class Binomial
    {
    public:
      Binomial(unsigned int Max)
      {
        max = Max+1;
        table = new unsigned long long * [max]();
        for (unsigned int i=0; i < max; i++)
        {
          table[i] = new unsigned long long[max]();

          for (unsigned int j = 0; j < max; j++)
          {
            table[i][j] = 0;
          }
        }
      }

      ~Binomial()
      {
        for (unsigned int i =0; i < max; i++)
        {
          delete table[i];
        }
        delete table;
      }

      unsigned long long choose(unsigned int n, unsigned int k);

    private:
      bool contains(unsigned int n, unsigned int k);

      int max;
      unsigned long long **table;
    };

    unsigned long long Binomial::choose(unsigned int n, unsigned int k)
    {
      if (n < k) return 0;
      if (k == 0 || n==1 ) return 1;
      if (n==2 && k==1) return 2;
      if (n==2 && k==2) return 1;
      if (n==k) return 1;

      if (contains(n,k))
      {
        return table[n][k];
      }
      table[n][k] = choose(n-1,k) + choose(n-1,k-1);
      return table[n][k];
    }

    bool Binomial::contains(unsigned int n, unsigned int k)
    {
      if (table[n][k] == 0)
      {
        return false;
      }
      return true;
    }

  }

}

#endif /* STAT_UTILS_H */
