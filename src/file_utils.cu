
/*
* File:   file_utils.cu
* Author: Emilio Vicari, Michele Amoretti
*/


#include "file_utils.h"

using namespace std;

namespace dci
{
  namespace FileUtils
  {

    /*
    * Computes file size in bytes using C system functions
    */
    long fileSize(const char* filename)
    {
      struct stat stat_buf;
      int rc = stat(filename, &stat_buf);
      return rc == 0 ? stat_buf.st_size : -1;
    }

    /*
    * Counts number of characters in string
    */
    unsigned int count(const string& str, const char& c)
    {
      unsigned int cnt = 0;
      for (unsigned int i = 0; i != str.length(); ++i) if (str[i] == c) cnt++;
      return cnt;
    }

    /*
    * Collects system parameters examining first sample and file size
    */
    void collectSystemParameters(const string& filename, unsigned int& N, unsigned int& M, unsigned int& S, unsigned long long& L,
      unsigned int& NA, unsigned int& SA, unsigned int& NO, unsigned int& SO, unsigned int& NBO, unsigned int& SBO, bool& has_mi_mask)
      {
        ifstream in_file(filename);                     // open file for reading
        string line;
        long size_offset = 0;
        NA = 0;
        NO = 0;
        NBO = 0;
        if (in_file.peek() == EOF)
        {
          cout << strerror(errno) << endl << errno << endl;
          return;
        }
        getline(in_file, line); // first sample or %% line or -- line
        has_mi_mask = false;
        if (line[0] == '-' && line[1] == '-') // mutual information mask
        {
          has_mi_mask = true;
          size_offset += line.length() + 1;
          getline(in_file, line);
        }
        if (line[0] == '%' && line[1] == '%') // %% line -> agents explicitly specified
        {
          do
          {
            NA++;
            size_offset += line.length() + 1;

            // original agent masks specified
            if (NBO == 0 && count(line, '%') == 6)
            {
              // set starting pos
              unsigned int pos = line.find_last_of('%') + 1;

              // consume whitespaces
              while (line[pos] == ' ' || line[pos] == '\t') pos++;

              // consume first set of 0s and 1s
              while (line[pos] == '0' || line[pos] == '1') { pos++; NBO++; }
            }

            getline(in_file, line);

            // sieving input - check for comment delimiter (only first row)
            if (NO == 0 && line.find_last_of('%') != std::string::npos)
            {
              // start from the character following the %%
              for (unsigned int i = line.find_last_of('%') + 1; i < line.length(); ++i)
              if (line[i] == '0' || line[i] == '1') NO++;
            }

          }
          while (line[0] != '%');
          NA--;
          size_offset += line.length() + 1;
          getline(in_file, line); // this is now the first sample
        }
        unsigned int NBO_offset = NBO ? (NBO + 4) : 0; // " %% " between current and original sample
        N = line.length() - NBO_offset;
        M = (fileSize(filename.c_str()) - size_offset + 1) / (N + NBO_offset + 1); // compute M according to sample size and file size
        S = dci::RegisterUtils::getNumberOfRegsFromNumberOfBits(N);

        if (NA)
        {
          L = ((unsigned long long)1 << NA) - 1;                                     // L = 2^NA-1
          SA = dci::RegisterUtils::getNumberOfRegsFromNumberOfBits(NA);
        }
        else
        {
          L = ((unsigned long long)1 << N) - 1;                                     // L = 2^N-1
          SA = S;
        }

        SO = NO ? dci::RegisterUtils::getNumberOfRegsFromNumberOfBits(NO) : 0;
        SBO = NBO ? dci::RegisterUtils::getNumberOfRegsFromNumberOfBits(NBO) : 0;
      }

    /*
    * Load system data from file
    */
    void loadSystemData(const string& filename, const unsigned int& N, const unsigned int& M,
      const unsigned int& S, const unsigned int& SA, const unsigned int& SO, const unsigned int& SBO,
      const unsigned int& NA, const unsigned int& NO, const unsigned int& NBO,
      register_t* system_data, register_t* original_system_data,
      register_t* agent_pool, register_t* original_agent_pool, register_t* starting_agent_pool,
      const bool& implicit_agents, vector<string>& agent_names, vector<string>& starting_agent_names,
      const bool& has_mi_mask, register_t* mutual_information_mask)
      {
        ifstream in_file(filename);
        register_t mask;
        string dummy;

        // check if very first line contains mutual information mask
        if (has_mi_mask)
        {
          getline(in_file, dummy);
          // reset mask
          dci::RegisterUtils::setAllBits<false>(mutual_information_mask, NA, SA);

          // set single bits
          for (unsigned int i = 0; i != NA; ++i)
          dci::RegisterUtils::setBitAtPos(mutual_information_mask, i, dummy[i+3] == '1');
        }

        // check if first line contains agent names
        if (!implicit_agents)
        {
          // reset flag
          bool is_starting_agent = false;

          // read line
          getline(in_file, dummy);

          // agents
          string temp_agent;

          // last character to examine for original agent names
          unsigned int last_char = NBO ? (dummy.find_last_of('%') - 2) : dummy.size();

          // retrieve agent names (if present) - skip leading %%
          for (unsigned int i = 2; i != last_char; ++i)
          if (dummy[i] == '%') // starting system agent section (sieving)
          is_starting_agent = true;
          else if (dummy[i] != ' ' && dummy[i] != '\t')
          temp_agent.push_back(dummy[i]);
          else if (temp_agent.size() > 0)
          {
            //cout << temp_agent << endl;
            if (is_starting_agent)
            starting_agent_names.push_back(temp_agent);
            else
            agent_names.push_back(temp_agent);
            temp_agent.resize(0);
          }

          // add last agent
          if (temp_agent.size() > 0)
          {
            if (is_starting_agent)
            starting_agent_names.push_back(temp_agent);
            else
            agent_names.push_back(temp_agent);
          }

          // retrieve original agent pool if specified
          if (NBO)
          {
            // store current position in string
            last_char = dummy.find_last_of('%') + 1;

            // for each agent
            for (unsigned int oa = 0; oa != NO; ++oa)
            {
              // skip whitespaces
              for (; dummy[last_char] == ' ' || dummy[last_char] == '\t'; ++last_char);

              // get current system agent
              for (unsigned int reg = 0; reg != SBO; ++reg)
              {
                *original_agent_pool = 0; // reset current register
                mask = 1; // reset mask
                for (unsigned int col = reg * BITS_PER_REG; col != (reg + 1) * BITS_PER_REG && col != NBO; ++col)
                {
                  if (!(dummy[last_char++] - '0'))
                  *original_agent_pool &= ~mask; // set current bit to zero
                  else
                  *original_agent_pool |= mask; // set current bit to one
                  mask <<= 1;
                }
                original_agent_pool++; // advance to next register
              }
            }
          }
        }

        // load agents
        for (unsigned int a = 0; a != NA; ++a)
        {
          // check if agent is implicit
          if (implicit_agents)
          {
            // reset agent
            dci::RegisterUtils::setAllBits<false>(agent_pool, N, S);

            // set bit for single variable
            dci::RegisterUtils::setBitAtPos(agent_pool, a, 1);

            // advance to next
            agent_pool += S;
          }
          else
          {
            // get current system agent
            for (unsigned int reg = 0; reg != S; ++reg)
            {
              *agent_pool = 0; // reset current register
              mask = 1; // reset mask
              for (unsigned int col = reg * BITS_PER_REG; col != (reg + 1) * BITS_PER_REG && col != N; ++col)
              {
                if (!(in_file.get() - '0'))
                *agent_pool &= ~mask; // set current bit to zero
                else
                *agent_pool |= mask; // set current bit to one
                mask <<= 1;
              }
              agent_pool++; // advance to next register
            }

            // get starting system agent (sieving) if necessary
            if (NO)
            {
              // remove useless characters
              while (in_file.peek() == '%' || in_file.peek() == ' ' || in_file.peek() == '\t') in_file.get();

              for (unsigned int reg = 0; reg != SO; ++reg)
              {
                *starting_agent_pool = 0; // reset current register
                mask = 1; // reset mask
                for (unsigned int col = reg * BITS_PER_REG; col != (reg + 1) * BITS_PER_REG && col != NO; ++col)
                {
                  if (!(in_file.get() - '0'))
                  *starting_agent_pool &= ~mask; // set current bit to zero
                  else
                  *starting_agent_pool |= mask; // set current bit to one
                  mask <<= 1;
                }
                starting_agent_pool++; // advance to next register
              }
            }

            in_file.get(); // consume newline
          }
        }

        // discard second %% line if necessary
        if (!implicit_agents) getline(in_file, dummy);

        // load data and compute frequencies
        for (unsigned int row = 0; row != M; ++row)
        {
          for (unsigned int reg = 0; reg != S; ++reg)
          {
            *system_data = 0; // reset current register
            mask = 1; // reset mask
            for (unsigned int col = reg * BITS_PER_REG; col != (reg + 1) * BITS_PER_REG && col != N; ++col)
            {
              if (in_file.get() - '0')
              *system_data |= mask; // set current bit to one
              mask <<= 1;
            }
            system_data++; // advance to next register
          }
          in_file.get(); // consume newline or whitespace

          // check if original system data is there as well
          if (NBO)
          {
            // get other useless characters
            in_file.get(); in_file.get(); in_file.get();

            for (unsigned int reg = 0; reg != SBO; ++reg)
            {
              *original_system_data = 0; // reset current register
              mask = 1; // reset mask
              for (unsigned int col = reg * BITS_PER_REG; col != (reg + 1) * BITS_PER_REG && col != NBO; ++col)
              {
                if (in_file.get() - '0')
                *original_system_data |= mask; // set current bit to one
                mask <<= 1;
              }
              original_system_data++; // advance to next register
            }

            // consume newline
            in_file.get();
          }
        }
      }

    /*
    * Load system stats from file
    */
    void loadSystemStats(const string& filename, const unsigned int& N, float* system_stats)
    {
      ifstream in_file(filename);

      for (unsigned int col = 0; col != N; ++col)
      in_file >> system_stats[2 * col] >> system_stats[2 * col + 1];
    }

    /*
    * Save system data to file
    */
    void saveSystemData(const string& filename, const register_t* system_data, const unsigned int& N, const unsigned int& M, const unsigned int& S)
    {
      ofstream out_file(filename);
      register_t mask;

      // load data and compute frequencies
      for (unsigned int row = 0; row != M; ++row)
      {
        for (unsigned int reg = 0; reg != S; ++reg)
        {
          mask = 1; // reset mask
          for (unsigned int col = reg * BITS_PER_REG; col != (reg + 1) * BITS_PER_REG && col != N; ++col)
          {
            out_file << ((*system_data & mask) ? '1' : '0');
            mask <<= 1;
          }
          system_data++; // advance to next register
        }
        out_file << endl; // newline
      }
    }

    /*
    * Load raw system data to file (N, M and S must be known beforehand)
    */
    void loadRawSystemData(const string& filename, register_t* system_data, const unsigned int& N, const unsigned int& M, const unsigned int& S)
    {
      ifstream in_file(filename);
      register_t mask;

      // load data and compute frequencies
      for (unsigned int row = 0; row != M; ++row)
      {
        for (unsigned int reg = 0; reg != S; ++reg)
        {
          *system_data = 0; // reset current register
          mask = 1; // reset mask
          for (unsigned int col = reg * BITS_PER_REG; col != (reg + 1) * BITS_PER_REG && col != N; ++col)
          {
            if (in_file.get() - '0')
            {
              cout << '1';
              *system_data |= mask; // set current bit to one
            }
            else cout << '0';
            mask <<= 1;
          }
          cout << ' ' << (*system_data);
          system_data++; // advance to next register
        }
        in_file.get(); // consume newline
        cout << '\n';
      }
    }

  }
}
