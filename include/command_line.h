/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   command_line.h
 * Author: e.vicari
 *
 * Created on 1 marzo 2016, 12.13
 *
 * Utilities for command line parsing
 *
 */

#ifndef COMMAND_LINE_H
#define COMMAND_LINE_H

#include <string>
#include <memory>
#include <iostream>
#include <ctime>

namespace dci
{

    // struct holding execution configuration coming from command-line
    struct RunInfo
    {
        bool tc;                           // if true, use Tc as index
        bool zi;                            // if true, use ZI as index
        bool si;                            // if true, use the strength index as index
        bool si2;

        bool show_device_stats;             // if true, device stats are shown at startup
        bool verbose;                       // verbose computation
        bool silent;                        // silent computation (overrides verbose flag)
        bool tune;                          // function tuning
        bool sieving;                       // sieving algorithm
        bool profile;                       // turn on profiling
        int sieving_mode;                   // sieving mode (1 = fixed size, 2 = differential, 3 = mean, 4 = mixed)
        int sieving_keep_top;               // how many clusters are to keep in case of sieving (fixed number/mixed mode)
        int sieving_diff;                   //
        int sieving_diff_num;               //
        std::string input_file_name;        // file containing input data
        std::string output_file_name;       // output file for results (cout if empty)
        std::string hs_input_file_name;     // file containing homogeneous system stats
        std::string hs_output_file_name;    // output file for homogeneous system stats
        std::string hs_data_input_file_name;     // file containing homogeneous system data
        std::string hs_data_output_file_name;    // output file for homogeneous system data
        std::string metric;                 // metric to be used to rank clusters (values: rel, int; default: rel)
        int rand_seed;                      // seed for random generator
        int res;                            // number of results to keep
        int sieving_max;                    // number of results to keep internally before sieving
        bool good_config;                   // true if command-line arguments are correct
        std::string error_message;          // error message (relevant only if good_config is false)
        unsigned int num_blocks;            // number of parallel execution blocks
        unsigned int num_threads;           // number of parallel threads per block

        unsigned int hs_count;              // number of homogeneous system clusters for each class

        std::string sieving_out;         // output file for new system after sieving procedure
        bool delete_input;                  // if true, input file is deleted after data is loaded
        bool tc_index;
        bool zi_index;
        bool strength_index;
        bool strength2_index;


        RunInfo()
        {
            tc_index = false;
            zi_index=false;
            strength_index=false;
            strength2_index=false;
            show_device_stats = false;
            verbose = false;
            silent = false;
            tune = false;
            sieving = false;
            profile = true;
            sieving_max = 100000;
            sieving_mode = 2;
            sieving_keep_top = 1;
            sieving_diff = 50;
            sieving_diff_num = 5;
            input_file_name = "";
            output_file_name = "";
            hs_input_file_name = "";
            hs_output_file_name = "";
            hs_data_input_file_name = "";
            hs_data_output_file_name = "";
            metric = "rel";
            rand_seed = std::time(0);
            res = 30;
            num_blocks = 128;
            num_threads = 32;
            hs_count = 0;
            sieving_out = "";
            delete_input = false;
        }
    };

    // typedef for struct pointer
    using RunInfo_p = std::unique_ptr<RunInfo>;

    RunInfo_p ProcessCommandLine(int argc, char** argv);

    void PrintUsage(char* command);

    void PrintRunInfo(const RunInfo& configuration);
}

#endif /* COMMAND_LINE_H */
