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
        bool zi_index;
        bool strength_index;
        bool strength2_index;
        
        
        RunInfo() 
        { 
            
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
    
    /*
     * Processes command-line arguments to retrieve execution configuration
     */
    RunInfo_p ProcessCommandLine(int argc, char** argv)
    {
        
        // constructor sets default values
        RunInfo_p pointer_to_data(new RunInfo());
        int pos;
        
        // cycle all command-line arguments
        for (int i = 1; i < argc; ++i)
        {
            std::string arg(argv[i]);
            
            /*
             * input filename retrieval
             */
            if (arg.length() >= 2 && (arg[0] != '-' || arg[1] != '-')) // this is a filename
            {
                if (pointer_to_data->input_file_name != "") // file was already specified, error
                {
                    pointer_to_data->error_message = "more than one file specified or unknown argument: " + arg;
                    return pointer_to_data;
                }
                pointer_to_data->input_file_name = arg;
            }
            
            /*
             * flags
             */
            else if (arg == "--device-info") // turn on device info flag
                pointer_to_data->show_device_stats = true;
            else if (arg == "--verbose") // turn on verbose flag
                pointer_to_data->verbose = true;
            else if (arg == "--tune") // turn on tuning flag
                pointer_to_data->tune = true;
            else if (arg == "--profile") // turn on profiling flag
                pointer_to_data->profile = true;
            else if (arg == "--silent") // turn on silent flag
                pointer_to_data->silent = true;
            else if (arg == "--sv") // turn on sieving
                pointer_to_data->sieving = true;
            else if (arg == "--delete-input") // delete input file after reading
                pointer_to_data->delete_input = true;
            else if (arg == "--si") 
                pointer_to_data->strength_index = true;
            else if (arg == "--zi") 
                pointer_to_data->zi_index = true;
            else if (arg == "--si2") 
                pointer_to_data->strength2_index = true;
            /*
             * valued parameters
             */
            else if (arg.length() >= 2 && (pos = arg.find_first_of(':')) != std::string::npos) // this is a --XXX:VVV parameter
            {

                if (pos == arg.length() - 1) // empty argument value
                {
                    pointer_to_data->error_message = "no value specified for argument: " + arg;
                    return pointer_to_data;
                }
                
                std::string name = arg.substr(0, pos);
                std::string value = arg.substr(pos + 1);
                
                if (name ==  "--out")
                    pointer_to_data->output_file_name = value;
                else if (name == "--hs-in")
                    pointer_to_data->hs_input_file_name = value;
                else if (name == "--hs-out")
                    pointer_to_data->hs_output_file_name = value;
                else if (name == "--hs-data-in")
                    pointer_to_data->hs_data_input_file_name = value;
                else if (name == "--hs-data-out")
                    pointer_to_data->hs_data_output_file_name = value;
                else if (name == "--sv-out")
                    pointer_to_data->sieving_out = value;
                else if (name == "--metric")
                    pointer_to_data->metric = value;
                else if (name == "--rand-seed")
                {
                    if ((pointer_to_data->rand_seed = std::atoi(value.data())) == 0)
                    {
                        pointer_to_data->error_message = "rand seed invalid or zero: " + value;
                        return pointer_to_data;
                    }
                }
                else if (name == "--res")
                {
                    if ((pointer_to_data->res = std::atoi(value.data())) == 0)
                    {
                        pointer_to_data->error_message = "number of results invalid or zero: " + value;
                        return pointer_to_data;
                    }
                }
                else if (name == "--sv-max")
                {
                    if ((pointer_to_data->sieving_max = std::atoi(value.data())) == 0)
                    {
                        pointer_to_data->error_message = "number of internal results invalid or zero: " + value;
                        return pointer_to_data;
                    }
                }
                else if (name == "--sv-mode")
                {
                    if ((pointer_to_data->sieving_mode = std::atoi(value.data())) < 1 || pointer_to_data->sieving_mode > 4)
                    {
                        pointer_to_data->error_message = "sieving mode unknown: " + value;
                        return pointer_to_data;
                    }
                }
                else if (name == "--sv-keep-top")
                {
                    if ((pointer_to_data->sieving_keep_top = std::atoi(value.data())) == 0)
                    {
                        pointer_to_data->error_message = "number of super clusters to keep invalid or zero: " + value;
                        return pointer_to_data;
                    }
                }
                else if (name == "--sv-diff")
                {
                    if ((pointer_to_data->sieving_diff = std::atoi(value.data())) == 0)
                    {
                        pointer_to_data->error_message = "sieving differential coefficient invalid or zero: " + value;
                        return pointer_to_data;
                    }
                }
                else if (name == "--sv-diff-num")
                {
                    if ((pointer_to_data->sieving_diff_num = std::atoi(value.data())) == 0)
                    {
                        pointer_to_data->error_message = "differential sieving set size invalid or zero: " + value;
                        return pointer_to_data;
                    }
                }
                else if (name == "--nb")
                {
                    if ((pointer_to_data->num_blocks = std::atoi(value.data())) == 0)
                    {
                        pointer_to_data->error_message = "number of blocks invalid or zero: " + value;
                        return pointer_to_data;
                    }
                }
                else if (name == "--nt")
                {
                    if ((pointer_to_data->num_threads = std::atoi(value.data())) == 0)
                    {
                        pointer_to_data->error_message = "number of threads invalid or zero: " + value;
                        return pointer_to_data;
                    }
                }
                else if (name == "--hs-count")
                {
                    pointer_to_data->hs_count = std::atoi(value.data());
                }
                else
                {
                    pointer_to_data->error_message = "unknown argument: " + name;
                    return pointer_to_data;
                }
            }
            
            /*
             * unknown parameters
             */
            else
            {
                pointer_to_data->error_message = "unknown argument: " + arg;
                return pointer_to_data;
            }
        }
        
        if (pointer_to_data->input_file_name == "") // check if input file has been specified
        {
            pointer_to_data->error_message = "input file not specified";
            return pointer_to_data;
        }
        
        // everything OK
        pointer_to_data->good_config = true;        
        
        return pointer_to_data;
        
    }
    
    /*
     * Prints usage
     */
    void PrintUsage(char* command)
    {
        
        std::cout << "USAGE:\n" << command << " input_file [--out:file] [--hs-in:file] [--hs-out:file] [--rand-seed:number] [--res:number] [--device-info] [--verbose]\n\n";
        std::cout << "PARAMETERS:\n";
        std::cout << "input_file           path to input data file (required)\n";
        std::cout << "--zi                 use ZI = 2*M*I - g/ sqrt(2*g) as index  (Default)\n";
        std::cout << "--si                 use the strength index SI = 2*M*I/g as index\n";
        std::cout << "--si2                use the strength index SI2 = I/Imax as index\n";
        std::cout << "--out:file           writes results to file\n";
        std::cout << "--nb:file            number of parallel execution blocks (default 128)\n";
        std::cout << "--nt:file            number of threads per block (default 8)\n";
        std::cout << "--hs-in:file         reads homogeneous system stats from file\n";
        std::cout << "--hs-out:file        writes homogeneous system stats to file\n";
        std::cout << "--hs-data-in:file    reads homogeneous system data from file\n";
        std::cout << "--hs-data-out:file   writes homogeneous system data to file\n";
        std::cout << "--hs-count:number    number of clusters to be used in homogeneous system statistics computation for each cluster size\n";
        std::cout << "--metric:value       metric used to rank clusters ('rel' for relevance index, 'int' for integration; default: 'rel')\n";
        std::cout << "--rand-seed:number   sets random number generator seed\n";
        std::cout << "--res:number         sets number of results to keep (default 30)\n";
        std::cout << "--sv                 applies sieving algorithm\n";
        std::cout << "--sv-max:number      sets number of results to keep internally (default 100000). Only with --sv option\n";
        std::cout << "--sv-mode:number     sets sieving mode for super-cluster selection (1 = fixed | 2 = differential | 3 = mean | 4 = manual, default 2)\n";
        std::cout << "--sv-keep-top:number sets number of results to keep for next cycle (fixed number/differential mode, default 1)\n";
        std::cout << "--sv-diff:number     sets min difference (%) for differential sieving mode (differential mode, default 100)\n";
        std::cout << "--sv-diff-num:number sets number of clusters to search for differential sieving mode (differential mode, default 5)\n";
        std::cout << "--sv-out:file        sets output file for new system data after sieving\n";
        std::cout << "--delete-input       deletes input file after reading\n";
        std::cout << "--device-info        writes device statistics at startup\n";
        std::cout << "--verbose            turn on debug messages\n";
        std::cout << "--silent             turn off all output except final results (overrides verbose flag)\n";
        std::cout << "--tune               turn on function tuning\n";
        std::cout << "--profile            turn on CUDA profiling\n";
        std::cout << "\n";
        
    }
    
    /*
     * Prints execution configuration parameters
     */
    void PrintRunInfo(const RunInfo& configuration)
    {
        std::cout << "Input file           " << configuration.input_file_name << '\n';
        std::cout << "Output file          " << configuration.output_file_name << '\n';
        std::cout << "N. blocks            " << configuration.num_blocks << '\n';
        std::cout << "N. threads           " << configuration.num_threads << '\n';
        std::cout << "HS input file        " << configuration.hs_input_file_name << '\n';
        std::cout << "HS output file       " << configuration.hs_output_file_name << '\n';
        std::cout << "HS data input file   " << configuration.hs_data_input_file_name << '\n';
        std::cout << "HS data output file  " << configuration.hs_data_output_file_name << '\n';
        std::cout << "HS sample count      " << configuration.hs_count << '\n';
        std::cout << "Metric               " << configuration.metric << '\n';
        std::cout << "Random seed          " << configuration.rand_seed << '\n';
        std::cout << "Results to keep      " << configuration.res << '\n';
        std::cout << "Use sieving          " << configuration.sieving << '\n';
        std::cout << "Results for sieving  " << configuration.sieving_max << '\n';
        std::cout << "Sieving output       " << configuration.sieving_out << '\n';
        std::cout << "Sieving mode         " << configuration.sieving_mode << '\n';
        std::cout << "Sieving keep top     " << configuration.sieving_keep_top << '\n';
        std::cout << "Sieving diff         " << configuration.sieving_diff << '\n';
        std::cout << "Sieving diff num     " << configuration.sieving_diff_num << '\n';
        std::cout << "Delete input file    " << configuration.delete_input << '\n';
        std::cout << "Show device info     " << configuration.show_device_stats << '\n';
        std::cout << "Show debug messages  " << configuration.verbose << '\n';
        std::cout << "Silent               " << configuration.silent << '\n';
        std::cout << "Function tuning      " << configuration.tune << '\n';
        std::cout << "CUDA profiling       " << configuration.profile << '\n';
    }

}

#endif /* COMMAND_LINE_H */

