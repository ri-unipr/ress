/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   dci.cu
 * Author: e.vicari
 *
 * Created on 1 marzo 2016, 12.12
 */

#include <cstdlib>
#include "dci.h"

using namespace std;

/*
 * Entry point: retrieves configuration from command-line parameters
 * and runs main application if a good configuration was supplied.
 */
int main(int argc, char** argv) {
    
    // process command line to obtain configuration
    dci::RunInfo_p configuration = dci::ProcessCommandLine(argc, argv);
    
    if (!configuration->good_config) // bad command line parameters, print error and usage
    {
        cout << "Error: " << configuration->error_message << "\n\n";
        dci::PrintUsage(argv[0]);
        return 1;
    }
    
    if (configuration->profile) cudaProfilerStart();
    
    // store start/end time
    clock_t start = clock(), stop;
        
    // create application object
    dci::Application* app = new dci::Application(*configuration);

    app->Init(); // initialize application
    int res = app->Run(); // run main application
   
    // get end time
    stop = clock();

    // get duration
    cout << "Computing time:  " << app->elapsedTimeMilliseconds(start, stop) << " ms" << endl;
    
    delete app;
    
    if (configuration->profile) { cudaProfilerStop(); cudaDeviceSynchronize(); }
    
    return res;

}

