
/*
* File:   dci.cu
* Author: Emilio Vicari, Michele Amoretti
*/

#include <cstdlib>
#include "application.h"

using namespace std;

/*
 * Entry point: retrieves configuration from command-line parameters
 * and runs main application if a good configuration was supplied.
 */
int main(int argc, char** argv) {

    // process command line to obtain configuration
    dci::RunInfo_p configuration = dci::processCommandLine(argc, argv);

    if (!configuration->good_config) // bad command line parameters, print error and usage
    {
        cout << "Error: " << configuration->error_message << "\n\n";
        dci::printUsage(argv[0]);
        return 1;
    }

    if (configuration->profile) cudaProfilerStart();

    // store start/end time
    clock_t start = clock(), stop;

    // create application object
    dci::Application* app = new dci::Application(*configuration);

    app->init(); // initialize application
    int res = app->run(); // run main application

    // get end time
    stop = clock();

    // get duration
    cout << "Computing time:  " << app->elapsedTimeMilliseconds(start, stop) << " ms" << endl;

    delete app;

    if (configuration->profile) { cudaProfilerStop(); cudaDeviceSynchronize(); }

    return res;

}
