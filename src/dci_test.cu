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
#include "application.h"


using namespace std;

/*
 *
 */
int main(int argc, char** argv) {

    // ********************************************************************* //
    // INITIALIZATION SECTION - call only once, store app object globally    //
    // ********************************************************************* //

    // create default configuration
    dci::RunInfo configuration = dci::RunInfo();

    // set configuration parameters
	  configuration.input_file_name = "../scripts/systems/cstr_26.txt"; 
    configuration.rand_seed = 123456;

    //configuration.hs_output_file_name = "../scripts/hs-kmpso-1.txt";
    configuration.hs_input_file_name = "../scripts/hs-dci-26var.txt";


    //configuration.hs_count=100000;

    //configuration.hs_input_file_name = "tesths.txt";

    // create application object
    dci::Application* app = new dci::Application(configuration);

    // initialize application
    app->Init();


    // ********************************************************************* //
    // COMPUTATION SECTION - repeat as needed                                //
    // ********************************************************************* //

    // allocate memory for clusters
    vector<register_t*> clusters(2);

    // allocate memory for cluster indexes
    vector<float> output(2);

    // create agent list for clusters

    vector<unsigned int> cluster1 = { 15, 18 };
    vector<unsigned int> cluster2 = { 6, 7, 13, 16, 25 };

    // allocate cluster bitmasks
    clusters[0] = (register_t*)malloc(app->getAgentSizeInBytes());
    clusters[1] = (register_t*)malloc(app->getAgentSizeInBytes());

    // set bitmasks from agent lists
    dci::ClusterUtils::setClusterFromPosArray(clusters[0], cluster1, app->getNumberOfAgents());
    dci::ClusterUtils::setClusterFromPosArray(clusters[1], cluster2, app->getNumberOfAgents());

    // perform computation
    app->ComputeIndex(clusters, output);

    // print clusters and results
    dci::ClusterUtils::print(cout, clusters[0], app->getNumberOfAgents());
    cout << " --> " << output[0] << endl;
    dci::ClusterUtils::print(cout, clusters[1], app->getNumberOfAgents());
    cout << " --> " << output[1] << endl;

    // free memory
    free(clusters[0]);
    free(clusters[1]);


    // ********************************************************************* //
    // SHUTDOWN SECTION - call once before exit                              //
    // ********************************************************************* //

    // delete app object
    delete app;

    return 0;

}
