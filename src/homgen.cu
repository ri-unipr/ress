
/*
* File:   hom_gen.cu
* Author: Michele Amoretti
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
  //dci::RunInfo_p configuration = dci::processCommandLine(argc, argv);
  dci::RunInfo_p configuration(new dci::RunInfo());
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
      if (configuration->input_file_name != "") // file was already specified, error
      {
        configuration->error_message = "more than one file specified or unknown argument: " + arg;
        cout << "Error: " << configuration->error_message << "\n\n";
        return -1;
      }
      configuration->input_file_name = arg;
    }
    else if (arg.length() >= 2 && (pos = arg.find_first_of(':')) != std::string::npos) // this is a --XXX:VVV parameter
    {

      if (pos == arg.length() - 1) // empty argument value
      {
        configuration->error_message = "no value specified for argument: " + arg;
        cout << "Error: " << configuration->error_message << "\n\n";
        return -1;
      }

      std::string name = arg.substr(0, pos);
      std::string value = arg.substr(pos + 1);

      if (name == "--hs-out")
      configuration->hs_output_file_name = value;
      else if (name == "--hs-data-out")
      configuration->hs_data_output_file_name = value;
      /*
      else if (name == "--rand-seed")
      {
        if ((configuration->rand_seed = std::atoi(value.data())) == 0)
        {
          configuration->error_message = "rand seed invalid or zero: " + value;
          return -1;
        }
      }
      */
      else
      {
        configuration->error_message = "unknown argument: " + name;
        cout << "Error: " << configuration->error_message << "\n\n";
        return -1;
      }
    }
    /*
    * unknown parameters
    */
    else
    {
      configuration->error_message = "unknown argument: " + arg;
      cout << "Error: " << configuration->error_message << "\n\n";
      return -1;
    }
  }

  configuration->verbose = true;
  configuration->tc_index = true;
  srand(time(NULL));
  configuration->rand_seed = rand();

  // store start/end time
  clock_t start = clock(), stop;

  // create application object
  dci::Application* app = new dci::Application(*configuration);

  app->init(); // initialize application

  // get end time
  stop = clock();

  // get duration
  cout << "Computing time:  " << app->elapsedTimeMilliseconds(start, stop) << " ms" << endl;

  delete app;

  return 1;

}
