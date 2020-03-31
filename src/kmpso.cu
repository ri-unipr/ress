//
//  kmpso.cu
//
//  Created by Gianluigi Silvestri on 5/12/2016.
//  Updated by Michele Amoretti on 29/3/2020.
//  Copyright © 2020 University of Parma. All rights reserved.
//

#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <random>
#include <vector>
#include <cfloat>
#include <limits>
#include <curand_kernel.h>
#include <string>
#include "dci.h"

using  namespace  std;

#define	D_max 500  // Max number of dimensions of the search space
#define	S_max 10000 // Max swarm size
#define K_max 3000 //Max number of clusters

// Structures


// Sub-programs
void identify_niches();
void k_means();
double alea( double a, double b );
vector<float> perf(int S, int D); // Fitness evaluation
void update();


// Global variables

double pi; // Useful for some test functions
int D; // Search space dimension
int S; // Swarm size
int K; // Number of seeds
int N; // number of results to keep
double *v; // vector for distance
double *V;
double *X;
double *P;
int *seed;
double *fx;
double *fp;
int *size;
double *M;
double *sigma;
double *bp;
double *fm;
bool *best;
vector < vector<unsigned int> > group(S_max, vector<unsigned int> (D_max));

int a;
double *d_X;
double *d_V;
double *d_P;
int *d_seed;
double *d_sigma;
double *d_bp;
curandState* devStates;
int TPB, NB;



double xmin, xmax; // Intervals defining the search space
int T;
vector<float> output1(S);
double x;
double c; // acceleration
double w; // constriction factor
dci::Application* app;
int fitness;
vector <double> results;
vector < vector<unsigned int> > g;
int r1, r2;
int b;



__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init ( seed, id, 0, &state[id] );
}

__device__ float generate( curandState* globalState, int ind )
{
  curandState localState = globalState[ind];
  float RANDOM = curand_uniform( &localState );
  globalState[ind] = localState;
  return RANDOM;
}



__global__ void compute(double* V,double* X,double* P,int* seed,double* bp,double* sigma,int xmin,int xmax,int S,int D, double c,curandState_t *state)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if(i < S*D)
  {
    if(seed[i/D]!=-1) // if belongs to a cluster
    {
      V[i] = V[i] +  (generate(state,i)*c) * ( P[i] - X[i] );
      V[i] = V[i] + (generate(state,i)*c) * ( bp[seed[i/D]*D+(i%D)] - X[i] );
      V[i] = 0.73*V[i];
      if(V[i]>2*(sqrt(sigma[seed[i/D]]))) V[i]=2*(sqrt(sigma[seed[i/D]]));
      if(V[i]<-2*(sqrt(sigma[seed[i/D]]))) V[i]=-2*(sqrt(sigma[seed[i/D]]));
      X[i] = X[i] + V[i];
    }
    else
    {
      V[i] = V[i] + (generate(state,i)*c) * ( P[i] - X[i] ); // cognition only
      V[i] = 0.73*V[i];
      X[i] = X[i] + V[i];
    }
    if ( X[i] < xmin )
    {
      X[i] = xmin; V[i] = 0;
    }
    if ( X[i] > xmax )
    {
      X[i] = xmax; V[i] = 0;
    }
  }
  else return;


}



int main(int argc, const char * argv[]) {

  clock_t tStart = clock();
  int d; // Current dimension
  int s; // Rank of the current particle
  int c1; // intervals for identify niches
  int interv; // print interval


  pi = acos( -1 ); // for rastrigin function
  if (argc < 10)
  { // We expect 5 arguments: the program name, the source path and the destination path
    cerr << "Usage: dimension swarm_size  n_seeds  range  n_iterations kmeans_interv print_interv  N_results inputfile outputfile" << endl;
    return 1;
  }
  else
  {
    D =atoi(argv[1]); // Search space dimension
    S=atoi(argv[2]);
    K=atoi(argv[3]);
    x=atof(argv[4]);
    T=atoi(argv[5]);
    c1=atoi(argv[6]);
    interv=atoi(argv[7]);
    N=atoi(argv[8]);
  }

  results.resize(N);
  g.resize(N);
  X= (double*) malloc(S*D*sizeof(double));
  V= (double*) malloc(S*D*sizeof(double));
  P= (double*) malloc(S*D*sizeof(double));
  v= (double*) malloc(D*sizeof(double));
  seed=(int*) malloc(S*sizeof(int));
  fx= (double*) malloc(S*sizeof(double));
  fp= (double*) malloc(S*sizeof(double));
  size=(int*) malloc(K*sizeof(int));
  M= (double*) malloc(K*D*sizeof(double));
  bp= (double*) malloc(K*D*sizeof(double));
  sigma= (double*) malloc(K*sizeof(double));
  fm= (double*) malloc(K*sizeof(double));
  best=(bool*) malloc(K*sizeof(bool));

  cudaMalloc((void **)&d_X, sizeof(double*)*S*D);
  cudaMalloc((void **)&d_V, sizeof(double*)*S*D);
  cudaMalloc((void **)&d_P, sizeof(double*)*S*D);
  cudaMalloc((void **)&d_seed, sizeof(int*)*S);
  cudaMalloc((void **)&d_sigma, sizeof(double*)*K);
  cudaMalloc((void **)&d_bp, sizeof(double*)*K*D);
  a=1024/D;
  TPB=512;
  NB=S*D/512;
  cudaMalloc ( &devStates, S*D*sizeof( curandState ) );

  // ********************************************************************* //
  // INITIALIZATION SECTION - call only once, store app object globally    //
  // ********************************************************************* //

  // create default configuration
  dci::RunInfo configuration = dci::RunInfo();

  // set configuration parameters
  configuration.input_file_name = argv[9];
  string output_file = argv[10];
  configuration.rand_seed = 123456;
  //configuration.hs_count=10000;
  string chosen_index = argv[11];
  if (chosen_index.compare("tc"))
    configuration.tc_index = true;
  else if (chosen_index.compare("zi"))
    configuration.zi_index = true;
  //configuration.zi_index = false;

  //configuration.hs_input_file_name = argv[9];
  //configuration.hs_output_file_name = "tesths.txt";
  // create application object
  app = new dci::Application(configuration);

  // initialize application
  app->Init();

  fitness=0;
  w = 0.73; // defined in thesis passaro
  c=2.05;
  // D-cube data
  xmin = -x; xmax = x;

  //-----------------------INITIALIZATION
  setup_kernel <<< NB,TPB >>> ( devStates,unsigned(time(NULL)) );
  srand( static_cast<unsigned int>(time(NULL)));

  for ( s = 0; s < S; s++ ) // create S particles
  {
    b=rand()%3;

    if(b==0)
    {
      do
      {
        r1=rand()%D;
        r2=rand()%D;
      }while(r1==r2);
      for ( d = 0; d < D; d++ )
      {
        if(r1==d || r2==d)
        {
          X[s*D+d] = alea( 0, xmax );
        }
        else
          X[s*D+d] = alea(xmin,0);
        V[s*D+d] = (alea( xmin, xmax ) - X[s*D+d])/2; // Non uniform
        P[s*D+d]=X[s*D+d];
      }
    }
    else if(b==1)
    {
      r1=rand()%D;
      for ( d = 0; d < D; d++ )
      {
        X[s*D+d] = alea( xmin, 0);
      }
      for(d=0; d<r1; d++)
      {
        r2=rand()%D;
        X[s*D+r2] = alea(0,xmax);
      }

      for ( d = 0; d < D; d++ )
      {
        V[s*D+d] = (alea( xmin, xmax ) - X[s*D+d])/2; // Non uniform
        P[s*D+d]=X[s*D+d];
      }
    }
    else
    {
      for ( d = 0; d < D; d++ )
      {
        X[s*D+d] = alea( xmin, xmax );
        V[s*D+d] = (alea( xmin, xmax ) - X[s*D+d])/2; // Non uniform
        P[s*D+d]=X[s*D+d];
      }
    }
  }
  cudaMemcpy(d_X, X, S*D*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, V, S*D*sizeof(double), cudaMemcpyHostToDevice);
  output1=perf(S,D);
  for (s=0; s<S; s++)
  {
    if(!(output1[s]<= DBL_MAX))output1[s]=0;
    fx[s]=output1[s];
    fp[s]=fx[s];
    for (int u=0; u<N; u++)
    {

      if(fx[s]>results[u])
      {
        for(int q=N-1; q>u; q--)
        {

          results[q]=results[q-1];
          g[q]=g[q-1];
        }
        results[u]=fx[s];
        g[u]=group[s];
        break;
      }
      else if(fx[s]==results[u])
      {
        if (g[u]==group[s]) break;
      }
    }
  }
  cudaMemcpy(d_P, P, S*D*sizeof(double), cudaMemcpyHostToDevice);

  k_means(); //k-means algorithm
  cudaMemcpy(d_seed, seed, S*sizeof(int), cudaMemcpyHostToDevice);

  //--------------------ITERATIONS
  for (int t=1; t<T; t++)
  {
    update();
    if (t % c1 ==0)
    {
      k_means();
      identify_niches();
    }

    //PRINT ON SCREEN

    std::ofstream outfile;
    std::ofstream outfile2;

    //outfile.open("dati_sistemi/sistema_test.txt", std::ios_base::app);
    //outfile.open("risultati/risultato_1.txt", std::ios_base::app);
    outfile.open(output_file, std::ios_base::app);
    //outfile2.open("dati_sistemi/sistema_test2.txt", std::ios_base::app);

    int var_count = 0;
    if(t%interv==0 || t==T-1){
      //for(int u=N-1; u>=0;u--) {
      for(int u=0; u<N;u++) {
        for (d=0; d<g[u].size(); d++) {
          for (int i=var_count; i<g[u][d]; i++)
          outfile << "0" << "\t";
          outfile << "1" << "\t";
          var_count = g[u][d]+1;
          outfile2 <<"["<< g[u][d] << "]";
          //outfile <<"["<< g[u][d] << "]";
        }
        while (var_count < D) {
          outfile << "0" << "\t";
          var_count++;
        }
        outfile << results[u]<< "\n";
        outfile2 << results[u]<< "\n";
        //cout <<" " << results[u]<< "\n";
        var_count = 0;
      }

      cout << "fitness computed " << fitness << " times\n";
      printf("Time taken: %.4fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
      cout <<"------------------------\n\n";
    }
    outfile.close();
    outfile2.close();

  }
  // delete app object
  cudaFree(d_X);
  cudaFree(d_V);
  cudaFree(d_P);
  cudaFree(d_bp);
  cudaFree(d_seed);
  cudaFree(d_sigma);
  delete app;

  return 0;
}


double alea( double a, double b )
{ // random number (uniform distribution) in [a b]
  double r;
  r=(double)rand(); r=r/RAND_MAX;
  return a + r * ( b - a );
}

vector<float> perf(int S, int D)
{

  // ********************************************************************* //
  // COMPUTATION SECTION - repeat as needed                                //
  // ********************************************************************* //

  // create agent list for clusters
  vector<unsigned int> cluster1(D);
  cluster1.clear();
  output1.clear();

  // allocate memory for clusters
  vector<register_t*> clusters(S);
  // allocate memory for cluster indexes
  vector<float> output(S);

  for( int s=0; s<S; s++)
  {
    group[s].clear();
    fitness++;
    for (int d=0; d<D; d++)
    {
      if (X[s*D+d]>=0)
      {
        cluster1.push_back(d);
        group[s].push_back(d);
      }
    }
    // allocate cluster bitmasks
    clusters[s] = (register_t*)malloc(app->getAgentSizeInBytes());
    // set bitmasks from agent lists
    dci::ClusterUtils::setClusterFromPosArray(clusters[s], cluster1, app->getNumberOfAgents());
    cluster1.clear();

  }

  // perform computation
  //app->ComputeStatisticalIndex(clusters, output);
  app->ComputeIndex(clusters, output);

  for (int s=0;s<S; s++)
  {
    // free memory
    free(clusters[s]);
  }

  return output;
}

void k_means()
{
  int k, d, s;
  int count=0;
  double k1, kt;
  bool change;
  bool insert;
  int seed1=-1;
  for (s=0;s<S;s++) seed[s]=-1;

  for (k=0; k<K; k++) //initialize seeds
  {
    for (d=0; d<D; d++)
    {
      M[k*D+d]= alea( xmin, xmax );
    }
    best[k]=false;
  }

  do
  {

    count++;
    change =false;
    for (k=0; k<K; k++)size[k]=0;
    for(s=0; s<S; s++) // for each particle i do
    {
      k1=0;
      insert=false; //doesn't belong to a cluster
      for (k=0; k<K; k++) // find the nearest seed mk
      {
        for (d=0; d<D; d++)
        {
          v[d] = P[s*D+d]-M[k*D+d];
        }
        kt=sqrt(inner_product(v, v+D, v, 0.0L)); // calculate distance p-m
        if((insert==false ) || kt<k1 )
        // if is the first evaluation or a smaller distance found
        {
          insert=true;
          k1=kt; // set the smallest distance
          seed1=k;
        }
      }
      // assign i to the cluster ck
      if(seed[s]!=seed1) // if found a nearer seed set it
      {
        seed[s]=seed1;
        change=true; // something has changed
      }
      size[seed[s]]+=1;// increase the size of the cluster

    }
    for(k=0; k<K; k++) // for each cluster recalculate the new mean
    {
      if(size[k]>0)
      {
        for(d=0; d<D; d++)
        {
          M[k*D+d]=0; // set the position to 0 to calculate the new one
          for (s=0; s<S; s++)
          {
            if (seed[s]==k)M[k*D+d]+=P[s*D+d];// for each particle in the cluster add the PB position
          }
          M[k*D+d]=M[k*D+d]/size[k]; // final new position
        }
      }
    }
  }while(change==true && count<=3);

  for(k=0;k<K;k++)
  {
    sigma[k]=0;
    if(size)
    for(s=0; s<S; s++)
    {
      if (seed[s]==k)
      {
        for (d=0; d<D; d++)
        {
          v[d] = P[s*D+d]-M[k*D+d];
        }
        sigma[k]+=inner_product(v, v+D, v, 0.0L); // distance (p-m)^2
      }
    }
    sigma[k]=sigma[k]/(size[k]-1);
  }
  cudaMemcpy(d_sigma, sigma, K*sizeof(double), cudaMemcpyHostToDevice);
  for(s=0; s<S; s++)
  {
    if(best[seed[s]]==false||fp[s]>fm[seed[s]])
    {
      fm[seed[s]]=fp[s];
      for(d=0; d<D; d++) bp[seed[s]*D+d]=P[s*D+d];
      best[seed[s]]=true;
    }

  }
  cudaMemcpy(d_bp, bp, K*D*sizeof(double), cudaMemcpyHostToDevice);
}

void update()
{
  int s, d;
  compute<<<NB, TPB>>>(d_V,d_X,d_P,d_seed,d_bp,d_sigma,xmin,xmax,S,D,c,devStates);
  cudaMemcpy(X, d_X, S*D*sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(V, d_V, S*D*sizeof(double), cudaMemcpyDeviceToHost);

  output1=perf(S,D);
  for (s=0; s<S; s++)
  {
    if(!(output1[s]<= DBL_MAX))output1[s]=0;
    fx[s]=output1[s];
    if (seed[s]!=-1)
    {
      if (fx[s]>fp[s])
      {
        for(d=0; d<D; d++) P[s*D+d]=X[s*D+d];
        fp[s]=fx[s];
        if(fp[s]>fm[seed[s]])
        {
          fm[seed[s]]=fp[s];
          for(d=0; d<D; d++) bp[seed[s]*D+d]=P[s*D+d];
        }
      }

    }
    else
    {
      if (fx[s]>fp[s])
      {
        for(d=0; d<D; d++) P[s*D+d]=X[s*D+d];
        fp[s]=fx[s];

      }
    }
    for (int u=0; u<N; u++)
    {
      if(fx[s]>results[u])
      {
        for(int q=N-1; q>u; q--)
        {
          results[q]=results[q-1];
          g[q]=g[q-1];
        }
        results[u]=fx[s];
        g[u]=group[s];
        break;
      }
      else if(fx[s]==results[u])
      {
        if (g[u]==group[s]) break;
      }
    }


  }
  cudaMemcpy(d_P, P, S*D*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bp, bp, K*D*sizeof(double), cudaMemcpyHostToDevice);

}

void identify_niches()
{
  int navg=0; // avarage number of particles per cluster
  int nu;
  double wf; //worst fitness
  double worst=-1; // worst particle
  bool empty;
  int k, s, d;

  for(k=0; k<K; k++)
  {
    navg+=size[k];
  }
  navg=navg/K; // calculate average number of particles per cluster
  nu=0;
  for(k=0; k<K; k++)
  {
    if (size[k]>navg)
    {
      for(int z=0; z<size[k]-navg; z++)
      {
        empty=true;
        wf=0;
        for(s=0;s<S;s++)
        {
          if(seed[s]==k)
          {
            if (fx[s]<wf || empty)
            {
              wf=fx[s];
              worst=s;
              empty=false;
            }
          }
        }
        for(s=worst;s<S;s++) // remove the nj-navg worst particles from cj
        {
          for(d=0; d<D; d++)  X[s*D+d]=X[(s+1)*D+d];

          for(d=0; d<D; d++)  P[s*D+d]=P[(s+1)*D+d];


          for(d=0; d<D; d++)  V[s*D+d]=V[(s+1)*D+d];

          fx[s]=fx[s+1];
          fp[s]=fp[s+1];
          seed[s]=seed[s+1];
          group[s]=group[s+1];
        }
      }
      nu+=size[k]-navg;
      size[k]-=size[k]-navg;
    }
  }
  for(s=S-nu;s<S;s++) // reinitialize the nu un-niched particles
  {
    b=rand()%3;
    if (b==0)
    {

      do
      {
        r1=rand()%D;
        r2=rand()%D;
      }while(r1==r2);
      for ( d = 0; d < D; d++ )
      {

        if(r1==d || r2==d)
        {
          X[s*D+d] = alea( 0, xmax );
        }
        else X[s*D+d] = alea(xmin,0);
        V[s*D+d] = (alea( xmin, xmax ) - X[s*D+d])/2; // Non uniform
      }
    }
    else if(b==1)
    {
      r1=rand()%D;
      for ( d = 0; d < D; d++ )
      {
        X[s*D+d] = alea( xmin, 0);
      }
      for(d=0; d<r1; d++)
      {
        r2=rand()%D;
        X[s*D+r2] = alea(0,xmax);
      }

      for ( d = 0; d < D; d++ )
      {
        V[s*D+d] = (alea( xmin, xmax ) - X[s*D+d])/2; // Non uniform
        P[s*D+d]=X[s*D+d];
      }
    }
    else
    {
      for ( d = 0; d < D; d++ )
      {
        X[s*D+d] = alea( xmin, xmax );
        V[s*D+d] = (alea( xmin, xmax ) - X[s*D+d])/2; // Non uniform
      }

    }

    seed[s]=-1;
  }
  cudaMemcpy(d_seed, seed, S*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_X, X, S*D*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, V, S*D*sizeof(double), cudaMemcpyHostToDevice);
  output1=perf(S,D);
  for (s=0; s<S; s++)
  {
    if(!(output1[s]<= DBL_MAX))output1[s]=0;
    if (seed[s]==-1)
    {
      fx[s]=output1[s];
      for(d=0; d<D; d++) P[s*D+d]=X[s*D+d]; // Best position = current one
      fp[s]=fx[s];
    }

    for (int u=0; u<N; u++)
    {
      if(fx[s]>results[u])
      {
        for(int q=N-1; q>u; q--)
        {
          results[q]=results[q-1];
          g[q]=g[q-1];
        }
        results[u]=fx[s];
        g[u]=group[s];
        break;
      }
      else if(fx[s]==results[u])
      {
        if (g[u]==group[s]) break;
      }
    }

  }
  cudaMemcpy(d_P, P, S*D*sizeof(double), cudaMemcpyHostToDevice);

}
