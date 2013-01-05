//main.cpp

#include "args.h"
#include "dtnode.h"
#include "tuple.h"
#include "dataset.h"
#include "boostedtrees.h"
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>
using namespace boost;
#include <stdio.h>
#include <time.h>




int main(int argc, char* argv[]) {
  int i;
  srand(time(NULL));


  // get command line args

  ARGS myargs;
  if (!myargs.GetArgs(argc, argv)) {
    printf("RT-Rank Version 1.5 (alpha) usage: [-options] train.txt test.txt output.txt\n");
    printf("\nRequired flags:\n");
    printf("-a float\t stepsize.\n");
    printf("-d int \tmax treep depth (for gradient boosting trees are typically limited to a small depth, e.g. d=4).\n");
    printf("-p int\tnumber of processors/threads to use.\n");
    printf("-k \tnumber of randomly selected features used for building each trees of a random forest.\n");
    printf("-t int \tnumber of trees for random forest.\n");
    printf("-m \tuse mode for prediction at leaf nodes (default is mean)\n");
    printf("-z \tsubstitute missing features with zeros (recommended if missing features exist).\n");
    printf("-o \tlearning start from input (default false).\n");
    printf("-e \tuse entropy to measure impurity for CART (default is squared loss).\n");
    printf("\nOperation in wrapper mode (e.g. wih Python scripts):\n");
    printf("-w \tread in weights.\n");
    printf("-r \tnumber of trees (/iterations).\n");
    printf("-s \tprint the set of features used to built the tree to stdout.\n");
    printf("\n\n");
    return 0;
  }

  // load data from input files

  DataSet train(myargs);
  DataSet test(myargs);

  if (!train.checkData(myargs.train_file)){
    fprintf(stderr, "could not load data files\n");
    return 0;
  }
  if (!train.loadData(myargs.train_file)) {
    fprintf(stderr, "could not load training data files\n");
    return 0;
  }
  myargs.num_train = train.rawData.size();

  if (!test.loadData(myargs.test_file)) {
    fprintf(stderr, "could not load test data files\n");
    return 0;
  }
  myargs.num_test = test.rawData.size();

  fprintf(stderr, "data loaded successfully.\n");

  fprintf(stderr, "building boosted tree with %d rounds, %d trees per round.\n", myargs.rounds, myargs.trees);
  if (myargs.learn_from_input) {
    fprintf(stderr, "Learning start from input\n");
  }

  //train the trees
  BoostedTrees bdt(myargs);
  bdt.train(train, test);

  return 0;

}
