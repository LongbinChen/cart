//parse command line arguments
#ifndef AM_RT_ARGS_H
#define AM_RT_ARGS_H

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "tuple.h"
#include <vector>
using namespace std;

typedef vector<Tuple*> data_t;
typedef vector<double> preds_t;

enum alg_t { ALG_BOOST, ALG_FOREST, ALG_REGRESSION };
enum alg_loss { ALG_SQUARED_LOSS, ALG_ENTROPY };
enum alg_pred { ALG_MEAN, ALG_MODE };

class ARGS {
public:
  //data statistic

  char* train_file;
  char* test_outs;
  char* test_file;
  int   num_train; // number of rows in train file
  int   num_test; // number of rows in test file

  //added by lchen
  vector<bool> skip_features; // feature index that has only missing values; skip this in training;
  int pos_training_count;
  int neg_training_count;
  int max_feature_index;
  int valid_feature_count;

  //training options
  int trees;
  int processors;
  double alpha;
  int depth;
  int kfeatures;
  int rounds;
  alg_t alg;
  alg_loss loss;
  alg_pred pred;
  bool missing;
  bool print_features;
  bool learn_from_input;


  int verbose;

  ARGS()
  {
    processors = 1;
    trees = 1;
    alpha = 1.0;
    max_feature_index = 0;
    valid_feature_count = 0;
    depth = 1000;
    num_test = 1;
    kfeatures = -1;
    alg = ALG_REGRESSION;
    verbose = 0;
    rounds = 1;
    loss = ALG_SQUARED_LOSS;
    pred = ALG_MEAN;
    missing = true;
    print_features = false;
    learn_from_input = false;

  }

  int GetArgs(int argc, char* argv[]) {
    int index, c, i = 0;

    // option arguments
    opterr = 0;
    while ((c = getopt (argc, argv, "a:d:ef:i:k:t:l:mop:r:svwzBFR")) != -1)
      switch (c) {
        case 'a': alpha = atof(optarg); break;
        case 'd': depth = atoi(optarg); break;
        case 'e': loss = ALG_ENTROPY; break;
        case 'o': learn_from_input = true; break;
        case 't': trees = atoi(optarg); break;
        case 'p': processors = atoi(optarg); break;
        case 'k': kfeatures = atoi(optarg); break;
        case 'r': rounds = atoi(optarg);   break;
        case 's': print_features=1; break;
        case 'B': alg = ALG_BOOST; break;
        case 'F': alg = ALG_FOREST; break;
        case 'R': alg = ALG_REGRESSION; break;
        case 'm': pred = ALG_MODE; break;
        case 'v': verbose = 1; break;
        case 'z': missing = 0; break;
        case '?':
    if (optopt == 'c')
      fprintf (stderr, "Option -%c requires an argument.\n", optopt);
    else if (isprint (optopt))
      fprintf (stderr, "Unknown option `-%c'.\n", optopt);
    else
      fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
    return 0;
        default:
    return 0;
  }

  // non option arguments
  if (argc-optind < 3 || argc-optind > 3)
    return 0;
  train_file = argv[optind];
  test_file = argv[optind + 1];
  test_outs = argv[optind + 2];
  return 1;
 }
};


#endif
