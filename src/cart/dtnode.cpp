#include "dtnode.h"

// input:  a Tuple, and decision tree
// output: class guess for the Tuple based on the dt

//TODO : use iteration or recursive to improve the performance
double DTNode::eval(const Tuple* const instance) {
  if (this->leaf)
    return this->pred;
  int f = this->feature;
  double v = this->value;
  if (instance->features[f] == UNKNOWN)
    if (this->child[MISSING] == 0)
      return this->pred;
    else
      return this->child[MISSING]->eval(instance);
  else
    if (instance->features[f] <= v)
      return this->child[YES]->eval(instance);
    else
      return this->child[NO]->eval(instance);
  return 0;
}



DTNode::DTNode(const vector<Tuple*>& data,
          ARGS& myargs,
          int maxdepth,  // the maximum depth of the tree
          int depth, // current depth in the tree
          int fk,
          bool par)
    : leaf(0), feature(0), value(UNKNOWN), leaf_depth(depth), pred(-1), args(myargs){
    // initialize fields
    int i;
    for (i = 0; i < CHILDTYPES; i++) {
      child[i] = 0;
    }

   // fprintf(stderr, "DTNode: 1\n");
    // get prediction
    pred = (args.pred == ALG_MEAN) ? average(data) : mode(data);

    // check if leaf node
    if (depth == maxdepth || same(data)) {
      leaf = true;
      return;
    }
  // fprintf(stderr, "DTNode: 2\n");
    // find criterion to split data, if cant make this leaf node
    int f_split; double v_split;

    if (!findSplitByHistogram(data,
         myargs.max_feature_index, f_split, v_split, fk, par, args)) {
      leaf = true;
      return;
    }
 //  fprintf(stderr, "DTNode: 3\n");
    // split data into 3 parts, based on criteria found
    vector<Tuple*> child_data[CHILDTYPES];
    splitData(data, child_data, f_split, v_split, args);

    if (!(child_data[YES].size() && child_data[NO].size())) {
      leaf = true;
      return;
    }
 //  fprintf(stderr, "DTNode: 4\n");
    // remember where we splitted, and recurse
    feature = f_split;
    value = v_split;
    child[YES] = new DTNode(child_data[YES], myargs, maxdepth, depth + 1, fk, par);
    child[NO] = new DTNode(child_data[NO], myargs, maxdepth, depth + 1, fk, par);


    if (child_data[MISSING].size()) {
      child[MISSING] = new DTNode(child_data[MISSING], myargs, maxdepth, depth + 1, fk, par);
    } else {
      child[MISSING] = 0;
    }
  }


  // input: vector of rawData (data), children vector, feature and value to split on
  // output: split rawData in data into appropriate children vectors
 void DTNode::splitData(const vector<Tuple*>& data, vector<Tuple*> child[CHILDTYPES],
    int f, double v, const ARGS& args) {

    int n = data.size();
    int i;
  //  fprintf(stderr, "split_data: 1\n");
    for (i = 0; i < CHILDTYPES; i++) {
      while(child[i].size()) {
        child[i].pop_back();
      }
    }


    for (i = 0; i < n; i++) {
     // fprintf(stderr, "(f: %d, number %d, data %f)", f, i, data[i]->features[f]);
      if (data[i]->features[f] == UNKNOWN) {
        child[MISSING].push_back(data[i]);
      } else if (data[i]->features[f] <= v) {
        child[YES].push_back(data[i]);
      } else {
        child[NO].push_back(data[i]);
      }
    }
  //  fprintf(stderr, "split_data: 2\n");

  }


DTNode::~DTNode() {
    if (!leaf) {
      delete child[YES];
      delete child[NO];
      if (child[MISSING] != 0) {
        delete child[MISSING];
      }
    }
  }
bool DTNode::getHistogram(vector<Tuple*> data,  // the data
                int feature_idx,  // the feature
                vector<double>& weight, // summation of weight of the bin
                vector<double>& weight_target, //sum of weight * target of the bin
                vector<double>& weight_target2, // sum of the weight * target * target of the bin
                int& missing_count, // number of rawData missing the feature
                double& missing_loss,  // regression loss of the missing values
                double& min_value, // minimal value of the features, except UNKNOWN
                double& max_value //maximal value of the features
                ) {
  min_value = 1e20;
  max_value = -1e20;
  int i, N = data.size();
  missing_count = 0;
  double missing_weight = 0;
  double missing_target = 0;
  double missing_target2 = 0;
  for (i = 0; i < N; i ++) {
    double val = data[i]->features[feature_idx];
    if (val == UNKNOWN)
      continue;
    if (data[i]->features[feature_idx] > max_value) {
      max_value = data[i]->features[feature_idx];
    }
    if (data[i]->features[feature_idx] < min_value) {
      min_value = data[i]->features[feature_idx];
    }
  }
   if (max_value == min_value) {
    return false;
   }
   double bin_width = (max_value - min_value) / HISTOGRAM_BIN;
   max_value = max_value +  bin_width/ 8;
   min_value = min_value - bin_width / 8;
   bin_width = (max_value - min_value) / HISTOGRAM_BIN;


  for (i = 0; i < N; i ++) {
    double val = data[i]->features[feature_idx];
    double w = data[i]->weight;
    double target  = data[i]-> target;

    if ( val == UNKNOWN) {
      missing_count += 1;
      missing_weight += w;
      missing_target +=  w * target;
      missing_target2 += w * target * target;
    } else {
      int bin_idx = int((val - min_value) / bin_width);
      weight[bin_idx] += w;
      weight_target[bin_idx] += w * target;
      weight_target2[bin_idx] += w * target * target;
     // printf("value %f, bin %d, %f %f %f\n", val, bin_idx, w, w* target, w * target * target);
    }
  }
  //printf("\nfeature %d ,  %d values ranges (%f %f), missing %d\n", feature_idx, (int)data.size(), min_value, max_value, missing_count);
  //printf("histogram of values ===\n");
  for (i = 0; i < HISTOGRAM_BIN; i ++) {
    //printf("bin %d, %f %f %f\n", i, weight[i], weight_target[i], weight_target2[i]);
  }
  //printf("missing_coutn %d, missing_weight %f, missing_target %f\n", missing_count, missing_weight, missing_target);


  //TODO: to verify this formula
  if (missing_weight > 0) {
    missing_loss = missing_target2 - missing_target * missing_target / missing_weight;
  } else {
    missing_loss = 0.0;
  }
  //printf("missing loss = %f\n", missing_loss);
  return true;
}



bool DTNode::findSplitByHistogram(
    vector<Tuple*> data,
    int NF,
    int& f_split,
    double& v_split,
    int K,
    bool par,
    const ARGS& args
  ) {
  f_split = -1;
  double min_loss = MY_DBL_MAX;
  int n = data.size(), i;

  vector<bool> skip;

  //printf(" pick %d  out of %d random features to split on, if specified\n", K, NF);
  for (i = 0; i <= NF; i++) {
    skip.push_back( (K > 0) ? true : false);
  }
  for (i = 0; i < K; i++) {
    int f;
    do
      f = rand() % (NF-2) + 1;
    while (!skip[f]);
    skip[f] = false;
  }

  //printf("total feature %d\n", NF - 1);
  for (int f = 1; f < NF; f ++) {
    //if (skip[f]) {
     // continue;
   // }
    if (skip[f] || args.skip_features[f]) {
      continue;
    }

    vector<double> weight(HISTOGRAM_BIN, 0);
    vector<double> weight_target(HISTOGRAM_BIN, 0);
    vector<double> weight_target2(HISTOGRAM_BIN, 0);
    double M = 0.0;
    int missing = 0;
    double min_value = 0.0, max_value = 0.0;
  //  fprintf(stderr, "getting histogram\n");
    if (!getHistogram(data, f, weight, weight_target, weight_target2, missing, M, min_value, max_value)){
      continue;
    }
  //  fprintf(stderr, "got histogram ok\n");
     if (missing == n) { // all data are missing,  set as leaf nodes, return value as the mean values
      continue;
    }

    double ybl, ybr, s, r, L, R, I, ywl, ywr, WL, WR;
    ybl = ybr = s = r = ywl = ywr = WL = WR = 0.0;

    for (i = 0 ; i < HISTOGRAM_BIN; i ++) {
      r += weight_target2[i];
      ywr += weight_target[i];
      WR += weight[i];
    }
    for (i = 0; i < HISTOGRAM_BIN; i ++) {

      s += weight_target2[i];
      r -= weight_target2[i];
      ywr -= weight_target[i];
      ywl += weight_target[i];
      WL += weight[i];
      WR -= weight[i];
   //   fprintf(stderr, "ok 2");
      r = r < 0 ? 0: r;
      L = s - ywl * ywl / WL;
      R = r - ywr * ywr / WR;
      L = L < 0 ? 0 : L;
      R = R < 0 ? 0 : R;
      // L, R are the mean loss of the left/right hand side
      I = L + R + M;
   //   fprintf(stderr, "ok 3");
     // printf("step %d, Missing Loss %f, Left Loss %f, Right Loss %f, total %f\n", i, M, L, R, I);
      if (I < min_loss) {
        min_loss = I;
        f_split = f;
        v_split = min_value + (max_value - min_value) / HISTOGRAM_BIN  * (i + 1);
      }

    }
  //  printf("f_split %d, v_split %f, loss : %f\n", f_split, v_split, min_loss);

  }
  if (f_split == -1 ) return false;
  return true;


}
