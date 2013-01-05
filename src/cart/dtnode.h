#ifndef AM_DECISION_TREE_ML_H
#define AM_DECISION_TREE_ML_H

#include "tuple.h"
#include "args.h"

#include <map>
using namespace std;

double static mymax(double a, double b) {return a > b ? a : b; }
double static mymin(double a, double b) {return a < b ? a : b; }
inline double static squared(double a) {return a * a; }

#define HISTOGRAM_BIN 256



class DTNode
{
public:
  enum {YES, NO, MISSING, CHILDTYPES};

  DTNode* child[CHILDTYPES];
  int feature;
  double value; // feature and value this node splits on
  bool leaf;
  double pred; // what this node predicts if leaf node
  int leaf_depth;

  ARGS& args;

  DTNode(ARGS& myargs): leaf(0), feature(0), value(UNKNOWN), leaf_depth(0), pred(-1), args(myargs) {

  }
  DTNode(const vector<Tuple*>& data,
          ARGS& myargs,
          int maxdepth,  // the maximum depth of the tree
          int depth, // current depth in the tree
          int fk,
          bool par) ;


  // input: vector of rawData (data), children vector, feature and value to split on
  // output: split rawData in data into appropriate children vectors
  static void splitData(const vector<Tuple*>& data, vector<Tuple*> child[CHILDTYPES],
    int f, double v, const ARGS& args) ;

  // destructor: free up allocated memory
  ~DTNode();

  // input: vector of rawData
  // output: bool indicating if all instances in data share same label
  static bool same(const vector<Tuple*>& d) {
    double first = d[0]->target;
    for (int i = 1; i < d.size(); i++) {
      if (d[i]->weight > 0 && d[i]->target != first) {
         return false;
      }
    }
    return true;
  }

  // input: list of rawData
  // output: average of classes
  static double average(const vector<Tuple*>& data) {
    double sum = 0.0;
    double sumweight = 0.0;
    int n = data.size();
    for (int i = 0; i < n; i++){
      sum += data[i]->target * data[i]->weight;
      sumweight += data[i]->weight;
    }
    return sum/sumweight;
  }

  static double mode(const vector<Tuple*>& data) {
    double best, max_count = -1;
    map<double, double> count;
    for (int i = 0; i < data.size(); i++) {
      double t = data[i]->target;
      count[t] += data[i]->weight;
      double t_count = count[t];
      if (t_count > max_count) {
        max_count = t_count, best = t;
      }
    }
    return best;
  }

  // print features the tree has split on
  void printFeature(){
    cout << "(" << leaf_depth << ", " << feature
        << ", " << value << ", "<< pred << " )" << endl;
    if(child[YES]!=0) {
      child[YES]->printFeature();
    } else {
      cout<<"X ";
    }
    if(child[NO]!=0) {
      child[NO]->printFeature();
    } else {
      cout<<"X ";
    }
    return;
  }

  bool findSplitByHistogram(
    vector<Tuple*> data,
    int NF,
    int& f_split,
    double& v_split,
    int K,
    bool par,
    const ARGS& args
  );

  bool  getHistogram(vector<Tuple*> data,  // the data
    int feature_idx,  // the feature
    vector<double>& weight, // summation of weight of the bin
    vector<double>& weight_target, //sum of weight * target of the bin
    vector<double>& weight_target2, // sum of the weight * target * target of the bin
    int& missing_count, // number of rawData missing the feature
    double& missing_loss,  // regression loss of the missing values
    double& min_value, // minimal value of the features, except UNKNOWN
    double& max_value //maximal value of the features
    );

  static bool findSplit(vector<Tuple*> data,
    vector<int> countData,
    vector<int> invertIdx,
    int NF, int& fsplit,
    double& vsplit,
    int K, bool par,
    const ARGS& args);

  double eval(const Tuple* const instance);

};


#endif //AM_DECISION_TREE_ML_H
