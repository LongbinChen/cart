#ifndef AM_BOOSTED_TREE_ML_H
#define AM_BOOSTED_TREE_ML_H
#include "dtnode.h"
#include "tuple.h"
#include "dataset.h"
#include "args.h"

class BoostedTrees {
  public:
    vector<DTNode*> trees;
    ARGS& args;
    BoostedTrees(ARGS& args);
    ~BoostedTrees();

    double eval(Tuple* data);
    int eval(DataSet& data, vector<double> & pred);
    double train(DataSet&data, DataSet& test_data);

  protected:
    void learnBiasConstant(DataSet& train_data);
    void initVector(vector<double>& vects, int size);

    void updateTarget(DataSet& train_data, vector<double>& train_pred);

};

#endif //AM_BOOSTED_TREE_ML_H
