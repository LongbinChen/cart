#include "boostedtrees.h"
#include "args.h"

BoostedTrees::BoostedTrees(ARGS& ar) : args(ar)
{

}

BoostedTrees::~BoostedTrees()
{

}
int BoostedTrees::eval(DataSet& data, vector<double> & pred){
    int nt = trees.size();
    double rslt = 0.0;
    pred.clear();
    for (int j = 0; j < data.size; j ++ ) {
      rslt = args.learn_from_input ? data.rawData[j]->baseline:0.0;
      for (int i = 0; i < nt; i++) {
          //rslt +=  args.alpha * classify(data, trees[i]);
          rslt +=  args.alpha * trees[i]->eval(data.rawData[j]);
      }
      pred.push_back(rslt);
    }
    return data.size;
}
double BoostedTrees::eval(Tuple* data){
    int nt = trees.size();
    double rslt = 0.0;
    rslt = args.learn_from_input ? data->baseline:0.0;
    for (int i = 0; i < nt; i++) {
        //rslt +=  args.alpha * classify(data, trees[i]);
        rslt +=  args.alpha * trees[i]->eval(data);
    }
    return rslt;
}

double BoostedTrees::train(DataSet& train_data, DataSet& test_data){
    double ce, rce, rmse, tce, trce, trmse;
    learnBiasConstant(train_data);
    clock_t clock_begin = clock();
    fprintf(stderr, "training data initialized  ... ok \n");
     //fprintf(stderr, "ok -2\n");
    vector<double> train_preds;
    vector<double> test_preds;
    initVector(train_preds, train_data.size);
    initVector(test_preds, test_data.size);
    eval(train_data, train_preds);
    eval(test_data, test_preds);
    clock_t clock_end = clock();
    train_data.calRCE(train_preds, ce, rce, rmse);
    test_data.calRCE(test_preds, tce, trce, trmse);
    fprintf(stderr, "%d\t%.1f\t%f\t%f\t%f\t%f\t%f\t%f\n", 0,
              double(clock_end - clock_begin)/CLOCKS_PER_SEC,(float)rmse,
              (float)ce, (float) rce, (float)trmse, (float) tce, (float) trce);
    // eval_bdt(bdt, train, args);
     //  fprintf(stderr, "ok -1\n");
    fprintf(stderr, "training ... \n");
    fprintf(stderr, "round\ttime(s)\ttrain_rmse\ttrain_ce\ttrain_rce\ttest_rmse\ttest_ce \ttest_rce\n");


    for (int iter = 0; iter < args.rounds; iter++)  {
        // single regression tree
        DTNode* t = new DTNode(train_data.rawData, args, args.depth, 1, args.kfeatures, false);
        trees.push_back(t);
        //fprintf(stderr, "ok 1\n");
        // get classification on training data

        eval(train_data, train_preds);
        eval(test_data, test_preds);
        train_data.calRCE(train_preds, ce, rce, rmse);
        test_data.calRCE(test_preds, tce, trce, trmse);
        //fprintf(stderr, "ok 4\n");

        clock_end = clock();
        fprintf(stderr, "%d\t%.1f\t%f\t%f\t%f\t%f\t%f\t%f\n", iter,
                double(clock_end - clock_begin)/CLOCKS_PER_SEC,(float)rmse,
                (float)ce, (float) rce, (float)trmse, (float) tce, (float) trce);
        //fprintf(stderr, "ok 5\n");
        //train.update_target(train_preds);

        updateTarget(train_data, train_preds);

        if (args.print_features){
            cout << endl;
            t->printFeature();
            cout << endl;
        }
    }

}

// proected methods
void BoostedTrees::initVector(vector<double>& vects, int size){
    for (int i = 0; i < size; i++) {
        vects.push_back(0.0);
    }
    return;
}

void BoostedTrees::updateTarget(DataSet& train_data, vector<double>& train_preds){
    for (int i = 0; i < args.num_train; i++) {
        //  fprintf(stderr, "ok %d of %d\n", i, args.num_train);
        train_data.rawData[i]->target = train_data.rawData[i]->label - train_preds[i];
    }
    return;
}

void BoostedTrees::learnBiasConstant(DataSet& train_data){
    if (!args.learn_from_input) {
        DTNode* thead = new DTNode(args);
        thead->leaf = true;
        thead->pred =  train_data.avg_target / args.alpha;

        trees.push_back(thead);

        fprintf(stderr, "building the first tree using average target value %f\n", thead -> pred * args.alpha);

        for (int i = 0; i < train_data.size; i++) {
            train_data.rawData[i]->target = train_data.rawData[i]->label - thead->pred * args.alpha;
        }
    }
}




