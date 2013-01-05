#ifndef AM_ML_DataSet_H
#define AM_ML_DataSet_H
#include "args.h"
#include "tuple.h"


class DataSet {
  public:

    DataSet(ARGS&);
    ~DataSet();

    bool checkData(char* filename);
    bool loadData(char* filename);
    bool loadWeight(char* weightfile);

    void calAverageTarget();
    bool calRCE(const vector<double>& preds, double& ce, double& rce, double& rmse) ;

    //data members

    vector<Tuple*> rawData;
    ARGS& args;
    int size;
    int dimension;
    double avg_target;

};



#endif //AM_ML_DataSet_H
