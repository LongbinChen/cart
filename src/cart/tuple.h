#ifndef AM_ML_Tuple_H
#define AM_ML_Tuple_H

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

#define MY_DBL_MAX 99999999999999.999999
#define MY_DBL_MIN (-MY_DBL_MAX)
#define UNKNOWN MY_DBL_MIN

class Tuple {// represents a data instance
  public:
    double* features;
    double label;
    double weight;
    int qid;
    double pred;
    double target;
    double baseline;
    string impressionid;
  	int idx;

    Tuple(int num_features, int dealwithmissing)
    : weight(1.0), label(-1), qid(-1), pred(-1), target(-1),  baseline(0.0) {
      features = new double[num_features];
      for (int i = 0; i < num_features; i++) {
        if (dealwithmissing) {
          features[i] =  UNKNOWN;
        } else {
  	      features[i] = 0.0;
        }
      }
    }

    ~Tuple() {
      if (features != NULL)
        delete[] features;
    }

};


#endif //AM_ML_Tuple_H
