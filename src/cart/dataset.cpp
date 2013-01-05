#include "dtnode.h"
#include "tuple.h"
#include "dataset.h"

DataSet::DataSet(ARGS& a): args(a){
    size = 0;
    dimension = 0;
}
DataSet::~DataSet(){
	int i = 0;
    for (i = 0; i < rawData.size(); i ++){
    	if (rawData[i] != NULL) {
    		delete rawData[i];
    	}
    }
}

bool DataSet::checkData(char* file){
//	int Tuple::check_input_ads(char* file, int& max_feature_index, int& num_features,
//  int& row_count, int& pos_count, int& neg_count, vector<int>& missing_features){
  fprintf(stderr, "Checking data file %s ...\n", file);

  ifstream input(file);


  if (input.fail()) {
    fprintf(stderr, "can't initialize the  file %s ", file);
    return 0;
  }


  int max_feature_index = 0;
  int valid_feature_count = 0;
  int row_count = 0;
  int pos_count = 0;
  int neg_count = 0;
  int num_features = 0;
  string strline;
  string feature_str;
  int idx=0;

  vector<int> feature_count;
  while( getline(input, strline)) {
    if (idx % 1000 == 0){
        fprintf(stderr, "\rchecking line %d ..", idx);
    }
    idx ++;
    char* line = strdup(strline.c_str()); // easier to get a line as string, but to use strtok on a char*
    char* tok = NULL;
    string impressionid, feature_str;
    double baseline, label;
    // Extract impressionid (first item)
    tok = strtok(line, "\t");
    if (tok) {
      impressionid = tok;
    }
    tok = strtok(NULL, "\t");
    if (tok) {
      feature_str = tok;
    }
    tok = strtok(NULL, "\t");
    if (tok) {
      baseline = atof(tok);
    }
    tok = strtok(NULL, "\t");
    if (tok) {
      label = atof(tok);
      if (label >= 1.0) pos_count ++;
      if (label == 0.0) neg_count ++;
    }

    row_count ++;
    char* feature_line = strdup(feature_str.c_str());
    //printf("feature line: %s\n", feature_line);
    tok = strtok(feature_line, ",");
    while (tok != NULL) {
        string bit = tok;

        int colon_index = bit.find(":");

        string feature = bit.substr(0, colon_index);


        string value = bit.substr(colon_index + 1, bit.length() - colon_index - 1);


        int f = atoi(feature.c_str());
        double v = (double)atof(value.c_str());
        if (f > max_feature_index)
          max_feature_index = f;
        while (feature_count.size() <= f)
          feature_count.push_back(0);

        feature_count[f] += 1;
        tok = strtok(NULL, ",");
    }
    free(line);
  }
  fprintf(stderr, "max feature index : %d\n", max_feature_index);
  int i = 0;
  for (i = 0; i < feature_count.size(); i ++) {
    if (feature_count[i] == 0){
      args.skip_features.push_back(i);
      fprintf(stderr, "feature %d has only missing feature values \n", i);
    } else {
      num_features ++;
    }
  }

  args.pos_training_count = pos_count;
  args.neg_training_count = neg_count;
  args.valid_feature_count =  valid_feature_count;
  args.max_feature_index = max_feature_index;
  size = row_count;
  return true;
}



bool DataSet::loadData(char* file){
  ifstream input(file);
  if (input.fail()) {
    fprintf(stderr, "can't initialize the  file %s ", file);
    return false;
  }

  string strline;
  string feature_str;

  int idx=0;
  double label = 0.0;
  //printf("mssing = %d \n", missing);
  while( getline(input, strline) ) {

    Tuple* t = new Tuple(args.max_feature_index, args.missing);

    if (t == NULL)
      cout << "out of memory" << endl;

    char* line = strdup(strline.c_str()); // easier to get a line as string, but to use strtok on a char*
    char* tok = NULL;
    t->weight = 1;
    // Extract impressionid (first item)
    tok = strtok(line, "\t");
    if (tok) {
      t->impressionid = tok;
    }
    tok = strtok(NULL, "\t");
    if (tok) {
      feature_str = tok;
    }
    tok = strtok(NULL, "\t");
    if (tok) {
      t->baseline = atof(tok);
    }
    tok = strtok(NULL, "\t");
    if (tok) {
      label = atof(tok);
      if (label >= 1.0) label = 1.0;
      t->label = label;
      t->target = t->label;
    }
    //printf("impressid %s, baseline %f, label %f, ", t->impressionid.c_str(), t->baseline, t->label);
    char* feature_line = strdup(feature_str.c_str());
    //printf("feature line: %s\n", feature_line);
    tok = strtok(feature_line, ",");
    int max_feature_idx = -1;
    while (tok != NULL) {

      string bit = tok;
      int colon_index = bit.find(":");
      string feature = bit.substr(0, colon_index);
      string value = bit.substr(colon_index + 1, bit.length() - colon_index - 1);
      int f = atoi(feature.c_str());
      double v = (double)atof(value.c_str());
      //printf("(%s, %d, %4.2f),", tok, f, v);
      if (v != 0) {// added
        t->features[f] = v;
        max_feature_idx = f;
      }
      tok = strtok(NULL, ",");
    }
    //printf("max feature idx : %d\n", max_feature_idx);
    t->features[0]=idx++;
    free(line);
    rawData.push_back(t);
  }
  size = rawData.size();
  calAverageTarget();
  return 1;
}

bool DataSet::loadWeight(char* weightfile){
  return true;

}


void DataSet::calAverageTarget(){
  double sum = 0;
  int i = 0;
  if (rawData.size() == 0) return;
  for (i = 0; i < size; i ++ ){
    sum += rawData[i]->label;
  }
  avg_target = sum / size;
  return ;
}


// given preds and labels, get relative Cross Entropy
bool DataSet::calRCE(const vector<double>& preds, double& ce, double& rce, double& rmse) {
  double r = 0;
  double avg = avg_target;
  rmse = 0.0;
  int i;
  //fprintf(stderr, "datasize %d , avg = %f\n", size, avg);
  for (i = 0; i < size; i ++) {
    double v = preds[i] > 0.999999 ? 0.999999 : preds[i];
    v = v < 0.000001 ? 0.000001 : v;
    r += rawData[i]->label == 1 ? (log(v) / log(2.0)) : (log(1 - v) / log(2.0));
    rmse += squared(v - rawData[i]->label);
  }
  double entropy = - avg * log(avg)/log(2.0) - (1.0 - avg) * log (1.0 - avg)/log(2.0);
  ce = -r / size;
  rce = (entropy - ce) / entropy;
  rmse = rmse / size;

  //fprintf(stderr, "mean %f, entropy %f, ce %f, rce %f n: %d\n", avg, entropy, ce, rce, N);
  return  true;
}
