

#ifndef WRAPPER_H
#define WRAPPER_H

// #include <time.h>

#include <atomic>
#include <memory>
#include <set>
#include <map>
#include <mutex>

#include "src/fasttext.h"

using fasttext::Args;
using fasttext::Dictionary;
using fasttext::FastText;
using fasttext::Matrix;
using fasttext::Model;
using fasttext::QMatrix;
using fasttext::real;
using fasttext::Vector;

struct PredictResult {
    std::string label;
    double value;
};

class Wrapper {
private:
    std::shared_ptr<Args> args_;
    std::shared_ptr<Dictionary> dict_;

    std::shared_ptr<Matrix> input_;
    std::shared_ptr<Matrix> output_;

    std::shared_ptr<QMatrix> qinput_;
    std::shared_ptr<QMatrix> qoutput_;

    std::shared_ptr<Model> model_;
    Matrix wordVectors_;
    FastText fastText_;

    // std::atomic<int64_t> tokenCount;
    // clock_t start;

    void signModel(std::ostream&);
    bool checkModel(std::istream&);

    std::map<std::string, std::string> loadModel(std::istream&);

    bool quant_;
    std::string modelFilename_;
    std::mutex mtx_;
    std::mutex precomputeMtx_;

    bool isLoaded_;
    bool isPrecomputed_;

    bool isModelLoaded() { return isLoaded_; }
    bool fileExist(const std::string& filename);
    std::map<std::string, std::string> getModelInfo();

public:
    Wrapper(std::string modelFilename);

    void getVector(Vector&, const std::string&);
    void getSentenceVector(std::istream& in, fasttext::Vector& svec);

    std::vector<PredictResult> predict(std::string sentence, int32_t k);
    std::map<std::string, std::string> train(const std::vector<std::string> args);
    std::map<std::string, std::string> quantize(const std::vector<std::string> args);

    std::vector<double> getWordVector(std::string word);
    std::vector<double> getSentenceVector(std::string text);

    void precomputeWordVectors();
    std::map<std::string, std::string> loadModel();
    std::map<std::string, std::string> loadModel(std::string filename);
};

#endif
