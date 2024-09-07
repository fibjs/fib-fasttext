#ifndef FASTTEXT_MODEL_H
#define FASTTEXT_MODEL_H

#include <napi.h>
#include "node-argument.h"
#include "wrapper.h"

class FasttextModel : public Napi::ObjectWrap<FasttextModel> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    FasttextModel(const Napi::CallbackInfo& info);

private:
    static Napi::FunctionReference constructor;

    Napi::Value LoadModel(const Napi::CallbackInfo& info);
    Napi::Value Predict(const Napi::CallbackInfo& info);
    Napi::Value Train(const Napi::CallbackInfo& info);
    Napi::Value Quantize(const Napi::CallbackInfo& info);
    Napi::Value getWordVector(const Napi::CallbackInfo& info);
    Napi::Value getSentenceVector(const Napi::CallbackInfo& info);

    std::shared_ptr<Wrapper> wrapper_;
};

#endif
