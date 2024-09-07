#include "model.h"
#include "AsyncWorker.h"
#include <iostream>

Napi::FunctionReference FasttextModel::constructor;

Napi::Object FasttextModel::Init(Napi::Env env, Napi::Object exports)
{
    Napi::HandleScope scope(env);
    Napi::Function func = DefineClass(env, "FasttextModel",
        { InstanceMethod("loadModel", &FasttextModel::LoadModel),
            InstanceMethod("predict", &FasttextModel::Predict),
            InstanceMethod("train", &FasttextModel::Train),
            InstanceMethod("quantize", &FasttextModel::Quantize),
            InstanceMethod("get_word_vector", &FasttextModel::getWordVector),
            InstanceMethod("get_sentence_vector", &FasttextModel::getSentenceVector) });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("Model", func);
    return exports;
}

FasttextModel::FasttextModel(const Napi::CallbackInfo& info)
    : Napi::ObjectWrap<FasttextModel>(info)
{
    Napi::Env env = info.Env();
    Napi::HandleScope scope(env);
    std::string modelFileName = "";

    if (info.Length() > 0 && info[0].IsString()) {
        modelFileName = info[0].As<Napi::String>().Utf8Value();
    }

    this->wrapper_ = std::make_shared<Wrapper>(modelFileName);
}
