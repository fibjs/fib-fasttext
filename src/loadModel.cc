#include "model.h"
#include "AsyncWorker.h"

Napi::Value FasttextModel::LoadModel(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    Napi::HandleScope scope(env);

    if (info.Length() < 1)
        Napi::TypeError::New(env, "Path to model file is missing!").ThrowAsJavaScriptException();
    else if (!info[0].IsString())
        Napi::TypeError::New(env, "Model file path must be a string!").ThrowAsJavaScriptException();
    std::string filename = info[0].As<Napi::String>();

    return async_execute(env, [this, filename]() {
        return wrapper_->loadModel(filename);
    });
}
