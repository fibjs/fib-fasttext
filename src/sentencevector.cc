#include "model.h"
#include "AsyncWorker.h"

Napi::Value FasttextModel::getSentenceVector(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    Napi::HandleScope scope(env);

    if (info.Length() == 0 || !info[0].IsString())
        Napi::TypeError::New(env, "text must be a string").ThrowAsJavaScriptException();

    std::string text = info[0].As<Napi::String>();

    return async_execute(env, [this, text]() {
        wrapper_->loadModel();
        wrapper_->precomputeWordVectors();
        return wrapper_->getSentenceVector(text);
    });
}
