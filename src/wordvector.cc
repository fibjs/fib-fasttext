#include "model.h"
#include "AsyncWorker.h"

Napi::Value FasttextModel::getWordVector(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    Napi::HandleScope scope(env);

    if (info.Length() == 0 || !info[0].IsString())
        Napi::TypeError::New(env, "word must be a string").ThrowAsJavaScriptException();

    std::string word = info[0].As<Napi::String>();

    return async_execute(env, [this, word]() {
        wrapper_->loadModel();
        wrapper_->precomputeWordVectors();
        return wrapper_->getWordVector(word);
    });
}
