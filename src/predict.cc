#include "model.h"
#include "AsyncWorker.h"

Napi::Value FasttextModel::Predict(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    Napi::HandleScope scope(env);

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "sentence must be a string").ThrowAsJavaScriptException();
    }
    std::string sentence = info[0].As<Napi::String>().Utf8Value();

    int32_t k = 1;
    if (info.Length() > 1 && info[1].IsNumber())
        k = info[1].As<Napi::Number>().Int32Value();

    return async_execute(env, [this, sentence, k]() {
        wrapper_->loadModel();

        auto result_ = wrapper_->predict(sentence, k);

        nlohmann::json retVal = nlohmann::json::array();
        for (auto& r : result_)
            retVal.push_back({ { "label", r.label }, { "value", r.value } });

        return retVal;
    });
}
