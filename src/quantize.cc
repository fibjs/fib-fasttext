#include "model.h"
#include "AsyncWorker.h"
#include "node-argument.h"

Napi::Value FasttextModel::Quantize(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    Napi::HandleScope scope(env);

    if (info.Length() < 1 || !info[0].IsObject())
        Napi::TypeError::New(env, "options must be an object").ThrowAsJavaScriptException();

    NodeArgument::NodeArgument nodeArg;
    NodeArgument::CArgument c_argument;

    try {
        Napi::Object confObj = info[1].As<Napi::Object>();
        c_argument = nodeArg.NapiObjectToCArgument(env, confObj);
    } catch (std::string errorMessage) {
        Napi::TypeError::New(env, errorMessage.c_str()).ThrowAsJavaScriptException();
    }

    int count = c_argument.argc;
    char** argument = c_argument.argv;

    std::vector<std::string> args;
    args.push_back("-command");
    args.push_back("quantize");

    for (int j = 0; j < count; j++) {
        args.push_back(argument[j]);
    }

    return async_execute(env, [this, args]() {
        return wrapper_->quantize(args);
    });
}
