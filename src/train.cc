#include "model.h"
#include "AsyncWorker.h"
#include "node-argument.h"

Napi::Value FasttextModel::Train(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    Napi::HandleScope scope(env);

    if (info.Length() < 2)
        Napi::TypeError::New(env, "requires at least 2 parameters").ThrowAsJavaScriptException();
    else if (!info[0].IsString())
        Napi::TypeError::New(env, "command must be a string").ThrowAsJavaScriptException();
    else if (!info[1].IsObject())
        Napi::TypeError::New(env, "options must be an object").ThrowAsJavaScriptException();

    std::string command = info[0].As<Napi::String>().Utf8Value();

    if (!(command == "cbow" || command == "quantize" || command == "skipgram" || command == "supervised")) {
        Napi::TypeError::New(env, "Permitted command types are ['cbow', 'quantize', 'skipgram', 'supervised").ThrowAsJavaScriptException();
    }

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
    args.push_back(command.c_str());

    for (int j = 0; j < count; j++) {
        args.push_back(argument[j]);
    }

    return async_execute(env, [this, args, command]() {
        if (command == "quantize")
            return wrapper_->quantize(args);
        else
            return wrapper_->train(args);
    });
}
