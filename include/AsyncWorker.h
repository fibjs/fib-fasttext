
#ifndef ASYNC_WORKER_H
#define ASYNC_WORKER_H

#include <napi.h>
#include "napi_value.h"

inline Napi::Value async_execute(Napi::Env env, std::function<nlohmann::json()> exec)
{
    class AsyncWorker : public Napi::AsyncWorker {
    public:
        AsyncWorker(Napi::Env env, std::function<nlohmann::json()> exec)
            : Napi::AsyncWorker(Napi::Function::New(env, EmptyCallback))
            , m_exec(exec)
            , deferred_(Napi::Promise::Deferred::New(env))
        {
        }

    public:
        Napi::Value Queue()
        {
            Napi::AsyncWorker::Queue();
            return deferred_.Promise();
        }

    private:
        virtual void Execute()
        {
            try {
                result = m_exec();
            } catch (std::string errorMessage) {
                SetError(errorMessage.c_str());
            } catch (const char* str) {
                SetError(str);
            } catch (const std::exception& e) {
                SetError(e.what());
            }
        }

        virtual void OnOK()
        {
            Napi::HandleScope scope(Env());
            Napi::Value _result = to_value(Env(), result);

            deferred_.Resolve(_result);
        }

        virtual void OnError(const Napi::Error& e)
        {
            Napi::HandleScope scope(Env());
            Napi::String error = Napi::String::New(Env(), e.Message());

            deferred_.Reject(error);
        }

        static Napi::Value EmptyCallback(const Napi::CallbackInfo& info)
        {
            Napi::Env env = info.Env();
            Napi::HandleScope scope(env);

            return env.Undefined();
        }

    private:
        std::function<nlohmann::json()> m_exec;
        Napi::Promise::Deferred deferred_;
        nlohmann::json result;
    };

    return (new AsyncWorker(env, exec))->Queue();
}

#endif
