
#include <napi.h>
#include "model.h"

Napi::Object Initialize(Napi::Env env, Napi::Object exports)
{
    FasttextModel::Init(env, exports);
    return exports;
}

NODE_API_MODULE(addon, Initialize)
