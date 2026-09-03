#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "llama.h"

#define LOG_TAG "MarciaNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static const llama_vocab* g_vocab = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_offline_1ai_MainActivity_initLlamaEngine(
    JNIEnv* env, jobject thiz, jstring model_path_jstr
) {
    const char* model_path = env->GetStringUTFChars(model_path_jstr, nullptr);
    LOGI("Loading GGUF model from path: %s", model_path);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    g_model = llama_model_load_from_file(model_path, mparams);
    env->ReleaseStringUTFChars(model_path_jstr, model_path);

    if (!g_model) {
        LOGE("Failed to load Llama model.");
        return JNI_FALSE;
    }

    g_vocab = llama_model_get_vocab(g_model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        LOGE("Failed to create context.");
        return JNI_FALSE;
    }

    LOGI("Marcia engine initialized successfully.");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_offline_1ai_MainActivity_runLlamaInference(
    JNIEnv* env, jobject thiz, jstring prompt_jstr
) {
    if (!g_ctx || !g_model) {
        return env->NewStringUTF("Error: Engine Not Initialized");
    }

    const char* prompt = env->GetStringUTFChars(prompt_jstr, nullptr);
    std::string prompt_str(prompt);
    env->ReleaseStringUTFChars(prompt_jstr, prompt);

    int n_prompt_tokens = -llama_tokenize(g_vocab, prompt_str.c_str(), prompt_str.size(),
                                            nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt_tokens);
    if (llama_tokenize(g_vocab, prompt_str.c_str(), prompt_str.size(),
                        tokens.data(), tokens.size(), true, true) < 0) {
        return env->NewStringUTF("Tokenization failed.");
    }

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(g_ctx, batch) != 0) {
        return env->NewStringUTF("Initial decode failed.");
    }

    std::string result;
    const int max_tokens = 200;

    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample(sampler, g_ctx, -1);

        if (llama_vocab_is_eog(g_vocab, new_token)) {
            LOGI("EOS reached after %d tokens", i);
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(g_vocab, new_token, buf, sizeof(buf), 0, true);
        if (n > 0) {
            result.append(buf, n);
        }

        llama_batch next_batch = llama_batch_get_one(&new_token, 1);
        if (llama_decode(g_ctx, next_batch) != 0) {
            LOGE("Decode failed at token %d", i);
            break;
        }
    }

    llama_sampler_free(sampler);
    return env->NewStringUTF(result.c_str());
}
