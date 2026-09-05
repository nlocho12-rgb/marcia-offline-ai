package com.example.offline_ai

import android.os.Handler
import android.os.Looper
import androidx.annotation.NonNull
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.example.offline_ai/native"
    private lateinit var onnxAudioHandler: OnnxAudioHandler
    private val mainHandler = Handler(Looper.getMainLooper())

    companion object {
        init {
            System.loadLibrary("marcia_engine")
        }
    }

    private external fun initLlamaEngine(modelPath: String): Boolean
    private external fun runLlamaInference(prompt: String): String

    // Runs [work] on a background thread, then delivers the result back on
    // the main thread, since MethodChannel.Result must be called on the
    // platform (UI) thread. This keeps model loading / inference / ONNX
    // calls from blocking the UI and triggering an ANR.
    private fun <T> runInBackground(
        result: MethodChannel.Result,
        work: () -> T,
        onSuccess: (T) -> Unit
    ) {
        Thread {
            try {
                val value = work()
                mainHandler.post { onSuccess(value) }
            } catch (e: Exception) {
                mainHandler.post {
                    result.error("NATIVE_ERROR", e.message ?: "Unknown native error", null)
                }
            }
        }.start()
    }

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        onnxAudioHandler = OnnxAudioHandler(applicationContext)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "initEngines" -> {
                    val ggufPath = call.argument<String>("ggufPath") ?: ""
                    val whisperPath = call.argument<String>("whisperPath") ?: ""
                    val ttsPath = call.argument<String>("ttsPath") ?: ""

                    runInBackground(
                        result,
                        work = {
                            val llamaOk = initLlamaEngine(ggufPath)
                            onnxAudioHandler.initialize(whisperPath, ttsPath)
                            llamaOk
                        },
                        onSuccess = { llamaOk ->
                            if (llamaOk) {
                                result.success(true)
                            } else {
                                result.error("INIT_FAIL", "Failed initializing native Llama C++ runtime", null)
                            }
                        }
                    )
                }
                "transcribe" -> {
                    val pcmData = call.argument<ByteArray>("pcm") ?: ByteArray(0)

                    runInBackground(
                        result,
                        work = {
                            val shortArray = ShortArray(pcmData.size / 2)
                            java.nio.ByteBuffer.wrap(pcmData).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                                .asShortBuffer().get(shortArray)
                            onnxAudioHandler.transcribePcm(shortArray)
                        },
                        onSuccess = { transcribedText ->
                            result.success(transcribedText)
                        }
                    )
                }
                "inferLLM" -> {
                    val prompt = call.argument<String>("prompt") ?: ""

                    runInBackground(
                        result,
                        work = { runLlamaInference(prompt) },
                        onSuccess = { response ->
                            result.success(response)
                        }
                    )
                }
                "synthesize" -> {
                    val text = call.argument<String>("text") ?: ""

                    runInBackground(
                        result,
                        work = { onnxAudioHandler.synthesizeSpeech(text) },
                        onSuccess = { pcmFloatArray ->
                            result.success(pcmFloatArray.toList())
                        }
                    )
                }
                else -> result.notImplemented()
            }
        }
    }
}
