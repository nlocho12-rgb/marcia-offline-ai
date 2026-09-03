package com.example.offline_ai

import androidx.annotation.NonNull
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity: FlutterActivity() {
    private val CHANNEL = "com.example.offline_ai/native"
    private lateinit var onnxAudioHandler: OnnxAudioHandler

    companion object {
        init {
            System.loadLibrary("marcia_engine")
        }
    }

    private external fun initLlamaEngine(modelPath: String): Boolean
    private external fun runLlamaInference(prompt: String): String

    override fun configureFlutterEngine(@NonNull flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        onnxAudioHandler = OnnxAudioHandler(applicationContext)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL).setMethodCallHandler { call, result ->
            when (call.method) {
                "initEngines" -> {
                    val ggufPath = call.argument<String>("ggufPath") ?: ""
                    val whisperPath = call.argument<String>("whisperPath") ?: ""
                    val ttsPath = call.argument<String>("ttsPath") ?: ""

                    val llamaOk = initLlamaEngine(ggufPath)
                    onnxAudioHandler.initialize(whisperPath, ttsPath)

                    if (llamaOk) {
                        result.success(true)
                    } else {
                        result.error("INIT_FAIL", "Failed initializing native Llama C++ runtime", null)
                    }
                }
                "transcribe" -> {
                    val pcmData = call.argument<ByteArray>("pcm") ?: ByteArray(0)
                    val shortArray = ShortArray(pcmData.size / 2)
                    java.nio.ByteBuffer.wrap(pcmData).order(java.nio.ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(shortArray)
                    val transcribedText = onnxAudioHandler.transcribePcm(shortArray)
                    result.success(transcribedText)
                }
                "inferLLM" -> {
                    val prompt = call.argument<String>("prompt") ?: ""
                    val response = runLlamaInference(prompt)
                    result.success(response)
                }
                "synthesize" -> {
                    val text = call.argument<String>("text") ?: ""
                    val pcmFloatArray = onnxAudioHandler.synthesizeSpeech(text)
                    result.success(pcmFloatArray.toList())
                }
                else -> result.notImplemented()
            }
        }
    }
}
