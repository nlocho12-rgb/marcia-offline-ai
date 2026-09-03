package com.example.offline_ai

import android.content.Context
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import java.nio.FloatBuffer

class OnnxAudioHandler(private val context: Context) {
    private var env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var whisperSession: OrtSession? = null
    private var ttsSession: OrtSession? = null

    fun initialize(whisperPath: String, ttsPath: String) {
        val opts = OrtSession.SessionOptions()
        opts.addNnapi()
        whisperSession = env.createSession(whisperPath, opts)
        ttsSession = env.createSession(ttsPath, opts)
    }

    fun transcribePcm(pcmShorts: ShortArray): String {
        if (whisperSession == null) return "Whisper Session Not Loaded"

        val floatBuffer = FloatBuffer.allocate(pcmShorts.size)
        for (s in pcmShorts) {
            floatBuffer.put(s.toFloat() / 32768.0f)
        }
        floatBuffer.flip()

        val tensor = OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, pcmShorts.size.toLong()))
        val inputs = mapOf("audio_pcm" to tensor)
        val output = whisperSession?.run(inputs)

        val rawOutput = output?.get(0)?.value
        tensor.close()
        output?.close()

        return rawOutput?.toString() ?: "Transcription failed"
    }

    fun synthesizeSpeech(text: String): FloatArray {
        if (ttsSession == null) return FloatArray(0)
        // Real TTS tensor wiring depends on your specific ONNX model's
        // input/output signature — placeholder until Phase 3 model integration
        return FloatArray(24000 * 2)
    }
}
