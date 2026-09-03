import 'dart:typed_data';
import 'package:flutter/services.dart';

class NativeAIService {
  static const _channel = MethodChannel('com.example.offline_ai/native');

  static Future<bool> initializeEngines({
    required String ggufPath,
    required String whisperPath,
    required String ttsPath,
  }) async {
    try {
      final bool success = await _channel.invokeMethod('initEngines', {
        'ggufPath': ggufPath,
        'whisperPath': whisperPath,
        'ttsPath': ttsPath,
      });
      return success;
    } catch (e) {
      print("Native binding error: $e");
      return false;
    }
  }

  static Future<String> transcribe(Uint8List pcmBytes) async {
    final String result = await _channel.invokeMethod('transcribe', {'pcm': pcmBytes});
    return result;
  }

  static Future<String> generateText(String prompt) async {
    final String response = await _channel.invokeMethod('inferLLM', {'prompt': prompt});
    return response;
  }

  static Future<Float32List> synthesize(String text) async {
    final List<dynamic> result = await _channel.invokeMethod('synthesize', {'text': text});
    return Float32List.fromList(result.cast<double>());
  }
}

