import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'services/llama_service.dart';

void main() {
  runApp(const MaterialApp(home: MarciaApp()));
}

class MarciaApp extends StatefulWidget {
  const MarciaApp({super.key});

  @override
  State<MarciaApp> createState() => _MarciaAppState();
}

class _MarciaAppState extends State<MarciaApp> {
  final AudioRecorder _audioRecorder = AudioRecorder();
  bool _isInitialized = false;
  bool _isRecording = false;
  String _systemStatus = "Initializing Core Engines...";
  String _sttOutput = "";
  String _llmOutput = "";

  @override
  void initState() {
    super.initState();
    _setupPermissionsAndInit();
  }

  Future<void> _setupPermissionsAndInit() async {
    await Permission.microphone.request();
    await Permission.storage.request();

    final dir = await getApplicationDocumentsDirectory();
    final ggufFile = File('${dir.path}/qwen2.5-1.5b-instruct-q4_k_m.gguf');
    final whisperFile = File('${dir.path}/whisper_tiny.onnx');
    final ttsFile = File('${dir.path}/pocket_tts_int8.onnx');

    if (!await ggufFile.exists()) {
      setState(() {
        _systemStatus = "Missing model assets inside ${dir.path}. Place files to continue.";
      });
      return;
    }

    bool ready = await NativeAIService.initializeEngines(
      ggufPath: ggufFile.path,
      whisperPath: whisperFile.path,
      ttsPath: ttsFile.path,
    );

    setState(() {
      _isInitialized = ready;
      _systemStatus = ready ? "System Fully Ready (100% Offline)" : "Failed initialization.";
    });
  }

  Future<void> _toggleVoiceInput() async {
    if (!_isRecording) {
      if (await _audioRecorder.hasPermission()) {
        final stream = await _audioRecorder.startStream(
          const RecordConfig(encoder: AudioEncoder.pcm16bits, sampleRate: 16000, numChannels: 1),
        );
        setState(() {
          _isRecording = true;
          _sttOutput = "Listening...";
        });
        stream.listen((Uint8List pcmChunk) async {
          String text = await NativeAIService.transcribe(pcmChunk);
          setState(() {
            _sttOutput = text;
          });
        });
      }
    } else {
      await _audioRecorder.stop();
      setState(() {
        _isRecording = false;
      });
      _processConversationLoop(_sttOutput);
    }
  }

  Future<void> _processConversationLoop(String inputText) async {
    if (inputText.isEmpty) return;

    String reply = await NativeAIService.generateText(inputText);
    setState(() {
      _llmOutput = reply;
    });

    await NativeAIService.synthesize(reply);
    // Audio playback wiring comes in Phase 4
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        title: const Text("MARCIA", style: TextStyle(letterSpacing: 3)),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              color: _isInitialized ? Colors.green.shade900 : Colors.amber.shade900,
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: Text(_systemStatus, style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.white)),
              ),
            ),
            const SizedBox(height: 20),
            const Text("Your Voice:", style: TextStyle(color: Colors.grey)),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(border: Border.all(color: Colors.grey), borderRadius: BorderRadius.circular(8)),
              child: Text(_sttOutput.isEmpty ? "Tap the mic and talk..." : _sttOutput, style: const TextStyle(color: Colors.white)),
            ),
            const SizedBox(height: 20),
            const Text("Marcia:", style: TextStyle(color: Colors.grey)),
            Expanded(
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(border: Border.all(color: Colors.grey), borderRadius: BorderRadius.circular(8)),
                child: SingleChildScrollView(child: Text(_llmOutput, style: const TextStyle(color: Colors.white))),
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: _isInitialized ? _toggleVoiceInput : null,
              icon: Icon(_isRecording ? Icons.stop : Icons.mic),
              label: Text(_isRecording ? "Stop Listening" : "Start Speaking"),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 20),
                backgroundColor: _isRecording ? Colors.red : Colors.grey.shade800,
                foregroundColor: Colors.white,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
