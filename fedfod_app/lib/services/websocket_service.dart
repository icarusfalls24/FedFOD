import 'dart:async';
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';
import '../config/app_config.dart';
import '../models/round_metrics.dart';

class WebSocketService {
  WebSocketChannel? _metricsChannel;
  WebSocketChannel? _logsChannel;

  final _metricsController = StreamController<RoundMetrics>.broadcast();
  final _logsController = StreamController<String>.broadcast();
  final _connectionController = StreamController<bool>.broadcast();

  Stream<RoundMetrics> get metricsStream => _metricsController.stream;
  Stream<String> get logsStream => _logsController.stream;
  Stream<String> get logStream => _logsController.stream;
  Stream<bool> get connectionStream => _connectionController.stream;

  bool _isConnected = false;
  bool get isConnected => _isConnected;

  Timer? _reconnectTimer;

  void connectMetrics() {
    _disconnectMetrics();
    try {
      _metricsChannel = WebSocketChannel.connect(
        Uri.parse(AppConfig.metricsWsUrl),
      );
      _isConnected = true;
      _connectionController.add(true);

      _metricsChannel!.stream.listen(
        (data) {
          try {
            final json = jsonDecode(data as String) as Map<String, dynamic>;
            final metrics = RoundMetrics.fromJson(json);
            _metricsController.add(metrics);
          } catch (_) {}
        },
        onError: (error) {
          _isConnected = false;
          _connectionController.add(false);
          _scheduleReconnect();
        },
        onDone: () {
          _isConnected = false;
          _connectionController.add(false);
          _scheduleReconnect();
        },
      );
    } catch (_) {
      _isConnected = false;
      _connectionController.add(false);
      _scheduleReconnect();
    }
  }

  void connectLogs() {
    _disconnectLogs();
    try {
      _logsChannel = WebSocketChannel.connect(
        Uri.parse(AppConfig.logsWsUrl),
      );

      _logsChannel!.stream.listen(
        (data) {
          _logsController.add(data as String);
        },
        onError: (_) {},
        onDone: () {},
      );
    } catch (_) {}
  }

  void _scheduleReconnect() {
    _reconnectTimer?.cancel();
    _reconnectTimer = Timer(const Duration(seconds: 5), () {
      if (!_isConnected) {
        connectMetrics();
      }
    });
  }

  void _disconnectMetrics() {
    _metricsChannel?.sink.close();
    _metricsChannel = null;
  }

  void _disconnectLogs() {
    _logsChannel?.sink.close();
    _logsChannel = null;
  }

  void dispose() {
    _reconnectTimer?.cancel();
    _disconnectMetrics();
    _disconnectLogs();
    _metricsController.close();
    _logsController.close();
    _connectionController.close();
  }
}
