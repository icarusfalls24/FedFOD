class AppConfig {
  static const String apiBaseUrl = 'http://localhost:8000';
  static const String wsBaseUrl = 'ws://localhost:8000';
  static const String metricsWsUrl = '$wsBaseUrl/ws/metrics';
  static const String logsWsUrl = '$wsBaseUrl/ws/logs';

  // Design tokens
  static const int primaryColor = 0xFF6C63FF;
  static const int secondaryColor = 0xFF00D9FF;
  static const int surfaceColor = 0xFF1A1A2E;
  static const int cardColor = 0xFF16213E;
  static const int errorColor = 0xFFFF6B6B;
  static const int successColor = 0xFF00E676;
  static const int warningColor = 0xFFFFB74D;

  // Targets (publication thresholds)
  static const double targetMAP50 = 0.79;
  static const double targetFARPerHour = 2.0;
  static const double targetCommPayloadMB = 2.0;
  static const double targetDPEpsilon = 4.0;
  static const int targetLatencySec = 45;
  static const double targetGiniMax = 0.35;
}
