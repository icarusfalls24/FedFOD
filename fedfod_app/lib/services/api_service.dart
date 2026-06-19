import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config/app_config.dart';
import '../models/global_config.dart';
import '../models/airport_config.dart';
import '../models/round_metrics.dart';
import '../models/training_state.dart';
import '../models/simulation_report.dart';

class ApiService {
  final String baseUrl;
  final http.Client _client;

  ApiService({String? baseUrl})
      : baseUrl = baseUrl ?? AppConfig.apiBaseUrl,
        _client = http.Client();

  // ---- Status ----

  Future<Map<String, dynamic>> getStatus() async {
    final response = await _get('/api/status');
    return response;
  }

  // ---- Global Config ----

  Future<GlobalConfig> getGlobalConfig() async {
    final response = await _get('/api/config/global');
    return GlobalConfig.fromJson(response);
  }

  Future<void> updateGlobalConfig(GlobalConfig config) async {
    await _put('/api/config/global', config.toJson());
  }

  // ---- Airport Configs ----

  Future<List<AirportConfig>> getAirportConfigs() async {
    final response = await _get('/api/config/airports');
    final list = response['airports'] as List? ?? [];
    return list.map((item) {
      final id = item['id'] ?? '';
      return AirportConfig.fromJson(id, item);
    }).toList();
  }

  Future<AirportConfig> getAirportConfig(String airportId) async {
    final response = await _get('/api/config/airports/$airportId');
    return AirportConfig.fromJson(airportId, response);
  }

  Future<void> updateAirportConfig(String airportId, Map<String, dynamic> config) async {
    await _put('/api/config/airports/$airportId', config);
  }

  // ---- Metrics ----

  Future<SimulationReport> getSimulationReport() async {
    final response = await _get('/api/metrics/report');
    return SimulationReport.fromJson(response);
  }

  Future<List<RoundMetrics>> getRoundMetrics() async {
    final response = await _get('/api/metrics/rounds');
    final list = response['rounds'] as List? ?? [];
    return list.map((item) => RoundMetrics.fromJson(item)).toList();
  }

  // ---- Training ----

  Future<TrainingState> getTrainingState() async {
    final response = await _get('/api/training/state');
    return TrainingState.fromJson(response);
  }

  Future<void> startTraining({
    int rounds = 90,
    int minClients = 2,
    int port = 50055,
    bool dummyModel = true,
    List<Map<String, String>>? clients,
  }) async {
    await _post('/api/training/start', {
      'rounds': rounds,
      'min_clients': minClients,
      'port': port,
      'dummy_model': dummyModel,
      'clients': clients ??
          [
            {'id': '0', 'airport_config': 'config/airport_configs/airport_A.yaml'},
            {'id': '1', 'airport_config': 'config/airport_configs/airport_B.yaml'},
            {'id': '2', 'airport_config': 'config/airport_configs/airport_N.yaml'},
          ],
    });
  }

  Future<void> stopTraining() async {
    await _post('/api/training/stop', {});
  }

  // ---- Logs ----

  Future<List<String>> getLogs({int lines = 200}) async {
    final response = await _get('/api/logs?lines=$lines');
    return (response['lines'] as List?)?.cast<String>() ?? [];
  }

  // ---- HTTP helpers ----

  Future<Map<String, dynamic>> _get(String path) async {
    try {
      final response = await _client.get(
        Uri.parse('$baseUrl$path'),
        headers: {'Content-Type': 'application/json'},
      );
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      throw ApiException('GET $path failed: ${response.statusCode}', response.statusCode);
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException('Connection failed: $e', 0);
    }
  }

  Future<Map<String, dynamic>> _post(String path, Map<String, dynamic> body) async {
    try {
      final response = await _client.post(
        Uri.parse('$baseUrl$path'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(body),
      );
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      throw ApiException('POST $path failed: ${response.statusCode}', response.statusCode);
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException('Connection failed: $e', 0);
    }
  }

  Future<void> _put(String path, Map<String, dynamic> body) async {
    try {
      final response = await _client.put(
        Uri.parse('$baseUrl$path'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(body),
      );
      if (response.statusCode != 200) {
        throw ApiException('PUT $path failed: ${response.statusCode}', response.statusCode);
      }
    } catch (e) {
      if (e is ApiException) rethrow;
      throw ApiException('Connection failed: $e', 0);
    }
  }

  void dispose() {
    _client.close();
  }
}

class ApiException implements Exception {
  final String message;
  final int statusCode;

  ApiException(this.message, this.statusCode);

  @override
  String toString() => 'ApiException($statusCode): $message';
}
