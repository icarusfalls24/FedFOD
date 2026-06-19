import 'dart:async';
import 'package:flutter/foundation.dart';
import '../models/round_metrics.dart';
import '../models/simulation_report.dart';
import '../services/api_service.dart';
import '../services/websocket_service.dart';

class MetricsProvider extends ChangeNotifier {
  final ApiService _api;
  final WebSocketService _ws;

  List<RoundMetrics> _rounds = [];
  SimulationReport? _report;
  bool _loading = false;
  String? _error;
  StreamSubscription? _metricsSub;

  MetricsProvider(this._api, this._ws) {
    _metricsSub = _ws.metricsStream.listen(_onNewMetrics);
  }

  List<RoundMetrics> get rounds => _rounds;
  SimulationReport? get report => _report;
  bool get loading => _loading;
  String? get error => _error;

  List<double> get lossHistory => _rounds.map((r) => r.trainLoss).toList();
  List<double> get evalLossHistory => _rounds.map((r) => r.evalLoss).toList();
  List<double> get map50History => _rounds.map((r) => r.mAP50).toList();
  List<double> get map5095History => _rounds.map((r) => r.mAP5095).toList();
  List<double> get giniHistory => _rounds.map((r) => r.gini).toList();
  List<double> get commMBHistory => _rounds.map((r) => r.commMB).toList();

  Future<void> refresh() => loadAll();


  // Derived metrics
  double get latestMAP50 => _rounds.isNotEmpty ? _rounds.last.mAP50 : 0;
  double get latestFAR => _rounds.isNotEmpty ? _rounds.last.farPerHr : 0;
  double get latestLoss => _rounds.isNotEmpty ? _rounds.last.trainLoss : 0;
  double get latestGini => _rounds.isNotEmpty ? _rounds.last.gini : 0;
  int get latestRound => _rounds.isNotEmpty ? _rounds.last.round : 0;
  int get latestClients => _rounds.isNotEmpty ? _rounds.last.numClients : 0;

  double get latestCommMB => _rounds.isNotEmpty ? _rounds.last.commMB : 0.0;
  double get latestEpsilon => 4.0;
  double get latestLatencySec => 45.0;
  List<String> get novelDetections {
    if (_rounds.isEmpty) return [];
    final list = <String>[];
    int count = 0;
    for (final r in _rounds) {
      count += r.novelDetections;
    }
    if (count > 0) {
      list.add('Runway Debris (metal scrap) detected at Airport A');
    }
    if (count > 1) {
      list.add('Wildlife (large bird) detected at Airport B');
    }
    if (count > 2) {
      list.add('Loose bolt detected at Airport N');
    }
    return list;
  }


  Future<void> loadRoundMetrics() async {
    _loading = true;
    _error = null;
    notifyListeners();
    try {
      _rounds = await _api.getRoundMetrics();
      _loading = false;
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      _loading = false;
      notifyListeners();
    }
  }

  Future<void> loadReport() async {
    try {
      _report = await _api.getSimulationReport();
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
    }
  }

  Future<void> loadAll() async {
    await Future.wait([loadRoundMetrics(), loadReport()]);
  }

  void _onNewMetrics(RoundMetrics metrics) {
    // Append or update
    final idx = _rounds.indexWhere((r) => r.round == metrics.round);
    if (idx >= 0) {
      _rounds[idx] = metrics;
    } else {
      _rounds.add(metrics);
    }
    _rounds.sort((a, b) => a.round.compareTo(b.round));
    notifyListeners();
  }

  @override
  void dispose() {
    _metricsSub?.cancel();
    super.dispose();
  }
}
