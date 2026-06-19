import 'dart:async';
import 'package:flutter/foundation.dart';
import '../models/training_state.dart';
import '../models/round_metrics.dart';
import '../services/api_service.dart';
import 'metrics_provider.dart';

class TrainingProvider extends ChangeNotifier {
  final ApiService _api;
  final MetricsProvider _metrics;

  TrainingState _state = TrainingState();
  bool _loading = false;
  String? _error;
  Timer? _pollTimer;

  TrainingProvider(this._api, this._metrics);

  TrainingState get state => _state;
  bool get loading => _loading;
  String? get error => _error;
  bool get isTraining => _state.isActive;

  // Screen compatibility getters
  String get trainingState {
    switch (_state.status) {
      case TrainingStatus.training:
        return 'training';
      case TrainingStatus.completed:
        return 'completed';
      case TrainingStatus.error:
        return 'error';
      case TrainingStatus.starting:
        return 'starting';
      case TrainingStatus.stopping:
        return 'stopping';
      default:
        return 'idle';
    }
  }

  int get totalRounds => _state.totalRounds;
  int get currentRound => _state.currentRound;
  int get activeClients => _state.connectedClients;
  bool get isServerConnected => _state.status != TrainingStatus.idle && _state.status != TrainingStatus.error;
  List<ClientState> get clientStatuses => _state.clients;
  List<RoundMetrics> get roundResults => _metrics.rounds;

  Future<void> loadState() async {
    try {
      _state = await _api.getTrainingState();
      _error = null;
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
    }
  }

  Future<void> startTraining({
    int numRounds = 90,
    int minClients = 2,
    int port = 50055,
    bool dummyModel = true,
  }) async {
    _loading = true;
    _error = null;
    _state = TrainingState(
      status: TrainingStatus.starting,
      totalRounds: numRounds,
      port: port,
      dummyModel: dummyModel,
    );
    notifyListeners();

    try {
      await _api.startTraining(
        rounds: numRounds,
        minClients: minClients,
        port: port,
        dummyModel: dummyModel,
      );
      _loading = false;
      _startPolling();
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      _loading = false;
      _state = TrainingState(
        status: TrainingStatus.error,
        errorMessage: e.toString(),
      );
      notifyListeners();
    }
  }

  Future<void> stopTraining() async {
    _loading = true;
    notifyListeners();
    try {
      await _api.stopTraining();
      _state = TrainingState(status: TrainingStatus.idle);
      _loading = false;
      _stopPolling();
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      _loading = false;
      notifyListeners();
    }
  }

  void _startPolling() {
    _stopPolling();
    _pollTimer = Timer.periodic(const Duration(seconds: 3), (_) async {
      await loadState();
      if (!_state.isActive && _state.status != TrainingStatus.idle) {
        _stopPolling();
      }
    });
  }

  void _stopPolling() {
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  @override
  void dispose() {
    _stopPolling();
    super.dispose();
  }
}
