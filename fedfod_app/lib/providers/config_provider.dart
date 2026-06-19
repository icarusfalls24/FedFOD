import 'package:flutter/foundation.dart';
import '../models/global_config.dart';
import '../models/airport_config.dart';
import '../services/api_service.dart';

class ConfigProvider extends ChangeNotifier {
  final ApiService _api;

  GlobalConfig _globalConfig = GlobalConfig();
  List<AirportConfig> _airports = [];
  bool _loading = false;
  String? _error;
  bool _dirty = false;

  ConfigProvider(this._api);

  GlobalConfig get globalConfig => _globalConfig;
  List<AirportConfig> get airports => _airports;
  bool get loading => _loading;
  String? get error => _error;
  bool get dirty => _dirty;

  // Direct config field getters for screen edits
  int get numRounds => _globalConfig.fl.numRounds;
  int get numClients => _globalConfig.fl.numClients;
  int get minClients => _globalConfig.fl.minClientsPerRound;
  int get localEpochs => _globalConfig.fl.localEpochs;
  double get learningRate => _globalConfig.fl.learningRate;
  bool get scaffoldCorrection => _globalConfig.fl.scaffoldCorrection;

  double get dpEpsilon => _globalConfig.privacy.dpEpsilon;
  double get dpDelta => _globalConfig.privacy.dpDelta;
  double get gradientClipNorm => _globalConfig.privacy.gradientClipNorm;

  double get maxPayloadMb => _globalConfig.communication.maxPayloadMb;
  double get sparsificationTopKPct => _globalConfig.communication.sparsificationTopKPct;
  int get quantizationBits => _globalConfig.communication.quantizationBits;

  String get backbone => _globalConfig.model.backbone;
  int get numClasses => _globalConfig.model.numClasses;
  double get confThreshold => _globalConfig.model.confThreshold;
  double get nmsIouThreshold => _globalConfig.model.nmsIouThreshold;

  double get map50Target => _globalConfig.targets.map50Known;
  double get farTarget => _globalConfig.targets.falseAlarmRatePerHour;
  int get latencyTarget => _globalConfig.targets.endToEndAlertLatencySec;

  Future<void> loadGlobalConfig() async {
    _loading = true;
    _error = null;
    notifyListeners();
    try {
      _globalConfig = await _api.getGlobalConfig();
      _dirty = false;
      _loading = false;
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      _loading = false;
      notifyListeners();
    }
  }

  Future<void> loadAirports() async {
    try {
      _airports = await _api.getAirportConfigs();
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
    }
  }

  Future<void> loadAll() async {
    _loading = true;
    notifyListeners();
    await Future.wait([loadGlobalConfig(), loadAirports()]);
    _loading = false;
    notifyListeners();
  }

  void updateGlobalConfig(GlobalConfig config) {
    _globalConfig = config;
    _dirty = true;
    notifyListeners();
  }

  Future<bool> saveGlobalConfig({
    int? numRounds,
    int? numClients,
    int? minClients,
    int? localEpochs,
    double? learningRate,
    bool? scaffoldCorrection,
    double? dpEpsilon,
    double? dpDelta,
    double? gradientClipNorm,
    double? maxPayloadMb,
    double? sparsificationTopKPct,
    int? quantizationBits,
    String? backbone,
    int? numClasses,
    double? confThreshold,
    double? nmsIouThreshold,
    double? map50Target,
    double? farTarget,
    int? latencyTarget,
  }) async {
    final updated = GlobalConfig(
      project: _globalConfig.project,
      version: _globalConfig.version,
      fl: FlConfig(
        numRounds: numRounds ?? _globalConfig.fl.numRounds,
        numClients: numClients ?? _globalConfig.fl.numClients,
        minClientsPerRound: minClients ?? _globalConfig.fl.minClientsPerRound,
        localEpochs: localEpochs ?? _globalConfig.fl.localEpochs,
        localStepsK: _globalConfig.fl.localStepsK,
        learningRate: learningRate ?? _globalConfig.fl.learningRate,
        proximalMu: _globalConfig.fl.proximalMu,
        scaffoldCorrection: scaffoldCorrection ?? _globalConfig.fl.scaffoldCorrection,
      ),
      model: ModelConfig(
        backbone: backbone ?? _globalConfig.model.backbone,
        inputSize: _globalConfig.model.inputSize,
        numClasses: numClasses ?? _globalConfig.model.numClasses,
        embedDim: _globalConfig.model.embedDim,
        confThreshold: confThreshold ?? _globalConfig.model.confThreshold,
        nmsIouThreshold: nmsIouThreshold ?? _globalConfig.model.nmsIouThreshold,
      ),
      privacy: PrivacyConfig(
        dpEpsilon: dpEpsilon ?? _globalConfig.privacy.dpEpsilon,
        dpDelta: dpDelta ?? _globalConfig.privacy.dpDelta,
        gradientClipNorm: gradientClipNorm ?? _globalConfig.privacy.gradientClipNorm,
      ),
      communication: CommunicationConfig(
        grpcPort: _globalConfig.communication.grpcPort,
        maxPayloadMb: maxPayloadMb ?? _globalConfig.communication.maxPayloadMb,
        sparsificationTopKPct: sparsificationTopKPct ?? _globalConfig.communication.sparsificationTopKPct,
        quantizationBits: quantizationBits ?? _globalConfig.communication.quantizationBits,
      ),
      aggregation: _globalConfig.aggregation,
      targets: TargetsConfig(
        map50Known: map50Target ?? _globalConfig.targets.map50Known,
        map50NovelOpenworld: _globalConfig.targets.map50NovelOpenworld,
        falseAlarmRatePerHour: farTarget ?? _globalConfig.targets.falseAlarmRatePerHour,
        endToEndAlertLatencySec: latencyTarget ?? _globalConfig.targets.endToEndAlertLatencySec,
        communicationPayloadMb: _globalConfig.targets.communicationPayloadMb,
      ),
      classMap: _globalConfig.classMap,
    );

    try {
      await _api.updateGlobalConfig(updated);
      _globalConfig = updated;
      _dirty = false;
      _error = null;
      notifyListeners();
      return true;
    } catch (e) {
      _error = e.toString();
      notifyListeners();
      return false;
    }
  }

  Future<void> updateAirportConfig({
    required int airportIndex,
    double? learningRate,
    int? localEpochs,
    int? batchSize,
  }) async {
    final airport = _airports[airportIndex];
    final updated = AirportConfig(
      id: airport.id,
      airport: airport.airport,
      network: airport.network,
      cameraInfo: airport.cameraInfo,
      compute: airport.compute,
      data: airport.data,
      training: TrainingInfo(
        localEpochs: localEpochs ?? airport.training.localEpochs,
        localBatchSize: batchSize ?? airport.training.localBatchSize,
        learningRate: learningRate ?? airport.training.learningRate,
        augmentationLevel: airport.training.augmentationLevel,
      ),
      environment: airport.environment,
    );
    try {
      await _api.updateAirportConfig(airport.id, updated.toJson());
      _airports[airportIndex] = updated;
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
      rethrow;
    }
  }

  AirportConfig? getAirport(String id) {
    try {
      return _airports.firstWhere((a) => a.id == id);
    } catch (_) {
      return null;
    }
  }
}
