class GlobalConfig {
  final String project;
  final String version;
  final FlConfig fl;
  final ModelConfig model;
  final PrivacyConfig privacy;
  final CommunicationConfig communication;
  final AggregationConfig aggregation;
  final TargetsConfig targets;
  final Map<String, String> classMap;

  GlobalConfig({
    this.project = 'FedFOD',
    this.version = '1.0.0',
    FlConfig? fl,
    ModelConfig? model,
    PrivacyConfig? privacy,
    CommunicationConfig? communication,
    AggregationConfig? aggregation,
    TargetsConfig? targets,
    this.classMap = const {},
  })  : fl = fl ?? FlConfig(),
        model = model ?? ModelConfig(),
        privacy = privacy ?? PrivacyConfig(),
        communication = communication ?? CommunicationConfig(),
        aggregation = aggregation ?? AggregationConfig(),
        targets = targets ?? TargetsConfig();

  factory GlobalConfig.fromJson(Map<String, dynamic> json) {
    return GlobalConfig(
      project: json['project'] ?? 'FedFOD',
      version: json['version'] ?? '1.0.0',
      fl: FlConfig.fromJson(json['fl'] ?? {}),
      model: ModelConfig.fromJson(json['model'] ?? {}),
      privacy: PrivacyConfig.fromJson(json['privacy'] ?? {}),
      communication: CommunicationConfig.fromJson(json['communication'] ?? {}),
      aggregation: AggregationConfig.fromJson(json['aggregation'] ?? {}),
      targets: TargetsConfig.fromJson(json['targets'] ?? {}),
      classMap: (json['class_map'] as Map<String, dynamic>?)
              ?.map((k, v) => MapEntry(k, v.toString())) ??
          {},
    );
  }

  Map<String, dynamic> toJson() => {
        'project': project,
        'version': version,
        'fl': fl.toJson(),
        'model': model.toJson(),
        'privacy': privacy.toJson(),
        'communication': communication.toJson(),
        'aggregation': aggregation.toJson(),
        'targets': targets.toJson(),
        'class_map': classMap,
      };
}

class FlConfig {
  final int numRounds;
  final int numClients;
  final int minClientsPerRound;
  final int localEpochs;
  final int localStepsK;
  final double learningRate;
  final double proximalMu;
  final bool scaffoldCorrection;

  FlConfig({
    this.numRounds = 90,
    this.numClients = 3,
    this.minClientsPerRound = 2,
    this.localEpochs = 3,
    this.localStepsK = 10,
    this.learningRate = 0.001,
    this.proximalMu = 0.05,
    this.scaffoldCorrection = true,
  });

  factory FlConfig.fromJson(Map<String, dynamic> json) => FlConfig(
        numRounds: json['num_rounds'] ?? 90,
        numClients: json['num_clients'] ?? 3,
        minClientsPerRound: json['min_clients_per_round'] ?? 2,
        localEpochs: json['local_epochs'] ?? 3,
        localStepsK: json['local_steps_K'] ?? 10,
        learningRate: (json['learning_rate'] ?? 0.001).toDouble(),
        proximalMu: (json['proximal_mu'] ?? 0.05).toDouble(),
        scaffoldCorrection: json['scaffold_correction'] ?? true,
      );

  Map<String, dynamic> toJson() => {
        'num_rounds': numRounds,
        'num_clients': numClients,
        'min_clients_per_round': minClientsPerRound,
        'local_epochs': localEpochs,
        'local_steps_K': localStepsK,
        'learning_rate': learningRate,
        'proximal_mu': proximalMu,
        'scaffold_correction': scaffoldCorrection,
      };
}

class ModelConfig {
  final String backbone;
  final List<int> inputSize;
  final int numClasses;
  final int embedDim;
  final double confThreshold;
  final double nmsIouThreshold;

  ModelConfig({
    this.backbone = 'rtdetr-l',
    this.inputSize = const [640, 640],
    this.numClasses = 15,
    this.embedDim = 512,
    this.confThreshold = 0.35,
    this.nmsIouThreshold = 0.45,
  });

  factory ModelConfig.fromJson(Map<String, dynamic> json) => ModelConfig(
        backbone: json['backbone'] ?? 'rtdetr-l',
        inputSize: (json['input_size'] as List?)?.cast<int>() ?? [640, 640],
        numClasses: json['num_classes'] ?? 15,
        embedDim: json['embed_dim'] ?? 512,
        confThreshold: (json['conf_threshold'] ?? 0.35).toDouble(),
        nmsIouThreshold: (json['nms_iou_threshold'] ?? 0.45).toDouble(),
      );

  Map<String, dynamic> toJson() => {
        'backbone': backbone,
        'input_size': inputSize,
        'num_classes': numClasses,
        'embed_dim': embedDim,
        'conf_threshold': confThreshold,
        'nms_iou_threshold': nmsIouThreshold,
      };
}

class PrivacyConfig {
  final double dpEpsilon;
  final double dpDelta;
  final double gradientClipNorm;

  PrivacyConfig({
    this.dpEpsilon = 4.0,
    this.dpDelta = 1e-6,
    this.gradientClipNorm = 1.0,
  });

  factory PrivacyConfig.fromJson(Map<String, dynamic> json) => PrivacyConfig(
        dpEpsilon: (json['dp_epsilon'] ?? 4.0).toDouble(),
        dpDelta: (json['dp_delta'] ?? 1e-6).toDouble(),
        gradientClipNorm: (json['gradient_clip_norm'] ?? 1.0).toDouble(),
      );

  Map<String, dynamic> toJson() => {
        'dp_epsilon': dpEpsilon,
        'dp_delta': dpDelta,
        'gradient_clip_norm': gradientClipNorm,
      };
}

class CommunicationConfig {
  final int grpcPort;
  final double maxPayloadMb;
  final double sparsificationTopKPct;
  final int quantizationBits;

  CommunicationConfig({
    this.grpcPort = 50051,
    this.maxPayloadMb = 2.0,
    this.sparsificationTopKPct = 0.05,
    this.quantizationBits = 8,
  });

  factory CommunicationConfig.fromJson(Map<String, dynamic> json) =>
      CommunicationConfig(
        grpcPort: json['grpc_port'] ?? 50051,
        maxPayloadMb: (json['max_payload_mb'] ?? 2.0).toDouble(),
        sparsificationTopKPct:
            (json['sparsification_top_k_pct'] ?? 0.05).toDouble(),
        quantizationBits: json['quantization_bits'] ?? 8,
      );

  Map<String, dynamic> toJson() => {
        'grpc_port': grpcPort,
        'max_payload_mb': maxPayloadMb,
        'sparsification_top_k_pct': sparsificationTopKPct,
        'quantization_bits': quantizationBits,
      };
}

class AggregationConfig {
  final String strategy;
  final double stalenessPenaltyBase;
  final int maxToleratedStalenessRounds;

  AggregationConfig({
    this.strategy = 'fa_weighted_scaffold',
    this.stalenessPenaltyBase = 0.85,
    this.maxToleratedStalenessRounds = 15,
  });

  factory AggregationConfig.fromJson(Map<String, dynamic> json) =>
      AggregationConfig(
        strategy: json['strategy'] ?? 'fa_weighted_scaffold',
        stalenessPenaltyBase:
            (json['staleness_penalty_base'] ?? 0.85).toDouble(),
        maxToleratedStalenessRounds:
            json['max_tolerated_staleness_rounds'] ?? 15,
      );

  Map<String, dynamic> toJson() => {
        'strategy': strategy,
        'staleness_penalty_base': stalenessPenaltyBase,
        'max_tolerated_staleness_rounds': maxToleratedStalenessRounds,
      };
}

class TargetsConfig {
  final double map50Known;
  final double map50NovelOpenworld;
  final double falseAlarmRatePerHour;
  final int endToEndAlertLatencySec;
  final double communicationPayloadMb;

  TargetsConfig({
    this.map50Known = 0.79,
    this.map50NovelOpenworld = 0.51,
    this.falseAlarmRatePerHour = 2.0,
    this.endToEndAlertLatencySec = 45,
    this.communicationPayloadMb = 2.0,
  });

  factory TargetsConfig.fromJson(Map<String, dynamic> json) => TargetsConfig(
        map50Known: (json['map_50_known'] ?? 0.79).toDouble(),
        map50NovelOpenworld:
            (json['map_50_novel_openworld'] ?? 0.51).toDouble(),
        falseAlarmRatePerHour:
            (json['false_alarm_rate_per_hour'] ?? 2.0).toDouble(),
        endToEndAlertLatencySec: json['end_to_end_alert_latency_sec'] ?? 45,
        communicationPayloadMb:
            (json['communication_payload_mb'] ?? 2.0).toDouble(),
      );

  Map<String, dynamic> toJson() => {
        'map_50_known': map50Known,
        'map_50_novel_openworld': map50NovelOpenworld,
        'false_alarm_rate_per_hour': falseAlarmRatePerHour,
        'end_to_end_alert_latency_sec': endToEndAlertLatencySec,
        'communication_payload_mb': communicationPayloadMb,
      };
}
