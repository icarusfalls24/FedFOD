class AirportConfig {
  final String id;
  final AirportInfo airport;
  final NetworkInfo network;
  final CameraInfo cameraInfo;
  final ComputeInfo compute;
  final DataInfo data;
  final TrainingInfo training;
  final EnvironmentInfo environment;

  AirportConfig({
    required this.id,
    AirportInfo? airport,
    NetworkInfo? network,
    CameraInfo? cameraInfo,
    ComputeInfo? compute,
    DataInfo? data,
    TrainingInfo? training,
    EnvironmentInfo? environment,
  })  : airport = airport ?? AirportInfo(),
        network = network ?? NetworkInfo(),
        cameraInfo = cameraInfo ?? CameraInfo(),
        compute = compute ?? ComputeInfo(),
        data = data ?? DataInfo(),
        training = training ?? TrainingInfo(),
        environment = environment ?? EnvironmentInfo();

  factory AirportConfig.fromJson(String id, Map<String, dynamic> json) {
    return AirportConfig(
      id: id,
      airport: AirportInfo.fromJson(json['airport'] ?? {}),
      network: NetworkInfo.fromJson(json['network'] ?? {}),
      cameraInfo: CameraInfo.fromJson(json['cameras'] ?? {}),
      compute: ComputeInfo.fromJson(json['compute'] ?? {}),
      data: DataInfo.fromJson(json['data'] ?? {}),
      training: TrainingInfo.fromJson(json['training'] ?? {}),
      environment: EnvironmentInfo.fromJson(json['environment'] ?? {}),
    );
  }

  Map<String, dynamic> toJson() => {
        'airport': airport.toJson(),
        'network': network.toJson(),
        'cameras': cameraInfo.toJson(),
        'compute': compute.toJson(),
        'data': data.toJson(),
        'training': training.toJson(),
        'environment': environment.toJson(),
      };

  // Screen compatibility getters
  String get name => airport.name;
  String get icaoCode => airport.icaoCode;
  String get type => airport.type;
  String get gpu => compute.gpu;
  int get cameras => cameraInfo.count;
  String get bandwidth => "${network.bandwidthMbps} Mbps";
  int get latencyMs => network.latencyMs;
  double get qualityScore => data.qualityScoreQ;
  List<String> get dominantFodClasses => data.dominantClasses;
  String get connectivity => network.connectivity;
  double get learningRate => training.learningRate;
  int get localEpochs => training.localEpochs;
  int get batchSize => training.localBatchSize;
  int get lastRound => 0;

  bool get isOnline => network.reliable;
  String get connectivityColor {
    switch (network.connectivity) {
      case 'fibre':
        return 'green';
      case 'LTE':
        return 'amber';
      case 'VSAT_satellite':
        return 'red';
      default:
        return 'grey';
    }
  }
}

class AirportInfo {
  final String name;
  final String type;
  final String icaoCode;
  final String description;

  AirportInfo({
    this.name = 'Unknown',
    this.type = 'unknown',
    this.icaoCode = 'XXXX',
    this.description = '',
  });

  factory AirportInfo.fromJson(Map<String, dynamic> json) => AirportInfo(
        name: json['name'] ?? 'Unknown',
        type: json['type'] ?? 'unknown',
        icaoCode: json['icao_code'] ?? 'XXXX',
        description: json['description'] ?? '',
      );

  Map<String, dynamic> toJson() => {
        'name': name,
        'type': type,
        'icao_code': icaoCode,
        'description': description,
      };
}

class NetworkInfo {
  final String connectivity;
  final double bandwidthMbps;
  final int latencyMs;
  final bool reliable;
  final String syncMode;

  NetworkInfo({
    this.connectivity = 'unknown',
    this.bandwidthMbps = 0,
    this.reliable = false,
    this.latencyMs = 0,
    this.syncMode = 'synchronous',
  });

  factory NetworkInfo.fromJson(Map<String, dynamic> json) => NetworkInfo(
        connectivity: json['connectivity'] ?? 'unknown',
        bandwidthMbps: (json['bandwidth_mbps'] ?? 0).toDouble(),
        reliable: json['reliable'] ?? false,
        latencyMs: json['latency_ms'] ?? 0,
        syncMode: json['sync_mode'] ?? 'synchronous',
      );

  Map<String, dynamic> toJson() => {
        'connectivity': connectivity,
        'bandwidth_mbps': bandwidthMbps,
        'reliable': reliable,
        'latency_ms': latencyMs,
        'sync_mode': syncMode,
      };
}

class CameraInfo {
  final int count;
  final String resolution;
  final int fps;
  final String type;

  CameraInfo({
    this.count = 0,
    this.resolution = 'unknown',
    this.fps = 0,
    this.type = 'optical',
  });

  factory CameraInfo.fromJson(Map<String, dynamic> json) => CameraInfo(
        count: json['count'] ?? 0,
        resolution: json['resolution_label'] ?? json['resolution'] ?? 'unknown',
        fps: json['fps'] ?? 0,
        type: json['type'] ?? 'optical',
      );

  Map<String, dynamic> toJson() => {
        'count': count,
        'resolution_label': resolution,
        'fps': fps,
        'type': type,
      };
}

class ComputeInfo {
  final String gpu;
  final double gpuMemoryGb;
  final int cpuCores;
  final int ramGb;

  ComputeInfo({
    this.gpu = 'unknown',
    this.gpuMemoryGb = 0,
    this.cpuCores = 0,
    this.ramGb = 0,
  });

  factory ComputeInfo.fromJson(Map<String, dynamic> json) => ComputeInfo(
        gpu: json['gpu'] ?? 'unknown',
        gpuMemoryGb: (json['gpu_memory_gb'] ?? 0).toDouble(),
        cpuCores: json['cpu_cores'] ?? 0,
        ramGb: json['ram_gb'] ?? 0,
      );

  Map<String, dynamic> toJson() => {
        'gpu': gpu,
        'gpu_memory_gb': gpuMemoryGb,
        'cpu_cores': cpuCores,
        'ram_gb': ramGb,
      };
}

class DataInfo {
  final double qualityScoreQ;
  final double annotationCompleteness;
  final int imagesPerDay;
  final List<String> dominantClasses;
  final List<String> rareClasses;

  DataInfo({
    this.qualityScoreQ = 0,
    this.annotationCompleteness = 0,
    this.imagesPerDay = 0,
    this.dominantClasses = const [],
    this.rareClasses = const [],
  });

  factory DataInfo.fromJson(Map<String, dynamic> json) => DataInfo(
        qualityScoreQ: (json['quality_score_q'] ?? 0).toDouble(),
        annotationCompleteness:
            (json['annotation_completeness'] ?? 0).toDouble(),
        imagesPerDay: json['images_per_day'] ?? 0,
        dominantClasses:
            (json['dominant_classes'] as List?)?.cast<String>() ?? [],
        rareClasses: (json['rare_classes'] as List?)?.cast<String>() ?? [],
      );

  Map<String, dynamic> toJson() => {
        'quality_score_q': qualityScoreQ,
        'annotation_completeness': annotationCompleteness,
        'images_per_day': imagesPerDay,
        'dominant_classes': dominantClasses,
        'rare_classes': rareClasses,
      };
}

class TrainingInfo {
  final int localEpochs;
  final int localBatchSize;
  final double learningRate;
  final String augmentationLevel;

  TrainingInfo({
    this.localEpochs = 3,
    this.localBatchSize = 16,
    this.learningRate = 0.001,
    this.augmentationLevel = 'standard',
  });

  factory TrainingInfo.fromJson(Map<String, dynamic> json) => TrainingInfo(
        localEpochs: json['local_epochs'] ?? 3,
        localBatchSize: json['local_batch_size'] ?? 16,
        learningRate: (json['learning_rate'] ?? 0.001).toDouble(),
        augmentationLevel: json['augmentation_level'] ?? 'standard',
      );

  Map<String, dynamic> toJson() => {
        'local_epochs': localEpochs,
        'local_batch_size': localBatchSize,
        'learning_rate': learningRate,
        'augmentation_level': augmentationLevel,
      };
}

class EnvironmentInfo {
  final List<String> weatherConditions;
  final List<String> surfaceTypes;
  final List<String> lighting;

  EnvironmentInfo({
    this.weatherConditions = const [],
    this.surfaceTypes = const [],
    this.lighting = const [],
  });

  factory EnvironmentInfo.fromJson(Map<String, dynamic> json) =>
      EnvironmentInfo(
        weatherConditions:
            (json['weather_conditions'] as List?)?.cast<String>() ?? [],
        surfaceTypes:
            (json['surface_types'] as List?)?.cast<String>() ?? [],
        lighting: (json['lighting'] as List?)?.cast<String>() ?? [],
      );

  Map<String, dynamic> toJson() => {
        'weather_conditions': weatherConditions,
        'surface_types': surfaceTypes,
        'lighting': lighting,
      };
}
