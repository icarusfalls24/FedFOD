class RoundMetrics {
  final int round;
  final double trainLoss;
  final double evalLoss;
  final double mAP50;
  final double mAP5095;
  final double gini;
  final bool giniValid;
  final int numClients;
  final int novelDetections;
  final double commMB;
  final double timeS;
  final double farPerHr;

  RoundMetrics({
    required this.round,
    this.trainLoss = 0,
    this.evalLoss = 0,
    this.mAP50 = 0,
    this.mAP5095 = 0,
    this.gini = 0,
    this.giniValid = true,
    this.numClients = 0,
    this.novelDetections = 0,
    this.commMB = 0,
    this.timeS = 0,
    this.farPerHr = 0,
  });

  double get loss => trainLoss;
  double get map50 => mAP50;
  double get timeSec => timeS;


  factory RoundMetrics.fromJson(Map<String, dynamic> json) => RoundMetrics(
        round: json['round'] ?? 0,
        trainLoss: (json['train_loss'] ?? json['loss'] ?? 0).toDouble(),
        evalLoss: (json['eval_loss'] ?? 0).toDouble(),
        mAP50: (json['mAP50'] ?? json['mAP@50'] ?? 0).toDouble(),
        mAP5095: (json['mAP50_95'] ?? json['mAP@50-95'] ?? 0).toDouble(),
        gini: (json['gini'] ?? 0).toDouble(),
        giniValid: json['gini_valid'] ?? true,
        numClients: json['num_clients'] ?? 0,
        novelDetections: json['novel_detections'] ?? 0,
        commMB: (json['comm_MB'] ?? json['comm_mb'] ?? 0).toDouble(),
        timeS: (json['time_s'] ?? json['time'] ?? 0).toDouble(),
        farPerHr: (json['FAR_per_hr'] ?? json['far_per_hr'] ?? json['FAR/hr'] ?? 0).toDouble(),
      );

  Map<String, dynamic> toJson() => {
        'round': round,
        'train_loss': trainLoss,
        'eval_loss': evalLoss,
        'mAP50': mAP50,
        'mAP50_95': mAP5095,
        'gini': gini,
        'gini_valid': giniValid,
        'num_clients': numClients,
        'novel_detections': novelDetections,
        'comm_MB': commMB,
        'time_s': timeS,
        'FAR_per_hr': farPerHr,
      };
}
