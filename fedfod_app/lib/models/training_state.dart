enum TrainingStatus { idle, starting, training, stopping, completed, error }

class TrainingState {
  final TrainingStatus status;
  final int currentRound;
  final int totalRounds;
  final int connectedClients;
  final int targetClients;
  final int port;
  final bool dummyModel;
  final String? errorMessage;
  final DateTime? startedAt;
  final List<ClientState> clients;

  TrainingState({
    this.status = TrainingStatus.idle,
    this.currentRound = 0,
    this.totalRounds = 90,
    this.connectedClients = 0,
    this.targetClients = 3,
    this.port = 50055,
    this.dummyModel = true,
    this.errorMessage,
    this.startedAt,
    this.clients = const [],
  });

  factory TrainingState.fromJson(Map<String, dynamic> json) => TrainingState(
        status: _parseStatus(json['status'] ?? 'idle'),
        currentRound: json['current_round'] ?? 0,
        totalRounds: json['total_rounds'] ?? 90,
        connectedClients: json['connected_clients'] ?? 0,
        targetClients: json['target_clients'] ?? 3,
        port: json['port'] ?? 50055,
        dummyModel: json['dummy_model'] ?? true,
        errorMessage: json['error_message'],
        startedAt: json['started_at'] != null
            ? DateTime.tryParse(json['started_at'])
            : null,
        clients: (json['clients'] as List?)
                ?.map((c) => ClientState.fromJson(c))
                .toList() ??
            [],
      );

  double get progress =>
      totalRounds > 0 ? currentRound / totalRounds : 0;

  bool get isActive =>
      status == TrainingStatus.training || status == TrainingStatus.starting;

  static TrainingStatus _parseStatus(String s) {
    switch (s.toLowerCase()) {
      case 'training':
        return TrainingStatus.training;
      case 'starting':
        return TrainingStatus.starting;
      case 'stopping':
        return TrainingStatus.stopping;
      case 'completed':
        return TrainingStatus.completed;
      case 'error':
        return TrainingStatus.error;
      default:
        return TrainingStatus.idle;
    }
  }
}

class ClientState {
  final String clientId;
  final String airportName;
  final double lastLoss;
  final int lastSamples;
  final int lastRound;
  final String device;
  final bool connected;

  ClientState({
    required this.clientId,
    this.airportName = '',
    this.lastLoss = 0,
    this.lastSamples = 0,
    this.lastRound = 0,
    this.device = 'cpu',
    this.connected = false,
  });

  double get loss => lastLoss;
  int get samples => lastSamples;


  factory ClientState.fromJson(Map<String, dynamic> json) => ClientState(
        clientId: json['client_id'] ?? '',
        airportName: json['airport_name'] ?? '',
        lastLoss: (json['last_loss'] ?? 0).toDouble(),
        lastSamples: json['last_samples'] ?? 0,
        lastRound: json['last_round'] ?? 0,
        device: json['device'] ?? 'cpu',
        connected: json['connected'] ?? false,
      );
}
