class SimulationReport {
  final int totalRounds;
  final double finalMAP50;
  final double finalFARPerHr;
  final double totalCommunicationMB;
  final int convergenceRound;
  final double privacyBudgetEpsilon;
  final int seed;
  final String device;

  SimulationReport({
    this.totalRounds = 0,
    this.finalMAP50 = 0,
    this.finalFARPerHr = 0,
    this.totalCommunicationMB = 0,
    this.convergenceRound = 0,
    this.privacyBudgetEpsilon = 0,
    this.seed = 42,
    this.device = 'cpu',
  });

  factory SimulationReport.fromJson(Map<String, dynamic> json) =>
      SimulationReport(
        totalRounds: json['total_rounds'] ?? 0,
        finalMAP50: (json['final_mAP@50'] ?? json['final_mAP50'] ?? 0).toDouble(),
        finalFARPerHr:
            (json['final_FAR_per_hr'] ?? json['final_FAR/hr'] ?? 0).toDouble(),
        totalCommunicationMB:
            (json['total_communication_MB'] ?? 0).toDouble(),
        convergenceRound: json['convergence_round_mAP50'] ?? 0,
        privacyBudgetEpsilon:
            (json['privacy_budget_epsilon'] ?? 0).toDouble(),
        seed: json['seed'] ?? 42,
        device: json['device'] ?? 'cpu',
      );
}
