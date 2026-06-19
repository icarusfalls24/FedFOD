import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import '../../providers/metrics_provider.dart';
import '../../providers/training_provider.dart';
import '../../providers/config_provider.dart';
import '../../config/app_config.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  // ── Design tokens ──
  static const _primary = Color(AppConfig.primaryColor);
  static const _secondary = Color(AppConfig.secondaryColor);
  static const _surface = Color(AppConfig.surfaceColor);
  static const _card = Color(AppConfig.cardColor);
  static const _success = Color(AppConfig.successColor);
  static const _error = Color(AppConfig.errorColor);

  @override
  Widget build(BuildContext context) {
    final metrics = context.watch<MetricsProvider>();
    final training = context.watch<TrainingProvider>();
    final config = context.watch<ConfigProvider>();

    return Scaffold(
      backgroundColor: _surface,
      appBar: _buildAppBar(training),
      body: RefreshIndicator(
        color: _primary,
        backgroundColor: _card,
        onRefresh: () async => metrics.refresh(),
        child: ListView(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          children: [
            // ── KPI Cards ──
            _buildKPIRow(metrics, training),
            const SizedBox(height: 20),

            // ── Training Loss Chart ──
            _sectionTitle('Training Loss'),
            const SizedBox(height: 8),
            _buildLossChart(metrics),
            const SizedBox(height: 24),

            // ── Airport Fleet ──
            _sectionTitle('Airport Fleet'),
            const SizedBox(height: 8),
            _buildAirportFleet(config),
            const SizedBox(height: 80), // space for FAB
          ],
        ),
      ),
      floatingActionButton: _buildTrainingFAB(training),
    );
  }

  // ══════════════════════════════════════════
  //  APP BAR
  // ══════════════════════════════════════════
  PreferredSizeWidget _buildAppBar(TrainingProvider training) {
    final isConnected = training.isServerConnected;
    return AppBar(
      backgroundColor: _card.withValues(alpha: 0.9),
      elevation: 0,
      centerTitle: false,
      title: Row(
        children: [
          AnimatedBuilder(
            animation: _pulseController,
            builder: (_, __) {
              final opacity =
                  0.4 + 0.6 * _pulseController.value; // 0.4 → 1.0
              return Container(
                width: 10,
                height: 10,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: (isConnected ? _success : _error)
                      .withValues(alpha: opacity),
                  boxShadow: [
                    BoxShadow(
                      color: (isConnected ? _success : _error)
                          .withValues(alpha: 0.4 * opacity),
                      blurRadius: 8,
                      spreadRadius: 2,
                    ),
                  ],
                ),
              );
            },
          ),
          const SizedBox(width: 12),
          Text(
            'FedFOD Command Center',
            style: GoogleFonts.outfit(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),
        ],
      ),
      actions: [
        IconButton(
          icon: const Icon(Icons.refresh, color: Colors.white70),
          onPressed: () => context.read<MetricsProvider>().refresh(),
        ),
      ],
    );
  }

  // ══════════════════════════════════════════
  //  KPI ROW
  // ══════════════════════════════════════════
  Widget _buildKPIRow(MetricsProvider m, TrainingProvider t) {
    return SizedBox(
      height: 140,
      child: Row(
        children: [
          _kpiRingCard(
            label: 'mAP@50',
            value: m.latestMAP50,
            maxValue: 1.0,
            color: _primary,
            format: (v) => '${(v * 100).toStringAsFixed(1)}%',
          ),
          const SizedBox(width: 10),
          _kpiRingCard(
            label: 'FAR/hr',
            value: m.latestFAR,
            maxValue: 5.0,
            color: _secondary,
            format: (v) => v.toStringAsFixed(2),
            invert: true,
          ),
          const SizedBox(width: 10),
          _kpiValueCard(
            label: 'Active Clients',
            value: '${t.activeClients}',
            icon: Icons.devices,
          ),
          const SizedBox(width: 10),
          _kpiValueCard(
            label: 'Current Round',
            value: '${t.currentRound}/${t.totalRounds}',
            icon: Icons.loop,
          ),
        ]
            .animate(interval: 80.ms)
            .fadeIn(duration: 400.ms)
            .slideX(begin: 0.05, end: 0),
      ),
    );
  }

  Widget _kpiRingCard({
    required String label,
    required double value,
    required double maxValue,
    required Color color,
    required String Function(double) format,
    bool invert = false,
  }) {
    final progress = (value / maxValue).clamp(0.0, 1.0);
    return Expanded(
      child: _glassCard(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            SizedBox(
              width: 56,
              height: 56,
              child: Stack(
                fit: StackFit.expand,
                children: [
                  CircularProgressIndicator(
                    value: progress,
                    strokeWidth: 5,
                    backgroundColor: Colors.white10,
                    valueColor: AlwaysStoppedAnimation(color),
                  ),
                  Center(
                    child: Text(
                      format(value),
                      style: GoogleFonts.inter(
                        fontSize: 11,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 8),
            Text(
              label,
              textAlign: TextAlign.center,
              style: GoogleFonts.inter(
                fontSize: 11,
                color: Colors.white60,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _kpiValueCard({
    required String label,
    required String value,
    required IconData icon,
  }) {
    return Expanded(
      child: _glassCard(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, color: _secondary, size: 28),
            const SizedBox(height: 8),
            Text(
              value,
              style: GoogleFonts.outfit(
                fontSize: 20,
                fontWeight: FontWeight.w700,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              label,
              textAlign: TextAlign.center,
              style: GoogleFonts.inter(fontSize: 11, color: Colors.white60),
            ),
          ],
        ),
      ),
    );
  }

  // ══════════════════════════════════════════
  //  TRAINING LOSS CHART
  // ══════════════════════════════════════════
  Widget _buildLossChart(MetricsProvider m) {
    final spots = m.lossHistory
        .asMap()
        .entries
        .map((e) => FlSpot(e.key.toDouble(), e.value))
        .toList();

    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 500),
      child: _glassCard(
        key: ValueKey(spots.length),
        padding: const EdgeInsets.fromLTRB(12, 16, 16, 12),
        child: SizedBox(
          height: 250,
          child: spots.isEmpty
              ? Center(
                  child: Text(
                    'No training data yet',
                    style: GoogleFonts.inter(color: Colors.white38),
                  ),
                )
              : LineChart(
                  LineChartData(
                    gridData: FlGridData(
                      show: true,
                      drawVerticalLine: false,
                      getDrawingHorizontalLine: (_) => FlLine(
                        color: Colors.white10,
                        strokeWidth: 0.5,
                      ),
                    ),
                    titlesData: FlTitlesData(
                      leftTitles: AxisTitles(
                        sideTitles: SideTitles(
                          showTitles: true,
                          reservedSize: 40,
                          getTitlesWidget: (v, _) => Text(
                            v.toStringAsFixed(2),
                            style: GoogleFonts.inter(
                              fontSize: 10,
                              color: Colors.white38,
                            ),
                          ),
                        ),
                      ),
                      bottomTitles: AxisTitles(
                        axisNameWidget: Text(
                          'Round',
                          style: GoogleFonts.inter(
                            fontSize: 11,
                            color: Colors.white38,
                          ),
                        ),
                        sideTitles: SideTitles(
                          showTitles: true,
                          interval: (spots.length / 6).ceilToDouble().clamp(1, 50),
                          getTitlesWidget: (v, _) => Text(
                            v.toInt().toString(),
                            style: GoogleFonts.inter(
                              fontSize: 10,
                              color: Colors.white38,
                            ),
                          ),
                        ),
                      ),
                      topTitles: const AxisTitles(
                        sideTitles: SideTitles(showTitles: false),
                      ),
                      rightTitles: const AxisTitles(
                        sideTitles: SideTitles(showTitles: false),
                      ),
                    ),
                    borderData: FlBorderData(show: false),
                    lineTouchData: LineTouchData(
                      touchTooltipData: LineTouchTooltipData(
                        getTooltipColor: (_) => _card,
                        getTooltipItems: (touchedSpots) => touchedSpots
                            .map((s) => LineTooltipItem(
                                  'R${s.x.toInt()}: ${s.y.toStringAsFixed(4)}',
                                  GoogleFonts.inter(
                                    fontSize: 12,
                                    color: Colors.white,
                                  ),
                                ))
                            .toList(),
                      ),
                    ),
                    lineBarsData: [
                      LineChartBarData(
                        spots: spots,
                        isCurved: true,
                        curveSmoothness: 0.25,
                        color: _primary,
                        barWidth: 2.5,
                        dotData: const FlDotData(show: false),
                        belowBarData: BarAreaData(
                          show: true,
                          gradient: LinearGradient(
                            begin: Alignment.topCenter,
                            end: Alignment.bottomCenter,
                            colors: [
                              _primary.withValues(alpha: 0.3),
                              _primary.withValues(alpha: 0.0),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
        ),
      ),
    )
        .animate()
        .fadeIn(duration: 500.ms)
        .slideY(begin: 0.04, end: 0);
  }

  // ══════════════════════════════════════════
  //  AIRPORT FLEET
  // ══════════════════════════════════════════
  Widget _buildAirportFleet(ConfigProvider config) {
    final airports = config.airports;
    if (airports.isEmpty) {
      return _glassCard(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Center(
            child: Text(
              'No airports connected',
              style: GoogleFonts.inter(color: Colors.white38),
            ),
          ),
        ),
      );
    }
    return Column(
      children: airports.asMap().entries.map((entry) {
        final i = entry.key;
        final a = entry.value;
        return Padding(
          padding: const EdgeInsets.only(bottom: 12),
          child: _airportCard(a),
        )
            .animate()
            .fadeIn(delay: (100 * i).ms, duration: 400.ms)
            .slideX(begin: 0.05, end: 0);
      }).toList(),
    );
  }

  Widget _airportCard(dynamic airport) {
    final name = airport.name as String;
    final type = airport.type as String; // hub / regional / remote
    final isOnline = airport.isOnline as bool;
    final qualityScore = (airport.qualityScore as double).clamp(0.0, 1.0);
    final lastRound = airport.lastRound as int;

    Color typeBadgeColor;
    switch (type) {
      case 'hub':
        typeBadgeColor = _primary;
        break;
      case 'regional':
        typeBadgeColor = _secondary;
        break;
      default:
        typeBadgeColor = const Color(AppConfig.warningColor);
    }

    return _glassCard(
      padding: const EdgeInsets.all(16),
      child: Row(
        children: [
          // Status dot
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: isOnline ? _success : _error,
              boxShadow: [
                BoxShadow(
                  color: (isOnline ? _success : _error).withValues(alpha: 0.5),
                  blurRadius: 6,
                ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          // Info
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      name,
                      style: GoogleFonts.outfit(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(width: 8),
                    _badge(type.toUpperCase(), typeBadgeColor),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Text(
                      'Quality',
                      style: GoogleFonts.inter(
                        fontSize: 11,
                        color: Colors.white54,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: qualityScore,
                          minHeight: 6,
                          backgroundColor: Colors.white10,
                          valueColor:
                              AlwaysStoppedAnimation(_qualityColor(qualityScore)),
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '${(qualityScore * 100).toInt()}%',
                      style: GoogleFonts.inter(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: Colors.white70,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  'Last round: $lastRound',
                  style: GoogleFonts.inter(fontSize: 11, color: Colors.white38),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Color _qualityColor(double q) {
    if (q >= 0.8) return _success;
    if (q >= 0.5) return const Color(AppConfig.warningColor);
    return _error;
  }

  // ══════════════════════════════════════════
  //  TRAINING FAB
  // ══════════════════════════════════════════
  Widget _buildTrainingFAB(TrainingProvider t) {
    final isTraining = t.isTraining;
    return AnimatedBuilder(
      animation: _pulseController,
      builder: (_, child) {
        final scale = isTraining ? (1.0 + 0.06 * _pulseController.value) : 1.0;
        return Transform.scale(
          scale: scale,
          child: FloatingActionButton.extended(
            backgroundColor: isTraining ? _error : _primary,
            icon: Icon(
              isTraining ? Icons.stop_rounded : Icons.play_arrow_rounded,
              color: Colors.white,
            ),
            label: Text(
              isTraining ? 'Stop' : 'Start Training',
              style: GoogleFonts.inter(
                fontWeight: FontWeight.w600,
                color: Colors.white,
              ),
            ),
            onPressed: () {
              if (isTraining) {
                t.stopTraining();
              } else {
                t.startTraining();
              }
            },
          ),
        );
      },
    );
  }

  // ══════════════════════════════════════════
  //  SHARED HELPERS
  // ══════════════════════════════════════════
  Widget _sectionTitle(String text) {
    return Text(
      text,
      style: GoogleFonts.outfit(
        fontSize: 18,
        fontWeight: FontWeight.w700,
        color: Colors.white,
      ),
    );
  }

  Widget _badge(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: color.withValues(alpha: 0.4)),
      ),
      child: Text(
        text,
        style: GoogleFonts.inter(
          fontSize: 10,
          fontWeight: FontWeight.w600,
          color: color,
        ),
      ),
    );
  }

  Widget _glassCard({
    required Widget child,
    EdgeInsets padding = const EdgeInsets.all(12),
    Key? key,
  }) {
    return Container(
      key: key,
      padding: padding,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            _card.withValues(alpha: 0.85),
            _card.withValues(alpha: 0.6),
          ],
        ),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.3),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: child,
    );
  }
}
