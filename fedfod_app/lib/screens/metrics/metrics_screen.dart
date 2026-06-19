import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import '../../providers/metrics_provider.dart';
import '../../config/app_config.dart';

class MetricsScreen extends StatefulWidget {
  const MetricsScreen({super.key});

  @override
  State<MetricsScreen> createState() => _MetricsScreenState();
}

class _MetricsScreenState extends State<MetricsScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  // ── Design tokens ──
  static const _primary = Color(AppConfig.primaryColor);
  static const _secondary = Color(AppConfig.secondaryColor);
  static const _surface = Color(AppConfig.surfaceColor);
  static const _card = Color(AppConfig.cardColor);
  static const _success = Color(AppConfig.successColor);
  static const _error = Color(AppConfig.errorColor);
  static const _warning = Color(AppConfig.warningColor);

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: _surface,
      appBar: AppBar(
        backgroundColor: _card.withValues(alpha: 0.9),
        elevation: 0,
        title: Text(
          'Metrics Explorer',
          style: GoogleFonts.outfit(
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
        bottom: TabBar(
          controller: _tabController,
          isScrollable: false,
          indicatorColor: _primary,
          indicatorWeight: 3,
          labelColor: Colors.white,
          unselectedLabelColor: Colors.white38,
          labelStyle: GoogleFonts.inter(
            fontSize: 13,
            fontWeight: FontWeight.w600,
          ),
          tabs: const [
            Tab(text: 'Loss'),
            Tab(text: 'mAP'),
            Tab(text: 'Fairness'),
            Tab(text: 'Comm'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _LossTab(),
          _MAPTab(),
          _FairnessTab(),
          _CommTab(),
        ],
      ),
    );
  }
}

// ═══════════════════════════════════════════════════
//  SHARED CHART HELPERS
// ═══════════════════════════════════════════════════

const _primary = Color(AppConfig.primaryColor);
const _secondary = Color(AppConfig.secondaryColor);
const _card = Color(AppConfig.cardColor);
const _success = Color(AppConfig.successColor);
const _error = Color(AppConfig.errorColor);
const _warning = Color(AppConfig.warningColor);

Widget _chartContainer({
  required Widget chart,
  required List<_LegendItem> legend,
  Key? key,
}) {
  return AnimatedSwitcher(
    duration: const Duration(milliseconds: 500),
    child: Container(
      key: key,
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.fromLTRB(12, 16, 16, 12),
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
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Legend
          Wrap(
            spacing: 16,
            runSpacing: 6,
            children: legend
                .map((l) => Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Container(
                          width: 12,
                          height: 3,
                          decoration: BoxDecoration(
                            color: l.color,
                            borderRadius: BorderRadius.circular(2),
                          ),
                        ),
                        const SizedBox(width: 6),
                        Text(
                          l.label,
                          style: GoogleFonts.inter(
                            fontSize: 11,
                            color: Colors.white54,
                          ),
                        ),
                      ],
                    ))
                .toList(),
          ),
          const SizedBox(height: 16),
          Expanded(child: chart),
        ],
      ),
    ),
  ).animate().fadeIn(duration: 500.ms);
}

class _LegendItem {
  final String label;
  final Color color;
  const _LegendItem(this.label, this.color);
}

FlTitlesData _defaultTitles({String xLabel = 'Round'}) {
  return FlTitlesData(
    leftTitles: AxisTitles(
      sideTitles: SideTitles(
        showTitles: true,
        reservedSize: 44,
        getTitlesWidget: (v, _) => Text(
          v.toStringAsFixed(2),
          style: GoogleFonts.inter(fontSize: 10, color: Colors.white38),
        ),
      ),
    ),
    bottomTitles: AxisTitles(
      axisNameWidget: Text(
        xLabel,
        style: GoogleFonts.inter(fontSize: 11, color: Colors.white38),
      ),
      sideTitles: SideTitles(
        showTitles: true,
        getTitlesWidget: (v, _) => Text(
          v.toInt().toString(),
          style: GoogleFonts.inter(fontSize: 10, color: Colors.white38),
        ),
      ),
    ),
    topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
    rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
  );
}

FlGridData _defaultGrid() {
  return FlGridData(
    show: true,
    drawVerticalLine: false,
    getDrawingHorizontalLine: (_) => FlLine(
      color: Colors.white10,
      strokeWidth: 0.5,
    ),
  );
}

LineTouchData _defaultTouch() {
  return LineTouchData(
    touchTooltipData: LineTouchTooltipData(
      getTooltipColor: (_) => _card,
      getTooltipItems: (spots) => spots
          .map((s) => LineTooltipItem(
                'R${s.x.toInt()}: ${s.y.toStringAsFixed(4)}',
                GoogleFonts.inter(fontSize: 12, color: Colors.white),
              ))
          .toList(),
    ),
  );
}

LineChartBarData _lineBarData({
  required List<FlSpot> spots,
  required Color color,
  bool showDots = false,
  double barWidth = 2.5,
  bool gradient = true,
}) {
  return LineChartBarData(
    spots: spots,
    isCurved: true,
    curveSmoothness: 0.25,
    color: color,
    barWidth: barWidth,
    dotData: FlDotData(show: showDots),
    belowBarData: gradient
        ? BarAreaData(
            show: true,
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [
                color.withValues(alpha: 0.25),
                color.withValues(alpha: 0.0),
              ],
            ),
          )
        : BarAreaData(show: false),
  );
}

Widget _emptyState(String text) {
  return Center(
    child: Text(
      text,
      style: GoogleFonts.inter(color: Colors.white38, fontSize: 14),
    ),
  );
}

// ═══════════════════════════════════════════════════
//  TAB 1: LOSS
// ═══════════════════════════════════════════════════
class _LossTab extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final m = context.watch<MetricsProvider>();
    final trainLoss = m.lossHistory;
    final evalLoss = m.evalLossHistory;

    if (trainLoss.isEmpty) return _emptyState('No loss data yet');

    final trainSpots = trainLoss
        .asMap()
        .entries
        .map((e) => FlSpot(e.key.toDouble(), e.value))
        .toList();
    final evalSpots = evalLoss
        .asMap()
        .entries
        .map((e) => FlSpot(e.key.toDouble(), e.value))
        .toList();

    return _chartContainer(
      key: ValueKey('loss_${trainSpots.length}'),
      legend: const [
        _LegendItem('Train Loss', _primary),
        _LegendItem('Eval Loss', _secondary),
      ],
      chart: LineChart(
        LineChartData(
          gridData: _defaultGrid(),
          titlesData: _defaultTitles(),
          borderData: FlBorderData(show: false),
          lineTouchData: _defaultTouch(),
          lineBarsData: [
            _lineBarData(spots: trainSpots, color: _primary),
            if (evalSpots.isNotEmpty)
              _lineBarData(spots: evalSpots, color: _secondary, gradient: false),
          ],
        ),
      ),
    );
  }
}

// ═══════════════════════════════════════════════════
//  TAB 2: mAP
// ═══════════════════════════════════════════════════
class _MAPTab extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final m = context.watch<MetricsProvider>();
    final map50 = m.map50History;
    final map5095 = m.map5095History;

    if (map50.isEmpty) return _emptyState('No mAP data yet');

    final spots50 = map50
        .asMap()
        .entries
        .map((e) => FlSpot(e.key.toDouble(), e.value))
        .toList();
    final spots5095 = map5095
        .asMap()
        .entries
        .map((e) => FlSpot(e.key.toDouble(), e.value))
        .toList();

    final maxX = spots50.length.toDouble() - 1;

    return _chartContainer(
      key: ValueKey('map_${spots50.length}'),
      legend: const [
        _LegendItem('mAP@50', _primary),
        _LegendItem('mAP@50-95', _secondary),
        _LegendItem('Target (0.79)', _success),
      ],
      chart: LineChart(
        LineChartData(
          gridData: _defaultGrid(),
          titlesData: _defaultTitles(),
          borderData: FlBorderData(show: false),
          lineTouchData: _defaultTouch(),
          extraLinesData: ExtraLinesData(
            horizontalLines: [
              HorizontalLine(
                y: AppConfig.targetMAP50,
                color: _success.withValues(alpha: 0.5),
                strokeWidth: 1.5,
                dashArray: [6, 4],
                label: HorizontalLineLabel(
                  show: true,
                  alignment: Alignment.topRight,
                  style: GoogleFonts.inter(
                    fontSize: 10,
                    color: _success,
                  ),
                  labelResolver: (_) => 'Target 0.79',
                ),
              ),
            ],
          ),
          lineBarsData: [
            _lineBarData(spots: spots50, color: _primary),
            if (spots5095.isNotEmpty)
              _lineBarData(
                  spots: spots5095, color: _secondary, gradient: false),
          ],
        ),
      ),
    );
  }
}

// ═══════════════════════════════════════════════════
//  TAB 3: FAIRNESS
// ═══════════════════════════════════════════════════
class _FairnessTab extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final m = context.watch<MetricsProvider>();
    final gini = m.giniHistory;

    if (gini.isEmpty) return _emptyState('No fairness data yet');

    final spots = gini
        .asMap()
        .entries
        .map((e) => FlSpot(e.key.toDouble(), e.value))
        .toList();

    return _chartContainer(
      key: ValueKey('gini_${spots.length}'),
      legend: const [
        _LegendItem('Gini Coefficient', _warning),
        _LegendItem('Threshold (0.35)', _error),
      ],
      chart: LineChart(
        LineChartData(
          gridData: _defaultGrid(),
          titlesData: _defaultTitles(),
          borderData: FlBorderData(show: false),
          lineTouchData: _defaultTouch(),
          minY: 0,
          maxY: 1,
          extraLinesData: ExtraLinesData(
            horizontalLines: [
              HorizontalLine(
                y: AppConfig.targetGiniMax,
                color: _error.withValues(alpha: 0.5),
                strokeWidth: 1.5,
                dashArray: [6, 4],
                label: HorizontalLineLabel(
                  show: true,
                  alignment: Alignment.topRight,
                  style: GoogleFonts.inter(fontSize: 10, color: _error),
                  labelResolver: (_) => 'Threshold 0.35',
                ),
              ),
            ],
          ),
          lineBarsData: [
            _lineBarData(spots: spots, color: _warning),
          ],
        ),
      ),
    );
  }
}

// ═══════════════════════════════════════════════════
//  TAB 4: COMMUNICATION
// ═══════════════════════════════════════════════════
class _CommTab extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    final m = context.watch<MetricsProvider>();
    final commMB = m.commMBHistory;

    if (commMB.isEmpty) return _emptyState('No communication data yet');

    final barGroups = commMB.asMap().entries.map((e) {
      return BarChartGroupData(
        x: e.key,
        barRods: [
          BarChartRodData(
            toY: e.value,
            width: 14,
            borderRadius: const BorderRadius.vertical(top: Radius.circular(4)),
            gradient: LinearGradient(
              begin: Alignment.bottomCenter,
              end: Alignment.topCenter,
              colors: [
                _primary.withValues(alpha: 0.6),
                _secondary,
              ],
            ),
          ),
        ],
      );
    }).toList();

    return _chartContainer(
      key: ValueKey('comm_${commMB.length}'),
      legend: const [
        _LegendItem('Comm (MB)', _secondary),
      ],
      chart: BarChart(
        BarChartData(
          gridData: _defaultGrid(),
          titlesData: FlTitlesData(
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 40,
                getTitlesWidget: (v, _) => Text(
                  '${v.toStringAsFixed(1)}',
                  style:
                      GoogleFonts.inter(fontSize: 10, color: Colors.white38),
                ),
              ),
            ),
            bottomTitles: AxisTitles(
              axisNameWidget: Text('Round',
                  style:
                      GoogleFonts.inter(fontSize: 11, color: Colors.white38)),
              sideTitles: SideTitles(
                showTitles: true,
                getTitlesWidget: (v, _) => Text(
                  v.toInt().toString(),
                  style:
                      GoogleFonts.inter(fontSize: 10, color: Colors.white38),
                ),
              ),
            ),
            topTitles:
                const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            rightTitles:
                const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          ),
          borderData: FlBorderData(show: false),
          barTouchData: BarTouchData(
            touchTooltipData: BarTouchTooltipData(
              getTooltipColor: (_) => _card,
              getTooltipItem: (group, gIndex, rod, rIndex) =>
                  BarTooltipItem(
                'R${group.x}: ${rod.toY.toStringAsFixed(2)} MB',
                GoogleFonts.inter(fontSize: 12, color: Colors.white),
              ),
            ),
          ),
          barGroups: barGroups,
        ),
      ),
    );
  }
}
