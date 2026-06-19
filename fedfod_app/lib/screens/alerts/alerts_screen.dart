import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import '../../providers/metrics_provider.dart';
import '../../config/app_config.dart';

class AlertsScreen extends StatelessWidget {
  const AlertsScreen({super.key});

  // ── Design tokens ──
  static const _primary = Color(AppConfig.primaryColor);
  static const _secondary = Color(AppConfig.secondaryColor);
  static const _surface = Color(AppConfig.surfaceColor);
  static const _card = Color(AppConfig.cardColor);
  static const _success = Color(AppConfig.successColor);
  static const _error = Color(AppConfig.errorColor);
  static const _warning = Color(AppConfig.warningColor);

  @override
  Widget build(BuildContext context) {
    final m = context.watch<MetricsProvider>();

    // ── Publication checklist items ──
    final checks = <_CheckItem>[
      _CheckItem(
        label: 'mAP@50 ≥ ${AppConfig.targetMAP50}',
        target: AppConfig.targetMAP50,
        current: m.latestMAP50,
        pass: m.latestMAP50 >= AppConfig.targetMAP50,
      ),
      _CheckItem(
        label: 'FAR/hr ≤ ${AppConfig.targetFARPerHour}',
        target: AppConfig.targetFARPerHour,
        current: m.latestFAR,
        pass: m.latestFAR <= AppConfig.targetFARPerHour,
      ),
      _CheckItem(
        label: 'Comm payload ≤ ${AppConfig.targetCommPayloadMB} MB',
        target: AppConfig.targetCommPayloadMB,
        current: m.latestCommMB,
        pass: m.latestCommMB <= AppConfig.targetCommPayloadMB,
      ),
      _CheckItem(
        label: 'Privacy ε = ${AppConfig.targetDPEpsilon}',
        target: AppConfig.targetDPEpsilon,
        current: m.latestEpsilon,
        pass: m.latestEpsilon <= AppConfig.targetDPEpsilon,
      ),
      _CheckItem(
        label: 'E2E latency ≤ ${AppConfig.targetLatencySec}s',
        target: AppConfig.targetLatencySec.toDouble(),
        current: m.latestLatencySec,
        pass: m.latestLatencySec <= AppConfig.targetLatencySec,
      ),
      _CheckItem(
        label: 'Gini coefficient < ${AppConfig.targetGiniMax}',
        target: AppConfig.targetGiniMax,
        current: m.latestGini,
        pass: m.latestGini < AppConfig.targetGiniMax,
      ),
    ];

    final passCount = checks.where((c) => c.pass).length;
    final readiness = checks.isEmpty ? 0.0 : passCount / checks.length;

    final violations = checks.where((c) => !c.pass).toList();
    final novelDetections = m.novelDetections; // List<String>

    return Scaffold(
      backgroundColor: _surface,
      appBar: AppBar(
        backgroundColor: _card.withValues(alpha: 0.9),
        elevation: 0,
        title: Text(
          'Alerts & Readiness',
          style: GoogleFonts.outfit(
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // ── Readiness Ring ──
          _buildReadinessRing(readiness, passCount, checks.length),
          const SizedBox(height: 20),

          // ── Publication Targets ──
          _buildPublicationCard(checks),
          const SizedBox(height: 20),

          // ── Active Alerts ──
          _buildActiveAlerts(violations),
          const SizedBox(height: 20),

          // ── Novel Detections ──
          _buildNovelDetections(novelDetections),
          const SizedBox(height: 24),
        ],
      ),
    );
  }

  // ══════════════════════════════════════════
  //  READINESS RING
  // ══════════════════════════════════════════
  Widget _buildReadinessRing(double readiness, int pass, int total) {
    final pct = (readiness * 100).toInt();
    final ringColor = readiness >= 1.0
        ? _success
        : readiness >= 0.6
            ? _warning
            : _error;

    return _glassCard(
      child: Column(
        children: [
          SizedBox(
            width: 120,
            height: 120,
            child: Stack(
              fit: StackFit.expand,
              children: [
                CircularProgressIndicator(
                  value: readiness,
                  strokeWidth: 8,
                  backgroundColor: Colors.white10,
                  valueColor: AlwaysStoppedAnimation(ringColor),
                ),
                Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        '$pct%',
                        style: GoogleFonts.outfit(
                          fontSize: 28,
                          fontWeight: FontWeight.w700,
                          color: Colors.white,
                        ),
                      ),
                      Text(
                        '$pass/$total',
                        style: GoogleFonts.inter(
                          fontSize: 12,
                          color: Colors.white54,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          Text(
            'Publication Readiness',
            style: GoogleFonts.outfit(
              fontSize: 16,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            readiness >= 1.0
                ? 'All targets met — ready for publication!'
                : '${total - pass} target(s) not yet met',
            style: GoogleFonts.inter(
              fontSize: 12,
              color: readiness >= 1.0 ? _success : Colors.white54,
            ),
          ),
        ],
      ),
    )
        .animate()
        .fadeIn(duration: 500.ms)
        .scale(begin: const Offset(0.95, 0.95), end: const Offset(1, 1));
  }

  // ══════════════════════════════════════════
  //  PUBLICATION TARGETS
  // ══════════════════════════════════════════
  Widget _buildPublicationCard(List<_CheckItem> checks) {
    return _glassCard(
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.fact_check, color: _primary, size: 20),
              const SizedBox(width: 8),
              Text(
                'Publication Targets',
                style: GoogleFonts.outfit(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          ...checks.asMap().entries.map((entry) {
            final i = entry.key;
            final c = entry.value;
            return _checkRow(c, i);
          }),
        ],
      ),
    );
  }

  Widget _checkRow(_CheckItem c, int index) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          // Pass/fail icon
          AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            width: 28,
            height: 28,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: (c.pass ? _success : _error).withValues(alpha: 0.15),
            ),
            child: Icon(
              c.pass ? Icons.check : Icons.close,
              size: 16,
              color: c.pass ? _success : _error,
            ),
          ),
          const SizedBox(width: 12),

          // Label
          Expanded(
            child: Text(
              c.label,
              style: GoogleFonts.inter(
                fontSize: 13,
                color: Colors.white70,
              ),
            ),
          ),

          // Current value
          Text(
            c.current.toStringAsFixed(2),
            style: GoogleFonts.inter(
              fontSize: 13,
              fontWeight: FontWeight.w700,
              color: c.pass ? _success : _error,
            ),
          ),
        ],
      ),
    )
        .animate()
        .fadeIn(delay: (60 * index).ms, duration: 350.ms)
        .slideX(begin: 0.03, end: 0);
  }

  // ══════════════════════════════════════════
  //  ACTIVE ALERTS
  // ══════════════════════════════════════════
  Widget _buildActiveAlerts(List<_CheckItem> violations) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(Icons.warning_amber_rounded, color: _warning, size: 20),
            const SizedBox(width: 8),
            Text(
              'Active Alerts',
              style: GoogleFonts.outfit(
                fontSize: 16,
                fontWeight: FontWeight.w700,
                color: Colors.white,
              ),
            ),
            const Spacer(),
            if (violations.isNotEmpty)
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                decoration: BoxDecoration(
                  color: _error.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  '${violations.length}',
                  style: GoogleFonts.inter(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: _error,
                  ),
                ),
              ),
          ],
        ),
        const SizedBox(height: 8),
        if (violations.isEmpty)
          _glassCard(
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.check_circle_outline, color: _success, size: 20),
                const SizedBox(width: 8),
                Text(
                  'No active alerts',
                  style: GoogleFonts.inter(
                    fontSize: 13,
                    color: _success,
                  ),
                ),
              ],
            ),
          )
        else
          ...violations.asMap().entries.map((entry) {
            final i = entry.key;
            final v = entry.value;
            return Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: _alertCard(v),
            )
                .animate()
                .fadeIn(delay: (80 * i).ms, duration: 400.ms)
                .slideX(begin: 0.04, end: 0);
          }),
      ],
    );
  }

  Widget _alertCard(_CheckItem v) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12),
        gradient: LinearGradient(
          colors: [
            _error.withValues(alpha: 0.08),
            _card.withValues(alpha: 0.7),
          ],
        ),
        border: Border.all(color: _error.withValues(alpha: 0.25)),
        boxShadow: [
          BoxShadow(
            color: _error.withValues(alpha: 0.1),
            blurRadius: 8,
          ),
        ],
      ),
      child: Row(
        children: [
          Icon(Icons.error_outline, color: _error, size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  v.label,
                  style: GoogleFonts.inter(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'Current: ${v.current.toStringAsFixed(2)}  •  Target: ${v.target.toStringAsFixed(2)}',
                  style:
                      GoogleFonts.inter(fontSize: 11, color: Colors.white54),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ══════════════════════════════════════════
  //  NOVEL DETECTIONS
  // ══════════════════════════════════════════
  Widget _buildNovelDetections(List<String> detections) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(Icons.new_releases_outlined, color: _secondary, size: 20),
            const SizedBox(width: 8),
            Text(
              'Novel Detections',
              style: GoogleFonts.outfit(
                fontSize: 16,
                fontWeight: FontWeight.w700,
                color: Colors.white,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        if (detections.isEmpty)
          _glassCard(
            child: Center(
              child: Text(
                'No novel FOD classes detected',
                style: GoogleFonts.inter(fontSize: 13, color: Colors.white38),
              ),
            ),
          )
        else
          ...detections.asMap().entries.map((entry) {
            final i = entry.key;
            final d = entry.value;
            return Padding(
              padding: const EdgeInsets.only(bottom: 6),
              child: _novelDetectionTile(d),
            )
                .animate()
                .fadeIn(delay: (60 * i).ms, duration: 350.ms)
                .slideY(begin: 0.05, end: 0);
          }),
      ],
    );
  }

  Widget _novelDetectionTile(String detection) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(10),
        gradient: LinearGradient(
          colors: [
            _secondary.withValues(alpha: 0.06),
            _card.withValues(alpha: 0.6),
          ],
        ),
        border: Border.all(color: _secondary.withValues(alpha: 0.2)),
      ),
      child: Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _secondary,
              boxShadow: [
                BoxShadow(
                  color: _secondary.withValues(alpha: 0.4),
                  blurRadius: 6,
                ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              detection,
              style: GoogleFonts.inter(
                fontSize: 13,
                color: Colors.white,
              ),
            ),
          ),
          Text(
            'NEW',
            style: GoogleFonts.inter(
              fontSize: 10,
              fontWeight: FontWeight.w700,
              color: _secondary,
              letterSpacing: 1,
            ),
          ),
        ],
      ),
    );
  }

  // ══════════════════════════════════════════
  //  SHARED HELPERS
  // ══════════════════════════════════════════
  Widget _glassCard({
    required Widget child,
    EdgeInsets padding = const EdgeInsets.all(16),
  }) {
    return Container(
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

// ── Helper data class ──
class _CheckItem {
  final String label;
  final double target;
  final double current;
  final bool pass;

  const _CheckItem({
    required this.label,
    required this.target,
    required this.current,
    required this.pass,
  });
}
