import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import '../../providers/training_provider.dart';
import '../../config/app_config.dart';

class TrainingScreen extends StatefulWidget {
  const TrainingScreen({super.key});

  @override
  State<TrainingScreen> createState() => _TrainingScreenState();
}

class _TrainingScreenState extends State<TrainingScreen> {
  // ── Design tokens ──
  static const _primary = Color(AppConfig.primaryColor);
  static const _secondary = Color(AppConfig.secondaryColor);
  static const _surface = Color(AppConfig.surfaceColor);
  static const _card = Color(AppConfig.cardColor);
  static const _success = Color(AppConfig.successColor);
  static const _error = Color(AppConfig.errorColor);
  static const _warning = Color(AppConfig.warningColor);

  // ── Local config state ──
  double _numRounds = 50;
  double _minClients = 2;
  int _port = 8000;
  bool _dummyModel = false;

  @override
  Widget build(BuildContext context) {
    final training = context.watch<TrainingProvider>();

    return Scaffold(
      backgroundColor: _surface,
      appBar: AppBar(
        backgroundColor: _card.withValues(alpha: 0.9),
        elevation: 0,
        title: Text(
          'Training Control',
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
          // ── State Badge ──
          _buildStateBadge(training),
          const SizedBox(height: 16),

          // ── Configuration ──
          _buildConfigSection(training),
          const SizedBox(height: 16),

          // ── Action Buttons ──
          _buildActionButtons(training),
          const SizedBox(height: 16),

          // ── Progress ──
          _buildProgressSection(training),
          const SizedBox(height: 20),

          // ── Per-Round Results ──
          _buildRoundResultsSection(training),
          const SizedBox(height: 20),

          // ── Per-Client Status ──
          _buildClientStatusSection(training),
          const SizedBox(height: 24),
        ],
      ),
    );
  }

  // ══════════════════════════════════════════
  //  STATE BADGE
  // ══════════════════════════════════════════
  Widget _buildStateBadge(TrainingProvider t) {
    final state = t.trainingState; // 'idle' | 'training' | 'completed'
    Color badgeColor;
    IconData badgeIcon;
    String label;

    switch (state) {
      case 'training':
        badgeColor = _secondary;
        badgeIcon = Icons.model_training;
        label = 'Training';
        break;
      case 'completed':
        badgeColor = _success;
        badgeIcon = Icons.check_circle_outline;
        label = 'Completed';
        break;
      default:
        badgeColor = Colors.white38;
        badgeIcon = Icons.pause_circle_outline;
        label = 'Idle';
    }

    return _glassCard(
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(badgeIcon, color: badgeColor, size: 28),
          const SizedBox(width: 12),
          Text(
            label,
            style: GoogleFonts.outfit(
              fontSize: 22,
              fontWeight: FontWeight.w700,
              color: badgeColor,
            ),
          ),
          const SizedBox(width: 12),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: badgeColor.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: badgeColor.withValues(alpha: 0.4)),
            ),
            child: Text(
              state.toUpperCase(),
              style: GoogleFonts.inter(
                fontSize: 11,
                fontWeight: FontWeight.w700,
                color: badgeColor,
                letterSpacing: 1,
              ),
            ),
          ),
        ],
      ),
    ).animate().fadeIn(duration: 400.ms).slideY(begin: -0.05, end: 0);
  }

  // ══════════════════════════════════════════
  //  CONFIGURATION
  // ══════════════════════════════════════════
  Widget _buildConfigSection(TrainingProvider t) {
    return _glassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Configuration',
            style: GoogleFonts.outfit(
              fontSize: 16,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 12),

          // Rounds slider
          _sliderRow(
            label: 'Rounds',
            value: _numRounds,
            min: 1,
            max: 200,
            divisions: 199,
            onChanged: (v) => setState(() => _numRounds = v),
          ),

          // Min clients slider
          _sliderRow(
            label: 'Min Clients',
            value: _minClients,
            min: 1,
            max: 3,
            divisions: 2,
            onChanged: (v) => setState(() => _minClients = v),
          ),

          // Port
          Row(
            children: [
              Text('Port',
                  style: GoogleFonts.inter(
                      fontSize: 13, color: Colors.white70)),
              const Spacer(),
              SizedBox(
                width: 100,
                child: TextField(
                  style: GoogleFonts.inter(
                    fontSize: 14,
                    color: Colors.white,
                  ),
                  decoration: InputDecoration(
                    isDense: true,
                    contentPadding:
                        const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    filled: true,
                    fillColor: Colors.white.withValues(alpha: 0.06),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                      borderSide: BorderSide.none,
                    ),
                    hintText: '$_port',
                    hintStyle: GoogleFonts.inter(color: Colors.white30),
                  ),
                  keyboardType: TextInputType.number,
                  onChanged: (v) {
                    final parsed = int.tryParse(v);
                    if (parsed != null) setState(() => _port = parsed);
                  },
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),

          // Dummy model toggle
          Row(
            children: [
              Text('Dummy Model',
                  style: GoogleFonts.inter(
                      fontSize: 13, color: Colors.white70)),
              const Spacer(),
              Switch(
                value: _dummyModel,
                activeColor: _primary,
                onChanged: (v) => setState(() => _dummyModel = v),
              ),
            ],
          ),
        ],
      ),
    ).animate().fadeIn(delay: 100.ms, duration: 400.ms);
  }

  Widget _sliderRow({
    required String label,
    required double value,
    required double min,
    required double max,
    required int divisions,
    required ValueChanged<double> onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          SizedBox(
            width: 80,
            child: Text(label,
                style:
                    GoogleFonts.inter(fontSize: 13, color: Colors.white70)),
          ),
          Expanded(
            child: SliderTheme(
              data: SliderThemeData(
                activeTrackColor: _primary,
                inactiveTrackColor: Colors.white12,
                thumbColor: _primary,
                overlayColor: _primary.withValues(alpha: 0.15),
                trackHeight: 3,
              ),
              child: Slider(
                value: value,
                min: min,
                max: max,
                divisions: divisions,
                onChanged: onChanged,
              ),
            ),
          ),
          SizedBox(
            width: 40,
            child: Text(
              value.toInt().toString(),
              textAlign: TextAlign.right,
              style: GoogleFonts.inter(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: Colors.white,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ══════════════════════════════════════════
  //  ACTION BUTTONS
  // ══════════════════════════════════════════
  Widget _buildActionButtons(TrainingProvider t) {
    final isTraining = t.isTraining;
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            style: ElevatedButton.styleFrom(
              backgroundColor: _primary,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              elevation: 4,
            ),
            icon: const Icon(Icons.play_arrow_rounded),
            label: Text('Start',
                style: GoogleFonts.inter(fontWeight: FontWeight.w600)),
            onPressed: isTraining
                ? null
                : () => t.startTraining(
                      numRounds: _numRounds.toInt(),
                      minClients: _minClients.toInt(),
                      port: _port,
                      dummyModel: _dummyModel,
                    ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: ElevatedButton.icon(
            style: ElevatedButton.styleFrom(
              backgroundColor: _error,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              elevation: 4,
            ),
            icon: const Icon(Icons.stop_rounded),
            label: Text('Stop',
                style: GoogleFonts.inter(fontWeight: FontWeight.w600)),
            onPressed: isTraining ? () => t.stopTraining() : null,
          ),
        ),
      ],
    ).animate().fadeIn(delay: 200.ms, duration: 400.ms);
  }

  // ══════════════════════════════════════════
  //  PROGRESS
  // ══════════════════════════════════════════
  Widget _buildProgressSection(TrainingProvider t) {
    final progress = t.totalRounds > 0
        ? (t.currentRound / t.totalRounds).clamp(0.0, 1.0)
        : 0.0;
    return _glassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Progress',
                style: GoogleFonts.outfit(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                ),
              ),
              Text(
                '${t.currentRound} / ${t.totalRounds}',
                style: GoogleFonts.inter(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: _secondary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(6),
            child: LinearProgressIndicator(
              value: progress,
              minHeight: 8,
              backgroundColor: Colors.white10,
              valueColor: AlwaysStoppedAnimation(_primary),
            ),
          ),
        ],
      ),
    );
  }

  // ══════════════════════════════════════════
  //  PER-ROUND RESULTS
  // ══════════════════════════════════════════
  Widget _buildRoundResultsSection(TrainingProvider t) {
    final rounds = t.roundResults; // List<RoundResult>
    return _glassCard(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
      child: ExpansionTile(
        tilePadding: EdgeInsets.zero,
        iconColor: Colors.white54,
        collapsedIconColor: Colors.white38,
        title: Text(
          'Per-Round Results (${rounds.length})',
          style: GoogleFonts.outfit(
            fontSize: 15,
            fontWeight: FontWeight.w600,
            color: Colors.white,
          ),
        ),
        children: rounds.isEmpty
            ? [
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Text('No results yet',
                      style:
                          GoogleFonts.inter(color: Colors.white38, fontSize: 13)),
                ),
              ]
            : rounds.asMap().entries.map((entry) {
                final i = entry.key;
                final r = entry.value;
                return _roundResultTile(r, i);
              }).toList(),
      ),
    );
  }

  Widget _roundResultTile(dynamic r, int index) {
    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        children: [
          // Round number
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _primary.withValues(alpha: 0.2),
            ),
            child: Center(
              child: Text(
                '${r.round}',
                style: GoogleFonts.inter(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: _primary,
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    _miniStat('Loss', r.loss.toStringAsFixed(4)),
                    const SizedBox(width: 12),
                    _miniStat('mAP50', r.map50.toStringAsFixed(3)),
                    const SizedBox(width: 12),
                    _miniStat('Gini', r.gini.toStringAsFixed(3)),
                  ],
                ),
                const SizedBox(height: 4),
                Row(
                  children: [
                    _miniStat('Clients', '${r.numClients}'),
                    const SizedBox(width: 12),
                    _miniStat('Time', '${r.timeSec.toStringAsFixed(1)}s'),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    )
        .animate()
        .fadeIn(delay: (50 * index).ms, duration: 300.ms);
  }

  Widget _miniStat(String label, String value) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          '$label: ',
          style: GoogleFonts.inter(fontSize: 11, color: Colors.white38),
        ),
        Text(
          value,
          style: GoogleFonts.inter(
            fontSize: 11,
            fontWeight: FontWeight.w600,
            color: Colors.white70,
          ),
        ),
      ],
    );
  }

  // ══════════════════════════════════════════
  //  PER-CLIENT STATUS
  // ══════════════════════════════════════════
  Widget _buildClientStatusSection(TrainingProvider t) {
    final clients = t.clientStatuses; // List<ClientStatus>
    if (clients.isEmpty) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Client Status',
          style: GoogleFonts.outfit(
            fontSize: 16,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
        const SizedBox(height: 8),
        ...clients.asMap().entries.map((entry) {
          final i = entry.key;
          final c = entry.value;
          return Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: _glassCard(
              child: Row(
                children: [
                  Icon(Icons.computer, color: _secondary, size: 22),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          c.clientId,
                          style: GoogleFonts.outfit(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            _miniStat('Loss', c.loss.toStringAsFixed(4)),
                            const SizedBox(width: 12),
                            _miniStat('Samples', '${c.samples}'),
                            const SizedBox(width: 12),
                            _miniStat('Device', c.device),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          )
              .animate()
              .fadeIn(delay: (80 * i).ms, duration: 350.ms)
              .slideX(begin: 0.04, end: 0);
        }),
      ],
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
