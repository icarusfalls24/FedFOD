import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import '../../providers/config_provider.dart';
import '../../config/app_config.dart';

class AirportsScreen extends StatelessWidget {
  const AirportsScreen({super.key});

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
    final config = context.watch<ConfigProvider>();
    final airports = config.airports; // List<AirportConfig>

    return Scaffold(
      backgroundColor: _surface,
      appBar: AppBar(
        backgroundColor: _card.withValues(alpha: 0.9),
        elevation: 0,
        title: Text(
          'Airport Fleet',
          style: GoogleFonts.outfit(
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
      ),
      body: airports.isEmpty
          ? Center(
              child: Text(
                'No airports configured',
                style: GoogleFonts.inter(color: Colors.white38, fontSize: 15),
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(16),
              itemCount: airports.length,
              itemBuilder: (ctx, i) {
                final a = airports[i];
                return Padding(
                  padding: const EdgeInsets.only(bottom: 16),
                  child: _AirportCard(airport: a, index: i),
                )
                    .animate()
                    .fadeIn(delay: (120 * i).ms, duration: 500.ms)
                    .slideY(begin: 0.06, end: 0);
              },
            ),
    );
  }
}

// ═══════════════════════════════════════════════
//  AIRPORT CARD (Stateless inner widget)
// ═══════════════════════════════════════════════
class _AirportCard extends StatelessWidget {
  final dynamic airport;
  final int index;

  const _AirportCard({required this.airport, required this.index});

  static const _primary = Color(AppConfig.primaryColor);
  static const _secondary = Color(AppConfig.secondaryColor);
  static const _card = Color(AppConfig.cardColor);
  static const _success = Color(AppConfig.successColor);
  static const _error = Color(AppConfig.errorColor);
  static const _warning = Color(AppConfig.warningColor);

  @override
  Widget build(BuildContext context) {
    final name = airport.name as String;
    final icao = airport.icaoCode as String;
    final type = airport.type as String; // hub / regional / remote
    final gpu = airport.gpu as String;
    final cameras = airport.cameras as int;
    final bandwidth = airport.bandwidth as String;
    final latency = airport.latencyMs as int;
    final qualityScore = (airport.qualityScore as double).clamp(0.0, 1.0);
    final dominantFOD = airport.dominantFodClasses as List<String>;
    final connectivity = airport.connectivity as String; // fibre / LTE / satellite

    IconData typeIcon;
    Color typeColor;
    switch (type) {
      case 'hub':
        typeIcon = Icons.apartment;
        typeColor = _primary;
        break;
      case 'regional':
        typeIcon = Icons.flight;
        typeColor = _secondary;
        break;
      default:
        typeIcon = Icons.satellite_alt;
        typeColor = _warning;
    }

    Color connColor;
    switch (connectivity.toLowerCase()) {
      case 'fibre':
        connColor = _success;
        break;
      case 'lte':
        connColor = _warning;
        break;
      default:
        connColor = _error;
    }

    return GestureDetector(
      onTap: () => _showDetailSheet(context),
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              _card.withValues(alpha: 0.9),
              _card.withValues(alpha: 0.6),
            ],
          ),
          border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.35),
              blurRadius: 16,
              offset: const Offset(0, 6),
            ),
          ],
        ),
        child: Column(
          children: [
            // ── Hero area ──
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                borderRadius:
                    const BorderRadius.vertical(top: Radius.circular(16)),
                gradient: LinearGradient(
                  colors: [
                    typeColor.withValues(alpha: 0.15),
                    Colors.transparent,
                  ],
                ),
              ),
              child: Row(
                children: [
                  Hero(
                    tag: 'airport_icon_$index',
                    child: Container(
                      width: 56,
                      height: 56,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: typeColor.withValues(alpha: 0.2),
                        border:
                            Border.all(color: typeColor.withValues(alpha: 0.4)),
                      ),
                      child: Icon(typeIcon, color: typeColor, size: 28),
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          name,
                          style: GoogleFonts.outfit(
                            fontSize: 20,
                            fontWeight: FontWeight.w700,
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(height: 2),
                        Row(
                          children: [
                            Text(
                              icao,
                              style: GoogleFonts.inter(
                                fontSize: 12,
                                color: Colors.white54,
                                letterSpacing: 1,
                              ),
                            ),
                            const SizedBox(width: 8),
                            _badge(type.toUpperCase(), typeColor),
                          ],
                        ),
                      ],
                    ),
                  ),
                  // Edit button
                  IconButton(
                    icon:
                        const Icon(Icons.tune, color: Colors.white38, size: 20),
                    onPressed: () => _showEditDialog(context),
                    tooltip: 'Edit Hyperparameters',
                  ),
                ],
              ),
            ),

            Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ── Specs Grid ──
                  Row(
                    children: [
                      _specTile(Icons.memory, 'GPU', gpu),
                      _specTile(Icons.videocam, 'Cameras', '$cameras'),
                      _specTile(Icons.speed, 'BW', bandwidth),
                      _specTile(Icons.timer, 'Latency', '${latency}ms'),
                    ],
                  ),
                  const SizedBox(height: 16),

                  // ── Quality bar ──
                  Row(
                    children: [
                      Text(
                        'Data Quality',
                        style: GoogleFonts.inter(
                          fontSize: 12,
                          color: Colors.white54,
                        ),
                      ),
                      const Spacer(),
                      Text(
                        '${(qualityScore * 100).toInt()}%',
                        style: GoogleFonts.inter(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: Colors.white70,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: qualityScore,
                      minHeight: 6,
                      backgroundColor: Colors.white10,
                      valueColor: AlwaysStoppedAnimation(
                        _qualityColor(qualityScore),
                      ),
                    ),
                  ),
                  const SizedBox(height: 14),

                  // ── Dominant FOD Classes ──
                  Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: dominantFOD
                        .map((cls) => _fodChip(cls))
                        .toList(),
                  ),
                  const SizedBox(height: 14),

                  // ── Connectivity ──
                  Row(
                    children: [
                      Icon(Icons.wifi, color: connColor, size: 16),
                      const SizedBox(width: 6),
                      Text(
                        connectivity,
                        style: GoogleFonts.inter(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: connColor,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Helpers ──
  Widget _specTile(IconData icon, String label, String value) {
    return Expanded(
      child: Column(
        children: [
          Icon(icon, color: Colors.white38, size: 18),
          const SizedBox(height: 4),
          Text(
            value,
            style: GoogleFonts.inter(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
            textAlign: TextAlign.center,
          ),
          Text(
            label,
            style: GoogleFonts.inter(fontSize: 10, color: Colors.white38),
          ),
        ],
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

  Widget _fodChip(String label) {
    final colors = [_primary, _secondary, _warning, _success, _error];
    final color = colors[label.hashCode.abs() % colors.length];
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Text(
        label,
        style: GoogleFonts.inter(
          fontSize: 11,
          fontWeight: FontWeight.w500,
          color: color,
        ),
      ),
    );
  }

  Color _qualityColor(double q) {
    if (q >= 0.8) return _success;
    if (q >= 0.5) return _warning;
    return _error;
  }

  // ── Bottom Sheet Detail ──
  void _showDetailSheet(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => DraggableScrollableSheet(
        initialChildSize: 0.7,
        maxChildSize: 0.95,
        minChildSize: 0.4,
        builder: (ctx, controller) => Container(
          decoration: BoxDecoration(
            color: _card,
            borderRadius:
                const BorderRadius.vertical(top: Radius.circular(24)),
            border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
          ),
          child: ListView(
            controller: controller,
            padding: const EdgeInsets.all(24),
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  margin: const EdgeInsets.only(bottom: 20),
                  decoration: BoxDecoration(
                    color: Colors.white24,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              Text(
                airport.name as String,
                style: GoogleFonts.outfit(
                  fontSize: 24,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'ICAO: ${airport.icaoCode}',
                style: GoogleFonts.inter(fontSize: 13, color: Colors.white54),
              ),
              const Divider(color: Colors.white12, height: 32),

              _detailRow('Type', airport.type),
              _detailRow('GPU', airport.gpu),
              _detailRow('Cameras', '${airport.cameras}'),
              _detailRow('Bandwidth', airport.bandwidth),
              _detailRow('Latency', '${airport.latencyMs}ms'),
              _detailRow('Connectivity', airport.connectivity),
              _detailRow(
                  'Quality Score', '${(airport.qualityScore * 100).toInt()}%'),
              _detailRow('Dominant FOD',
                  (airport.dominantFodClasses as List).join(', ')),

              if (airport.localEpochs != null)
                _detailRow('Local Epochs', '${airport.localEpochs}'),
              if (airport.learningRate != null)
                _detailRow('Learning Rate', '${airport.learningRate}'),
              if (airport.batchSize != null)
                _detailRow('Batch Size', '${airport.batchSize}'),
            ],
          ),
        ),
      ),
    );
  }

  Widget _detailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          SizedBox(
            width: 130,
            child: Text(
              label,
              style: GoogleFonts.inter(fontSize: 13, color: Colors.white54),
            ),
          ),
          Expanded(
            child: Text(
              value,
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

  // ── Hyperparameter Edit Dialog ──
  void _showEditDialog(BuildContext context) {
    final lrController =
        TextEditingController(text: '${airport.learningRate ?? 0.001}');
    final epochsController =
        TextEditingController(text: '${airport.localEpochs ?? 5}');
    final batchController =
        TextEditingController(text: '${airport.batchSize ?? 16}');

    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: _card,
        shape:
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        title: Text(
          'Edit ${airport.name}',
          style: GoogleFonts.outfit(
            fontSize: 18,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _dialogField('Learning Rate', lrController),
            const SizedBox(height: 12),
            _dialogField('Local Epochs', epochsController),
            const SizedBox(height: 12),
            _dialogField('Batch Size', batchController),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: Text('Cancel',
                style: GoogleFonts.inter(color: Colors.white54)),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: _primary,
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10)),
            ),
            onPressed: () {
              context.read<ConfigProvider>().updateAirportConfig(
                    airportIndex: index,
                    learningRate: double.tryParse(lrController.text),
                    localEpochs: int.tryParse(epochsController.text),
                    batchSize: int.tryParse(batchController.text),
                  );
              Navigator.pop(ctx);
            },
            child: Text('Save',
                style: GoogleFonts.inter(
                    fontWeight: FontWeight.w600, color: Colors.white)),
          ),
        ],
      ),
    );
  }

  Widget _dialogField(String label, TextEditingController controller) {
    return TextField(
      controller: controller,
      style: GoogleFonts.inter(fontSize: 14, color: Colors.white),
      decoration: InputDecoration(
        labelText: label,
        labelStyle: GoogleFonts.inter(fontSize: 13, color: Colors.white54),
        filled: true,
        fillColor: Colors.white.withValues(alpha: 0.06),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(10),
          borderSide: BorderSide.none,
        ),
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      ),
      keyboardType: const TextInputType.numberWithOptions(decimal: true),
    );
  }
}
