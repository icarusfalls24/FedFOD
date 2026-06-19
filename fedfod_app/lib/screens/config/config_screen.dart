import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';

import '../../providers/config_provider.dart';
import '../../config/app_config.dart';

class ConfigScreen extends StatefulWidget {
  const ConfigScreen({super.key});

  @override
  State<ConfigScreen> createState() => _ConfigScreenState();
}

class _ConfigScreenState extends State<ConfigScreen> {
  // ── Design tokens ──
  static const _primary = Color(AppConfig.primaryColor);
  static const _secondary = Color(AppConfig.secondaryColor);
  static const _surface = Color(AppConfig.surfaceColor);
  static const _card = Color(AppConfig.cardColor);
  static const _success = Color(AppConfig.successColor);
  static const _error = Color(AppConfig.errorColor);

  // ── Local editing state ──
  late double _numRounds;
  late int _numClients;
  late int _minClients;
  late int _localEpochs;
  late TextEditingController _lrController;
  late bool _scaffoldCorrection;

  late double _dpEpsilon;
  late TextEditingController _dpDeltaController;
  late double _gradClipNorm;

  late double _maxPayloadMb;
  late double _sparsTopK;
  late int _quantBits;

  late String _backbone;
  late int _numClasses;
  late double _confThreshold;
  late double _nmsIou;

  late double _map50Target;
  late double _farTarget;
  late int _latencyTarget;

  bool _initialized = false;

  void _initFromProvider(ConfigProvider c) {
    if (_initialized) return;
    _initialized = true;

    _numRounds = c.numRounds.toDouble();
    _numClients = c.numClients;
    _minClients = c.minClients;
    _localEpochs = c.localEpochs;
    _lrController = TextEditingController(text: c.learningRate.toString());
    _scaffoldCorrection = c.scaffoldCorrection;

    _dpEpsilon = c.dpEpsilon;
    _dpDeltaController = TextEditingController(text: c.dpDelta.toString());
    _gradClipNorm = c.gradientClipNorm;

    _maxPayloadMb = c.maxPayloadMb;
    _sparsTopK = c.sparsificationTopKPct;
    _quantBits = c.quantizationBits;

    _backbone = c.backbone;
    _numClasses = c.numClasses;
    _confThreshold = c.confThreshold;
    _nmsIou = c.nmsIouThreshold;

    _map50Target = c.map50Target;
    _farTarget = c.farTarget;
    _latencyTarget = c.latencyTarget;
  }

  @override
  void dispose() {
    _lrController.dispose();
    _dpDeltaController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final config = context.watch<ConfigProvider>();
    _initFromProvider(config);

    return Scaffold(
      backgroundColor: _surface,
      appBar: AppBar(
        backgroundColor: _card.withValues(alpha: 0.9),
        elevation: 0,
        title: Text(
          'Configuration',
          style: GoogleFonts.outfit(
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.restart_alt, color: Colors.white54),
            tooltip: 'Reset to defaults',
            onPressed: _handleReset,
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // ── Federated Learning ──
          _section(
            title: 'Federated Learning',
            icon: Icons.hub,
            children: [
              _sliderTile('Num Rounds', _numRounds, 1, 200, 199,
                  (v) => setState(() => _numRounds = v)),
              _intRow('Num Clients', _numClients),
              _sliderTile('Min Clients', _minClients.toDouble(), 1, 3, 2,
                  (v) => setState(() => _minClients = v.toInt())),
              _intRow('Local Epochs', _localEpochs),
              _textFieldTile('Learning Rate', _lrController),
              _switchTile('SCAFFOLD Correction', _scaffoldCorrection,
                  (v) => setState(() => _scaffoldCorrection = v)),
            ],
          ),

          // ── Privacy ──
          _section(
            title: 'Privacy',
            icon: Icons.shield,
            children: [
              _sliderTile('DP Epsilon', _dpEpsilon, 0.1, 10, 99,
                  (v) => setState(() => _dpEpsilon = v),
                  decimals: 1),
              _textFieldTile('DP Delta', _dpDeltaController),
              _sliderTile('Gradient Clip Norm', _gradClipNorm, 0.1, 10, 99,
                  (v) => setState(() => _gradClipNorm = v),
                  decimals: 1),
            ],
          ),

          // ── Communication ──
          _section(
            title: 'Communication',
            icon: Icons.cell_tower,
            children: [
              _sliderTile('Max Payload (MB)', _maxPayloadMb, 0.1, 10, 99,
                  (v) => setState(() => _maxPayloadMb = v),
                  decimals: 1),
              _sliderTile('Sparsification Top-K %', _sparsTopK, 0, 100, 100,
                  (v) => setState(() => _sparsTopK = v)),
              _dropdownTile('Quantization Bits', _quantBits, [4, 8, 16],
                  (v) => setState(() => _quantBits = v)),
            ],
          ),

          // ── Model ──
          _section(
            title: 'Model',
            icon: Icons.model_training,
            children: [
              _dropdownStringTile(
                  'Backbone',
                  _backbone,
                  ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l'],
                  (v) => setState(() => _backbone = v)),
              _intRow('Num Classes', _numClasses),
              _sliderTile('Conf Threshold', _confThreshold, 0.01, 1.0, 99,
                  (v) => setState(() => _confThreshold = v),
                  decimals: 2),
              _sliderTile('NMS IoU Threshold', _nmsIou, 0.01, 1.0, 99,
                  (v) => setState(() => _nmsIou = v),
                  decimals: 2),
            ],
          ),

          // ── Targets ──
          _section(
            title: 'Targets',
            icon: Icons.flag,
            children: [
              _sliderTile('mAP@50 Known', _map50Target, 0.0, 1.0, 100,
                  (v) => setState(() => _map50Target = v),
                  decimals: 2),
              _sliderTile('FAR/hr Target', _farTarget, 0.0, 10.0, 100,
                  (v) => setState(() => _farTarget = v),
                  decimals: 1),
              _intRow('E2E Latency (sec)', _latencyTarget),
            ],
          ),

          const SizedBox(height: 20),

          // ── Action Buttons ──
          _buildActions(context),
          const SizedBox(height: 32),
        ],
      ),
    );
  }

  // ══════════════════════════════════════════
  //  SECTION
  // ══════════════════════════════════════════
  Widget _section({
    required String title,
    required IconData icon,
    required List<Widget> children,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Container(
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
              color: Colors.black.withValues(alpha: 0.25),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: ExpansionTile(
          tilePadding: const EdgeInsets.symmetric(horizontal: 16),
          childrenPadding:
              const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          leading: Icon(icon, color: _primary, size: 22),
          title: Text(
            title,
            style: GoogleFonts.outfit(
              fontSize: 16,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
          ),
          iconColor: Colors.white54,
          collapsedIconColor: Colors.white38,
          initiallyExpanded: false,
          children: children,
        ),
      ),
    ).animate().fadeIn(duration: 400.ms).slideY(begin: 0.03, end: 0);
  }

  // ══════════════════════════════════════════
  //  TILE BUILDERS
  // ══════════════════════════════════════════
  Widget _sliderTile(
    String label,
    double value,
    double min,
    double max,
    int divisions,
    ValueChanged<double> onChanged, {
    int decimals = 0,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          SizedBox(
            width: 140,
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
                overlayColor: _primary.withValues(alpha: 0.12),
                trackHeight: 3,
              ),
              child: Slider(
                value: value.clamp(min, max),
                min: min,
                max: max,
                divisions: divisions,
                onChanged: onChanged,
              ),
            ),
          ),
          SizedBox(
            width: 50,
            child: Text(
              value.toStringAsFixed(decimals),
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

  Widget _switchTile(
      String label, bool value, ValueChanged<bool> onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          Expanded(
            child: Text(label,
                style:
                    GoogleFonts.inter(fontSize: 13, color: Colors.white70)),
          ),
          Switch(value: value, activeColor: _primary, onChanged: onChanged),
        ],
      ),
    );
  }

  Widget _textFieldTile(String label, TextEditingController controller) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          SizedBox(
            width: 140,
            child: Text(label,
                style:
                    GoogleFonts.inter(fontSize: 13, color: Colors.white70)),
          ),
          Expanded(
            child: TextField(
              controller: controller,
              style: GoogleFonts.inter(fontSize: 14, color: Colors.white),
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
              ),
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
            ),
          ),
        ],
      ),
    );
  }

  Widget _dropdownTile(
      String label, int value, List<int> options, ValueChanged<int> onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          SizedBox(
            width: 140,
            child: Text(label,
                style:
                    GoogleFonts.inter(fontSize: 13, color: Colors.white70)),
          ),
          Expanded(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.06),
                borderRadius: BorderRadius.circular(8),
              ),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<int>(
                  value: value,
                  dropdownColor: _card,
                  style:
                      GoogleFonts.inter(fontSize: 14, color: Colors.white),
                  icon: const Icon(Icons.arrow_drop_down,
                      color: Colors.white38),
                  items: options
                      .map((o) =>
                          DropdownMenuItem(value: o, child: Text('$o')))
                      .toList(),
                  onChanged: (v) {
                    if (v != null) onChanged(v);
                  },
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _dropdownStringTile(String label, String value,
      List<String> options, ValueChanged<String> onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          SizedBox(
            width: 140,
            child: Text(label,
                style:
                    GoogleFonts.inter(fontSize: 13, color: Colors.white70)),
          ),
          Expanded(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              decoration: BoxDecoration(
                color: Colors.white.withValues(alpha: 0.06),
                borderRadius: BorderRadius.circular(8),
              ),
              child: DropdownButtonHideUnderline(
                child: DropdownButton<String>(
                  value: options.contains(value) ? value : options.first,
                  dropdownColor: _card,
                  style:
                      GoogleFonts.inter(fontSize: 14, color: Colors.white),
                  icon: const Icon(Icons.arrow_drop_down,
                      color: Colors.white38),
                  items: options
                      .map((o) =>
                          DropdownMenuItem(value: o, child: Text(o)))
                      .toList(),
                  onChanged: (v) {
                    if (v != null) onChanged(v);
                  },
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _intRow(String label, int value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        children: [
          SizedBox(
            width: 140,
            child: Text(label,
                style:
                    GoogleFonts.inter(fontSize: 13, color: Colors.white70)),
          ),
          Text(
            '$value',
            style: GoogleFonts.inter(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }

  // ══════════════════════════════════════════
  //  ACTION BUTTONS
  // ══════════════════════════════════════════
  Widget _buildActions(BuildContext context) {
    return Row(
      children: [
        Expanded(
          child: OutlinedButton.icon(
            style: OutlinedButton.styleFrom(
              foregroundColor: Colors.white54,
              side: BorderSide(color: Colors.white.withValues(alpha: 0.2)),
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
            ),
            icon: const Icon(Icons.restart_alt, size: 18),
            label: Text('Reset',
                style: GoogleFonts.inter(fontWeight: FontWeight.w600)),
            onPressed: _handleReset,
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          flex: 2,
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
            icon: const Icon(Icons.save, size: 18),
            label: Text('Save Configuration',
                style: GoogleFonts.inter(fontWeight: FontWeight.w600)),
            onPressed: () => _handleSave(context),
          ),
        ),
      ],
    ).animate().fadeIn(delay: 200.ms, duration: 400.ms);
  }

  // ══════════════════════════════════════════
  //  HANDLERS
  // ══════════════════════════════════════════
  Future<void> _handleSave(BuildContext context) async {
    try {
      await context.read<ConfigProvider>().saveGlobalConfig(
            numRounds: _numRounds.toInt(),
            numClients: _numClients,
            minClients: _minClients,
            localEpochs: _localEpochs,
            learningRate: double.tryParse(_lrController.text) ?? 0.001,
            scaffoldCorrection: _scaffoldCorrection,
            dpEpsilon: _dpEpsilon,
            dpDelta: double.tryParse(_dpDeltaController.text) ?? 1e-5,
            gradientClipNorm: _gradClipNorm,
            maxPayloadMb: _maxPayloadMb,
            sparsificationTopKPct: _sparsTopK,
            quantizationBits: _quantBits,
            backbone: _backbone,
            numClasses: _numClasses,
            confThreshold: _confThreshold,
            nmsIouThreshold: _nmsIou,
            map50Target: _map50Target,
            farTarget: _farTarget,
            latencyTarget: _latencyTarget,
          );
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Row(
              children: [
                const Icon(Icons.check_circle, color: Colors.white, size: 18),
                const SizedBox(width: 8),
                Text('Configuration saved successfully',
                    style: GoogleFonts.inter(color: Colors.white)),
              ],
            ),
            backgroundColor: _success.withValues(alpha: 0.85),
            behavior: SnackBarBehavior.floating,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Row(
              children: [
                const Icon(Icons.error_outline, color: Colors.white, size: 18),
                const SizedBox(width: 8),
                Expanded(
                  child: Text('Failed to save: $e',
                      style: GoogleFonts.inter(color: Colors.white)),
                ),
              ],
            ),
            backgroundColor: _error.withValues(alpha: 0.85),
            behavior: SnackBarBehavior.floating,
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
          ),
        );
      }
    }
  }

  void _handleReset() {
    setState(() => _initialized = false);
  }
}
