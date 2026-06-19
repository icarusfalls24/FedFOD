import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import 'package:flutter_animate/flutter_animate.dart';

import 'config/app_config.dart';
import 'services/api_service.dart';
import 'services/websocket_service.dart';
import 'providers/metrics_provider.dart';
import 'providers/training_provider.dart';
import 'providers/config_provider.dart';
import 'screens/dashboard/dashboard_screen.dart';
import 'screens/training/training_screen.dart';
import 'screens/airports/airports_screen.dart';
import 'screens/metrics/metrics_screen.dart';
import 'screens/config/config_screen.dart';
import 'screens/logs/logs_screen.dart';
import 'screens/alerts/alerts_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  Animate.restartOnHotReload = true;

  final apiService = ApiService();
  final wsService = WebSocketService();

  runApp(
    MultiProvider(
      providers: [
        Provider<ApiService>.value(value: apiService),
        Provider<WebSocketService>.value(value: wsService),
        ChangeNotifierProvider(
          create: (_) => MetricsProvider(apiService, wsService),
        ),
        ChangeNotifierProvider(
          create: (context) => TrainingProvider(
            apiService,
            context.read<MetricsProvider>(),
          ),
        ),
        ChangeNotifierProvider(
          create: (_) => ConfigProvider(apiService),
        ),
      ],
      child: const FedFODApp(),
    ),
  );
}

class FedFODApp extends StatelessWidget {
  const FedFODApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'FedFOD Command Center',
      debugShowCheckedModeBanner: false,
      theme: _buildDarkTheme(),
      home: const AppShell(),
    );
  }

  ThemeData _buildDarkTheme() {
    const primary = Color(AppConfig.primaryColor);
    const secondary = Color(AppConfig.secondaryColor);
    const surface = Color(AppConfig.surfaceColor);
    const cardColor = Color(AppConfig.cardColor);

    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      scaffoldBackgroundColor: surface,
      colorScheme: ColorScheme.dark(
        primary: primary,
        secondary: secondary,
        surface: surface,
        error: const Color(AppConfig.errorColor),
      ),
      cardTheme: CardThemeData(
        color: cardColor,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(color: Colors.white.withValues(alpha: 0.06)),
        ),
      ),
      textTheme: GoogleFonts.interTextTheme(
        ThemeData.dark().textTheme,
      ).copyWith(
        headlineLarge: GoogleFonts.outfit(
          fontSize: 28,
          fontWeight: FontWeight.w700,
          color: Colors.white,
        ),
        headlineMedium: GoogleFonts.outfit(
          fontSize: 22,
          fontWeight: FontWeight.w600,
          color: Colors.white,
        ),
        headlineSmall: GoogleFonts.outfit(
          fontSize: 18,
          fontWeight: FontWeight.w600,
          color: Colors.white,
        ),
        titleLarge: GoogleFonts.outfit(
          fontSize: 16,
          fontWeight: FontWeight.w600,
          color: Colors.white,
        ),
        bodyLarge: GoogleFonts.inter(
          fontSize: 14,
          color: Colors.white.withOpacity(0.87),
        ),
        bodyMedium: GoogleFonts.inter(
          fontSize: 13,
          color: Colors.white.withOpacity(0.7),
        ),
        bodySmall: GoogleFonts.inter(
          fontSize: 11,
          color: Colors.white.withOpacity(0.5),
        ),
        labelLarge: GoogleFonts.inter(
          fontSize: 14,
          fontWeight: FontWeight.w600,
          color: Colors.white,
        ),
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: surface,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: GoogleFonts.outfit(
          fontSize: 20,
          fontWeight: FontWeight.w700,
          color: Colors.white,
        ),
        iconTheme: const IconThemeData(color: Colors.white70),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primary,
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: primary,
          side: BorderSide(color: primary.withOpacity(0.5)),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: Colors.white.withOpacity(0.05),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: Colors.white.withOpacity(0.1)),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: Colors.white.withOpacity(0.1)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: primary),
        ),
      ),
      dividerTheme: DividerThemeData(
        color: Colors.white.withOpacity(0.06),
        thickness: 1,
      ),
      snackBarTheme: SnackBarThemeData(
        backgroundColor: cardColor,
        contentTextStyle: GoogleFonts.inter(color: Colors.white),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }
}

class AppShell extends StatefulWidget {
  const AppShell({super.key});

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _selectedIndex = 0;

  final List<_NavItem> _navItems = const [
    _NavItem(icon: Icons.dashboard_rounded, label: 'Dashboard'),
    _NavItem(icon: Icons.play_circle_outline_rounded, label: 'Training'),
    _NavItem(icon: Icons.flight_takeoff_rounded, label: 'Airports'),
    _NavItem(icon: Icons.insights_rounded, label: 'Metrics'),
    _NavItem(icon: Icons.tune_rounded, label: 'Config'),
    _NavItem(icon: Icons.terminal_rounded, label: 'Logs'),
    _NavItem(icon: Icons.notifications_active_rounded, label: 'Alerts'),
  ];

  @override
  void initState() {
    super.initState();
    _loadInitialData();
  }

  Future<void> _loadInitialData() async {
    final metrics = context.read<MetricsProvider>();
    final config = context.read<ConfigProvider>();
    final training = context.read<TrainingProvider>();
    final ws = context.read<WebSocketService>();

    // Try loading data — will fail silently if API server is down
    await Future.wait([
      metrics.loadAll().catchError((_) {}),
      config.loadAll().catchError((_) {}),
      training.loadState().catchError((_) {}),
    ]);

    // Connect WebSocket for live updates
    ws.connectMetrics();
    ws.connectLogs();
  }

  @override
  Widget build(BuildContext context) {
    final isWide = MediaQuery.of(context).size.width > 800;

    return Scaffold(
      body: Row(
        children: [
          // Navigation rail for desktop
          if (isWide) _buildNavRail(),
          if (isWide)
            VerticalDivider(
              width: 1,
              color: Colors.white.withOpacity(0.06),
            ),
          // Main content
          Expanded(
            child: AnimatedSwitcher(
              duration: const Duration(milliseconds: 300),
              child: _buildScreen(_selectedIndex),
            ),
          ),
        ],
      ),
      // Bottom nav for mobile
      bottomNavigationBar: isWide
          ? null
          : NavigationBar(
              selectedIndex: _selectedIndex,
              onDestinationSelected: (i) => setState(() => _selectedIndex = i),
              backgroundColor: const Color(AppConfig.surfaceColor),
              indicatorColor:
                  const Color(AppConfig.primaryColor).withOpacity(0.2),
              destinations: _navItems
                  .map((n) => NavigationDestination(
                        icon: Icon(n.icon),
                        label: n.label,
                      ))
                  .toList(),
            ),
    );
  }

  Widget _buildNavRail() {
    return Container(
      width: 220,
      color: const Color(AppConfig.cardColor),
      child: Column(
        children: [
          const SizedBox(height: 24),
          // Logo
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(
              children: [
                Container(
                  width: 36,
                  height: 36,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [
                        Color(AppConfig.primaryColor),
                        Color(AppConfig.secondaryColor),
                      ],
                    ),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Icon(Icons.radar_rounded,
                      color: Colors.white, size: 20),
                ),
                const SizedBox(width: 12),
                Text(
                  'FedFOD',
                  style: GoogleFonts.outfit(
                    fontSize: 20,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 32),
          // Nav items
          ...List.generate(_navItems.length, (i) {
            final item = _navItems[i];
            final selected = i == _selectedIndex;
            return Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 2),
              child: Material(
                color: Colors.transparent,
                borderRadius: BorderRadius.circular(12),
                child: InkWell(
                  borderRadius: BorderRadius.circular(12),
                  onTap: () => setState(() => _selectedIndex = i),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    padding: const EdgeInsets.symmetric(
                        horizontal: 16, vertical: 12),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(12),
                      color: selected
                          ? const Color(AppConfig.primaryColor).withOpacity(0.15)
                          : Colors.transparent,
                      border: selected
                          ? Border.all(
                              color: const Color(AppConfig.primaryColor)
                                  .withOpacity(0.3))
                          : null,
                    ),
                    child: Row(
                      children: [
                        Icon(
                          item.icon,
                          size: 20,
                          color: selected
                              ? const Color(AppConfig.primaryColor)
                              : Colors.white54,
                        ),
                        const SizedBox(width: 12),
                        Text(
                          item.label,
                          style: GoogleFonts.inter(
                            fontSize: 13,
                            fontWeight:
                                selected ? FontWeight.w600 : FontWeight.w400,
                            color: selected
                                ? Colors.white
                                : Colors.white60,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            );
          }),
          const Spacer(),
          // Server status
          Padding(
            padding: const EdgeInsets.all(20),
            child: Consumer<TrainingProvider>(
              builder: (_, tp, __) {
                final active = tp.isTraining;
                return Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(12),
                    color: Colors.white.withOpacity(0.04),
                  ),
                  child: Row(
                    children: [
                      _PulsingDot(active: active),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          active ? 'Training Active' : 'Server Idle',
                          style: GoogleFonts.inter(
                            fontSize: 12,
                            color: Colors.white70,
                          ),
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildScreen(int index) {
    switch (index) {
      case 0:
        return const DashboardScreen(key: ValueKey('dashboard'));
      case 1:
        return const TrainingScreen(key: ValueKey('training'));
      case 2:
        return const AirportsScreen(key: ValueKey('airports'));
      case 3:
        return const MetricsScreen(key: ValueKey('metrics'));
      case 4:
        return const ConfigScreen(key: ValueKey('config'));
      case 5:
        return const LogsScreen(key: ValueKey('logs'));
      case 6:
        return const AlertsScreen(key: ValueKey('alerts'));
      default:
        return const DashboardScreen(key: ValueKey('dashboard'));
    }
  }
}

class _NavItem {
  final IconData icon;
  final String label;

  const _NavItem({required this.icon, required this.label});
}

class _PulsingDot extends StatefulWidget {
  final bool active;
  const _PulsingDot({required this.active});

  @override
  State<_PulsingDot> createState() => _PulsingDotState();
}

class _PulsingDotState extends State<_PulsingDot>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    );
    if (widget.active) _controller.repeat(reverse: true);
  }

  @override
  void didUpdateWidget(_PulsingDot old) {
    super.didUpdateWidget(old);
    if (widget.active && !_controller.isAnimating) {
      _controller.repeat(reverse: true);
    } else if (!widget.active && _controller.isAnimating) {
      _controller.stop();
      _controller.reset();
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (_, __) {
        final opacity = widget.active ? 0.4 + 0.6 * _controller.value : 1.0;
        return Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: (widget.active
                    ? const Color(AppConfig.successColor)
                    : Colors.white30)
                .withOpacity(opacity),
            boxShadow: widget.active
                ? [
                    BoxShadow(
                      color: const Color(AppConfig.successColor)
                          .withOpacity(0.4 * _controller.value),
                      blurRadius: 8,
                      spreadRadius: 2,
                    ),
                  ]
                : null,
          ),
        );
      },
    );
  }
}
