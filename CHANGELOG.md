# Changelog

## [2.1.1] - 2026-02-06

### Added
- WayyFin broker integration (`WayyFinBroker`, `WayyFinBrokerSync`)
  - Deploy strategies to wayyFin paper trading with one command
  - Monitor paper trading performance and equity curves
  - Promote strategies through discovery -> paper -> live stages
  - Access public leaderboard
- CLI: `wrtrade wayyfin` subcommand group
  - `deploy <strategy.py>` -- Deploy to paper trading
  - `start <strategy_id>` -- Resume paper trading
  - `stop <strategy_id>` -- Pause paper trading
  - `status [strategy_id]` -- Get strategy metrics
  - `leaderboard` -- View top strategies

### Fixed
- Added missing `python-dotenv` dependency to setup.py
- Fixed port reference in README (5175 -> 5173)

## [2.1.0] - 2026-01-18

### Changed
- Simplified API: `Portfolio` is now a data type, not a framework
- `wrt.backtest(signal, prices)` one-liner
- `wrt.validate(signal, prices)` for permutation testing
- `wrt.optimize(signal, prices)` for Kelly optimization

## [2.0.0] - 2026-01-10

### Added
- wrchart integration for financial charting
- Options trading support
- `BacktestChart`, `price_chart`, `line_chart`, `area_chart`, `histogram`, `bar_chart`, `indicator_panel`, `plot_backtest`

## [1.0.0] - 2025-12-01

### Added
- Initial release
- N-dimensional portfolio builder
- Kelly criterion optimization
- Permutation testing for strategy validation
- Local deployment with broker integration (Alpaca, Robinhood)
- CLI for strategy management
