"""
wrtrade.charts - Backtest visualization with wrchart

All chart output from wrtrade uses wrchart for consistent, interactive visualization.
Charts auto-render in Jupyter notebooks.

Chart Types:
- BacktestChart: Equity curve, drawdown, rolling Sharpe from backtest returns
- price_chart: Price line with optional overlays (indicators)
- indicator_panel: Multi-panel indicator visualization
- histogram: Distribution histogram
- bar_chart: Category bar chart for regime analysis
"""

import numpy as np
import polars as pl
from typing import Optional, Union, List, Dict, Any, Tuple


def _to_numpy(data) -> np.ndarray:
    """Convert various data types to numpy array."""
    if hasattr(data, 'to_numpy'):
        return data.to_numpy()
    elif hasattr(data, 'values'):
        return data.values
    elif isinstance(data, list):
        return np.array(data)
    return np.asarray(data)


def _to_timestamps(timestamps) -> list:
    """Convert timestamps to Unix seconds for wrchart."""
    ts = _to_numpy(timestamps)

    # If already numeric, use as-is
    if np.issubdtype(ts.dtype, np.number):
        return [int(t) for t in ts]

    # Convert datetime64 to Unix seconds
    if np.issubdtype(ts.dtype, np.datetime64):
        return [int(t.astype('datetime64[s]').astype(int)) for t in ts]

    # Try datetime objects
    try:
        return [int(t.timestamp()) for t in ts]
    except:
        return list(range(len(ts)))


def _get_wrchart():
    """Import wrchart."""
    try:
        import wrchart as wrc
        return wrc
    except ImportError:
        raise ImportError("wrchart required: pip install wrchart")


# Wayy brand colors (with semantic meanings)
COLORS = {
    "primary": "#000000",      # Black - main line color (prices, indicators)
    "secondary": "#888888",    # Gray 400 - secondary/benchmark lines
    "accent": "#E53935",       # Red - accent/drawdown/negative
    "grid": "#e0e0e0",         # Gray 200 - grid/reference lines
    "equity": "#22863a",       # Green - equity curves (positive performance)
    "positive": "#22863a",     # Green - positive values
    "negative": "#E53935",     # Red - negative values
}


# =============================================================================
# Price and Line Charts
# =============================================================================

def price_chart(
    prices,
    timestamps=None,
    overlays: Optional[Dict[str, Any]] = None,
    title: str = "Price",
    width: int = 900,
    height: int = 400,
):
    """
    Price line chart with optional overlays.

    Args:
        prices: Price array
        timestamps: Optional timestamps
        overlays: Dict of {name: values} for overlay lines
        title: Chart title
        width: Chart width
        height: Chart height

    Returns:
        wrchart.Chart (auto-renders in Jupyter)

    Example:
        price_chart(prices, timestamps, overlays={'KAMA': kama_values})
    """
    wrc = _get_wrchart()

    prices_arr = _to_numpy(prices)
    n = len(prices_arr)
    ts = _to_timestamps(timestamps) if timestamps is not None else list(range(n))

    df = pl.DataFrame({"time": ts, "price": prices_arr})

    chart = wrc.Chart(width=width, height=height, theme=wrc.WayyTheme, title=title)
    chart.add_line(df, time_col="time", value_col="price", color=COLORS["primary"])

    # Add overlays
    if overlays:
        overlay_colors = [COLORS["accent"], COLORS["secondary"], "#555555"]
        for i, (name, values) in enumerate(overlays.items()):
            overlay_arr = _to_numpy(values)
            df_overlay = pl.DataFrame({"time": ts, name: overlay_arr})
            chart.add_line(df_overlay, time_col="time", value_col=name,
                          color=overlay_colors[i % len(overlay_colors)])

    return chart


def line_chart(
    values,
    timestamps=None,
    title: str = "Line Chart",
    color: str = None,
    width: int = 900,
    height: int = 300,
    h_lines: Optional[List[Tuple[float, str]]] = None,
):
    """
    Simple line chart.

    Args:
        values: Y values
        timestamps: Optional timestamps
        title: Chart title
        color: Line color (defaults to brand primary)
        width: Chart width
        height: Chart height
        h_lines: List of (value, color) tuples for horizontal lines

    Returns:
        wrchart.Chart
    """
    wrc = _get_wrchart()

    values_arr = _to_numpy(values)
    n = len(values_arr)
    ts = _to_timestamps(timestamps) if timestamps is not None else list(range(n))

    df = pl.DataFrame({"time": ts, "value": values_arr})

    chart = wrc.Chart(width=width, height=height, theme=wrc.WayyTheme, title=title)
    chart.add_line(df, time_col="time", value_col="value", color=color or COLORS["primary"])

    if h_lines:
        for val, col in h_lines:
            chart.add_horizontal_line(val, color=col, line_style=2)

    return chart


def area_chart(
    values,
    timestamps=None,
    title: str = "Area Chart",
    color: str = None,
    width: int = 900,
    height: int = 300,
):
    """
    Area chart (filled line).

    Args:
        values: Y values
        timestamps: Optional timestamps
        title: Chart title
        color: Area color (defaults to brand primary)
        width: Chart width
        height: Chart height

    Returns:
        wrchart.Chart
    """
    wrc = _get_wrchart()

    values_arr = _to_numpy(values)
    n = len(values_arr)
    ts = _to_timestamps(timestamps) if timestamps is not None else list(range(n))

    df = pl.DataFrame({"time": ts, "value": values_arr})
    line_color = color or COLORS["primary"]

    chart = wrc.Chart(width=width, height=height, theme=wrc.WayyTheme, title=title)
    chart.add_area(df, time_col="time", value_col="value",
                   line_color=line_color,
                   top_color=f"{line_color}11",
                   bottom_color=f"{line_color}44")

    return chart


# =============================================================================
# Histogram Charts
# =============================================================================

def histogram(
    values,
    bins: int = 50,
    title: str = "Distribution",
    color: str = None,
    width: int = 900,
    height: int = 300,
    v_lines: Optional[List[Tuple[float, str, str]]] = None,
):
    """
    Histogram chart for distributions.

    Args:
        values: Data values
        bins: Number of bins
        title: Chart title
        color: Bar color (defaults to brand secondary)
        width: Chart width
        height: Chart height
        v_lines: List of (value, color, label) tuples for vertical reference lines

    Returns:
        wrchart.Chart
    """
    wrc = _get_wrchart()

    values_arr = _to_numpy(values)

    # Compute histogram
    counts, bin_edges = np.histogram(values_arr, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create dataframe with bin indices as time
    df = pl.DataFrame({
        "time": list(range(len(bin_centers))),
        "count": counts.astype(float)
    })

    chart = wrc.Chart(width=width, height=height, theme=wrc.WayyTheme, title=title)
    chart.add_histogram(df, time_col="time", value_col="count", color=color or COLORS["secondary"])

    # Add vertical lines as horizontal lines (since x-axis is bin index)
    # We need to convert the value to bin index
    if v_lines:
        for val, col, label in v_lines:
            # Find closest bin index
            bin_idx = np.argmin(np.abs(bin_centers - val))
            chart.add_horizontal_line(counts.max() * 0.9, color=col, line_style=2, label=label)

    return chart


# =============================================================================
# Bar Charts
# =============================================================================

def bar_chart(
    categories: List[str],
    values: List[float],
    title: str = "Bar Chart",
    width: int = 900,
    height: int = 300,
    color_positive: str = None,
    color_negative: str = None,
):
    """
    Category bar chart.

    Args:
        categories: Category labels
        values: Bar values
        title: Chart title
        width: Chart width
        height: Chart height
        color_positive: Color for positive values (defaults to brand primary)
        color_negative: Color for negative values (defaults to brand accent/red)

    Returns:
        wrchart.Chart
    """
    wrc = _get_wrchart()

    values_arr = np.array(values)
    pos_color = color_positive or COLORS["positive"]
    neg_color = color_negative or COLORS["negative"]

    # Create colors based on value sign
    colors = [pos_color if v >= 0 else neg_color for v in values_arr]

    df = pl.DataFrame({
        "time": list(range(len(categories))),
        "value": values_arr,
        "color": colors
    })

    chart = wrc.Chart(width=width, height=height, theme=wrc.WayyTheme, title=title)
    chart.add_histogram(df, time_col="time", value_col="value", color_col="color")
    chart.add_horizontal_line(0, color=COLORS["grid"], line_style=0)

    return chart


# =============================================================================
# Backtest Charts
# =============================================================================

class BacktestChart:
    """
    Backtest visualization using wrchart.

    Example:
        results = backtest_strategy(prices, signals)
        chart = BacktestChart(results['returns'], timestamps)
        chart.equity()      # Equity curve (starts at 0)
        chart.drawdown()    # Drawdown chart
        chart.rolling_sharpe()  # Rolling Sharpe
    """

    def __init__(
        self,
        returns,
        timestamps=None,
        benchmark_returns=None,
        name: str = "Strategy"
    ):
        """
        Initialize backtest chart.

        Args:
            returns: Strategy returns
            timestamps: Optional timestamps
            benchmark_returns: Optional benchmark returns
            name: Strategy name
        """
        self.returns = _to_numpy(returns)
        self.n = len(self.returns)
        self.name = name

        # Timestamps
        self.timestamps = _to_timestamps(timestamps) if timestamps is not None else list(range(self.n))

        # Benchmark
        self._benchmark_returns = None
        if benchmark_returns is not None:
            self._benchmark_returns = _to_numpy(benchmark_returns)

        # Compute metrics
        self._compute()

    def _compute(self):
        """Compute all metrics."""
        # Equity curve (cumsum, truly starts at 0)
        cumulative = np.cumsum(self.returns)
        self._equity = np.concatenate([[0], cumulative])

        # Drawdown (from equity that starts at 0)
        running_max = np.maximum.accumulate(self._equity)
        self._drawdown = self._equity - running_max
        self._max_drawdown = np.min(self._drawdown)

        # Total return
        self._total_return = self._equity[-1] if len(self._equity) > 0 else 0

        # Annualized (assume hourly)
        ann = 252 * 24
        self._ann_return = np.mean(self.returns) * ann
        self._ann_vol = np.std(self.returns) * np.sqrt(ann)
        self._sharpe = self._ann_return / self._ann_vol if self._ann_vol > 0 else 0

        # Benchmark (also starts at 0)
        if self._benchmark_returns is not None:
            bench_cumulative = np.cumsum(self._benchmark_returns)
            self._benchmark_equity = np.concatenate([[0], bench_cumulative])

    def equity(self, title: str = None, width: int = 900, height: int = 400):
        """
        Equity curve chart (starts at 0).

        Returns:
            wrchart.Chart
        """
        wrc = _get_wrchart()

        if title is None:
            title = f"{self.name} | Return: {self._total_return:.2%} | Sharpe: {self._sharpe:.2f}"

        # Extend timestamps to match equity length (prepend t0)
        if len(self.timestamps) > 0:
            t0 = self.timestamps[0] - 1 if isinstance(self.timestamps[0], (int, float)) else 0
            equity_timestamps = [t0] + list(self.timestamps)
        else:
            equity_timestamps = list(range(len(self._equity)))

        df = pl.DataFrame({"time": equity_timestamps, "equity": self._equity})

        chart = wrc.Chart(width=width, height=height, theme=wrc.WayyTheme, title=title, value_format="percent")
        chart.add_line(df, time_col="time", value_col="equity", color=COLORS["equity"])

        # Benchmark (uses same extended timestamps)
        if self._benchmark_returns is not None:
            df_b = pl.DataFrame({"time": equity_timestamps, "benchmark": self._benchmark_equity})
            chart.add_line(df_b, time_col="time", value_col="benchmark", color=COLORS["secondary"])

        chart.add_horizontal_line(0, color=COLORS["grid"], line_style=0)

        return chart

    def drawdown(self, title: str = None, width: int = 900, height: int = 300):
        """
        Drawdown chart.

        Returns:
            wrchart.Chart
        """
        wrc = _get_wrchart()

        if title is None:
            title = f"Drawdown | Max: {self._max_drawdown:.2%}"

        # Extend timestamps to match drawdown length (prepend t0)
        if len(self.timestamps) > 0:
            t0 = self.timestamps[0] - 1 if isinstance(self.timestamps[0], (int, float)) else 0
            dd_timestamps = [t0] + list(self.timestamps)
        else:
            dd_timestamps = list(range(len(self._drawdown)))

        df = pl.DataFrame({"time": dd_timestamps, "dd": self._drawdown})

        chart = wrc.Chart(width=width, height=height, theme=wrc.WayyTheme, title=title, value_format="percent")
        chart.add_area(df, time_col="time", value_col="dd",
                       line_color=COLORS["accent"],
                       top_color="rgba(229,57,53,0.1)",
                       bottom_color="rgba(229,57,53,0.4)")
        chart.add_horizontal_line(0, color=COLORS["grid"], line_style=0)

        return chart

    def sharpe(self, window: int = 252 * 24, title: str = None,
               width: int = 900, height: int = 300):
        """
        Rolling Sharpe ratio chart.

        Args:
            window: Rolling window
            title: Chart title

        Returns:
            wrchart.Chart
        """
        wrc = _get_wrchart()

        ann = 252 * 24
        actual_window = min(window, self.n // 2)

        rolling = np.full(self.n, np.nan)
        for i in range(actual_window, self.n):
            w = self.returns[i - actual_window:i]
            m = np.mean(w) * ann
            s = np.std(w) * np.sqrt(ann)
            if s > 0:
                rolling[i] = m / s

        # Filter valid
        mask = ~np.isnan(rolling)
        valid_ts = [self.timestamps[i] for i in range(self.n) if mask[i]]
        valid_sr = rolling[mask]

        if title is None:
            years = actual_window / ann
            title = f"Rolling Sharpe | {years:.1f}Y Window"

        df = pl.DataFrame({"time": valid_ts, "sharpe": valid_sr})

        chart = wrc.Chart(width=width, height=height, theme=wrc.WayyTheme, title=title)
        chart.add_line(df, time_col="time", value_col="sharpe", color=COLORS["primary"])
        chart.add_horizontal_line(0, color=COLORS["grid"], line_style=0)
        chart.add_horizontal_line(1.0, color=COLORS["secondary"], line_style=2)
        chart.add_horizontal_line(2.0, color=COLORS["accent"], line_style=2)

        return chart

    # Alias for compatibility
    rolling_sharpe = sharpe

    def dashboard(self):
        """Show all charts."""
        self.equity().show()
        self.drawdown().show()
        self.rolling_sharpe().show()


# =============================================================================
# Indicator Panel (Multi-series)
# =============================================================================

def indicator_panel(
    timestamps,
    panels: List[Dict[str, Any]],
    width: int = 900,
    panel_height: int = 200,
):
    """
    Multi-panel indicator chart.

    Args:
        timestamps: Shared timestamps for all panels
        panels: List of panel configs, each with:
            - title: Panel title
            - series: Dict of {name: (values, color, type)}
                      type can be 'line', 'area', or 'histogram'
                      color is optional (uses brand colors if None)
            - h_lines: Optional list of (value, color) for horizontal lines
        width: Chart width
        panel_height: Height per panel

    Returns:
        List of wrchart.Chart objects

    Example:
        panels = [
            {
                'title': 'Price vs KAMA',
                'series': {
                    'Price': (prices, '#000000', 'line'),
                    'KAMA': (kama, '#E53935', 'line'),
                }
            },
            {
                'title': 'Efficiency Ratio',
                'series': {'ER': (er, '#888888', 'line')},
                'h_lines': [(0.5, '#e0e0e0')]
            }
        ]
        charts = indicator_panel(timestamps, panels)
        for c in charts:
            c.show()
    """
    wrc = _get_wrchart()

    ts = _to_timestamps(timestamps)
    charts = []
    default_colors = [COLORS["primary"], COLORS["accent"], COLORS["secondary"]]

    for panel in panels:
        title = panel.get('title', '')
        series = panel.get('series', {})
        h_lines = panel.get('h_lines', [])

        chart = wrc.Chart(width=width, height=panel_height, theme=wrc.WayyTheme, title=title)

        for i, (name, config) in enumerate(series.items()):
            values, color, series_type = config
            color = color or default_colors[i % len(default_colors)]
            values_arr = _to_numpy(values)
            df = pl.DataFrame({"time": ts, name: values_arr})

            if series_type == 'line':
                chart.add_line(df, time_col="time", value_col=name, color=color)
            elif series_type == 'area':
                chart.add_area(df, time_col="time", value_col=name,
                              line_color=color,
                              top_color=f"{color}11",
                              bottom_color=f"{color}44")
            elif series_type == 'histogram':
                chart.add_histogram(df, time_col="time", value_col=name, color=color)

        for val, col in h_lines:
            chart.add_horizontal_line(val, color=col or COLORS["grid"], line_style=2)

        charts.append(chart)

    return charts


# =============================================================================
# Convenience exports
# =============================================================================

def plot_backtest(returns, timestamps=None, benchmark_returns=None, name: str = "Strategy"):
    """Quick backtest visualization."""
    chart = BacktestChart(returns, timestamps, benchmark_returns, name)
    chart.dashboard()
