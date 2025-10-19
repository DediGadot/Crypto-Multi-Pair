#!/usr/bin/env python3
"""
Daily Trading Recommendations - Generate Today's Trade Signals

**Purpose**: Analyze current market conditions and generate actionable trading
recommendations for today using a specified strategy.

**Key Features**:
- Single-pair or multi-pair strategy support
- Real-time data fetching
- Signal generation with confidence levels
- Risk management parameters
- Entry/exit recommendations
- Position sizing suggestions
- Export to multiple formats (console, JSON, CSV)

**Third-party packages**:
- typer: https://typer.tiangolo.com/
- pandas: https://pandas.pydata.org/docs/
- loguru: https://loguru.readthedocs.io/en/stable/
- rich: https://rich.readthedocs.io/en/stable/

**Sample Input**:
```bash
python daily_recommendations.py --strategy SMA_Crossover
python daily_recommendations.py --strategy PortfolioRebalancer --multi-pair
python daily_recommendations.py --strategy RSI_MeanReversion --symbols BTC/USDT,ETH/USDT
python daily_recommendations.py --strategy MACD_Momentum --export json --risk-level medium
```

**Expected Output**:
- Console: Formatted table of trade recommendations
- File: daily_recommendations_YYYYMMDD.json (if --export json)
- File: daily_recommendations_YYYYMMDD.csv (if --export csv)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json

# Add src directory to Python path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import typer
import pandas as pd
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from crypto_trader.data.fetchers import BinanceDataFetcher
from crypto_trader.strategies import get_registry
import crypto_trader.strategies.library  # Register all strategies
from crypto_trader.features.factory import augment_with_features, DEFAULT_JOIN_CONFIG
from crypto_trader.data.alt.onchain_ingestor import ingest_onchain
from crypto_trader.data.alt.sentiment_ingestor import ingest_sentiment

# Initialize
app = typer.Typer(help="Daily trading recommendations generator")
console = Console()


@dataclass
class TradeRecommendation:
    """A single trade recommendation."""
    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-100
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size_pct: Optional[float] = None  # % of portfolio
    reasoning: str = ""
    risk_reward_ratio: Optional[float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class DailyReport:
    """Complete daily recommendation report."""
    date: str
    strategy_name: str
    recommendations: List[TradeRecommendation]
    market_conditions: Dict[str, Any]
    risk_level: str
    generated_at: str


class DailyRecommendationEngine:
    """
    Generate daily trading recommendations based on selected strategy.

    This engine fetches current market data, runs the specified strategy,
    and produces actionable trade recommendations with risk parameters.
    """

    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        timeframe: str = "1h",
        lookback_days: int = 30,
        risk_level: str = "medium",
        multi_pair: bool = False
    ):
        """
        Initialize the recommendation engine.

        Args:
            strategy_name: Name of strategy to use (e.g., 'SMA_Crossover')
            symbols: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
            timeframe: Candle timeframe (default: '1h')
            lookback_days: Days of historical data for context (default: 30)
            risk_level: Risk tolerance (low, medium, high)
            multi_pair: Whether strategy is multi-pair
        """
        logger.info("=" * 80)
        logger.info("🎯 DAILY RECOMMENDATION ENGINE INITIALIZATION")
        logger.info("=" * 80)

        self.strategy_name = strategy_name
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback_days = lookback_days
        self.risk_level = risk_level
        self.multi_pair = multi_pair

        logger.info(f"Strategy: {strategy_name}")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Risk Level: {risk_level}")

        # Initialize components
        self.fetcher = BinanceDataFetcher()
        self.registry = get_registry()

        # Risk parameters
        self.risk_params = self._get_risk_parameters(risk_level)
        logger.info(f"Risk Parameters: {self.risk_params}")

    def _get_risk_parameters(self, risk_level: str) -> Dict[str, float]:
        """Get risk parameters based on risk level."""
        risk_configs = {
            'low': {
                'max_position_size': 10.0,  # % of portfolio
                'stop_loss_pct': 2.0,        # % from entry
                'min_risk_reward': 2.0,      # Minimum R:R ratio
                'min_confidence': 70.0       # Minimum signal confidence
            },
            'medium': {
                'max_position_size': 20.0,
                'stop_loss_pct': 3.0,
                'min_risk_reward': 1.5,
                'min_confidence': 60.0
            },
            'high': {
                'max_position_size': 30.0,
                'stop_loss_pct': 5.0,
                'min_risk_reward': 1.2,
                'min_confidence': 50.0
            }
        }
        return risk_configs.get(risk_level.lower(), risk_configs['medium'])

    def fetch_current_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch current market data for a symbol.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')

        Returns:
            DataFrame with OHLCV data and features
        """
        logger.info(f"📥 Fetching current data for {symbol}...")

        # Fetch recent data for context
        limit = self._calculate_limit(self.timeframe, self.lookback_days)
        data = self.fetcher.get_ohlcv(symbol, self.timeframe, limit=limit)

        if data is None or len(data) == 0:
            raise ValueError(f"No data available for {symbol}")

        logger.info(f"✓ Fetched {len(data)} candles")

        # Prepare features
        self._prepare_features(symbol)

        # Augment with alternative data
        try:
            data = augment_with_features(
                market_df=data,
                symbol=symbol,
                timeframe=self.timeframe,
                config=DEFAULT_JOIN_CONFIG,
            )
            logger.info(f"✓ Augmented with {len(data.columns)} total columns")
        except Exception as e:
            logger.warning(f"Feature augmentation failed: {e}")

        return data

    def _calculate_limit(self, timeframe: str, days: int) -> int:
        """Calculate number of candles needed for given days."""
        timeframe_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440
        }
        minutes = timeframe_minutes.get(timeframe, 60)
        return int((days * 24 * 60) / minutes)

    def _prepare_features(self, symbol: str):
        """Prepare alternative data features."""
        try:
            ingest_onchain(symbol, self.timeframe)
            ingest_sentiment(symbol, self.timeframe)
        except Exception as e:
            logger.debug(f"Feature preparation: {e}")

    def generate_recommendations(self) -> List[TradeRecommendation]:
        """
        Generate trade recommendations for all symbols.

        Returns:
            List of TradeRecommendation objects
        """
        logger.info("=" * 80)
        logger.info("🔮 GENERATING RECOMMENDATIONS")
        logger.info("=" * 80)

        recommendations = []

        for symbol in self.symbols:
            try:
                logger.info(f"\n📊 Analyzing {symbol}...")
                rec = self._generate_symbol_recommendation(symbol)
                if rec:
                    recommendations.append(rec)
            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                continue

        logger.success(f"\n✅ Generated {len(recommendations)} recommendations")
        return recommendations

    def _generate_symbol_recommendation(self, symbol: str) -> Optional[TradeRecommendation]:
        """Generate recommendation for a single symbol."""
        # Fetch data
        data = self.fetch_current_data(symbol)

        # Load and initialize strategy
        strategy_class = self.registry.get_strategy(self.strategy_name)
        strategy = strategy_class()

        if hasattr(strategy, 'initialize'):
            strategy.initialize({})

        # Add required indicators
        data = self._add_required_indicators(strategy, data)

        # Generate signals
        logger.info(f"🎲 Generating signals...")
        signals = strategy.generate_signals(data)

        if signals is None or len(signals) == 0:
            logger.warning(f"No signals generated for {symbol}")
            return None

        # Get latest signal
        latest_signal = signals.iloc[-1]

        # Analyze signal
        action = self._interpret_signal(latest_signal)

        if action == 'HOLD':
            logger.info(f"💤 Recommendation: HOLD (no action needed)")
            return TradeRecommendation(
                symbol=symbol,
                action='HOLD',
                confidence=50.0,
                reasoning="No clear signal at current market conditions"
            )

        # Get current price
        current_price = data['close'].iloc[-1]

        # Calculate confidence
        confidence = self._calculate_confidence(latest_signal, data)

        # Skip if confidence too low
        if confidence < self.risk_params['min_confidence']:
            logger.info(f"⚠️  Signal confidence too low: {confidence:.1f}%")
            return None

        # Calculate entry, target, stop loss
        entry_price = current_price
        stop_loss, target_price = self._calculate_risk_reward(
            action, current_price, data
        )

        # Calculate position size
        position_size = self._calculate_position_size(
            action, current_price, stop_loss, confidence
        )

        # Calculate risk/reward ratio
        risk_reward = None
        if stop_loss and target_price:
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            risk_reward = reward / risk if risk > 0 else None

        # Build reasoning
        reasoning = self._build_reasoning(latest_signal, data, strategy)

        logger.success(f"✓ {action} signal - Confidence: {confidence:.1f}%")

        return TradeRecommendation(
            symbol=symbol,
            action=action,
            confidence=confidence,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            position_size_pct=position_size,
            reasoning=reasoning,
            risk_reward_ratio=risk_reward
        )

    def _add_required_indicators(self, strategy: Any, data: pd.DataFrame) -> pd.DataFrame:
        """Add indicators required by the strategy."""
        import pandas_ta as ta

        # Common indicators
        if 'sma_50' not in data.columns:
            data['sma_50'] = ta.sma(data['close'], length=50)
        if 'sma_200' not in data.columns:
            data['sma_200'] = ta.sma(data['close'], length=200)
        if 'rsi' not in data.columns:
            data['rsi'] = ta.rsi(data['close'], length=14)
        if 'atr' not in data.columns:
            atr_result = ta.atr(data['high'], data['low'], data['close'], length=14)
            if isinstance(atr_result, pd.DataFrame):
                data['atr'] = atr_result.iloc[:, 0]
            else:
                data['atr'] = atr_result

        return data

    def _interpret_signal(self, signal: pd.Series) -> str:
        """Interpret strategy signal into action."""
        signal_value = signal.get('signal', 0)

        if signal_value == 1:
            return 'BUY'
        elif signal_value == -1:
            return 'SELL'
        else:
            return 'HOLD'

    def _calculate_confidence(self, signal: pd.Series, data: pd.DataFrame) -> float:
        """Calculate confidence level for the signal."""
        confidence = 50.0  # Base confidence

        # Increase confidence based on trend alignment
        if 'sma_50' in data.columns and 'sma_200' in data.columns:
            sma_50 = data['sma_50'].iloc[-1]
            sma_200 = data['sma_200'].iloc[-1]

            signal_value = signal.get('signal', 0)
            if signal_value == 1 and sma_50 > sma_200:
                confidence += 15  # Bullish trend
            elif signal_value == -1 and sma_50 < sma_200:
                confidence += 15  # Bearish trend

        # Increase confidence based on RSI
        if 'rsi' in data.columns:
            rsi = data['rsi'].iloc[-1]
            signal_value = signal.get('signal', 0)

            if signal_value == 1 and rsi < 50:
                confidence += 10  # Not overbought
            elif signal_value == -1 and rsi > 50:
                confidence += 10  # Not oversold

        # Increase confidence if volume is high
        if 'volume' in data.columns:
            avg_volume = data['volume'].rolling(20).mean().iloc[-1]
            current_volume = data['volume'].iloc[-1]

            if current_volume > avg_volume * 1.2:
                confidence += 10  # High volume confirmation

        return min(confidence, 95.0)  # Cap at 95%

    def _calculate_risk_reward(
        self, action: str, entry_price: float, data: pd.DataFrame
    ) -> tuple:
        """Calculate stop loss and target price."""
        atr = data['atr'].iloc[-1] if 'atr' in data.columns else entry_price * 0.02

        if action == 'BUY':
            stop_loss = entry_price - (atr * 2)  # 2 ATR stop
            target_price = entry_price + (atr * self.risk_params['min_risk_reward'] * 2)
        else:  # SELL
            stop_loss = entry_price + (atr * 2)
            target_price = entry_price - (atr * self.risk_params['min_risk_reward'] * 2)

        return stop_loss, target_price

    def _calculate_position_size(
        self, action: str, entry_price: float, stop_loss: float, confidence: float
    ) -> float:
        """Calculate position size as percentage of portfolio."""
        # Base position size on confidence and risk level
        base_size = self.risk_params['max_position_size']

        # Adjust for confidence
        confidence_multiplier = confidence / 100.0

        position_size = base_size * confidence_multiplier

        # Cap at max
        return min(position_size, self.risk_params['max_position_size'])

    def _build_reasoning(
        self, signal: pd.Series, data: pd.DataFrame, strategy: Any
    ) -> str:
        """Build human-readable reasoning for the recommendation."""
        parts = []

        # Strategy name
        parts.append(f"Strategy: {self.strategy_name}")

        # Signal strength
        signal_value = signal.get('signal', 0)
        if signal_value == 1:
            parts.append("Bullish signal detected")
        elif signal_value == -1:
            parts.append("Bearish signal detected")

        # Trend context
        if 'sma_50' in data.columns and 'sma_200' in data.columns:
            sma_50 = data['sma_50'].iloc[-1]
            sma_200 = data['sma_200'].iloc[-1]

            if sma_50 > sma_200:
                parts.append("Price above long-term trend (bullish)")
            else:
                parts.append("Price below long-term trend (bearish)")

        # RSI context
        if 'rsi' in data.columns:
            rsi = data['rsi'].iloc[-1]
            if rsi > 70:
                parts.append(f"RSI overbought ({rsi:.1f})")
            elif rsi < 30:
                parts.append(f"RSI oversold ({rsi:.1f})")
            else:
                parts.append(f"RSI neutral ({rsi:.1f})")

        return "; ".join(parts)

    def get_market_conditions(self) -> Dict[str, Any]:
        """Get current market conditions summary."""
        conditions = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'symbols_analyzed': len(self.symbols),
            'timeframe': self.timeframe,
            'lookback_days': self.lookback_days,
        }

        # Add market sentiment if available
        try:
            data = self.fetch_current_data(self.symbols[0])
            conditions['market_trend'] = 'bullish' if data['close'].iloc[-1] > data['close'].iloc[-20] else 'bearish'
        except Exception:
            pass

        return conditions


def display_recommendations(recommendations: List[TradeRecommendation], strategy_name: str):
    """Display recommendations in a rich formatted table."""
    console.print("\n")
    console.print(Panel.fit(
        f"[bold cyan]Daily Trading Recommendations[/bold cyan]\n"
        f"Strategy: [yellow]{strategy_name}[/yellow]\n"
        f"Generated: [green]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/green]",
        border_style="cyan"
    ))

    if not recommendations:
        console.print("\n[yellow]⚠️  No recommendations generated (no clear signals)[/yellow]\n")
        return

    # Create table
    table = Table(show_header=True, header_style="bold magenta", border_style="cyan")
    table.add_column("Symbol", style="cyan", width=12)
    table.add_column("Action", justify="center", width=8)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Entry", justify="right", width=12)
    table.add_column("Target", justify="right", width=12)
    table.add_column("Stop Loss", justify="right", width=12)
    table.add_column("Size %", justify="right", width=8)
    table.add_column("R:R", justify="right", width=6)

    for rec in recommendations:
        # Color code action
        if rec.action == 'BUY':
            action_str = "[green]BUY[/green]"
        elif rec.action == 'SELL':
            action_str = "[red]SELL[/red]"
        else:
            action_str = "[yellow]HOLD[/yellow]"

        # Color code confidence
        if rec.confidence >= 70:
            conf_str = f"[green]{rec.confidence:.1f}%[/green]"
        elif rec.confidence >= 60:
            conf_str = f"[yellow]{rec.confidence:.1f}%[/yellow]"
        else:
            conf_str = f"[red]{rec.confidence:.1f}%[/red]"

        table.add_row(
            rec.symbol,
            action_str,
            conf_str,
            f"${rec.entry_price:.2f}" if rec.entry_price else "-",
            f"${rec.target_price:.2f}" if rec.target_price else "-",
            f"${rec.stop_loss:.2f}" if rec.stop_loss else "-",
            f"{rec.position_size_pct:.1f}%" if rec.position_size_pct else "-",
            f"{rec.risk_reward_ratio:.2f}" if rec.risk_reward_ratio else "-"
        )

    console.print(table)

    # Print detailed reasoning
    console.print("\n[bold]📋 Detailed Analysis:[/bold]\n")
    for rec in recommendations:
        if rec.action != 'HOLD':
            console.print(f"[cyan]{rec.symbol}[/cyan]: {rec.reasoning}")

    console.print()


@app.command()
def recommend(
    strategy: str = typer.Option(..., "--strategy", "-s", help="Strategy name to use"),
    symbols: str = typer.Option("BTC/USDT", "--symbols", help="Comma-separated symbols (e.g., BTC/USDT,ETH/USDT)"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Timeframe (1h, 4h, 1d)"),
    multi_pair: bool = typer.Option(False, "--multi-pair", help="Use multi-pair strategy"),
    risk_level: str = typer.Option("medium", "--risk-level", "-r", help="Risk level: low, medium, high"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="Export format: json, csv"),
    lookback: int = typer.Option(30, "--lookback", "-l", help="Days of historical data"),
):
    """
    Generate daily trading recommendations using specified strategy.

    Examples:
        daily_recommendations.py -s SMA_Crossover
        daily_recommendations.py -s RSI_MeanReversion --symbols BTC/USDT,ETH/USDT
        daily_recommendations.py -s MACD_Momentum --risk-level high --export json
    """
    # Parse symbols
    symbol_list = [s.strip() for s in symbols.split(',')]

    # Initialize engine
    engine = DailyRecommendationEngine(
        strategy_name=strategy,
        symbols=symbol_list,
        timeframe=timeframe,
        lookback_days=lookback,
        risk_level=risk_level,
        multi_pair=multi_pair
    )

    # Generate recommendations
    recommendations = engine.generate_recommendations()

    # Display
    display_recommendations(recommendations, strategy)

    # Export if requested
    if export and recommendations:
        export_recommendations(recommendations, strategy, export, engine.get_market_conditions())

    # Summary
    actionable = [r for r in recommendations if r.action != 'HOLD']
    if actionable:
        console.print(f"[bold green]✅ {len(actionable)} actionable trade(s) recommended[/bold green]\n")
    else:
        console.print(f"[bold yellow]⏸️  No actionable trades at this time[/bold yellow]\n")


def export_recommendations(
    recommendations: List[TradeRecommendation],
    strategy_name: str,
    format: str,
    market_conditions: Dict[str, Any]
):
    """Export recommendations to file."""
    timestamp = datetime.now().strftime("%Y%m%d")

    if format == 'json':
        filename = f"daily_recommendations_{timestamp}.json"
        report = DailyReport(
            date=datetime.now().strftime("%Y-%m-%d"),
            strategy_name=strategy_name,
            recommendations=recommendations,
            market_conditions=market_conditions,
            risk_level="medium",
            generated_at=datetime.now().isoformat()
        )

        with open(filename, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)

        console.print(f"[green]✓ Exported to {filename}[/green]")

    elif format == 'csv':
        filename = f"daily_recommendations_{timestamp}.csv"
        df = pd.DataFrame([asdict(r) for r in recommendations])
        df.to_csv(filename, index=False)
        console.print(f"[green]✓ Exported to {filename}[/green]")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "daily_recommendations.log",
        rotation="10 MB",
        level="DEBUG"
    )

    app()
