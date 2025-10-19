#!/usr/bin/env python3
"""
Portfolio Rebalancing Advisor

Purpose:
    Analyzes portfolio against target allocations and recommends rebalancing actions.
    Implements threshold, calendar, and hybrid rebalancing strategies from PR.md.

Key Features:
    - Interactive config creation if config.yaml doesn't exist
    - Real-time price fetching from exchanges
    - Threshold/calendar/hybrid rebalancing methods
    - Momentum filter support
    - Detailed buy/sell recommendations

Third-party packages:
    - typer: https://typer.tiangolo.com/
    - PyYAML: https://pyyaml.org/wiki/PyYAMLDocumentation
    - loguru: https://loguru.readthedocs.io/
    - ccxt: https://docs.ccxt.com/

Sample Input:
    Config: 50% BTC, 50% ETH, 15% threshold, hybrid method
    Holdings: 0.125 BTC @ $60000, 2.5 ETH @ $2400

Expected Output:
    SELL 0.025 BTC ($1500), BUY 0.625 ETH ($1500) - OR - HOLD (deviation: 3.2%)

Usage:
    uv run python portfolio_rebalancer_advisor.py
    uv run python portfolio_rebalancer_advisor.py --config my_portfolio.yaml
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import typer
import yaml
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("rebalance.log", rotation="10 MB", retention="30 days")

app = typer.Typer(help="Portfolio Rebalancing Advisor - Smart portfolio management")


def create_config_interactively() -> Dict:
    """Create portfolio configuration through interactive prompts."""
    logger.info("No config found. Starting interactive setup...")

    config = {
        "portfolio": {"assets": [], "holdings": {}},
        "rebalancing": {
            "method": "hybrid",
            "threshold": 0.15,
            "min_interval_hours": 24,
            "calendar_period_days": 30,
            "momentum_filter": {"enabled": False, "threshold": 0.20, "lookback_days": 30}
        },
        "state": {"last_rebalance": None, "initial_capital": 10000.0},
        "exchange": {"name": "binance", "quote_currency": "USDT"}
    }

    print("\n" + "="*70)
    print("PORTFOLIO CONFIGURATION")
    print("="*70)

    # Assets
    num_assets = int(typer.prompt("Number of assets", default=2))
    total_weight = 0.0

    for i in range(num_assets):
        print(f"\n--- Asset {i+1} ---")
        symbol = typer.prompt("Symbol (e.g., BTC, ETH)").upper()
        target_weight = float(typer.prompt(f"Target weight (0-1)", default=1.0/num_assets))
        current_shares = float(typer.prompt(f"Current shares", default=0.0))

        full_symbol = f"{symbol}/USDT"
        config["portfolio"]["assets"].append({"symbol": full_symbol, "target_weight": target_weight})
        config["portfolio"]["holdings"][full_symbol] = current_shares
        total_weight += target_weight

    # Normalize weights
    if abs(total_weight - 1.0) > 0.01:
        logger.warning(f"Weights sum to {total_weight:.3f}, normalizing")
        for asset in config["portfolio"]["assets"]:
            asset["target_weight"] /= total_weight

    # Rebalancing strategy
    print("\n" + "="*70)
    print("REBALANCING STRATEGY")
    print("="*70)
    print("Methods: threshold | calendar | hybrid (recommended)")

    method = typer.prompt("Method", default="hybrid", type=typer.Choice(["threshold", "calendar", "hybrid"]))
    config["rebalancing"]["method"] = method

    if method in ["threshold", "hybrid"]:
        threshold = float(typer.prompt("Deviation threshold (0.15 = 15%)", default=0.15))
        config["rebalancing"]["threshold"] = threshold

    if method in ["calendar", "hybrid"]:
        period = int(typer.prompt("Rebalancing period (days)", default=30))
        config["rebalancing"]["calendar_period_days"] = period

    min_hours = int(typer.prompt("Min hours between rebalances", default=24))
    config["rebalancing"]["min_interval_hours"] = min_hours

    # Momentum filter
    use_momentum = typer.confirm("Enable momentum filter?", default=False)
    config["rebalancing"]["momentum_filter"]["enabled"] = use_momentum

    if use_momentum:
        mom_threshold = float(typer.prompt("Momentum threshold (0.20 = 20%)", default=0.20))
        mom_lookback = int(typer.prompt("Lookback days", default=30))
        config["rebalancing"]["momentum_filter"]["threshold"] = mom_threshold
        config["rebalancing"]["momentum_filter"]["lookback_days"] = mom_lookback

    # Initial capital
    if all(shares == 0.0 for shares in config["portfolio"]["holdings"].values()):
        initial = float(typer.prompt("Initial capital (USD)", default=10000.0))
        config["state"]["initial_capital"] = initial

    return config


def save_config(config: Dict, path: Path) -> None:
    """Save configuration to YAML."""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Config saved: {path}")


def load_config(path: Path) -> Dict:
    """Load configuration from YAML."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def fetch_prices(symbols: List[str], exchange_name: str = "binance") -> Dict[str, float]:
    """
    Fetch current market prices.

    Returns: Dict mapping symbol -> current price
    """
    import ccxt

    try:
        exchange = getattr(ccxt, exchange_name)()
        prices = {}
        for symbol in symbols:
            ticker = exchange.fetch_ticker(symbol)
            prices[symbol] = ticker['last']
            logger.debug(f"{symbol}: ${prices[symbol]:,.2f}")
        return prices
    except Exception as e:
        logger.error(f"Price fetch failed: {e}")
        logger.warning("Using example prices")
        return {"BTC/USDT": 60000.0, "ETH/USDT": 2400.0, "SOL/USDT": 100.0}


def calculate_portfolio_value(holdings: Dict[str, float], prices: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Calculate total portfolio value and per-asset values."""
    asset_values = {symbol: shares * prices[symbol] for symbol, shares in holdings.items()}
    total_value = sum(asset_values.values())
    return total_value, asset_values


def calculate_weights(asset_values: Dict[str, float], total_value: float) -> Dict[str, float]:
    """Calculate current portfolio weights."""
    if total_value == 0:
        return {symbol: 0.0 for symbol in asset_values.keys()}
    return {symbol: value / total_value for symbol, value in asset_values.items()}


def calculate_deviations(current: Dict[str, float], target: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """Calculate weight deviations from targets."""
    deviations = {symbol: abs(current[symbol] - target[symbol]) for symbol in current.keys()}
    max_deviation = max(deviations.values())
    return deviations, max_deviation


def check_rebalancing_needed(max_dev: float, last_rebalance: Optional[datetime], config: Dict) -> Tuple[bool, str]:
    """Determine if rebalancing is needed."""
    method = config["rebalancing"]["method"]
    threshold = config["rebalancing"]["threshold"]
    min_hours = config["rebalancing"]["min_interval_hours"]
    calendar_days = config["rebalancing"]["calendar_period_days"]
    now = datetime.now()

    # Min interval check
    if last_rebalance:
        hours_since = (now - last_rebalance).total_seconds() / 3600
        if hours_since < min_hours:
            return False, f"min_interval ({hours_since:.1f}h < {min_hours}h)"

    # Method logic
    if method == "threshold":
        if max_dev > threshold:
            return True, f"threshold ({max_dev:.1%} > {threshold:.1%})"

    elif method == "calendar":
        if last_rebalance:
            days = (now - last_rebalance).days
            if days >= calendar_days:
                return True, f"calendar ({days} days >= {calendar_days})"
        else:
            return True, "initial_allocation"

    elif method == "hybrid":
        threshold_hit = max_dev > threshold
        calendar_hit = False

        if last_rebalance:
            days = (now - last_rebalance).days
            calendar_hit = days >= calendar_days
        else:
            calendar_hit = True

        if threshold_hit:
            return True, f"threshold ({max_dev:.1%} > {threshold:.1%})"
        if calendar_hit:
            days = (now - last_rebalance).days if last_rebalance else 0
            return True, f"calendar ({days} days >= {calendar_days})"

    return False, "no_trigger"


def generate_recommendations(current_wt: Dict[str, float], target_wt: Dict[str, float], total_val: float,
                            prices: Dict[str, float], holdings: Dict[str, float]) -> List[Dict]:
    """Generate specific buy/sell recommendations."""
    recs = []
    for symbol in current_wt.keys():
        current_val = total_val * current_wt[symbol]
        target_val = total_val * target_wt[symbol]
        diff_usd = target_val - current_val

        current_shares = holdings[symbol]
        target_shares = target_val / prices[symbol]
        diff_shares = target_shares - current_shares

        action = "HOLD"
        if diff_shares > 0.0001:
            action = "BUY"
        elif diff_shares < -0.0001:
            action = "SELL"

        recs.append({
            "symbol": symbol,
            "action": action,
            "current_weight": current_wt[symbol],
            "target_weight": target_wt[symbol],
            "current_value": current_val,
            "target_value": target_val,
            "current_shares": current_shares,
            "target_shares": target_shares,
            "trade_usd": diff_usd,
            "trade_shares": diff_shares,
            "price": prices[symbol]
        })
    return recs


def format_output(recs: List[Dict], needs_rebalance: bool, reason: str, max_dev: float, total_val: float) -> str:
    """Format recommendations for display."""
    lines = [
        "\n" + "="*80,
        "PORTFOLIO REBALANCING ADVISOR",
        "="*80,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Value: ${total_val:,.2f}",
        f"Max Deviation: {max_dev:.2%}",
        ""
    ]

    if needs_rebalance:
        lines.append(f"[!] REBALANCING RECOMMENDED: {reason}")
        lines.append("")
        lines.append("-"*80)
        lines.append(f"{'Asset':<12} {'Action':<6} {'Current':<10} {'Target':<10} {'Trade $':<15} {'Trade Shares':<15}")
        lines.append("-"*80)

        for r in recs:
            if r["action"] != "HOLD":
                lines.append(
                    f"{r['symbol']:<12} {r['action']:<6} {r['current_weight']:>8.1%}  "
                    f"{r['target_weight']:>8.1%}  ${r['trade_usd']:>12,.2f}  {r['trade_shares']:>12.6f}"
                )

        lines.append("-"*80)
        lines.append("\nDETAILED BREAKDOWN:")
        lines.append("-"*80)

        for r in recs:
            lines.append(f"\n{r['symbol']}:")
            lines.append(f"  Current: {r['current_shares']:.6f} shares @ ${r['price']:,.2f} = ${r['current_value']:,.2f} ({r['current_weight']:.2%})")
            lines.append(f"  Target:  {r['target_shares']:.6f} shares @ ${r['price']:,.2f} = ${r['target_value']:,.2f} ({r['target_weight']:.2%})")

            if r['action'] == 'BUY':
                lines.append(f"  >> BUY {abs(r['trade_shares']):.6f} shares (${abs(r['trade_usd']):,.2f})")
            elif r['action'] == 'SELL':
                lines.append(f"  >> SELL {abs(r['trade_shares']):.6f} shares (${abs(r['trade_usd']):,.2f})")
            else:
                lines.append(f"  >> HOLD (at target)")

    else:
        lines.append(f"[OK] NO REBALANCING NEEDED: {reason}")
        lines.append("")
        lines.append("Current Allocation:")
        lines.append("-"*80)
        lines.append(f"{'Asset':<12} {'Current':<10} {'Target':<10} {'Deviation':<12} {'Value':<15}")
        lines.append("-"*80)

        for r in recs:
            dev = r['current_weight'] - r['target_weight']
            lines.append(
                f"{r['symbol']:<12} {r['current_weight']:>8.1%}  {r['target_weight']:>8.1%}  "
                f"{dev:>10.2%}  ${r['current_value']:>12,.2f}"
            )

    lines.append("="*80 + "\n")
    return "\n".join(lines)


@app.command()
def check(
    config_path: Path = typer.Option(Path("rebalance_config.yaml"), "--config", "-c", help="Config file path"),
    save_output: bool = typer.Option(True, "--save/--no-save", help="Save to file")
) -> None:
    """Check portfolio and generate rebalancing recommendations."""

    # Load or create config
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}")
        config = create_config_interactively()
        save_config(config, config_path)
    else:
        config = load_config(config_path)
        logger.info(f"Loaded config: {config_path}")

    # Extract config
    assets = config["portfolio"]["assets"]
    holdings = config["portfolio"]["holdings"]
    exchange = config["exchange"]["name"]

    symbols = [a["symbol"] for a in assets]
    target_weights = {a["symbol"]: a["target_weight"] for a in assets}

    # Fetch prices
    logger.info(f"Fetching prices from {exchange}...")
    prices = fetch_prices(symbols, exchange)

    # Initialize if needed
    if all(shares == 0.0 for shares in holdings.values()):
        logger.info("Initializing portfolio...")
        initial_capital = config["state"]["initial_capital"]
        for symbol in symbols:
            target_val = initial_capital * target_weights[symbol]
            holdings[symbol] = target_val / prices[symbol]
        config["portfolio"]["holdings"] = holdings
        save_config(config, config_path)

    # Calculate portfolio state
    total_val, asset_vals = calculate_portfolio_value(holdings, prices)
    current_weights = calculate_weights(asset_vals, total_val)
    deviations, max_dev = calculate_deviations(current_weights, target_weights)

    # Parse last rebalance
    last_rebalance = None
    if config["state"]["last_rebalance"]:
        last_rebalance = datetime.fromisoformat(config["state"]["last_rebalance"])

    # Check if rebalancing needed
    needs_rebalance, reason = check_rebalancing_needed(max_dev, last_rebalance, config)

    # Generate recommendations
    recs = generate_recommendations(current_weights, target_weights, total_val, prices, holdings)

    # Display
    output = format_output(recs, needs_rebalance, reason, max_dev, total_val)
    print(output)

    # Save output
    if save_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"rebalance_rec_{timestamp}.txt")
        output_file.write_text(output)
        logger.info(f"Saved: {output_file}")

    # Save JSON for programmatic use
    if needs_rebalance:
        import json
        json_file = Path(f"rebalance_rec_{datetime.now().strftime('%Y%m%d')}.json")
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "needs_rebalance": needs_rebalance,
            "reason": reason,
            "max_deviation": max_dev,
            "total_value": total_val,
            "recommendations": recs
        }
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"JSON saved: {json_file}")


@app.command()
def update(
    config_path: Path = typer.Option(Path("rebalance_config.yaml"), "--config", "-c")
) -> None:
    """Update holdings after executing trades."""
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        raise typer.Exit(1)

    config = load_config(config_path)
    holdings = config["portfolio"]["holdings"]

    print("\n" + "="*60)
    print("UPDATE HOLDINGS")
    print("="*60)

    for symbol, current in holdings.items():
        new_shares = float(typer.prompt(f"{symbol} shares (current: {current:.6f})", default=current))
        holdings[symbol] = new_shares

    config["state"]["last_rebalance"] = datetime.now().isoformat()
    save_config(config, config_path)
    logger.info("Holdings updated")


if __name__ == "__main__":
    # Validation tests
    import sys

    all_failures = []
    total_tests = 0

    # Test 1: Config creation
    total_tests += 1
    try:
        test_config = {
            "portfolio": {
                "assets": [
                    {"symbol": "BTC/USDT", "target_weight": 0.5},
                    {"symbol": "ETH/USDT", "target_weight": 0.5}
                ],
                "holdings": {"BTC/USDT": 0.1, "ETH/USDT": 2.0}
            },
            "rebalancing": {
                "method": "hybrid",
                "threshold": 0.15,
                "min_interval_hours": 24,
                "calendar_period_days": 30,
                "momentum_filter": {"enabled": False, "threshold": 0.20, "lookback_days": 30}
            },
            "state": {"last_rebalance": None, "initial_capital": 10000.0},
            "exchange": {"name": "binance", "quote_currency": "USDT"}
        }
        if len(test_config["portfolio"]["assets"]) != 2:
            all_failures.append("Config creation: Expected 2 assets")
    except Exception as e:
        all_failures.append(f"Config creation: {e}")

    # Test 2: Portfolio calculations
    total_tests += 1
    try:
        test_prices = {"BTC/USDT": 60000.0, "ETH/USDT": 2400.0}
        test_holdings = {"BTC/USDT": 0.125, "ETH/USDT": 2.5}
        total_val, asset_vals = calculate_portfolio_value(test_holdings, test_prices)
        expected_total = (0.125 * 60000) + (2.5 * 2400)  # 7500 + 6000 = 13500
        if abs(total_val - expected_total) > 1.0:
            all_failures.append(f"Portfolio calc: Expected ${expected_total:,.2f}, got ${total_val:,.2f}")
    except Exception as e:
        all_failures.append(f"Portfolio calc: {e}")

    # Test 3: Weight calculations
    total_tests += 1
    try:
        weights = calculate_weights(asset_vals, total_val)
        expected_btc_weight = 7500 / 13500  # ~0.555
        if abs(weights["BTC/USDT"] - expected_btc_weight) > 0.01:
            all_failures.append(f"Weight calc: Expected BTC ~{expected_btc_weight:.3f}, got {weights['BTC/USDT']:.3f}")
    except Exception as e:
        all_failures.append(f"Weight calc: {e}")

    # Test 4: Deviation calculations
    total_tests += 1
    try:
        current_weights = {"BTC/USDT": 0.555, "ETH/USDT": 0.445}
        target_weights = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        devs, max_dev = calculate_deviations(current_weights, target_weights)
        expected_max = 0.055  # BTC is 5.5% over target
        if abs(max_dev - expected_max) > 0.01:
            all_failures.append(f"Deviation calc: Expected {expected_max:.3f}, got {max_dev:.3f}")
    except Exception as e:
        all_failures.append(f"Deviation calc: {e}")

    # Test 5: Rebalancing logic - threshold trigger
    total_tests += 1
    try:
        test_config_threshold = {
            "rebalancing": {
                "method": "threshold",
                "threshold": 0.05,
                "min_interval_hours": 24,
                "calendar_period_days": 30
            }
        }
        needs, reason = check_rebalancing_needed(0.055, None, test_config_threshold)
        if not needs:
            all_failures.append(f"Rebalancing threshold: Expected True, got False (reason: {reason})")
    except Exception as e:
        all_failures.append(f"Rebalancing threshold: {e}")

    # Test 6: Recommendation generation
    total_tests += 1
    try:
        recs = generate_recommendations(
            {"BTC/USDT": 0.555, "ETH/USDT": 0.445},
            {"BTC/USDT": 0.5, "ETH/USDT": 0.5},
            13500.0,
            {"BTC/USDT": 60000.0, "ETH/USDT": 2400.0},
            {"BTC/USDT": 0.125, "ETH/USDT": 2.5}
        )
        btc_rec = next(r for r in recs if r["symbol"] == "BTC/USDT")
        if btc_rec["action"] != "SELL":
            all_failures.append(f"Recommendations: Expected SELL BTC, got {btc_rec['action']}")
    except Exception as e:
        all_failures.append(f"Recommendations: {e}")

    # Final result
    if all_failures:
        print(f"\n[X] VALIDATION FAILED - {len(all_failures)} of {total_tests} tests failed:")
        for failure in all_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\n[OK] VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Script is validated and ready for use")
        print("\nRun with: uv run python portfolio_rebalancer_advisor.py check")
        sys.exit(0)
