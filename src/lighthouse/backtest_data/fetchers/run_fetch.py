"""
================================================================================
RUNNER FILE FOR FUNDING RATE FETCHERS
================================================================================

This script dynamically loads the appropriate funding rate fetcher for the specified exchange,
retrieves historical funding rate settlements, and saves them in the standardized DataFrame format
expected by BaseFundingFetcher and consumed by FundingModel_Historical.

Usage:
    python run_fetch.py --exchange EXCHANGE --symbol SYMBOL --start-date START --end-date END

Example:
    python run_fetch.py --exchange phemex --symbol .BTCFR8H --start-date 2024-01-01 --end-date 2024-12-31
================================================================================
"""

######### IMPORTS ##################
import argparse
import importlib

######## RUNNER FUNCTION ##################
def main():
    parser = argparse.ArgumentParser(description="Fetch funding rate data from an exchange")
    parser.add_argument("--exchange", required=True, help="Exchange name (e.g., 'phemex')")
    parser.add_argument("--symbol", required=True, help="Exchange funding rate symbol")
    parser.add_argument("--start-date", required=True, help="Start date in ISO format (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date in ISO format (YYYY-MM-DD)")
    args = parser.parse_args()

    # Dynamically import  the fetcher for the specified exchange
    try:
        module = importlib.import_module(f".{args.exchange}", package="exchanges.funding_fetchers")
        class_name = "".join(word.capitalize() for word in args.exchange.split("_")) + "FundingFetcher"
        fetcher_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Cannot find fetcher for exchange '{args.exchange}': {e}")

    print(f"Fetching {args.symbol} from {args.start_date} to {args.end_date} ({args.exchange})...")

    # Instantiate the fetcher and fetch the data
    fetcher = fetcher_class(timeout=10)
    df = fetcher.fetch(args.symbol, args.start_date, args.end_date)

    if df.empty:
        raise RuntimeError(f"No funding rate data returned for {args.symbol} from {args.start_date} to {args.end_date}.")
    
    path = fetcher.save(df, args.symbol)
    print(f"  Saved {len(df)} rows to {path}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print(f"  Funding rate range: {df['funding_rate'].min():.6f} to {df['funding_rate'].max():.6f}")

######## ENTRY POINT ##################
if __name__ == "__main__":
    main()
