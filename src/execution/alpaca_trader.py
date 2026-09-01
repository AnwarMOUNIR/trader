"""
Alpaca Trader module facade and backward compatibility.
Re-exports AlpacaPaperTrader and AlpacaClient from alpaca_client.
"""

from .alpaca_client import AlpacaPaperTrader, AlpacaClient, OrderResult

__all__ = ["AlpacaPaperTrader", "AlpacaClient", "OrderResult"]
