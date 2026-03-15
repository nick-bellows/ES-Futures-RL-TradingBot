#!/usr/bin/env python3
"""
External Market Data Feed
Alternative real-time data sources since AT Interface doesn't provide market data
"""

import logging
import threading
import time
import requests
from datetime import datetime
from typing import Optional, Callable, List
import websocket
import json

class ExternalDataFeed:
    """
    External market data feed for ES futures
    Provides real-time prices when AT Interface cannot
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.current_price: Optional[float] = None
        self.last_update_time: Optional[datetime] = None
        self.running = False
        
        # Callbacks for price updates
        self.price_callbacks: List[Callable[[float], None]] = []
        
        # Update thread
        self.update_thread: Optional[threading.Thread] = None
        self.update_interval = 5.0  # seconds
        
    def start_feed(self) -> bool:
        """Start external data feed"""
        try:
            self.logger.info("Starting external market data feed...")
            
            # Get initial price
            initial_price = self._get_current_es_price()
            if initial_price:
                self.current_price = initial_price
                self.last_update_time = datetime.now()
                self.logger.info(f"Initial ES price: {initial_price:.2f}")
            else:
                self.logger.warning("Could not get initial price")
            
            # Start update thread
            self.running = True
            self.update_thread = threading.Thread(target=self._update_worker, daemon=True)
            self.update_thread.start()
            
            self.logger.info("External data feed started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start external data feed: {e}")
            return False
    
    def stop_feed(self):
        """Stop external data feed"""
        try:
            self.logger.info("Stopping external data feed...")
            self.running = False
            
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5.0)
            
            self.logger.info("External data feed stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping external data feed: {e}")
    
    def _update_worker(self):
        """Background worker to update prices"""
        self.logger.info("External data feed worker started")
        
        while self.running:
            try:
                # Get updated price
                new_price = self._get_current_es_price()
                
                if new_price and new_price != self.current_price:
                    old_price = self.current_price
                    self.current_price = new_price
                    self.last_update_time = datetime.now()
                    
                    change = new_price - old_price if old_price else 0
                    self.logger.info(f"Price update: {new_price:.2f} ({change:+.2f})")
                    
                    # Notify callbacks
                    for callback in self.price_callbacks:
                        try:
                            callback(new_price)
                        except Exception as e:
                            self.logger.error(f"Price callback error: {e}")
                
                # Wait before next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"External data feed worker error: {e}")
                time.sleep(self.update_interval)
        
        self.logger.info("External data feed worker stopped")
    
    def _get_current_es_price(self) -> Optional[float]:
        """Get current ES price from external sources"""
        try:
            # Try multiple sources in order of preference
            sources = [
                self._get_price_from_yahoo,
                self._get_price_from_investing,
                self._get_price_from_marketwatch
            ]
            
            for get_price_func in sources:
                try:
                    price = get_price_func()
                    if price and 5000 <= price <= 8000:  # Validate ES range
                        return price
                except Exception as e:
                    self.logger.debug(f"Price source failed: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting ES price: {e}")
            return None
    
    def _get_price_from_yahoo(self) -> Optional[float]:
        """Get ES price from Yahoo Finance"""
        try:
            # Use Yahoo Finance API (if available) or scraping
            url = "https://query1.finance.yahoo.com/v8/finance/chart/ES=F"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'chart' in data and data['chart']['result']:
                    meta = data['chart']['result'][0]['meta']
                    if 'regularMarketPrice' in meta:
                        return float(meta['regularMarketPrice'])
            
            return None
            
        except Exception:
            return None
    
    def _get_price_from_investing(self) -> Optional[float]:
        """Get ES price from Investing.com (example)"""
        try:
            # This would require implementing their API or web scraping
            # For now, return None as placeholder
            return None
            
        except Exception:
            return None
    
    def _get_price_from_marketwatch(self) -> Optional[float]:
        """Get ES price from MarketWatch (example)"""
        try:
            # This would require web scraping MarketWatch
            # For now, return None as placeholder  
            return None
            
        except Exception:
            return None
    
    def add_price_callback(self, callback: Callable[[float], None]):
        """Add callback for price updates"""
        self.price_callbacks.append(callback)
        self.logger.info(f"Added price callback: {callback.__name__}")
    
    def get_current_price(self) -> Optional[float]:
        """Get current ES price"""
        return self.current_price
    
    def get_last_update_time(self) -> Optional[datetime]:
        """Get time of last price update"""
        return self.last_update_time
    
    def is_data_fresh(self, max_age_seconds: int = 60) -> bool:
        """Check if data is fresh (updated within max_age_seconds)"""
        if not self.last_update_time:
            return False
        
        age = (datetime.now() - self.last_update_time).total_seconds()
        return age <= max_age_seconds