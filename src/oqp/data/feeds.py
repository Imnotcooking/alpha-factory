import os
import pandas as pd
from abc import ABC, abstractmethod
from oqp.data.tick_data_adapter import TickUniverseSelector, parquet_columns

# ==========================================
# 2.1 ABSTRACT BASE CLASS
# ==========================================
class DataFeed(ABC):
    """
    The Universal Contract. Any asset class added to the fund MUST 
    inherit from this class and implement these exact methods.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Loads and standardizes the raw data into a strict schema."""
        pass

    @abstractmethod
    def get_multiplier(self) -> float:
        """Returns the contract point-value multiplier."""
        pass

    @abstractmethod
    def get_crisis_period(self) -> tuple[str, str]:
        """Returns a tuple of (start_date, end_date) representing the asset's worst historical shock."""
        pass

# ==========================================
# 2.2 EQUITIES MODULE
# ==========================================
class EquitiesFeed(DataFeed):
    """Handles corporate actions, splits, dividends, and survivorship bias."""
    
    def load_data(self) -> pd.DataFrame:
        print(f"   -> [DATA ENGINE] Loading Equities from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        # Future Logic: Apply split-adjustment multipliers here
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        self.data = df
        return self.data
        
    def get_multiplier(self) -> float:
        return 1.0  # 1 share = 1x PnL
    
    def get_crisis_period(self) -> tuple[str, str]:
        # US Equity Crisis: COVID-19 Crash
        return ("2020-02-19", "2020-03-23")

# ==========================================
# 2.3 FUTURES MODULE
# ==========================================
class FuturesFeed(DataFeed):
    """Handles continuous contract rolling, margin, and multipliers."""
    
    def __init__(self, data_path: str, multiplier: float = 300.0):
        super().__init__(data_path)
        self.multiplier = multiplier

    def load_data(self) -> pd.DataFrame:
        print(f"   -> [DATA ENGINE] Loading Futures/Index data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        # Standardize strictly to the pipeline schema
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        # Future Logic: Implement Panama Canal continuous roll logic here
        # to prevent fake price gaps during contract expirations.
            
        self.data = df
        return self.data
        
    def get_multiplier(self) -> float:
        return self.multiplier  # e.g., IF contract = 300 RMB per point
    
    def get_crisis_period(self) -> tuple[str, str]:
        # Chinese Market Crisis: 2024 "Quant Quake" (DMA/Snowball liquidations)
        return ("2024-01-01", "2024-02-28")


class FuturesTickFeed(FuturesFeed):
    """Handles Level-1 Chinese futures tick parquet without binding to one file name."""

    def load_data(self) -> pd.DataFrame:
        print(f"   -> [DATA ENGINE] Loading Futures Tick data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        df = TickUniverseSelector.normalize_schema(df)
        df.attrs["data_frequency"] = "tick"
        self.data = df
        return self.data

# ==========================================
# 2.4 OPTIONS MODULE
# ==========================================
class OptionsFeed(DataFeed):
    """Handles multi-dimensional Option chains (Strike, Expiry, Call/Put)."""
    
    def load_data(self) -> pd.DataFrame:
        print(f"   -> [DATA ENGINE] Loading Options from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        
        # Future Logic: Multi-index creation (Date, Underlying, Expiry, Strike)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        self.data = df
        return self.data
        
    def get_multiplier(self) -> float:
        return 100.0  # Standard US option contract represents 100 shares
    
    def get_crisis_period(self) -> tuple[str, str]:
        # Options: VIX Volmageddon
        return ("2018-01-26", "2018-02-09")

# ==========================================
# THE FACTORY DISPENSER
# ==========================================
class DataEngineFactory:
    """
    The router. Tell it what asset class you are trading, and it dispenses
    the correctly configured OOP object.
    """
    @staticmethod
    def create_feed(asset_class: str, data_path: str) -> DataFeed:
        target = asset_class.upper()
        
        if target == "EQUITY":
            return EquitiesFeed(data_path)
        elif target == "FUTURES":
            if TickUniverseSelector.looks_like_tick_columns(parquet_columns(data_path)):
                return FuturesTickFeed(data_path, multiplier=1.0)
            # For this specific Chinese Index project, we use Futures logic
            return FuturesFeed(data_path, multiplier=1.0) # Using 1.0 for indices temporarily
        elif target == "OPTIONS":
            return OptionsFeed(data_path)
        else:
            raise ValueError(f"❌ [DATA ENGINE ERROR] Unknown asset class: {asset_class}")
