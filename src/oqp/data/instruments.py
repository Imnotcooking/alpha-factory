import re
from dataclasses import dataclass
from typing import Dict, Any

from oqp.contracts.market_vertical import ASSET_TAXONOMY, normalize_market_vertical

# ---------------------------------------------------------
# 1. The Physical Instrument Blueprint
# ---------------------------------------------------------
@dataclass
class InstrumentProfile:
    ticker: str
    exchange: str         
    sector: str           
    multiplier: int       
    tick_size: float      
    margin_rate: float    
    fee_type: str         # 'ratio' or 'fixed'
    fee_open: float
    fee_close_history: float
    fee_close_today: float

# ---------------------------------------------------------
# 2. Base Registry (The Interface)
# ---------------------------------------------------------
class BaseRegistry:
    """Abstract base class for all asset registries."""
    def get_profile(self, ticker: str) -> InstrumentProfile:
        raise NotImplementedError
    
    def get_sector_map(self) -> Dict[str, str]:
        raise NotImplementedError

    def get_margin_map(self) -> Dict[str, float]:
        raise NotImplementedError

    def get_multiplier_map(self) -> Dict[str, int]:
        raise NotImplementedError

    def get_fee_profile_map(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def get_fee_type_map(self) -> Dict[str, str]:
        raise NotImplementedError

    def get_fee_open_map(self) -> Dict[str, float]:
        raise NotImplementedError

    def get_fee_close_history_map(self) -> Dict[str, float]:
        raise NotImplementedError

    def get_fee_close_today_map(self) -> Dict[str, float]:
        raise NotImplementedError

# ---------------------------------------------------------
# 3. Chinese Futures Registry (Your Goldmine Data)
# ---------------------------------------------------------
class ChineseFuturesRegistry(BaseRegistry):
    def __init__(self):   
        self.SYMBOL_DICT = {
            # --- 郑商所 (CZCE) ---
            'AP': [10, 1.0, 'CZCE', '生鲜'],     'CF': [5, 5.0, 'CZCE', '软商品'],
            'CJ': [5, 5.0, 'CZCE', '生鲜'],      'CY': [5, 5.0, 'CZCE', '软商品'],
            'FG': [20, 1.0, 'CZCE', '建材'],     'MA': [10, 1.0, 'CZCE', '化工'],
            'OI': [10, 1.0, 'CZCE', '油脂油料'], 'PF': [5, 2.0, 'CZCE', '化工'],
            'PK': [5, 2.0, 'CZCE', '生鲜'],      'PL': [20, 1.0, 'CZCE', '化工'],
            'PR': [15, 2.0, 'CZCE', '化工'],     'PX': [5, 2.0, 'CZCE', '化工'],
            'RM': [10, 1.0, 'CZCE', '油脂油料'], 'SA': [20, 1.0, 'CZCE', '建材'],
            'SF': [5, 2.0, 'CZCE', '黑色'],      'SH': [30, 1.0, 'CZCE', '化工'],
            'SM': [5, 2.0, 'CZCE', '黑色'],      'SR': [10, 1.0, 'CZCE', '软商品'],
            'TA': [5, 2.0, 'CZCE', '化工'],      'UR': [20, 1.0, 'CZCE', '化工'],

            # --- 大商所 (DCE) ---
            'a': [10, 1.0, 'DCE', '油脂油料'],   'b': [10, 1.0, 'DCE', '油脂油料'],
            'c': [10, 1.0, 'DCE', '软商品'],     'cs': [10, 1.0, 'DCE', '软商品'],
            'eb': [5, 1.0, 'DCE', '化工'],       'eg': [10, 1.0, 'DCE', '化工'],
            'i': [100, 0.5, 'DCE', '黑色'],      'j': [100, 0.5, 'DCE', '黑色'],
            'jd': [10, 1.0, 'DCE', '生鲜'],      'jm': [60, 0.5, 'DCE', '黑色'],
            'l': [5, 1.0, 'DCE', '化工'],        'lg': [90, 0.5, 'DCE', '建材'],
            'lh': [16, 5.0, 'DCE', '生鲜'],      'm': [10, 1.0, 'DCE', '油脂油料'],
            'p': [10, 2.0, 'DCE', '油脂油料'],   'pg': [20, 1.0, 'DCE', '能源'],
            'pp': [5, 1.0, 'DCE', '化工'],       'rr': [10, 1.0, 'DCE', '软商品'],
            'v': [5, 1.0, 'DCE', '化工'],        'y': [10, 2.0, 'DCE', '油脂油料'],
            'bz': [30, 1.0, 'DCE', '化工'],

            # --- 上期所 & 能源中心 (SHFE & INE) ---
            'ad': [10, 5.0, 'SHFE', '有色'],     'ag': [15, 1.0, 'SHFE', '贵金属'],
            'al': [5, 5.0, 'SHFE', '有色'],      'ao': [20, 1.0, 'SHFE', '有色'],
            'au': [1000, 0.02, 'SHFE', '贵金属'], 'bc': [5, 10.0, 'INE', '有色'],
            'br': [5, 5.0, 'SHFE', '化工'],      'bu': [10, 1.0, 'SHFE', '能源'],
            'cu': [5, 10.0, 'SHFE', '有色'],     'fu': [10, 1.0, 'SHFE', '能源'],
            'hc': [10, 1.0, 'SHFE', '黑色'],     'lu': [10, 1.0, 'INE', '能源'],
            'ni': [1, 10.0, 'SHFE', '有色'],     'nr': [10, 5.0, 'INE', '化工'],
            'op': [40, 2.0, 'SHFE', '软商品'],   'pb': [5, 5.0, 'SHFE', '有色'],
            'rb': [10, 1.0, 'SHFE', '黑色'],     'ru': [10, 5.0, 'SHFE', '化工'],
            'sc': [1000, 0.1, 'INE', '能源'],    'sn': [1, 10.0, 'SHFE', '有色'],
            'sp': [10, 2.0, 'SHFE', '软商品'],   'ss': [5, 5.0, 'SHFE', '黑色'],
            'zn': [5, 5.0, 'SHFE', '有色'],      'ec': [50, 0.1, 'INE', '航运'],

            # --- 广期所 (GFEX) ---
            'si': [5, 5.0, 'GFEX', '新能源'],    'ps': [3, 5.0, 'GFEX', '新能源'],
            'lc': [1, 20.0, 'GFEX', '新能源'],   'pd': [1000, 0.05, 'GFEX', '贵金属'],
            'pt': [1000, 0.05, 'GFEX', '贵金属'],

            # --- 中金所 (CFFEX) ---
            'IC': [200, 0.2, 'CFFEX', '股指'],   'IF': [300, 0.2, 'CFFEX', '股指'],
            'IH': [300, 0.2, 'CFFEX', '股指'],   'IM': [200, 0.2, 'CFFEX', '股指'],
            'T':  [10000, 0.005, 'CFFEX', '国债'], 'TF': [10000, 0.005, 'CFFEX', '国债'],
            'TL': [10000, 0.01, 'CFFEX', '国债'],  'TS': [20000, 0.002, 'CFFEX', '国债']
        }
        
        self.FEE_DICT = {
            # ================= 上海期货交易所 (SHFE) =================
            'rb': { 'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.09, 'fee_type': 'ratio', 'fee_open': 0.0001, 'fee_close_history': 0.0001, 'fee_close_today': 0.0001 },
            'hc': { 'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.09, 'fee_type': 'ratio', 'fee_open': 0.0001, 'fee_close_history': 0.0001, 'fee_close_today': 0.0001 },
            'cu': { 'multiplier': 5, 'tick_size': 10.0, 'margin_rate': 0.12, 'fee_type': 'ratio', 'fee_open': 0.00005, 'fee_close_history': 0.00005, 'fee_close_today': 0.0001 }, # 铜平今费率加倍
            'al': { 'multiplier': 5, 'tick_size': 5.0, 'margin_rate': 0.12, 'fee_type': 'fixed', 'fee_open': 3.0, 'fee_close_history': 3.0, 'fee_close_today': 3.0 },
            'zn': { 'multiplier': 5, 'tick_size': 5.0, 'margin_rate': 0.12, 'fee_type': 'fixed', 'fee_open': 3.0, 'fee_close_history': 3.0, 'fee_close_today': 3.0 },
            'pb': { 'multiplier': 5, 'tick_size': 5.0, 'margin_rate': 0.12, 'fee_type': 'ratio', 'fee_open': 0.00004, 'fee_close_history': 0.00004, 'fee_close_today': 0.00004 },
            'ao': { 'multiplier': 20, 'tick_size': 1.0, 'margin_rate': 0.12, 'fee_type': 'ratio', 'fee_open': 0.0001, 'fee_close_history': 0.0001, 'fee_close_today': 0.0001 },
            'ss': { 'multiplier': 5, 'tick_size': 5.0, 'margin_rate': 0.10, 'fee_type': 'fixed', 'fee_open': 2.0, 'fee_close_history': 2.0, 'fee_close_today': 2.0 },
            'ni': { 'multiplier': 1, 'tick_size': 10.0, 'margin_rate': 0.14, 'fee_type': 'fixed', 'fee_open': 3.0, 'fee_close_history': 3.0, 'fee_close_today': 3.0 },
            'sn': { 'multiplier': 1, 'tick_size': 10.0, 'margin_rate': 0.14, 'fee_type': 'fixed', 'fee_open': 3.0, 'fee_close_history': 3.0, 'fee_close_today': 3.0 },
            'au': { 'multiplier': 1000, 'tick_size': 0.02, 'margin_rate': 0.16, 'fee_type': 'fixed', 'fee_open': 20.0, 'fee_close_history': 20.0, 'fee_close_today': 20.0 }, # 标准合约费率
            'ag': { 'multiplier': 15, 'tick_size': 1.0, 'margin_rate': 0.21, 'fee_type': 'ratio', 'fee_open': 0.00005, 'fee_close_history': 0.00005, 'fee_close_today': 0.00005 },
            'ru': { 'multiplier': 10, 'tick_size': 5.0, 'margin_rate': 0.11, 'fee_type': 'fixed', 'fee_open': 3.0, 'fee_close_history': 3.0, 'fee_close_today': 3.0 },
            'bu': { 'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.11, 'fee_type': 'ratio', 'fee_open': 0.00005, 'fee_close_history': 0.00005, 'fee_close_today': 0.00005 },
            'sp': { 'multiplier': 10, 'tick_size': 2.0, 'margin_rate': 0.09, 'fee_type': 'ratio', 'fee_open': 0.00005, 'fee_close_history': 0.00005, 'fee_close_today': 0.00005 },
            'br': { 'multiplier': 5, 'tick_size': 5.0, 'margin_rate': 0.11, 'fee_type': 'ratio', 'fee_open': 0.00002, 'fee_close_history': 0.00002, 'fee_close_today': 0.00002 },
            'ad': { 'multiplier': 10, 'tick_size': 5.0, 'margin_rate': 0.07, 'fee_type': 'ratio', 'fee_open': 0.00005, 'fee_close_history': 0.00005, 'fee_close_today': 0.00005 },
            'op': { 'multiplier': 40, 'tick_size': 2.0, 'margin_rate': 0.09, 'fee_type': 'ratio', 'fee_open': 0.00005, 'fee_close_history': 0.00005, 'fee_close_today': 0.00005 },
            'fu': { 'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.11, 'fee_type': 'ratio', 'fee_open': 0.0001, 'fee_close_history': 0.0001, 'fee_close_today': 0.0003 }, # 燃油平今费率三倍

            # ----------------- 上海国际能源交易中心 (INE, 小写代码) -----------------
            'nr': {'multiplier': 10, 'tick_size': 5.0, 'margin_rate': 0.11, 'fee_type': 'ratio', 'fee_open': 0.00002,'fee_close_history': 0.00002, 'fee_close_today': 0.00002},
            'bc': {'multiplier': 5, 'tick_size': 10.0, 'margin_rate': 0.09, 'fee_type': 'ratio', 'fee_open': 0.00001,'fee_close_history': 0.00001, 'fee_close_today': 0.00001},
            'sc': {'multiplier': 1000, 'tick_size': 0.1, 'margin_rate': 0.11, 'fee_type': 'fixed', 'fee_open': 40.0, 'fee_close_history': 40.0, 'fee_close_today': 240.0},  # 原油平今费率 240 元
            'lu': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.11, 'fee_type': 'ratio', 'fee_open': 0.0001,'fee_close_history': 0.0001, 'fee_close_today': 0.0003},  # 低硫燃料油平今费率三倍
            'ec': {'multiplier': 50, 'tick_size': 0.1, 'margin_rate': 0.17, 'fee_type': 'ratio', 'fee_open': 0.0006,'fee_close_history': 0.0006, 'fee_close_today': 0.0012},  # 集运欧线平今费率翻倍

            # ----------------- 郑州商品交易所 (CZCE, 大写代码) -----------------
            'CF': {'multiplier': 5, 'tick_size': 5.0, 'margin_rate': 0.07, 'fee_type': 'fixed', 'fee_open': 4.3,'fee_close_history': 4.3, 'fee_close_today': 4.3},
            'CY': {'multiplier': 5, 'tick_size': 5.0, 'margin_rate': 0.08, 'fee_type': 'fixed', 'fee_open': 1.0,'fee_close_history': 1.0, 'fee_close_today': 1.0},
            'SR': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.08, 'fee_type': 'fixed', 'fee_open': 3.0, 'fee_close_history': 3.0, 'fee_close_today': 3.0},
            'TA': {'multiplier': 5, 'tick_size': 2.0, 'margin_rate': 0.10, 'fee_type': 'fixed', 'fee_open': 3.0, 'fee_close_history': 3.0, 'fee_close_today': 3.0},
            'PF': {'multiplier': 5, 'tick_size': 2.0, 'margin_rate': 0.10, 'fee_type': 'fixed', 'fee_open': 2.0,'fee_close_history': 2.0, 'fee_close_today': 2.0},
            'FG': {'multiplier': 20, 'tick_size': 1.0, 'margin_rate': 0.12, 'fee_type': 'fixed', 'fee_open': 6.0,'fee_close_history': 6.0, 'fee_close_today': 6.0},  # 玻璃手续费 6 元
            'MA': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.10, 'fee_type': 'ratio', 'fee_open': 0.0001,'fee_close_history': 0.0001, 'fee_close_today': 0.0001},
            'SA': {'multiplier': 20, 'tick_size': 1.0, 'margin_rate': 0.12, 'fee_type': 'ratio', 'fee_open': 0.0002,'fee_close_history': 0.0002, 'fee_close_today': 0.0002},
            'RM': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.10, 'fee_type': 'fixed', 'fee_open': 1.5, 'fee_close_history': 1.5, 'fee_close_today': 1.5},
            'OI': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.10, 'fee_type': 'fixed', 'fee_open': 2.0,'fee_close_history': 2.0, 'fee_close_today': 2.0},
            'ZC': {'multiplier': 100, 'tick_size': 0.2, 'margin_rate': 0.50, 'fee_type': 'fixed', 'fee_open': 150.0,'fee_close_history': 150.0, 'fee_close_today': 150.0},  # 动力煤
            'WH': {'multiplier': 20, 'tick_size': 1.0, 'margin_rate': 0.15, 'fee_type': 'fixed', 'fee_open': 30.0, 'fee_close_history': 30.0, 'fee_close_today': 30.0},  # 强麦
            'PM': {'multiplier': 50, 'tick_size': 1.0, 'margin_rate': 0.15, 'fee_type': 'fixed', 'fee_open': 30.0, 'fee_close_history': 30.0, 'fee_close_today': 30.0},  # 普麦
            'RI': {'multiplier': 20, 'tick_size': 1.0, 'margin_rate': 0.15, 'fee_type': 'fixed', 'fee_open': 2.5,'fee_close_history': 2.5, 'fee_close_today': 2.5},  # 早籼稻
            'LR': {'multiplier': 20, 'tick_size': 1.0, 'margin_rate': 0.15, 'fee_type': 'fixed', 'fee_open': 3.0, 'fee_close_history': 3.0, 'fee_close_today': 3.0},  # 晚籼稻
            'JR': {'multiplier': 20, 'tick_size': 1.0, 'margin_rate': 0.15, 'fee_type': 'fixed', 'fee_open': 3.0,'fee_close_history': 3.0, 'fee_close_today': 3.0},  # 粳稻
            'AP': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.12, 'fee_type': 'fixed', 'fee_open': 5.0,'fee_close_history': 5.0, 'fee_close_today': 20.0},  # 苹果平今手续费 20 元
            'UR': {'multiplier': 20, 'tick_size': 1.0, 'margin_rate': 0.10, 'fee_type': 'ratio', 'fee_open': 0.0001, 'fee_close_history': 0.0001, 'fee_close_today': 0.0001},
            'PK': {'multiplier': 5, 'tick_size': 2.0, 'margin_rate': 0.10, 'fee_type': 'fixed', 'fee_open': 4.0,'fee_close_history': 4.0, 'fee_close_today': 4.0},
            'RS': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.20, 'fee_type': 'fixed', 'fee_open': 2.0, 'fee_close_history': 2.0, 'fee_close_today': 2.0},  # 油菜籽
            'SM': {'multiplier': 5, 'tick_size': 2.0, 'margin_rate': 0.10, 'fee_type': 'fixed', 'fee_open': 3.0,'fee_close_history': 3.0, 'fee_close_today': 3.0},
            'SF': {'multiplier': 5, 'tick_size': 2.0, 'margin_rate': 0.10, 'fee_type': 'fixed', 'fee_open': 3.0,'fee_close_history': 3.0, 'fee_close_today': 3.0},
            'SH': {'multiplier': 30, 'tick_size': 1.0, 'margin_rate': 0.10, 'fee_type': 'ratio', 'fee_open': 0.0002,'fee_close_history': 0.0002, 'fee_close_today': 0.0002},
            'PR': {'multiplier': 15, 'tick_size': 2.0, 'margin_rate': 0.10, 'fee_type': 'ratio', 'fee_open': 0.00005, 'fee_close_history': 0.00005, 'fee_close_today': 0.00005},
            'PL': {'multiplier': 20, 'tick_size': 1.0, 'margin_rate': 0.10, 'fee_type': 'ratio', 'fee_open': 0.0001, 'fee_close_history': 0.0001, 'fee_close_today': 0.0001},
            'CJ': {'multiplier': 5, 'tick_size': 5.0, 'margin_rate': 0.12, 'fee_type': 'fixed', 'fee_open': 3.0,'fee_close_history': 3.0, 'fee_close_today': 3.0},
            'PX': {'multiplier': 5, 'tick_size': 2.0, 'margin_rate': 0.10, 'fee_type': 'ratio', 'fee_open': 0.0001, 'fee_close_history': 0.0001, 'fee_close_today': 0.0001},

            # ----------------- 大连商品交易所 (DCE, 小写代码) -----------------
            'a': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.09, 'fee_type': 'fixed', 'fee_open': 2.0,'fee_close_history': 2.0, 'fee_close_today': 2.0},
            'b': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.09, 'fee_type': 'fixed', 'fee_open': 1.0,'fee_close_history': 1.0, 'fee_close_today': 1.0},
            'm': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.09, 'fee_type': 'fixed', 'fee_open': 1.5,'fee_close_history': 1.5, 'fee_close_today': 1.5},
            'y': {'multiplier': 10, 'tick_size': 2.0, 'margin_rate': 0.10, 'fee_type': 'fixed', 'fee_open': 2.5,'fee_close_history': 2.5, 'fee_close_today': 2.5},
            'p': {'multiplier': 10, 'tick_size': 2.0, 'margin_rate': 0.11, 'fee_type': 'fixed', 'fee_open': 2.5,'fee_close_history': 2.5, 'fee_close_today': 2.5},
            'jm': {'multiplier': 60, 'tick_size': 0.5, 'margin_rate': 0.14, 'fee_type': 'ratio', 'fee_open': 0.0001,'fee_close_history': 0.0001, 'fee_close_today': 0.0001},
            'j': {'multiplier': 100, 'tick_size': 0.5, 'margin_rate': 0.20, 'fee_type': 'ratio', 'fee_open': 0.0001,'fee_close_history': 0.0001, 'fee_close_today': 0.0001},
            'i': {'multiplier': 100, 'tick_size': 0.5, 'margin_rate': 0.13, 'fee_type': 'ratio', 'fee_open': 0.0001,'fee_close_history': 0.0001, 'fee_close_today': 0.0001},
            'c': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.09, 'fee_type': 'fixed', 'fee_open': 1.2,'fee_close_history': 1.2, 'fee_close_today': 1.2},
            'cs': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.08, 'fee_type': 'fixed', 'fee_open': 1.5,'fee_close_history': 1.5, 'fee_close_today': 1.5},
            'eg': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.11, 'fee_type': 'fixed', 'fee_open': 3.0,'fee_close_history': 3.0, 'fee_close_today': 3.0},
            'eb': {'multiplier': 5, 'tick_size': 1.0, 'margin_rate': 0.11, 'fee_type': 'fixed', 'fee_open': 3.0,'fee_close_history': 3.0, 'fee_close_today': 3.0},
            'pg': {'multiplier': 20, 'tick_size': 1.0, 'margin_rate': 0.11, 'fee_type': 'fixed', 'fee_open': 6.0,'fee_close_history': 6.0, 'fee_close_today': 12.0},  # PG 平今手续费 12 元
            'rr': {'multiplier': 10, 'tick_size': 1.0, 'margin_rate': 0.08, 'fee_type': 'fixed', 'fee_open': 1.0,'fee_close_history': 1.0, 'fee_close_today': 1.0},
            'bb': {'multiplier': 500, 'tick_size': 0.05, 'margin_rate': 0.15, 'fee_type': 'ratio', 'fee_open': 0.0001,'fee_close_history': 0.0001, 'fee_close_today': 0.0001},
            'fb': {'multiplier': 10, 'tick_size': 0.05, 'margin_rate': 0.10, 'fee_type': 'ratio', 'fee_open': 0.0001,'fee_close_history': 0.0001, 'fee_close_today': 0.0001},
            'lg': {'multiplier': 90, 'tick_size': 0.5, 'margin_rate': 0.10, 'fee_type': 'ratio', 'fee_open': 0.0001,'fee_close_history': 0.0001, 'fee_close_today': 0.0001},
            'bz': {'multiplier': 30, 'tick_size': 1.0, 'margin_rate': 0.12, 'fee_type': 'ratio', 'fee_open': 0.0001,'fee_close_history': 0.0001, 'fee_close_today': 0.0001},
            'jd': {'multiplier': 5, 'tick_size': 1.0, 'margin_rate': 0.09, 'fee_type': 'ratio', 'fee_open': 0.00015,'fee_close_history': 0.00015, 'fee_close_today': 0.00015},
            'lh': {'multiplier': 16, 'tick_size': 5.0, 'margin_rate': 0.10, 'fee_type': 'ratio', 'fee_open': 0.0001,'fee_close_history': 0.0001, 'fee_close_today': 0.0001},
            'l': {'multiplier': 5, 'tick_size': 1.0, 'margin_rate': 0.10, 'fee_type': 'fixed', 'fee_open': 1.0,'fee_close_history': 1.0, 'fee_close_today': 1.0},
            'v': {'multiplier': 5, 'tick_size': 1.0, 'margin_rate': 0.10, 'fee_type': 'fixed', 'fee_open': 1.0,'fee_close_history': 1.0, 'fee_close_today': 1.0},
            'pp': {'multiplier': 5, 'tick_size': 1.0, 'margin_rate': 0.10, 'fee_type': 'fixed', 'fee_open': 1.0,'fee_close_history': 1.0, 'fee_close_today': 1.0},

            # ----------------- 广州期货交易所 (GFEX, 小写代码) -----------------
            'si': {'multiplier': 5, 'tick_size': 5.0, 'margin_rate': 0.13, 'fee_type': 'ratio', 'fee_open': 0.0001,'fee_close_history': 0.0001, 'fee_close_today': 0.0001},
            'lc': {'multiplier': 1, 'tick_size': 20.0, 'margin_rate': 0.17, 'fee_type': 'ratio', 'fee_open': 0.00032,'fee_close_history': 0.00032, 'fee_close_today': 0.00032},
            'ps': {'multiplier': 3, 'tick_size': 5.0, 'margin_rate': 0.15, 'fee_type': 'ratio', 'fee_open': 0.0005,'fee_close_history': 0.0005, 'fee_close_today': 0.0005},
            'pt': {'multiplier': 1000, 'tick_size': 0.05, 'margin_rate': 0.19, 'fee_type': 'ratio', 'fee_open': 0.0001,'fee_close_history': 0.0001, 'fee_close_today': 0.0001},
            'pd': {'multiplier': 1000, 'tick_size': 0.05, 'margin_rate': 0.19, 'fee_type': 'ratio', 'fee_open': 0.0001, 'fee_close_history': 0.0001, 'fee_close_today': 0.0001},

            # ----------------- 中国金融期货交易所 (CFFEX, 大写代码) -----------------
            'IF': {'multiplier': 300, 'tick_size': 0.2, 'margin_rate': 0.12, 'fee_type': 'ratio', 'fee_open': 0.000023, 'fee_close_history': 0.000023, 'fee_close_today': 0.00023},  # 股指平今费率约为普通平仓 10 倍
            'IH': {'multiplier': 300, 'tick_size': 0.2, 'margin_rate': 0.12, 'fee_type': 'ratio', 'fee_open': 0.000023,'fee_close_history': 0.000023, 'fee_close_today': 0.00023},  # 股指平今费率约为普通平仓 10 倍
            'IC': {'multiplier': 200, 'tick_size': 0.2, 'margin_rate': 0.12, 'fee_type': 'ratio', 'fee_open': 0.000023, 'fee_close_history': 0.000023, 'fee_close_today': 0.00023},  # 股指平今费率约为普通平仓 10 倍
            'IM': {'multiplier': 200, 'tick_size': 0.2, 'margin_rate': 0.12, 'fee_type': 'ratio', 'fee_open': 0.000023, 'fee_close_history': 0.000023, 'fee_close_today': 0.00023},  # 股指平今费率约为普通平仓 10 倍
            'TS': {'multiplier': 20000, 'tick_size': 0.002, 'margin_rate': 0.005, 'fee_type': 'fixed', 'fee_open': 3.0, 'fee_close_history': 3.0, 'fee_close_today': 3.0},
            'TF': {'multiplier': 10000, 'tick_size': 0.005, 'margin_rate': 0.012, 'fee_type': 'fixed', 'fee_open': 3.0,'fee_close_history': 3.0, 'fee_close_today': 3.0},
            'T': {'multiplier': 10000, 'tick_size': 0.005, 'margin_rate': 0.02, 'fee_type': 'fixed', 'fee_open': 3.0,'fee_close_history': 3.0, 'fee_close_today': 3.0},
            'TL': {'multiplier': 10000, 'tick_size': 0.01, 'margin_rate': 0.035, 'fee_type': 'fixed', 'fee_open': 3.0,'fee_close_history': 3.0, 'fee_close_today': 3.0}
        }

    def _strip_ticker(self, ticker: str) -> str:
        """Extracts the physical base symbol.

        Handles both tradable contracts (e.g. ``TA609`` -> ``TA``) and
        Chinese index labels (e.g. ``黄金(au)[指数]`` -> ``au``).
        """
        value = str(ticker or "").strip()
        paren_match = re.search(r"[\(（]([A-Za-z]+)[\)）]", value)
        if paren_match:
            return paren_match.group(1)
        return re.sub(r'\d+', '', value)

    def get_profile(self, ticker: str) -> InstrumentProfile:
        base = self._strip_ticker(ticker)
        
        sym_data = self.SYMBOL_DICT.get(base, [10, 1.0, 'UNKNOWN', 'Macro'])
        fee_data = self.FEE_DICT.get(base, {
            'margin_rate': 0.10, 'fee_type': 'ratio', 
            'fee_open': 0.0001, 'fee_close_history': 0.0001, 'fee_close_today': 0.0001
        })
        
        return InstrumentProfile(
            ticker=base,
            exchange=sym_data[2],
            sector=sym_data[3],
            multiplier=sym_data[0],
            tick_size=sym_data[1],
            margin_rate=fee_data['margin_rate'],
            fee_type=fee_data['fee_type'],
            fee_open=fee_data['fee_open'],
            fee_close_history=fee_data['fee_close_history'],
            fee_close_today=fee_data['fee_close_today']
        )

    def get_sector_map(self) -> Dict[str, str]:
        return {k: v[3] for k, v in self.SYMBOL_DICT.items()}
    
    def get_margin_map(self) -> Dict[str, float]:
        return {k: v['margin_rate'] for k, v in self.FEE_DICT.items()}
        
    def get_multiplier_map(self) -> Dict[str, int]:
        return {k: v[0] for k, v in self.SYMBOL_DICT.items()}

    def get_fee_profile_map(self) -> Dict[str, Dict[str, Any]]:
        return {k: dict(v) for k, v in self.FEE_DICT.items()}

    def get_fee_type_map(self) -> Dict[str, str]:
        return {k: v['fee_type'] for k, v in self.FEE_DICT.items()}

    def get_fee_open_map(self) -> Dict[str, float]:
        return {k: v['fee_open'] for k, v in self.FEE_DICT.items()}

    def get_fee_close_history_map(self) -> Dict[str, float]:
        return {k: v['fee_close_history'] for k, v in self.FEE_DICT.items()}

    def get_fee_close_today_map(self) -> Dict[str, float]:
        return {k: v['fee_close_today'] for k, v in self.FEE_DICT.items()}

# ---------------------------------------------------------
# 4. Equity / Options Registries (Safe Placeholders)
# ---------------------------------------------------------
class GenericEquitiesRegistry(BaseRegistry):
    def __init__(
        self,
        *,
        exchange: str,
        sector: str = "Equity",
        margin_rate: float = 0.25,
        fee_open: float = 0.00005,
        fee_close_history: float | None = None,
        fee_close_today: float | None = None,
        tick_size: float = 0.01,
    ) -> None:
        self.exchange = exchange
        self.sector = sector
        self.margin_rate = margin_rate
        self.fee_open = fee_open
        self.fee_close_history = fee_open if fee_close_history is None else fee_close_history
        self.fee_close_today = self.fee_close_history if fee_close_today is None else fee_close_today
        self.tick_size = tick_size

    def get_profile(self, ticker: str) -> InstrumentProfile:
        # Do not strip numbers here: symbols such as ``3M`` must remain intact.
        return InstrumentProfile(
            ticker=str(ticker),
            exchange=self.exchange,
            sector=self.sector,
            multiplier=1,
            tick_size=self.tick_size,
            margin_rate=self.margin_rate,
            fee_type="ratio",
            fee_open=self.fee_open,
            fee_close_history=self.fee_close_history,
            fee_close_today=self.fee_close_today,
        )

    def get_sector_map(self) -> Dict[str, str]: return {}
    def get_margin_map(self) -> Dict[str, float]: return {}
    def get_multiplier_map(self) -> Dict[str, int]: return {}
    def get_fee_profile_map(self) -> Dict[str, Dict[str, Any]]: return {}
    def get_fee_type_map(self) -> Dict[str, str]: return {}
    def get_fee_open_map(self) -> Dict[str, float]: return {}
    def get_fee_close_history_map(self) -> Dict[str, float]: return {}
    def get_fee_close_today_map(self) -> Dict[str, float]: return {}


class USEquitiesRegistry(GenericEquitiesRegistry):
    def __init__(self) -> None:
        super().__init__(exchange="US", margin_rate=0.25, fee_open=0.00005)


class CNEquitiesRegistry(GenericEquitiesRegistry):
    def __init__(self) -> None:
        super().__init__(
            exchange="CN",
            margin_rate=1.0,
            fee_open=0.0003,
            fee_close_history=0.0008,
            fee_close_today=0.0008,
        )


class HKEquitiesRegistry(GenericEquitiesRegistry):
    def __init__(self) -> None:
        super().__init__(exchange="HK", margin_rate=0.25, fee_open=0.0001)


class GenericOptionsRegistry(BaseRegistry):
    def __init__(
        self,
        *,
        exchange: str,
        multiplier: int,
        margin_rate: float,
        tick_size: float,
        fee_open: float,
    ) -> None:
        self.exchange = exchange
        self.multiplier = multiplier
        self.margin_rate = margin_rate
        self.tick_size = tick_size
        self.fee_open = fee_open

    def get_profile(self, ticker: str) -> InstrumentProfile:
        return InstrumentProfile(
            ticker=str(ticker),
            exchange=self.exchange,
            sector="Option",
            multiplier=self.multiplier,
            tick_size=self.tick_size,
            margin_rate=self.margin_rate,
            fee_type="fixed",
            fee_open=self.fee_open,
            fee_close_history=self.fee_open,
            fee_close_today=self.fee_open,
        )

    def get_sector_map(self) -> Dict[str, str]: return {}
    def get_margin_map(self) -> Dict[str, float]: return {}
    def get_multiplier_map(self) -> Dict[str, int]: return {}
    def get_fee_profile_map(self) -> Dict[str, Dict[str, Any]]: return {}
    def get_fee_type_map(self) -> Dict[str, str]: return {}
    def get_fee_open_map(self) -> Dict[str, float]: return {}
    def get_fee_close_history_map(self) -> Dict[str, float]: return {}
    def get_fee_close_today_map(self) -> Dict[str, float]: return {}


class USOptionsRegistry(GenericOptionsRegistry):
    def __init__(self) -> None:
        super().__init__(exchange="US", multiplier=100, margin_rate=1.0, tick_size=0.01, fee_open=0.65)


class CNOptionsRegistry(GenericOptionsRegistry):
    def __init__(self) -> None:
        super().__init__(exchange="CN", multiplier=10_000, margin_rate=1.0, tick_size=0.0001, fee_open=5.0)


# ---------------------------------------------------------
# 5. The Master Factory Router
# ---------------------------------------------------------
class InstrumentMaster:
    """
    Dynamically loads the correct physical rulebook based on the asset class.
    Prevents Chinese Futures logic from crashing US Equity pipelines.
    """
    def __init__(self, asset_class: str = "FUTURES_CN"):
        asset_class = normalize_market_vertical(asset_class)
        if asset_class not in ASSET_TAXONOMY:
            print(f"⚠️ WARNING: '{asset_class}' not in ASSET_TAXONOMY. Defaulting to FUTURES_CN.")
            asset_class = "FUTURES_CN"
            
        self.asset_class = asset_class
        self.taxonomy = ASSET_TAXONOMY[asset_class]
        
        # Route to the correct physics engine
        if "FUTURES_CN" in asset_class:
            self.registry = ChineseFuturesRegistry()
        elif "EQUITY_US" in asset_class:
            self.registry = USEquitiesRegistry()
        elif "EQUITY_CN" in asset_class:
            self.registry = CNEquitiesRegistry()
        elif "EQUITY_HK" in asset_class:
            self.registry = HKEquitiesRegistry()
        elif "OPTIONS_US" in asset_class:
            self.registry = USOptionsRegistry()
        elif "OPTIONS_CN" in asset_class:
            self.registry = CNOptionsRegistry()
        else:
            # Fallback
            self.registry = ChineseFuturesRegistry()

    # Pass-through methods to the active registry
    def get_profile(self, ticker: str) -> InstrumentProfile:
        return self.registry.get_profile(ticker)

    def get_sector_map(self) -> Dict[str, str]:
        return self.registry.get_sector_map()
    
    def get_margin_map(self) -> Dict[str, float]:
        return self.registry.get_margin_map()
        
    def get_multiplier_map(self) -> Dict[str, int]:
        return self.registry.get_multiplier_map()

    def get_fee_profile_map(self) -> Dict[str, Dict[str, Any]]:
        return self.registry.get_fee_profile_map()

    def get_fee_type_map(self) -> Dict[str, str]:
        return self.registry.get_fee_type_map()

    def get_fee_open_map(self) -> Dict[str, float]:
        return self.registry.get_fee_open_map()

    def get_fee_close_history_map(self) -> Dict[str, float]:
        return self.registry.get_fee_close_history_map()

    def get_fee_close_today_map(self) -> Dict[str, float]:
        return self.registry.get_fee_close_today_map()
