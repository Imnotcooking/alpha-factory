"""Canonical Chinese display names for Chinese futures product roots."""

from __future__ import annotations


CN_FUTURES_PRODUCT_NAMES_ZH = {
    "AP": "苹果",
    "CF": "棉花",
    "CJ": "红枣",
    "CY": "棉纱",
    "FG": "玻璃",
    "JR": "粳稻",
    "LR": "晚籼稻",
    "MA": "甲醇",
    "OI": "菜籽油",
    "PF": "短纤",
    "PK": "花生",
    "PL": "瓶片",
    "PM": "普麦",
    "PR": "瓶片",
    "PX": "对二甲苯",
    "RI": "早籼稻",
    "RM": "菜籽粕",
    "RS": "油菜籽",
    "SA": "纯碱",
    "SF": "硅铁",
    "SH": "烧碱",
    "SM": "锰硅",
    "SR": "白糖",
    "TA": "PTA",
    "UR": "尿素",
    "WH": "强麦",
    "ZC": "动力煤",
    "a": "豆一",
    "b": "豆二",
    "bb": "胶合板",
    "bz": "纯苯",
    "c": "玉米",
    "cs": "玉米淀粉",
    "eb": "苯乙烯",
    "eg": "乙二醇",
    "fb": "纤维板",
    "i": "铁矿石",
    "j": "焦炭",
    "jd": "鸡蛋",
    "jm": "焦煤",
    "l": "聚乙烯",
    "lg": "原木",
    "lh": "生猪",
    "m": "豆粕",
    "p": "棕榈油",
    "pg": "液化石油气",
    "pp": "聚丙烯",
    "rr": "粳米",
    "v": "PVC",
    "y": "豆油",
    "ad": "铸造铝合金",
    "ag": "白银",
    "al": "铝",
    "ao": "氧化铝",
    "au": "黄金",
    "bc": "国际铜",
    "br": "丁二烯橡胶",
    "bu": "沥青",
    "cu": "铜",
    "ec": "集运欧线",
    "fu": "燃料油",
    "hc": "热卷",
    "lu": "低硫燃料油",
    "ni": "镍",
    "nr": "20号胶",
    "op": "胶版印刷纸",
    "pb": "铅",
    "rb": "螺纹钢",
    "ru": "橡胶",
    "sc": "原油",
    "sn": "锡",
    "sp": "纸浆",
    "ss": "不锈钢",
    "wr": "线材",
    "zn": "锌",
    "lc": "碳酸锂",
    "pd": "钯",
    "ps": "多晶硅",
    "pt": "铂",
    "si": "工业硅",
    "IC": "中证500股指",
    "IF": "沪深300股指",
    "IH": "上证50股指",
    "IM": "中证1000股指",
    "T": "10年期国债",
    "TF": "5年期国债",
    "TL": "30年期国债",
    "TS": "2年期国债",
}


def futures_cn_product_name_zh(root: object) -> str | None:
    """Return a Chinese display name while respecting exchange symbol casing."""

    value = str(root or "").strip()
    if not value:
        return None
    return (
        CN_FUTURES_PRODUCT_NAMES_ZH.get(value)
        or CN_FUTURES_PRODUCT_NAMES_ZH.get(value.upper())
        or CN_FUTURES_PRODUCT_NAMES_ZH.get(value.lower())
    )


def format_futures_cn_product_zh(root: object) -> str:
    """Format a root for bilingual selectors without changing its stored value."""

    value = str(root or "").strip()
    name = futures_cn_product_name_zh(value)
    return f"{value} · {name}" if name else value
