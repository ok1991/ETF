"""ETF universe metadata used by factor neutralisation and portfolio constraints."""

from __future__ import annotations

from typing import Dict


# Broad economic groups deliberately contain several related ETFs.  The grouping is
# used to remove broad-industry bets from cross-sectional factor scores and to stop
# the backtest from filling every slot with highly correlated themes.
CODE_INDUSTRY_GROUP: Dict[str, str] = {
    "159326": "advanced_manufacturing",
    "588170": "technology",
    "513090": "financials",
    "159206": "advanced_manufacturing",
    "515880": "technology",
    "159869": "technology",
    "516150": "materials",
    "562950": "advanced_manufacturing",
    "562500": "advanced_manufacturing",
    "515220": "energy_materials",
    "515790": "clean_energy",
    "512660": "advanced_manufacturing",
    "159566": "technology",
    "515210": "materials",
    "159611": "utilities",
    "512690": "consumer",
    "159930": "energy_materials",
    "560280": "advanced_manufacturing",
    "512800": "financials",
    "159851": "financials",
    "513120": "healthcare",
    "513050": "technology",
    "159667": "advanced_manufacturing",
    "159259": "technology",
    "159996": "consumer",
    "518880": "precious_metals",
    "510300": "broad_market",
}


KEYWORD_INDUSTRY_GROUP = {
    "银行": "financials",
    "证券": "financials",
    "金融": "financials",
    "保险": "financials",
    "芯片": "technology",
    "半导体": "technology",
    "通信": "technology",
    "计算机": "technology",
    "软件": "technology",
    "游戏": "technology",
    "互联网": "technology",
    "人工智能": "technology",
    "机器人": "advanced_manufacturing",
    "军工": "advanced_manufacturing",
    "机床": "advanced_manufacturing",
    "制造": "advanced_manufacturing",
    "光伏": "clean_energy",
    "新能源": "clean_energy",
    "电池": "clean_energy",
    "煤炭": "energy_materials",
    "能源": "energy_materials",
    "钢铁": "materials",
    "有色": "materials",
    "稀土": "materials",
    "黄金": "precious_metals",
    "医药": "healthcare",
    "医疗": "healthcare",
    "创新药": "healthcare",
    "消费": "consumer",
    "白酒": "consumer",
    "家电": "consumer",
    "食品": "consumer",
    "电力": "utilities",
    "公用": "utilities",
}


def industry_group(code: str, name: str = "") -> str:
    """Return a stable broad-industry group for neutralisation and risk limits."""
    code_value = str(code)
    if code_value in CODE_INDUSTRY_GROUP:
        return CODE_INDUSTRY_GROUP[code_value]
    name_value = str(name)
    for keyword, group in KEYWORD_INDUSTRY_GROUP.items():
        if keyword in name_value:
            return group
    return "other"


__all__ = ["CODE_INDUSTRY_GROUP", "industry_group"]
