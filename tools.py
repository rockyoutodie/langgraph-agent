"""
工具集(复用第 4 周的思路,简化版)
"""

import math
import json


def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        allowed = {"sqrt": math.sqrt, "pow": pow, "pi": math.pi, "e": math.e}
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"计算错误: {e}"


def get_weather(city: str) -> str:
    """查询天气(模拟)"""
    data = {"北京": "晴, 8°C", "上海": "多云, 15°C", "广州": "小雨, 22°C"}
    return data.get(city, f"没有 {city} 的天气数据")


def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """查询汇率(模拟)"""
    rates = {("USD","CNY"): 7.18, ("EUR","CNY"): 7.85, ("CNY","USD"): 0.139}
    rate = rates.get((from_currency.upper(), to_currency.upper()))
    return f"1 {from_currency.upper()} = {rate} {to_currency.upper()}" if rate else "无此汇率"


# 工具 Schema (OpenAI 格式)
TOOLS_SCHEMA = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "精确计算数学表达式。任何数学运算都用它,不要心算。支持 sqrt。",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "数学表达式,如 '50 * 7.18'"}
        }, "required": ["expression"]}
    }},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查询城市天气。",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "城市名"}
        }, "required": ["city"]}
    }},
    {"type": "function", "function": {
        "name": "get_exchange_rate",
        "description": "查询两种货币的汇率。",
        "parameters": {"type": "object", "properties": {
            "from_currency": {"type": "string", "description": "源货币,如 USD"},
            "to_currency": {"type": "string", "description": "目标货币,如 CNY"}
        }, "required": ["from_currency", "to_currency"]}
    }},
]

# 函数映射
AVAILABLE_FUNCTIONS = {
    "calculator": calculator,
    "get_weather": get_weather,
    "get_exchange_rate": get_exchange_rate,
}