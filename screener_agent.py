import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
from pytdx.hq import TdxHq_API

SYSTEM_PROMPT = """你是一个选标Agent（ScreenerAgent），负责从输入的ETF列表中筛选“趋势突破观察候选”。

你的职责：
- 基于输入的结构化ETF特征数据，筛选值得观察的趋势突破候选（WATCH）
- 输出标准JSON结果，包含候选代码、推荐系数、观察等级、选择原因、风险标签
- 不输出交易指令（BUY/SELL）、仓位建议、止损止盈

你必须遵守以下规则：
1) 只能使用输入中提供的数据字段进行判断，不得编造任何数据。
2) 只输出 WATCH 候选，不输出 BUY / SELL / 仓位 / 止损止盈。
3) 必须严格输出 JSON，不能输出 JSON 之外的文字。
4) 如果没有符合条件的候选，必须返回合法空结果，并填写 no_candidate_reason。
5) selection_reasons 必须基于输入字段事实，简洁、可验证，不得使用“感觉”“猜测”“可能会涨”等主观措辞。
6) risk_tags 必须是数组，可为空数组 []。
7) candidates 必须按 recommendation_score 从高到低排序。
8) 输出数量不得超过 target_shortlist_size。
9) 如果 market_regime 是 RANGE 或 CHAOTIC，应提高筛选标准，谨慎输出候选。
10) 如果 data_status 非 OK，应降级或剔除，并在 risk_tags 或 no_candidate_reason 中体现。

筛选偏好（趋势突破观察）：
- 优先：流动性好、近期涨幅更强、量能放大、位于VWAP上方、接近日内高点、波动不过度失控
- 降级/剔除：数据异常、走势弱、无量、远离高点、不在VWAP上方、噪声过大

输出JSON字段要求：
顶层字段：
- cycle_ts
- market_regime
- candidates
- no_candidate_reason（可选）

每个 candidate 字段：
- symbol
- name
- group
- recommendation_score（0-100）
- watch_level（WATCH_LOW / WATCH_MID / WATCH_HIGH）
- selection_reasons（2-4条）
- risk_tags（数组）
"""


USER_PROMPT_TEMPLATE = """请基于下面输入数据，执行“黄金ETF / 跨境ETF 的趋势突破观察候选筛选”。

要求：
- 只输出 WATCH 候选，不输出交易指令
- 输出数量不超过 target_shortlist_size
- 严格输出 JSON（不要输出额外说明文字）

输入数据（JSON）：
{payload_json}
"""



# ====== 你自己的ETF池（先手工维护，最简单） ======
ETF_UNIVERSE = [
    # 黄金ETF（示例，按你的券商常见代码可再调整）
    {"symbol": "518880", "name": "黄金ETF", "group": "gold_etf"},
    {"symbol": "518800", "name": "黄金基金ETF", "group": "gold_etf"},
    {"symbol": "159934", "name": "黄金ETF", "group": "gold_etf"},

    # 跨境ETF（示例，宽基优先；你后续可替换）
    {"symbol": "513500", "name": "标普500ETF", "group": "cross_border_etf"},
    {"symbol": "513100", "name": "纳指ETF", "group": "cross_border_etf"},
    {"symbol": "513050", "name": "中概互联网ETF", "group": "cross_border_etf"},
    {"symbol": "513330", "name": "恒生互联网ETF", "group": "cross_border_etf"},
    {"symbol": "159866", "name": "日经ETF", "group": "cross_border_etf"},
]

# 常见可用通达信行情服务器（可轮询/切换）
TDX_SERVERS = [
    ("119.147.212.81", 7709),
    ("119.147.212.83", 7709),
    ("60.191.117.167", 7709),
    ("113.105.73.88", 7709),
]


def infer_market(symbol: str) -> int:
    """
    通达信 market 参数:
    0 = 深圳, 1 = 上海
    常见规则：
    - 上交所ETF多为 5xxxxx -> 上海
    - 深交所ETF多为 1xxxxx / 15xxxx / 16xxxx / 18xxxx -> 深圳
    """
    s = str(symbol)
    if s.startswith("5"):
        return 1  # 上海
    return 0  # 深圳


def connect_tdx() -> Tuple[TdxHq_API, Tuple[str, int]]:
    api = TdxHq_API()
    for host, port in TDX_SERVERS:
        try:
            ok = api.connect(host, port)
            if ok:
                return api, (host, port)
        except Exception:
            continue
    raise RuntimeError("无法连接到任何 TDX 行情服务器")


def fetch_bars_df(api: TdxHq_API, symbol: str, market: int, category: int, count: int = 120) -> pd.DataFrame:
    """
    category:
      7 -> 1分钟K线
      0 -> 5分钟K线
    参考 pytdx 标准行情文档（get_security_bars 支持分钟K线）.
    """
    # start=0 表示从最新开始取 count 根
    raw = api.get_security_bars(category, market, symbol, 0, count)
    if raw is None:
        return pd.DataFrame()
    df = api.to_df(raw)
    if df is None or df.empty:
        return pd.DataFrame()

    # 标准化列
    # pytdx常见列包含: open, close, high, low, vol, amount, year, month, day, hour, minute, datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        # 兜底拼时间（较少用到）
        if all(c in df.columns for c in ["year", "month", "day", "hour", "minute"]):
            df["datetime"] = pd.to_datetime(
                df[["year", "month", "day", "hour", "minute"]].assign(second=0)
            )
        else:
            # 实在没有就给顺序时间索引
            df["datetime"] = pd.RangeIndex(start=0, stop=len(df), step=1)

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def safe_pct_change(new: float, old: float) -> float:
    if old is None or old == 0 or pd.isna(old) or pd.isna(new):
        return 0.0
    return (float(new) / float(old) - 1.0) * 100.0


def calc_vwap(df_1m: pd.DataFrame) -> Optional[float]:
    if df_1m.empty:
        return None
    if "amount" in df_1m.columns and "vol" in df_1m.columns:
        # amount / vol 的单位在不同品种/接口可能有差异，但用于“是否高于VWAP”判断通常足够
        vol_sum = df_1m["vol"].replace(0, pd.NA).dropna().sum()
        amt_sum = df_1m["amount"].dropna().sum()
        if vol_sum and vol_sum > 0:
            return float(amt_sum / vol_sum)
    # 兜底：用收盘价均值近似
    return float(df_1m["close"].mean()) if "close" in df_1m.columns else None


def level_from_value(value: float, low_thr: float, high_thr: float) -> str:
    if value >= high_thr:
        return "HIGH"
    if value >= low_thr:
        return "MEDIUM"
    return "LOW"


def build_symbol_features_from_bars(symbol_info: Dict[str, str], df_1m: pd.DataFrame) -> Dict[str, Any]:
    symbol = symbol_info["symbol"]
    name = symbol_info["name"]
    group = symbol_info["group"]

    if df_1m.empty or len(df_1m) < 35:
        return {
            "symbol": symbol,
            "name": name,
            "group": group,
            "data_status": "BAD",
            "reason": "insufficient_1m_bars"
        }

    close = df_1m["close"].astype(float)
    high = df_1m["high"].astype(float)
    vol = df_1m["vol"].astype(float) if "vol" in df_1m.columns else pd.Series([0.0] * len(df_1m))

    latest_close = float(close.iloc[-1])

    # 15m / 30m 涨跌幅（用 1m bars）
    pct_change_15m = safe_pct_change(close.iloc[-1], close.iloc[-16] if len(close) >= 16 else close.iloc[0])
    pct_change_30m = safe_pct_change(close.iloc[-1], close.iloc[-31] if len(close) >= 31 else close.iloc[0])

    # 量能放大倍数：最近5根 vs 前30根均值（可按你后续再调）
    recent_vol_mean = float(vol.tail(5).mean()) if len(vol) >= 5 else 0.0
    hist_slice = vol.iloc[-35:-5] if len(vol) >= 35 else vol.iloc[:-5]
    hist_vol_mean = float(hist_slice.mean()) if len(hist_slice) > 0 else 0.0
    volume_spike_ratio = round((recent_vol_mean / hist_vol_mean), 2) if hist_vol_mean > 0 else 0.0

    # VWAP
    vwap = calc_vwap(df_1m)
    above_vwap = bool(vwap is not None and latest_close > vwap)

    # 距离当日高点百分比（越小越接近突破位）
    day_high = float(high.max())
    distance_to_day_high_pct = round(max(0.0, (day_high - latest_close) / day_high * 100.0), 2) if day_high > 0 else 999.0

    # 波动水平（用近30根1m收益率标准差粗略分层）
    ret_1m = close.pct_change().dropna()
    vol_std = float(ret_1m.tail(30).std() * 100.0) if len(ret_1m) >= 5 else 0.0
    if vol_std >= 0.6:
        volatility_level = "HIGH"
    elif vol_std >= 0.2:
        volatility_level = "MEDIUM"
    else:
        volatility_level = "LOW"

    # 流动性水平（先用近30根成交量均值粗略分层，后续你可改成 amount）
    vol_mean_30 = float(vol.tail(30).mean()) if len(vol) >= 30 else float(vol.mean())
    if vol_mean_30 >= 30000:
        liquidity_level = "HIGH"
    elif vol_mean_30 >= 8000:
        liquidity_level = "MEDIUM"
    else:
        liquidity_level = "LOW"

    return {
        "symbol": symbol,
        "name": name,
        "group": group,
        "pct_change_15m": round(pct_change_15m, 2),
        "pct_change_30m": round(pct_change_30m, 2),
        "volume_spike_ratio": volume_spike_ratio,
        "above_vwap": above_vwap,
        "distance_to_day_high_pct": distance_to_day_high_pct,
        "volatility_level": volatility_level,
        "liquidity_level": liquidity_level,
        "data_status": "OK",
        # 可选调试字段（给你后续排查）
        "_debug": {
            "latest_close": round(latest_close, 4),
            "vwap": round(vwap, 4) if vwap is not None else None,
            "day_high": round(day_high, 4),
            "vol_std_pct_1m_30": round(vol_std, 4),
            "vol_mean_30": round(vol_mean_30, 2),
        }
    }


def infer_market_regime_simple(symbol_features: List[Dict[str, Any]]) -> str:
    """
    超简版 Regime（Lite版够用）
    逻辑：看有效样本的 30m 涨幅中位数 + 强势数量
    """
    valid = [x for x in symbol_features if x.get("data_status") == "OK"]
    if len(valid) < 3:
        return "CHAOTIC"

    vals = [x["pct_change_30m"] for x in valid]
    median_30m = pd.Series(vals).median()

    strong_count = sum(
        1 for x in valid
        if x["pct_change_30m"] > 0.5 and x["above_vwap"] and x["volume_spike_ratio"] >= 1.2
    )

    if median_30m > 0.4 and strong_count >= max(2, len(valid) // 4):
        return "TREND_UP"
    if median_30m < -0.4:
        return "TREND_DOWN"
    if abs(median_30m) <= 0.2:
        return "RANGE"
    return "CHAOTIC"


def build_real_payload_from_pytdx(target_shortlist_size: int = 5) -> Dict[str, Any]:
    tz = timezone(timedelta(hours=8))
    cycle_ts = datetime.now(tz=tz).replace(microsecond=0).isoformat()

    api, server = connect_tdx()
    symbol_features: List[Dict[str, Any]] = []

    try:
        for item in ETF_UNIVERSE:
            symbol = item["symbol"]
            market = infer_market(symbol)

            try:
                # 1分钟K线，取120根（够算30分钟特征+留余量）
                df_1m = fetch_bars_df(api, symbol=symbol, market=market, category=7, count=120)
                feat = build_symbol_features_from_bars(item, df_1m)
            except Exception as e:
                feat = {
                    "symbol": item["symbol"],
                    "name": item["name"],
                    "group": item["group"],
                    "data_status": "BAD",
                    "reason": f"fetch_or_feature_error: {type(e).__name__}"
                }

            symbol_features.append(feat)
    finally:
        try:
            api.disconnect()
        except Exception:
            pass

    market_regime = infer_market_regime_simple(symbol_features)

    payload = {
        "cycle_ts": cycle_ts,
        "target_shortlist_size": target_shortlist_size,
        "market_regime": market_regime,
        "source": {
            "provider": "pytdx",
            "server": f"{server[0]}:{server[1]}"
        },
        "symbols": symbol_features
    }
    return payload

def call_model(system_prompt: str, user_prompt: str) -> str:
    """
    这里替换成你的实际模型调用代码。
    目前返回一个示例JSON字符串，方便你先跑通。
    """
    # TODO: 替换为你的API调用，例如 OpenAI / 本地模型 / OpenClaw 调用
    # 返回值必须是字符串（模型原始输出）
    mocked_output = {
        "cycle_ts": json.loads(user_prompt.split("输入数据（JSON）：\n", 1)[1])["cycle_ts"],
        "market_regime": "TREND_UP",
        "candidates": [
            {
                "symbol": "518880",
                "name": "黄金ETF",
                "group": "gold_etf",
                "recommendation_score": 87,
                "watch_level": "WATCH_HIGH",
                "selection_reasons": [
                    "近15分钟与30分钟涨幅在样本中较强",
                    "成交量放大倍数较高，具备量价共振特征",
                    "价格位于VWAP上方，走势相对稳健",
                    "接近日内高点，具备突破观察价值"
                ],
                "risk_tags": ["BREAKOUT_NOT_CONFIRMED"]
            },
            {
                "symbol": "513100",
                "name": "纳指ETF",
                "group": "cross_border_etf",
                "recommendation_score": 80,
                "watch_level": "WATCH_HIGH",
                "selection_reasons": [
                    "近期涨幅表现较强，趋势方向较一致",
                    "成交量较近期有放大，具备一定突破关注价值",
                    "价格位于VWAP上方，结构相对健康"
                ],
                "risk_tags": ["RANGE_REGIME_HIGHER_THRESHOLD"]
            },
            {
                "symbol": "518800",
                "name": "黄金基金ETF",
                "group": "gold_etf",
                "recommendation_score": 69,
                "watch_level": "WATCH_MID",
                "selection_reasons": [
                    "近期涨幅为正，且位于VWAP上方",
                    "量能有一定放大，但强度弱于头部候选",
                    "距离日内高点较近，仍可纳入观察"
                ],
                "risk_tags": ["BREAKOUT_NOT_CONFIRMED"]
            }
        ]
    }
    return json.dumps(mocked_output, ensure_ascii=False)


def try_parse_json(text: str) -> Dict[str, Any]:
    """尝试从模型输出中解析JSON。"""
    text = text.strip()
    # 常见情况：模型前后多了说明文字，尝试截取最外层JSON
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]

    return json.loads(text)


def validate_result(result: Dict[str, Any], payload: Dict[str, Any]) -> None:
    """最小字段校验（Lite版）。"""
    required_top = ["cycle_ts", "market_regime", "candidates"]
    for k in required_top:
        if k not in result:
            raise ValueError(f"Missing top-level field: {k}")

    if not isinstance(result["candidates"], list):
        raise ValueError("candidates must be a list")

    # 数量限制
    max_n = payload.get("target_shortlist_size", 10)
    if len(result["candidates"]) > max_n:
        raise ValueError(f"candidates exceeds target_shortlist_size: {len(result['candidates'])} > {max_n}")

    allowed_watch = {"WATCH_LOW", "WATCH_MID", "WATCH_HIGH"}

    for i, c in enumerate(result["candidates"]):
        for k in ["symbol", "name", "group", "recommendation_score", "watch_level", "selection_reasons", "risk_tags"]:
            if k not in c:
                raise ValueError(f"Candidate[{i}] missing field: {k}")

        if c["watch_level"] not in allowed_watch:
            raise ValueError(f"Candidate[{i}] invalid watch_level: {c['watch_level']}")

        if not isinstance(c["selection_reasons"], list):
            raise ValueError(f"Candidate[{i}] selection_reasons must be list")

        if not isinstance(c["risk_tags"], list):
            raise ValueError(f"Candidate[{i}] risk_tags must be list")

        score = c["recommendation_score"]
        if not isinstance(score, (int, float)):
            raise ValueError(f"Candidate[{i}] recommendation_score must be number")
        if score < 0 or score > 100:
            raise ValueError(f"Candidate[{i}] recommendation_score out of range: {score}")

    # 排序校验（降序）
    scores = [c["recommendation_score"] for c in result["candidates"]]
    if scores != sorted(scores, reverse=True):
        raise ValueError("candidates are not sorted by recommendation_score descending")


def save_result(result: Dict[str, Any], path: str = "screening_result.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main() -> None:
    payload = build_real_payload_from_pytdx(target_shortlist_size=5)
    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)

    user_prompt = USER_PROMPT_TEMPLATE.format(payload_json=payload_json)

    # 1) 调模型
    raw_output = call_model(SYSTEM_PROMPT, user_prompt)

    # 2) 解析与校验
    try:
        result = try_parse_json(raw_output)
        validate_result(result, payload)
    except Exception as e:
        print("模型输出解析/校验失败：", e)
        print("原始输出：")
        print(raw_output)
        raise

    # 3) 保存结果
    save_result(result, "screening_result.json")
    print("✅ 已生成 screening_result.json")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()