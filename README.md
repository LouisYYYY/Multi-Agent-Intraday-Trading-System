# Multi-Agent-Intraday-Trading-System
本项目是一个运行在 NAS（Docker）环境中的 多-Agent（Multi-Agent）日内交易系统，用于将实时/准实时市场数据转化为明确、可执行、可分级的交易信号，并支持快速调整策略参数与交易规则。
1. 项目定位

本项目是一个运行在 NAS（Docker）环境中的 多-Agent（Multi-Agent）日内交易系统，用于将实时/准实时市场数据转化为明确、可执行、可分级的交易信号，并支持快速调整策略参数与交易规则。

核心目标：

在日内频率（分钟级或更高频）生成交易信号

信号分级清晰：买入/卖出各三档（Low / Mid / Strong）

规则可配置、可快速迭代（两人小团队可维护）

过程可追溯：每次信号与关键依据落盘记录

2. 输出产物（你最终拿到什么）

系统每个决策周期输出一份标准化结果（JSON/日志/可选通知）：

交易动作（Action）：BUY / SELL / WAIT

信号等级（Grade）：LOW / MID / STRONG

交易计划（Trade Plan）（可选）：入场条件、退出条件、建议仓位、止损/止盈、超时退出

关键依据（Rationale）：以要点形式列出触发因子与风险门控结果

审计日志（Journal）：持久化记录每次输入、因子、分数、决策与风控处理

说明：MVP 可仅输出信号与日志；自动下单属于可选功能模块。

3. 系统组成（模块划分）
3.1 交易池管理（Universe Management）

负责确定“今天/当前窗口值得交易的标的集合”，减少噪声与资源消耗，提升日内执行质量。

输入：市场范围、筛选阈值（流动性/点差/波动/事件风险）

输出：universe（建议 ≤ 10–30 个标的）

3.2 数据接入（Market Data Ingestion）

负责从数据源持续拉取/订阅市场数据并标准化。

MVP：分钟级 K 线（1m/5m bars）+ 基本报价（L1）

增强：盘口深度（L2）、逐笔成交（ticks）、公告/事件流（filings/events）

3.3 因子与特征（Factor & Feature Engineering）

将原始数据加工为决策所需的指标向量，例如：

日内趋势/动量（Intra-day Trend/Momentum）

波动与回撤（Volatility/Drawdown）

VWAP 偏离、成交量异动（VWAP Deviation/Volume Spike）
-（可选）微观结构特征：点差、盘口不平衡（Spread/Order Book Imbalance）

3.4 决策与信号分级（Decision & Signal Grading）

把因子向量映射为可执行信号：

计算分数 S ∈ [-100, 100]

映射为 BUY/SELL/WAIT + LOW/MID/STRONG

生成可选交易计划（仓位、止损止盈、退出规则）

3.5 风控门控（Risk Gating）

对所有候选信号做硬约束检查与降级/拦截，确保日内交易可控：

最大仓位、最大订单频率

连续亏损熔断（Kill Switch）

重大事件窗口（财报/公告）降级或暂停

数据异常/延迟异常的自动停机与告警

3.6 执行与订单管理（Execution & Order Management，可选）

将批准后的信号转为订单计划并管理生命周期（下单/撤单/改单/超时退出）。
MVP 可先不启用自动执行，仅输出信号与建议。

3.7 复盘与调参（Review & Tuning）

对交易与信号表现做轻量统计，并支持可控调参：

记录胜率、平均盈亏、滑点、撤单率、策略触发频率等

允许在白名单内调整策略阈值与权重，并保留变更记录与回滚点

4. Agent 列表（专业命名与职责边界）

为保持小巧与可落地，项目默认采用 4 个核心 Agent（足以覆盖“选标的→因子→信号→风控”闭环）：

A1. 交易池 Agent — ScreenerAgent

职责：从全量标的中筛出适合日内交易的候选池（流动性/点差/波动/事件风险）

输入：市场范围、筛选阈值、候选上限

输出：Shortlist（≤ N 个 symbols + 核心筛选指标）

A2. 因子 Agent — FactorAgent

职责：对候选池计算/汇总关键因子（技术因子为主，后续可加入财务/事件标签）

输入：市场数据（bars/L1/L2）、候选池

输出：FactorVector（每个 symbol 一组结构化因子：value/normalized/confidence）

A3. 决策 Agent — DecisionAgent

职责：基于因子与策略配置生成信号与交易计划，并完成信号分级（Low/Mid/Strong）

输入：FactorVector + Strategy Config（权重、阈值、规则）

输出：TradeSignal（action+grade+entry/exit+size+reasons）

A4. 风控 Agent — RiskAgent

职责：硬风控门控：批准/降级/拦截交易信号，必要时触发熔断

输入：TradeSignal + 账户/持仓状态 + 风险配置

输出：ApprovedSignal（可能被降级为 WAIT 或降低仓位）

可选扩展（不影响 MVP）：

ExecutionAgent：订单拆分/撤单/追单

JudgeAgent：绩效评估与异常告警

TunerAgent：在白名单内自动调参（带审计与回滚）

5. 信号分级规范（业务口径）

系统输出的“低/中/强”必须可解释、可配置、可审计：

LOW（低等级）：边际优势较弱或一致性不足；允许小仓位试探或仅挂被动单

MID（中等级）：多个因子/策略一致，且流动性/风险条件良好；可按基准仓位执行

STRONG（强等级）：一致性强 + 风险门控通过 + 关键条件满足；允许加仓或更积极的执行策略

卖出信号同理（对称规则）。

6. 配置与可迭代点（面向小团队）

系统将“可变内容”集中在少量配置中，便于快速改策略：

Strategy Config：因子权重、阈值、入场/退出规则（可频繁调整）

Risk Config：最大仓位、熔断规则、事件窗口处理（谨慎调整）

Universe Config：候选池规模、筛选标准（适度调整）

所有配置变更必须记录到日志，以支持回滚与复盘。

7. MVP 范围与演进路线

MVP（最快跑起来）：

频率：1m/5m bars

标的：≤ 10–30

Agent：ScreenerAgent + FactorAgent + DecisionAgent + RiskAgent

输出：分级信号 + journal 日志

执行：可先关闭自动下单，仅输出建议

演进（逐步增强）：

引入 L2/逐笔，提高日内信号质量

引入公告/事件源，强化风险过滤

加 ExecutionAgent，提升成交率与滑点控制

加 Judge/Tuner，形成更自动化的优化闭环

NAS Multi-Agent Intraday Trading System (Business README)

English version follows the Chinese one. This README describes what the system does, its modules and agents, and what outputs it produces. Deployment and implementation details are documented separately in the developer documentation.

1. Overview

This project is a Multi-Agent Intraday Trading System designed to run on a NAS (Docker). It transforms market data into clear, actionable, graded trading signals and supports fast iteration of strategy parameters for small teams.

2. What You Get

On each decision cycle, the system produces:

Action: BUY / SELL / WAIT

Grade: LOW / MID / STRONG

Optional Trade Plan: entry/exit, sizing, stop loss/take profit, time-based exits

Rationale: concise bullet points of key drivers and risk gates

Journal: persistent logs for audit and review

3. Major Components

Universe Management: selects an intraday-friendly tradable set (liquidity/spread/volatility/event risk)

Market Data Ingestion: consumes bars (MVP), optionally L1/L2/ticks and event streams

Factor & Feature Engineering: produces intraday features (trend, momentum, volatility, VWAP deviation, etc.)

Decision & Signal Grading: maps features into S ∈ [-100, 100] and grades signals

Risk Gating: hard constraints, downgrades, kill switch, event windows

Optional Execution & OMS: converts approved signals into order plans and manages orders

Review & Tuning: lightweight performance metrics and controlled parameter updates

4. Core Agents (MVP)

ScreenerAgent: builds the tradable shortlist

FactorAgent: computes/aggregates factor vectors

DecisionAgent: generates graded trade signals and optional trade plans

RiskAgent: approves/downgrades/blocks signals and triggers kill switches

Optional extensions: ExecutionAgent, JudgeAgent, TunerAgent.

5. Signal Grades

LOW: weak edge or low agreement; small sizing / passive orders

MID: good agreement and conditions; baseline sizing

STRONG: strong agreement + risk gates passed; allows more aggressive execution

6. Configuration & Iteration

Strategy and risk parameters are centralized into a small set of configs (strategy/risk/universe). All changes are journaled for audit and rollback.

7. MVP Scope & Roadmap

MVP starts with 1m/5m bars, a small universe, 4 core agents, and signal + journal outputs; then evolves toward L2/ticks, event risk filters, execution optimization, and automated tuning.
