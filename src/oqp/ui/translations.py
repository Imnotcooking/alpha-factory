"""Shared bilingual text catalogs for Streamlit dashboards."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


LANGUAGE_ALIASES = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "EN": "en",
    "zh": "zh",
    "ZH": "zh",
    "cn": "zh",
    "CN": "zh",
    "chinese": "zh",
    "中文": "zh",
}


OPS_TEXT: dict[str, dict[str, Any]] = {
    "en": {
        "language_label": "Language / 语言",
        "language_caption": "Dashboard language",
        "checked_at": "Checked at {time}",
        "server_runtime_sync": "Server runtime sync: {synced_at} from {remote}",
        "overall": "Overall",
        "failures": "Failures",
        "warnings": "Warnings",
        "live_nav": "Live NAV",
        "paper_nav": "Paper NAV",
        "dry_run_tickets": "Dry-Run Tickets",
        "paper_submit": "Paper Submit",
        "action_items": "Action Items",
        "paper_submission_armed": "Paper broker submission is armed. Live trading remains disabled.",
        "live_trading_enabled": "Live trading is enabled in settings. Treat this dashboard as unsafe until reviewed.",
        "live_locked": "Live trading is locked. Paper trading can be reviewed here; paper submit stays separately gated.",
        "no_action_rows": "No action rows available.",
        "no_immediate_action": "No immediate action is waiting.",
        "no_pipeline_rows": "No pipeline rows available.",
        "no_policy_rows": "No policy rows available.",
        "no_account_summary_rows": "No account summary rows available.",
        "no_system_summary_rows": "No system summary rows available.",
        "all_ops_passing": "All collected ops checks are passing.",
        "unified_account_ledger": "Unified account ledger: {path}",
        "paper_submission_armed_readonly": "Paper broker submission is armed. This page is still read-only, but submitter jobs can send approved tickets.",
        "paper_submit_locked": "Paper submit is locked. Reviews and dry-run tickets can be monitored without sending broker orders.",
        "nav": "NAV",
        "cash": "Cash",
        "total_cash": "Total Cash",
        "daily_pnl": "Daily P&L",
        "live_daily_pnl": "Live Daily P&L",
        "paper_daily_pnl": "Paper Daily P&L",
        "live_pnl": "Live P&L",
        "paper_pnl": "Paper P&L",
        "unrealized_pnl": "Unrealized P&L",
        "positions": "Positions",
        "gross_nav": "Gross Exposure / NAV",
        "max_dd": "Max DD",
        "live_drawdown": "Live Drawdown",
        "paper_drawdown": "Paper Drawdown",
        "live_gross_nav": "Live Gross Exposure / NAV",
        "risk_flags": "Risk Flags",
        "snapshot": "Snapshot",
        "account": "Account",
        "review_gate": "Review Gate",
        "submit_gate": "Submit Gate",
        "daily_notional": "Daily Notional",
        "open": "open",
        "locked": "locked",
        "armed": "armed",
        "active": "active",
        "missing": "missing",
        "present": "present",
        "proposal_reviews": "proposal reviews",
        "broker_submit": "broker submit",
        "entries_today": "Entries Today",
        "all_entries": "All Entries",
        "paper_tickets": "Paper Tickets",
        "reviews": "Reviews",
        "journal_ledger": "Journal ledger: `{path}`",
        "daily_notes": "Daily Notes",
        "entries_for_selected_date": "Entries For Selected Date",
        "trade_thesis_log": "Trade Thesis Log",
        "end_of_day_summary": "End-of-Day Summary",
        "pnl_digest": "P&L Digest",
        "daily_activity_counts": "Daily Activity Counts",
        "mistake_review": "Mistake Review",
        "exported_reports": "Exported Reports And Runtime Artifacts",
        "manual_report_reference": "Manual Report Reference",
        "saved_report_references": "Saved Report References",
        "date": "Date",
        "environment": "Environment",
        "symbols": "Symbols",
        "tags": "Tags",
        "title": "Title",
        "daily_note_default": "Daily note",
        "daily_note_body": "What happened / what am I watching?",
        "follow_ups": "Follow-ups",
        "save_daily_note": "Save Daily Note",
        "symbol": "Symbol",
        "strategy_setup_id": "Strategy / setup ID",
        "direction": "Direction",
        "horizon": "Horizon",
        "status": "Status",
        "confidence": "Confidence",
        "thesis_title": "Thesis title",
        "thesis": "Thesis",
        "invalidation": "Invalidation / what proves me wrong",
        "expected_outcome": "Expected outcome / target behavior",
        "risk_sizing_notes": "Risk / sizing notes",
        "save_thesis": "Save Thesis",
        "generated_digest": "Generated digest",
        "summary_title": "Summary title",
        "human_reflection": "Human reflection / lessons from today",
        "tomorrow_follow_ups": "Tomorrow follow-ups",
        "save_eod_summary": "Save EOD Summary",
        "severity": "Severity",
        "mistake_title": "Mistake title",
        "what_went_wrong": "What went wrong?",
        "what_did_i_learn": "What did I learn?",
        "rule_process_change": "Rule/process change",
        "save_mistake_review": "Save Mistake Review",
        "report_title": "Report title",
        "path_or_url": "Path or URL",
        "why_report_matters": "Why this report matters",
        "save_report_reference": "Save Report Reference",
        "no_journal_entries": "No journal entries saved for this date yet.",
        "no_trade_thesis": "No trade thesis entries saved yet.",
        "no_mistake_reviews": "No mistake reviews saved yet.",
        "no_report_files": "No report/export files found under reports, runtime/exports, runtime/artifacts, or logs.",
        "no_report_references": "No report references saved yet. Discord daily summaries are posted externally today; this tab is ready to archive references when we start saving them locally.",
        "journal_tabs": [
            "Daily Journal",
            "Trade Thesis",
            "EOD Digest",
            "Mistakes & Lessons",
            "Reports Archive",
        ],
        "live_tabs": [
            "Overview",
            "Holdings",
            "Exposure",
            "Options Hub",
            "Reconciliation",
        ],
        "option_tabs": [
            "Book Audit",
            "Position Lab",
            "Vol & Model Audit",
        ],
        "paper_tabs": [
            "Overview",
            "Account & Risk",
            "Strategies",
            "Reviews",
            "Tickets",
            "Ledger",
            "Health",
        ],
        "risk_tabs": [
            "Overview",
            "Exposure & Concentration",
            "Drawdown & P&L",
            "Options & Greeks",
            "Stress Tests",
            "Allocation Lab",
        ],
        "execution_tabs": ["Command", "Strategies", "Orders", "Signals", "Boundaries"],
        "workbench_tabs": [
            "Overview",
            "Valuation",
            "Options",
            "News",
            "AI",
            "Decision",
        ],
        "workbench_nav_views": ["Market Monitor", "Watchlist", "Opportunity Hub", "API Status"],
        "workbench_option_tabs": [
            "Playbook",
            "Strategy Fit",
            "Scanner",
            "Payoff Lab",
            "Vol Models",
        ],
        "valuation_tabs": ["DCF", "Targets", "Peers", "Supply Chain", "Checklist"],
        "directional_lens_tabs": [
            "Horizon Summary",
            "Signal Contributions",
            "Sentiment Rows",
        ],
        "option_scanner_tabs": [
            "Long Calls",
            "Long Puts",
            "Cash-Secured Puts",
            "Verticals",
            "Calendars",
            "Condors",
            "Butterflies",
            "Ratios",
            "Backspreads",
        ],
        "strategy_fit_tabs": [
            "Strategy Fit",
            "Directional Lens",
            "Scanner",
            "Payoff Lab",
        ],
        "builder_tabs": ["Builder", "Saved Ideas"],
        "save_rows": "Save Rows",
        "reset_starter": "Reset Starter",
        "stock_valuation": "Stock Valuation",
        "valuation_ticker": "Valuation ticker",
        "load_valuation": "Load Valuation",
        "choose_ticker_load": "Choose a ticker and load valuation data.",
        "loading_fundamentals": "Loading {symbol} fundamentals and price history...",
        "valuation_snapshot_missing": "Could not load a usable valuation snapshot.",
        "checklist_unavailable": "Checklist data unavailable.",
        "write_draft_proposal": "Write Draft Paper Proposal",
        "proposal_slot_caption": "This only writes a proposal artifact. It does not approve, route, or submit an order.",
        "contracts_per_leg": "Contracts per leg unit",
        "candidate": "Candidate",
        "write_draft_button": "Write Draft Proposal",
        "could_not_write_proposal": "Could not write proposal: {error}",
        "draft_proposal_written": "Draft proposal written: {path}",
        "proposal_id": "Proposal ID: `{proposal_id}`",
        "save_idea": "Save Idea",
        "active_symbol": "Active Symbol",
        "watchlist": "Watchlist",
        "saved_ideas": "Saved Ideas",
        "option_package_leg": "Option package / leg",
        "greek_package_leg": "Greek package / leg",
        "registered": "Registered",
        "running": "Running",
        "kill_switches": "Kill Switches",
        "proposals": "Proposals",
        "ready_reviews": "Ready Reviews",
        "blocked_reviews": "Blocked Reviews",
        "execution_live_enabled": "Live trading is enabled. Treat this as a production boundary until reviewed.",
        "execution_paper_armed": "Paper submit is armed. Approved tickets may be eligible for guarded paper broker submission.",
        "execution_locked": "Live trading is locked and paper submit is locked. This page is monitoring proposals, tickets, and strategy state.",
    },
    "zh": {
        "language_label": "Language / 语言",
        "language_caption": "仪表盘语言",
        "checked_at": "检查时间：{time}",
        "server_runtime_sync": "服务器运行时同步：{synced_at}，来源：{remote}",
        "overall": "总体状态",
        "failures": "失败项",
        "warnings": "警告项",
        "live_nav": "实盘净值",
        "paper_nav": "模拟净值",
        "dry_run_tickets": "干跑票据",
        "paper_submit": "模拟提交",
        "action_items": "行动项",
        "paper_submission_armed": "模拟券商提交已武装；实盘交易仍保持关闭。",
        "live_trading_enabled": "配置中已开启实盘交易。复核前请将此仪表盘视为不安全状态。",
        "live_locked": "实盘交易已锁定。可在此复核模拟交易；模拟提交仍由独立闸门控制。",
        "no_action_rows": "暂无行动队列记录。",
        "no_immediate_action": "当前没有需要立即处理的事项。",
        "no_pipeline_rows": "暂无运营流水线记录。",
        "no_policy_rows": "暂无政策闸门记录。",
        "no_account_summary_rows": "暂无账户摘要记录。",
        "no_system_summary_rows": "暂无系统摘要记录。",
        "all_ops_passing": "所有已采集的运营检查均通过。",
        "unified_account_ledger": "统一账户账本：{path}",
        "paper_submission_armed_readonly": "模拟券商提交已武装。本页仍为只读，但提交任务可发送已批准票据。",
        "paper_submit_locked": "模拟提交已锁定。仍可监控复核与干跑票据，不会发送券商订单。",
        "nav": "净值",
        "cash": "现金",
        "total_cash": "总现金",
        "daily_pnl": "日内盈亏",
        "live_daily_pnl": "实盘日内盈亏",
        "paper_daily_pnl": "模拟日内盈亏",
        "live_pnl": "实盘盈亏",
        "paper_pnl": "模拟盈亏",
        "unrealized_pnl": "未实现盈亏",
        "positions": "持仓数",
        "gross_nav": "总敞口 / 净值",
        "max_dd": "最大回撤",
        "live_drawdown": "实盘回撤",
        "paper_drawdown": "模拟回撤",
        "live_gross_nav": "实盘总敞口 / 净值",
        "risk_flags": "风险标记",
        "snapshot": "快照",
        "account": "账户",
        "review_gate": "复核闸门",
        "submit_gate": "提交闸门",
        "daily_notional": "日度名义额",
        "open": "开放",
        "locked": "锁定",
        "armed": "已武装",
        "active": "启用",
        "missing": "缺失",
        "present": "存在",
        "proposal_reviews": "提案复核",
        "broker_submit": "券商提交",
        "entries_today": "今日记录",
        "all_entries": "全部记录",
        "paper_tickets": "模拟票据",
        "reviews": "复核",
        "journal_ledger": "日志账本：`{path}`",
        "daily_notes": "每日笔记",
        "entries_for_selected_date": "所选日期记录",
        "trade_thesis_log": "交易假设日志",
        "end_of_day_summary": "收盘总结",
        "pnl_digest": "盈亏摘要",
        "daily_activity_counts": "每日活动计数",
        "mistake_review": "错误复盘",
        "exported_reports": "导出报告与运行产物",
        "manual_report_reference": "手动报告引用",
        "saved_report_references": "已保存报告引用",
        "date": "日期",
        "environment": "环境",
        "symbols": "标的",
        "tags": "标签",
        "title": "标题",
        "daily_note_default": "每日笔记",
        "daily_note_body": "发生了什么 / 我正在观察什么？",
        "follow_ups": "后续事项",
        "save_daily_note": "保存每日笔记",
        "symbol": "标的",
        "strategy_setup_id": "策略 / 设置 ID",
        "direction": "方向",
        "horizon": "周期",
        "status": "状态",
        "confidence": "信心",
        "thesis_title": "假设标题",
        "thesis": "交易假设",
        "invalidation": "失效条件 / 什么能证明我错了",
        "expected_outcome": "预期结果 / 目标行为",
        "risk_sizing_notes": "风险 / 仓位笔记",
        "save_thesis": "保存交易假设",
        "generated_digest": "生成的摘要",
        "summary_title": "总结标题",
        "human_reflection": "人工复盘 / 今日经验",
        "tomorrow_follow_ups": "明日后续事项",
        "save_eod_summary": "保存收盘总结",
        "severity": "严重程度",
        "mistake_title": "错误标题",
        "what_went_wrong": "哪里出了问题？",
        "what_did_i_learn": "我学到了什么？",
        "rule_process_change": "规则 / 流程改动",
        "save_mistake_review": "保存错误复盘",
        "report_title": "报告标题",
        "path_or_url": "路径或 URL",
        "why_report_matters": "这份报告为何重要",
        "save_report_reference": "保存报告引用",
        "no_journal_entries": "所选日期尚无日志记录。",
        "no_trade_thesis": "尚无交易假设记录。",
        "no_mistake_reviews": "尚无错误复盘记录。",
        "no_report_files": "reports、runtime/exports、runtime/artifacts 或 logs 下未找到报告/导出文件。",
        "no_report_references": "尚无报告引用。Discord 每日总结目前发布在外部；开始本地保存后可在此归档引用。",
        "journal_tabs": ["每日日志", "交易假设", "收盘摘要", "错误与经验", "报告归档"],
        "live_tabs": ["总览", "持仓", "敞口", "期权中心", "对账"],
        "option_tabs": ["账簿审计", "持仓实验室", "波动率与模型审计"],
        "paper_tabs": ["总览", "账户与风险", "策略", "复核", "票据", "账本", "健康"],
        "risk_tabs": [
            "总览",
            "敞口与集中度",
            "回撤与盈亏",
            "期权与希腊值",
            "压力测试",
            "配置实验室",
        ],
        "execution_tabs": ["指挥", "策略", "订单", "信号", "边界"],
        "workbench_tabs": ["总览", "估值", "期权", "新闻", "AI", "决策"],
        "workbench_nav_views": ["市场监控", "自选股", "机会中心", "API 状态"],
        "workbench_option_tabs": [
            "剧本",
            "策略匹配",
            "扫描器",
            "收益图实验室",
            "波动率模型",
        ],
        "valuation_tabs": ["DCF", "目标价", "同业", "供应链", "检查清单"],
        "directional_lens_tabs": ["周期摘要", "信号贡献", "情绪记录"],
        "option_scanner_tabs": [
            "买入看涨",
            "买入看跌",
            "现金担保看跌",
            "价差",
            "日历",
            "铁鹰",
            "蝶式",
            "比率",
            "反向价差",
        ],
        "strategy_fit_tabs": ["策略匹配", "方向透镜", "扫描器", "收益图实验室"],
        "builder_tabs": ["构建器", "已保存想法"],
        "save_rows": "保存行",
        "reset_starter": "重置模板",
        "stock_valuation": "股票估值",
        "valuation_ticker": "估值标的",
        "load_valuation": "加载估值",
        "choose_ticker_load": "请选择一个标的并加载估值数据。",
        "loading_fundamentals": "正在加载 {symbol} 的基本面与价格历史...",
        "valuation_snapshot_missing": "无法加载可用的估值快照。",
        "checklist_unavailable": "检查清单数据不可用。",
        "write_draft_proposal": "撰写模拟交易草案",
        "proposal_slot_caption": "这只会写入提案产物；不会批准、路由或提交订单。",
        "contracts_per_leg": "每组腿合约数",
        "candidate": "候选",
        "write_draft_button": "写入草案提案",
        "could_not_write_proposal": "无法写入提案：{error}",
        "draft_proposal_written": "草案提案已写入：{path}",
        "proposal_id": "提案 ID：`{proposal_id}`",
        "save_idea": "保存想法",
        "active_symbol": "当前标的",
        "watchlist": "自选股",
        "saved_ideas": "已保存想法",
        "option_package_leg": "期权组合 / 单腿",
        "greek_package_leg": "希腊值组合 / 单腿",
        "registered": "已注册",
        "running": "运行中",
        "kill_switches": "熔断开关",
        "proposals": "提案",
        "ready_reviews": "可执行复核",
        "blocked_reviews": "阻断复核",
        "execution_live_enabled": "实盘交易已开启。复核前请将此视为生产边界。",
        "execution_paper_armed": "模拟提交已武装。已批准票据可能进入受控模拟券商提交。",
        "execution_locked": "实盘交易与模拟提交均已锁定。本页正在监控提案、票据与策略状态。",
    },
}

RESEARCH_TEXT: dict[str, dict[str, Any]] = {
    "en": {
        "title": "Research Home",
        "ledger": "## 📜 Run Ledger",
        "ledger_desc": "Historical backtest iterations.",
        "ledger_filter": "Ledger view",
        "ledger_completed": "✅ Completed backtests ({count})",
        "ledger_blocked": "⛔ Blocked before backtest ({count})",
        "ledger_missing": "⚠️ Missing return artifact ({count})",
        "ledger_all": "All ledger records ({count})",
        "ledger_not_backtested": "NOT BACKTESTED — blocked during preflight",
        "ledger_artifact_unavailable": "Return artifact unavailable",
        "db_empty": "Database empty or not found. Run the Evaluator first!",
        "metrics": [
            "Total Experiments",
            "Best Holdout IC",
            "Candidate Factors",
            "Research Warning Rate",
        ],
        "matrix_title": "### 🗄️ Candidate Matrix",
        "cols": {
            "Factor": "Factor",
            "Round": "Round",
            "Val_IC": "Val_IC",
            "Holdout_IC": "Holdout_IC",
            "Diagnostics": "Diagnostics",
            "Next_Step": "Next_Step",
        },
        "chart_title": "### 📈 Institutional Risk Analytics (Tear Sheet)",
        "chart_desc": "Out-of-sample portfolio simulation with drawdown constraints.",
        "trace_strat": "Factor Portfolio",
        "trace_bench": "Benchmark (SPY)",
        "risk_metrics": [
            "Ann. Return",
            "Ann. Volatility",
            "Sharpe Ratio",
            "Max Drawdown",
            "Calmar Ratio",
        ],
        "glossary_title": "📖 Metric Definitions & Institutional Bounds",
        "glossary": {
            "Information Coefficient (IC)": "Measures the correlation between factor scores and future "
            "returns.<br>• <b>< 2%:</b> Noise<br>• <b>2% - 5%:</b> Tradeable "
            "Alpha<br>• <b>> 10%:</b> Warning (Possible Lookahead Bias)",
            "Sharpe Ratio": "Return per unit of volatility.<br>• <b>< 1.0:</b> Poor<br>• <b>1.0 - 2.0:</b> "
            "Good<br>• <b>> 3.0:</b> Warning (Check transaction costs)",
            "Max Drawdown": "Worst peak-to-trough loss. CTAs strictly monitor this due to leverage.<br>• <b>< "
            "10%:</b> Excellent<br>• <b>> 20%:</b> Un-tradeable with high leverage",
            "Calmar Ratio": "Annualized Return divided by Max Drawdown (Return per unit of pain).<br>• <b>< "
            "1.0:</b> Too risky<br>• <b>> 2.0:</b> Exceptional",
        },
        "tab_tearsheet": "📊 Tear Sheet",
        "tab_correlation": "🔗 Correlation Matrix",
        "corr_title": "Factor Orthogonalization (Return Correlation)",
        "corr_desc": "Checking for multicollinearity across historical factor returns. High correlation (>0.7) "
        "indicates duplicate risk.",
        "heatmap_title": "🗓️ Monthly Return Matrix",
        "vol_title": "📉 Rolling 30-Day Volatility",
        "heatmap_help_title": "💡 How to read the Heatmap",
        "heatmap_help": "Look for **Consistency** over **Magnitude**. A robust factor should have a balanced mix of "
        "green months. If all profits are clustered in a single year or a specific month (e.g., only "
        "making money in March 2020), the model is likely overfit to a specific macro event rather "
        "than capturing true alpha.",
        "vol_help_title": "💡 How to read Rolling Volatility",
        "vol_help": "This tracks how much the strategy's risk fluctuates over time. Because our Execution Desk uses "
        "**Volatility Scaling**, this line should ideally remain relatively flat. Massive spikes indicate "
        "that the strategy lost control of its risk sizing during a market shock.",
        "ic_decay_title": "📉 IC Decay Curve",
        "ic_decay_desc": "Tests whether the selected feature's cross-sectional signal survives as the forward return "
        "horizon lengthens. Returns are measured from next open to future close to avoid same-close "
        "execution leakage.",
        "ic_decay_select_feature": "Feature to test",
        "ic_decay_missing": "IC decay unavailable",
        "ic_decay_no_features": "No engineered feature columns found for IC decay.",
        "ic_decay_no_data": "No valid IC decay observations for this feature.",
        "ic_decay_1d": "1D IC",
        "ic_decay_peak": "Peak |IC|",
        "ic_decay_days": "Valid Days",
        "ic_decay_xaxis": "Forward horizon (days)",
        "ic_decay_yaxis": "Mean daily Spearman IC",
        "ic_decay_help_title": "💡 How to read IC Decay",
        "ic_decay_help": "IC is the daily cross-sectional Spearman correlation between today's feature rank and future "
        "return rank. Rough guide for liquid futures: **<0.01 noise**, **0.01-0.03 weak**, "
        "**0.03-0.05 useful**, **>0.05 strong**. A fast drop toward zero means the alpha is "
        "short-lived; a stable curve means the signal survives longer. Negative IC means the feature "
        "is predictive in the opposite direction.",
        "dna_pnl_title": "💡 How to read PnL & Entry/Exit",
        "dna_pnl": "**PnL Distribution (Top Left):** Look at the tails. If the green (Loss) bars stretch much further "
        "to the left than the red (Win) bars stretch to the right, your losers are bigger than your winners "
        "(Negative Skew).<br><br>**Entry vs Exit (Bottom Left):** The dashed line is the break-even point. "
        "If you see massive clusters far away from the line, it means your holding periods or stop-losses "
        "are too loose.",
        "dna_time_title": "💡 How to read Holding Time",
        "dna_time": "**Holding Time (Top Right):** StatArb models should have tight, consistent holding times (e.g., "
        "24-48 hours). If you have a 'fat tail' of trades held for hundreds of hours, the model is trapped "
        "in a non-reverting spread.<br><br>**PnL vs Time (Bottom Right):** The holy grail is 'Cut losses "
        "early, let profits run.' If your largest red bubbles (losses) are clustered on the far right "
        "(long holding times), your model is stubbornly holding onto bad trades.",
        "dna_total_trades": "Trades",
        "dna_win_rate": "Win Rate",
        "dna_profit_factor": "Profit Factor",
        "dna_median_hold": "Median Hold",
        "dna_payoff_ratio": "Payoff Ratio",
        "dna_avg_win": "Avg Win",
        "dna_avg_loss": "Avg Loss",
        "dna_profit_concentration": "80% Profit Tickers",
        "dna_expectancy": "Expectancy / Trade",
        "tab_dna": "🧬 Strategy DNA",
        "tab_pareto": "📈 Pareto Frontier",
        "tab_ml": "🧠 ML Feature Importance",
        "tab_assumptions": "🧾 Assumptions",
        "assumptions_title": "🧾 Factor Assumptions",
        "assumptions_select_run": "Inspect run",
        "assumptions_missing": "No assumption manifest found for this run.",
        "assumptions_reconstructed": "No saved JSON manifest exists for this legacy run, so this view is reconstructed from the research ledger.",
        "assumptions_expected_path": "Expected JSON path",
        "assumptions_not_saved": "not saved; reconstructed in memory",
        "assumptions_reconstruction": "Reconstruction Note",
        "assumptions_available": "Available manifests",
        "assumptions_manifest_path": "Manifest",
        "assumptions_data": "Data",
        "assumptions_data_health": "Data Health",
        "assumptions_data_health_note": "Accounting views may forward-fill stale marks, while alpha views must block synthetic inputs.",
        "assumptions_data_health_missing": "Data health snapshot is unavailable.",
        "assumptions_health_status": "Status",
        "assumptions_health_asset": "Asset",
        "assumptions_health_timeframe": "Timeframe",
        "assumptions_health_file": "Sample file",
        "assumptions_health_fresh": "Fresh %",
        "assumptions_health_synthetic": "Synthetic %",
        "assumptions_health_expired": "Expired rows",
        "assumptions_health_policy": "Fill policy",
        "assumptions_signal": "Signal & Execution",
        "assumptions_engine": "Execution Engine",
        "assumptions_costs": "Costs & Slippage",
        "assumptions_cost_readiness": "Transaction Cost Readiness",
        "assumptions_cost_field": "Field",
        "assumptions_cost_value": "Value",
        "assumptions_cost_market": "Market",
        "assumptions_cost_profile": "Profile",
        "assumptions_cost_source": "Source",
        "assumptions_cost_source_frozen": "Frozen with run",
        "assumptions_cost_source_current": "Current default; not frozen with run",
        "assumptions_cost_use_case": "Claim level",
        "assumptions_cost_schedule": "Schedule status",
        "assumptions_cost_completeness": "Completeness",
        "assumptions_cost_engine_support": "Engine support",
        "assumptions_cost_net_ready": "Research-net ready",
        "assumptions_cost_production_ready": "Production ready",
        "assumptions_cost_ready_production": "This run used a profile approved for research-net and production claims.",
        "assumptions_cost_ready_research": "This profile supports research-net results but is not approved for production claims.",
        "assumptions_cost_gross_only": "This is a gross-only run. It must not be reported as net or production-ready.",
        "assumptions_cost_not_net_ready": "The fee schedule is not fully wired into the engine, so net performance is not validated.",
        "assumptions_cost_historical_unfrozen": "This historical run did not freeze a transaction-cost profile. The panel shows the current default for this market and does not validate the old net result.",
        "assumptions_cost_required_work": "Required before promotion",
        "assumptions_cost_fingerprint": "Frozen profile fingerprint",
        "assumptions_cost_current_fingerprint": "Current default profile fingerprint",
        "assumptions_cost_yes": "Yes",
        "assumptions_cost_no": "No",
        "assumptions_liquidity": "Liquidity Policy",
        "assumptions_option_selection": "Option Contract Selection",
        "assumptions_realized": "Realized Summary",
        "assumptions_raw_json": "Raw JSON",
        "assumptions_download": "Download manifest",
        "assumptions_asset": "Asset",
        "assumptions_mode": "Mode",
        "assumptions_alpha_col": "Alpha",
        "assumptions_trades": "Trades",
        "assumptions_no_values": "No recorded values.",
        "tab_oracle": "🔮 Oracle Regime Test",
        "oracle_title": "🔮 Oracle Regime Validation",
        "oracle_caption": "Compares GMM/HMM regime probabilities against hindsight trend/chop/panic labels. This "
        "validates the regime engine, not ordinary factor PnL.",
        "oracle_not_applicable": "Oracle validation is only shown for regime-aware factors.",
        "oracle_missing": "Oracle inputs are missing.",
        "oracle_scope": "Evaluation universe",
        "oracle_scope_run": "Selected run traded universe",
        "oracle_scope_all": "All GMM assets",
        "oracle_loading": "Running oracle regime validation...",
        "oracle_error": "Oracle validation failed",
        "oracle_accuracy": "Oracle Accuracy",
        "oracle_panic_auc": "Panic ROC-AUC",
        "oracle_samples": "Labeled Samples",
        "oracle_panic_rate": "Oracle / AI Panic Rate",
        "oracle_panic_box": "AI Panic Probability by Oracle State",
        "oracle_state_mix": "Predicted State Mix by Oracle Label",
        "oracle_select_asset": "Inspect asset",
        "oracle_asset_drilldown": "Asset Probability Drilldown",
        "oracle_state_map": "State alignment",
        "oracle_manual_title": "How to use the Oracle page",
        "oracle_manual": "**What this page answers:** whether the unsupervised regime engine is recognizing future "
        "trend/chop/panic structure, not whether the strategy made money.\n"
        "\n"
        "**Workflow:**\n"
        "1. Choose the evaluation universe. Use the selected run universe when you want to inspect "
        "the assets this factor actually traded; use all GMM assets when you want a broad model "
        "audit.\n"
        "2. Read the top metrics first. Accuracy is broad regime agreement. Panic ROC-AUC is the most "
        "important survival metric because it asks whether high AI panic probability separates future "
        "crash windows from normal windows.\n"
        "3. Use the confusion matrix to see what kind of mistakes the model makes. A regime model "
        "that misses Panic as Chop is much more dangerous than one that mistakes Trend for Chop.\n"
        "4. Use the asset dropdown last. It is a local drilldown for one contract, so the asset "
        "metrics and chart update when you switch assets.",
        "oracle_metrics_help_title": "How to read the summary metrics",
        "oracle_metrics_help": "**Oracle Accuracy** is calculated over the current evaluation universe selected above. "
        "It does not change when you switch the asset drilldown below.\n"
        "\n"
        "**How to judge accuracy:** With 3 states, ~33% is close to balanced random guessing. "
        "But because Chop often dominates the sample, also compare against the majority-class "
        "baseline. Rough guide: **<40% weak**, **40-50% modest**, **50-60% useful**, **>60% "
        "strong** for this noisy hindsight task. So an 80k-sample accuracy of **33.8%** means "
        "the current state semantics are still weak; the model is not consistently matching the "
        "Oracle labels.\n"
        "\n"
        "**Panic Rate** is not accuracy. It is the base frequency of future panic windows. For "
        "example, **19.2% Oracle Panic** means about 19 out of every 100 labeled 20-day windows "
        "later suffered an Oracle-defined panic drawdown. Compare it with AI Panic Rate: AI "
        "much higher means over-defensive false alarms; AI much lower means possible tail-risk "
        "blindness.\n"
        "\n"
        "**Panic ROC-AUC** is usually more important than total accuracy. **0.50=random**, "
        "**0.55=very weak edge**, **0.60-0.65=modest**, **0.65-0.70=useful**, **>0.70=strong**, "
        "**<0.50=inverted signal**. A value like **0.549** means the panic probability is only "
        "slightly better than random at ranking future panic windows above normal windows.",
        "oracle_confusion_help_title": "How to read the confusion matrix",
        "oracle_confusion_help": "Rows are the hindsight Oracle labels; columns are the AI's predicted labels. The "
        "diagonal is correct classification; off-diagonal cells are mistakes.\n"
        "\n"
        "**Numbers:** in Count mode, each cell is the number of samples in that "
        "true/predicted bucket. In Row % mode, each row adds to 100%, so you can read recall "
        "by true state. Example: True Panic -> Pred Panic is the percentage of Oracle panic "
        "windows caught as panic.\n"
        "\n"
        "**Color:** brighter/darker cells mean more samples or a higher row percentage, "
        "depending on the mode. A bright diagonal is good. A bright True Panic -> Pred "
        "Trend/Chop cell is dangerous because the shield stayed exposed during future "
        "crash-like windows.",
        "oracle_confusion_mode": "Confusion matrix values",
        "oracle_confusion_counts": "Counts",
        "oracle_confusion_row_pct": "Row %",
        "oracle_confusion_title_counts": "Confusion Matrix (Counts)",
        "oracle_confusion_title_pct": "Confusion Matrix (Row %)",
        "oracle_classification_report": "Classification Report",
        "oracle_probability_help_title": "How to read the probability diagnostics",
        "oracle_probability_help": "**AI Panic Probability by Oracle State:** Panic probabilities should be low for "
        "Oracle Trend/Chop and visibly higher for Oracle Panic. If all three boxes overlap "
        "heavily, the model is not separating crisis structure.\n"
        "\n"
        "**Predicted State Mix:** shows whether the AI is biased toward one state. If one "
        "color dominates every Oracle label, the GMM may have collapsed into a nearly "
        "static classifier.",
        "oracle_asset_help_title": "How to use the asset drilldown",
        "oracle_asset_help": "The asset dropdown changes only the lower drilldown section and the asset-specific "
        "metrics above that chart. The top summary metrics remain for the chosen universe.\n"
        "\n"
        "In the chart, the grey line is price. The background regions show the AI's dominant "
        "regime: green = Quiet/Trend, orange = Chop, red = Panic. Red x-marks show future windows "
        "that the Oracle labels as Panic. A useful shield should show red/orange defensive "
        "regions before or during clusters of red x-marks. Hover over the price line to see the "
        "underlying Quiet/Chop/Panic probabilities without cluttering the chart with extra lines.",
        "oracle_asset_scope_note": "Asset drilldown metrics update when you switch this dropdown; the top summary "
        "metrics stay at the selected universe scope.",
        "oracle_asset_accuracy": "Asset Accuracy",
        "oracle_asset_panic_auc": "Asset Panic ROC-AUC",
        "oracle_asset_samples": "Asset Samples",
        "oracle_asset_panic_rate": "Asset Oracle / AI Panic Rate",
        "oracle_asset_no_data": "No usable price/regime rows for this asset.",
        "tab_conditional_perf": "🏛️ Conditional Strategy Performance",
        "exec_dd_title": "### 📉 Drawdown Savings from Regime Model",
        "exec_dd_desc": "Compares baseline SMA/Bollinger max drawdown versus the final Layer 4 Router.",
        "exec_to_title": "### 🔁 Turnover Evolution (Binary → Continuous → Discretized)",
        "exec_to_desc": "Quantifies execution drag reduction through continuous math and discretization.",
        "exec_missing": "Insufficient run history for one or more required factors. Please run the baseline and Layer "
        "4 backtests first.",
        "exec_overlay_title": "### 📈 Overlay Equity Curve (Benchmark vs SMA vs Layer 4)",
        "exec_overlay_desc": "Compare cumulative equity paths to visualize crash behavior and capital preservation.",
        "exec_crisis_title": "### 🚨 Crisis Zoom-In (2024-01-01 to 2024-02-28)",
        "exec_crisis_desc": "Focused view of the designated crisis period to verify defensive routing behavior.",
        "exec_pareto2_title": "### 🎯 Ablation Pareto (Annualized Volatility vs Annualized Return)",
        "exec_pareto2_desc": "Five-step ablation map: risk on X-axis, return on Y-axis.",
        "exec_overlay_missing": "Could not build overlay equity curve. Ensure at least SMA baseline and Layer 4 runs "
        "have return logs.",
        "select_run_hint": "👈 Please select a completed backtest run from the Ledger to view detailed analytics.",
        "blocked_run_title": "⛔ Not backtested — this trial was blocked during preflight.",
        "blocked_run_reason": "**Blocking reason:** {reason}",
        "blocked_run_reason_missing": "No blocker explanation was recorded.",
        "blocked_run_explainer": "This is an audit record for a planned factor+sleeve trial, not a failed backtest. No return series should exist until the blocker is resolved.",
        "missing_artifact_title": "⚠️ This ledger row has no loadable return artifact, so analytics cannot be shown.",
        "ml_caption": "Visualizing the internal decision-making weights of the Machine Learning model.",
        "ml_missing": "ℹ️ No Feature Importance data found. This tab will populate when you select an ML-generated "
        "run.",
        "ts_no_returns": "⚠️ No return data found for this run. It may have failed execution.",
        "strategy_net": "#### 🟢 Strategy (Net of Fees)",
        "benchmark_eq": "#### 🟡 Benchmark",
        "avg_turnover": "Avg Turnover",
        "holdout_ic": "Holdout IC",
        "total_trades": "Total Trades",
        "test_scope": "Run: {run_id} | Backtest result window used by widgets/charts: {start} to {end} | Return rows: {rows:,} (~{years:.2f}y annualization) | Prepared data: {prepared_window} | Requested filter: {requested_window} | Frequency: {frequency} | Data: {role} | Return clock: {return_clock} | Source: {source}",
        "raw_entries": "Raw Entries",
        "raw_exits": "Raw Exits",
        "target_weight_changes": "Target-Weight Changes",
        "active_signal_rows": "Active Signal Rows",
        "context_asset_class": "Asset Class",
        "context_traded_universe": "Traded Universe",
        "context_run": "Run",
        "context_window": "Result window",
        "context_rows": "Return rows",
        "context_prepared_data": "Prepared data",
        "context_requested_filter": "Requested filter",
        "context_frequency": "Frequency",
        "context_data": "Data",
        "context_return_clock": "Return clock",
        "context_source": "Source",
        "tearsheet_benchmark_guide": "What the benchmarks mean",
        "tearsheet_benchmark_guide_intro": "The benchmark card above uses the primary `benchmark_return` series. Extra benchmark columns, when available, are plotted as dotted lines in the cumulative-return chart.",
        "tearsheet_benchmark_guide_headers": ["Slot", "Benchmark", "Return column", "Meaning"],
        "benchmark_slot_primary": "Primary",
        "benchmark_slot_secondary": "Secondary",
        "benchmark_slot_control": "Control",
        "benchmark_slot_additional": "Additional",
        "benchmark_mode_label_same_horizon": "Same horizon",
        "benchmark_mode_label_passive_close_to_close": "Passive close-to-close",
        "benchmark_mode_label_unknown": "Mode not recorded",
        "benchmark_role_active_universe_futures_basket": "Dynamic equal-weight basket of contracts active in the factor universe. Useful market-context benchmark, but it may not share an intraday signal clock.",
        "benchmark_role_same_horizon_universe_control": "Equal-weight active universe measured over the same signal/execution horizon as the strategy. This is usually the fairest comparator for next-bar or intraday factors.",
        "benchmark_role_broad_commodity_index": "Broad commodity-index context, such as Nanhua, when index data is available.",
        "benchmark_role_asset_class_market_index": "Asset-class market index buy-and-hold context.",
        "benchmark_role_style_context_index": "Secondary style or market index context.",
        "benchmark_role_full_universe_equal_weight": "Equal-weight basket across the full tradable universe, useful for separating instrument selection from broad universe drift.",
        "benchmark_role_active_crypto_universe": "Buy-and-hold basket across the active crypto universe.",
        "benchmark_mode_same_horizon": "Built from the same return clock as the strategy, so it helps audit whether alpha beats naive active-universe exposure over the exact tested horizon.",
        "benchmark_mode_passive_close_to_close": "Passive close-to-close holding return. Useful for regime context, but not a clean execution-horizon benchmark for intraday signals.",
        "benchmark_mode_unknown": "Return construction was not recorded in the manifest; use this as context and audit the source column.",
        "tearsheet_quality_guide": "How to read metric labels",
        "tearsheet_quality_guide_text": """
The chips are **diagnostic heuristics, not promotion gates**. Strategy return and volatility are compared with the displayed same-horizon benchmark when one is available. Sharpe, Calmar, drawdown, turnover, and holdout IC use absolute research ranges. Trade counts and signal rows describe sample coverage; they do not prove statistical independence or robustness. Entry, exit, and target-weight counts only confirm that execution diagnostics were recorded.

Key ranges: Sharpe `<0.5` weak, `1-2` good, `2-3` strong, `>=3` audit; holdout IC `<1%` weak, `1-3%` modest, `3-5%` good, `5-10%` strong, `>=10%` integrity audit; average daily turnover `<5%` low, `5-20%` moderate, `20-50%` high, `>=50%` very high. Always check costs, annualization, overlapping returns, trial count, and regime stability.
""",
        "tearsheet_quality_labels": {
            "not_available": "N/A",
            "not_applicable": "Not applicable",
            "negative": "Negative",
            "low": "Low",
            "positive": "Positive",
            "below_benchmark": "Below benchmark",
            "ahead": "Ahead",
            "in_line": "In line",
            "lower_risk": "Lower risk",
            "very_low": "Very low",
            "controlled": "Controlled",
            "moderate": "Moderate",
            "elevated": "Elevated",
            "high": "High",
            "very_high": "Very high",
            "weak": "Weak",
            "marginal": "Marginal",
            "modest": "Modest",
            "good": "Good",
            "strong": "Strong",
            "audit": "Audit",
            "high_audit": "High - audit",
            "extreme_audit": "Extreme - audit",
            "severe": "Severe",
            "none": "None",
            "sparse": "Sparse",
            "limited": "Limited",
            "adequate": "Adequate",
            "broad": "Broad sample",
            "missing": "Missing",
            "recorded": "Recorded",
            "reference": "Reference",
        },
        "tearsheet_quality_help": {
            "return_unavailable": "No finite annualized return is available.",
            "return_negative": "Annualized return is non-positive over the evaluated window.",
            "return_below_benchmark": "Positive return, but more than 20 bps below the displayed same-horizon benchmark.",
            "return_ahead": "Annualized return is more than 20 bps above the displayed same-horizon benchmark.",
            "return_in_line": "Annualized return is within 20 bps of the displayed same-horizon benchmark.",
            "return_low": "Positive annualized return below 2%; judge it together with risk and costs.",
            "return_positive": "Positive annualized return in a conventional range; risk-adjusted evidence still matters.",
            "return_high_audit": "Annualized return above 30%; audit annualization, leverage, costs, leakage, and the test window.",
            "volatility_unavailable": "No finite annualized volatility is available.",
            "volatility_flat_audit": "Near-zero volatility can indicate no exposure, stale prices, or a return-construction issue.",
            "volatility_lower": "Annualized volatility is at least 25% below the same-horizon benchmark; verify this is not simply low exposure.",
            "volatility_in_line": "Annualized volatility is within 25% of the same-horizon benchmark.",
            "volatility_very_low": "Annualized volatility is below 2%; confirm the strategy was genuinely active.",
            "volatility_controlled": "Annualized volatility is between 2% and 10%.",
            "volatility_moderate": "Annualized volatility is between 10% and 20%.",
            "volatility_elevated": "Volatility is elevated relative to the benchmark or above 20% annualized.",
            "volatility_high": "Volatility is high relative to the benchmark or above 35% annualized.",
            "sharpe_unavailable": "No finite Sharpe ratio is available.",
            "sharpe_negative": "Risk-adjusted return is non-positive.",
            "sharpe_weak": "Sharpe is below 0.5 and provides weak risk-adjusted evidence.",
            "sharpe_marginal": "Sharpe is between 0.5 and 1.0; potentially usable but fragile.",
            "sharpe_good": "Sharpe is between 1.0 and 2.0; credible if out of sample and after costs.",
            "sharpe_strong": "Sharpe is between 2.0 and 3.0; strong, but verify stability and trial selection.",
            "sharpe_high_audit": "Sharpe is between 3.0 and 5.0; unusually high and requires an integrity audit.",
            "sharpe_extreme_audit": "Sharpe is at least 5.0; audit leakage, annualization, overlapping returns, costs, and curve fitting.",
            "drawdown_unavailable": "No finite maximum drawdown is available.",
            "drawdown_none_audit": "Near-zero drawdown can reflect a short sample, low exposure, stale prices, or a construction issue.",
            "drawdown_controlled": "Maximum drawdown magnitude is at most 5%.",
            "drawdown_moderate": "Maximum drawdown magnitude is between 5% and 10%.",
            "drawdown_elevated": "Maximum drawdown magnitude is between 10% and 20%.",
            "drawdown_severe": "Maximum drawdown magnitude exceeds 20%.",
            "calmar_unavailable": "No finite Calmar ratio is available.",
            "calmar_negative": "Annualized return is non-positive relative to drawdown.",
            "calmar_weak": "Calmar is below 0.5; return is weak relative to peak-to-trough pain.",
            "calmar_modest": "Calmar is between 0.5 and 1.0.",
            "calmar_good": "Calmar is between 1.0 and 2.0.",
            "calmar_strong": "Calmar is between 2.0 and 5.0.",
            "calmar_high_audit": "Calmar is at least 5.0; verify the drawdown sample and return construction.",
            "turnover_unavailable": "No finite average daily turnover is available.",
            "turnover_low": "Average daily capital turnover is below 5%; execution drag is usually easier to control.",
            "turnover_moderate": "Average daily capital turnover is between 5% and 20%.",
            "turnover_high": "Average daily capital turnover is between 20% and 50%; costs and capacity need close review.",
            "turnover_very_high": "Average daily capital turnover is at least 50%; the strategy is highly exposed to costs, slippage, and capacity limits.",
            "ic_unavailable": "No finite untouched holdout IC is available.",
            "ic_negative": "Holdout IC is non-positive.",
            "ic_weak": "Holdout IC is positive but below 1%.",
            "ic_modest": "Holdout IC is between 1% and 3%.",
            "ic_good": "Holdout IC is between 3% and 5%.",
            "ic_strong": "Holdout IC is between 5% and 10%.",
            "ic_extreme_audit": "Holdout IC is at least 10%; audit look-ahead bias, target leakage, and trial selection.",
            "trades_unavailable": "No discrete trade count was recorded.",
            "trades_none": "No discrete trades were recorded.",
            "trades_sparse": "Fewer than 30 trades; inference is very sample-sensitive.",
            "trades_limited": "Between 30 and 99 trades; useful but still a limited sample.",
            "trades_adequate": "Between 100 and 299 trades; an adequate count, though independence still matters.",
            "trades_broad": "At least 300 trades; broad count coverage, but not proof of independent observations.",
            "diagnostic_missing": "This execution diagnostic was not recorded for the run.",
            "diagnostic_none": "The diagnostic was recorded with no events.",
            "diagnostic_recorded": "Execution events were recorded. The count is descriptive, not a quality score.",
            "signal_rows_missing": "Active signal-row coverage was not recorded.",
            "signal_rows_limited": "Fewer than 100 active signal rows; coverage is limited.",
            "signal_rows_moderate": "Between 100 and 499 active signal rows.",
            "signal_rows_broad": "At least 500 active signal rows; broad coverage, not necessarily independent evidence.",
            "reference_control": "Passive benchmark control; turnover and trade-count grading does not apply.",
            "not_applicable": "This metric is not defined for the passive benchmark control.",
        },
        "axis_cum_return": "Cumulative return (%)",
        "axis_drawdown": "Drawdown (%)",
        "axis_gross_leverage": "Gross exposure (x)",
        "axis_date": "Date",
        "leaderboard_title": "### 🏆 Alpha Leaderboard",
        "corr_select_factors": "🔍 Select Factors to Compare",
        "corr_select_min": "👆 Please select at least two factors from the dropdown above to view the correlation "
        "matrix.",
        "corr_waiting": "⏳ Waiting for more factors... You need at least 2 successful backtests to generate a "
        "correlation matrix.",
        "dna_trade_caption": "Trade-Level Analytics (StatArb / Pairs)",
        "dna_portfolio_caption": "Cross-Sectional Daily Net PnL Frequency (Portfolio-Level)",
        "dna_error": "⚠️ Error rendering trade analytics",
        "dna_no_returns": "⚠️ No return data found for this run.",
        "pareto_caption": "Visualizing the execution boundary of this factor across all historical iterations.",
        "pareto_help": "💡 How to read this: prefer iterations with high return, low drawdown magnitude, low "
        "turnover, and stronger holdout IC color. Bubble size indicates absolute Out-of-Sample IC.",
        "pareto_axis_drawdown": "Drawdown Magnitude (%)",
        "pareto_axis_return": "Return (%)",
        "pareto_axis_turnover": "Turnover (%)",
        "pareto_color_ic": "Holdout IC",
        "pareto_size_abs_ic": "Abs(IC)",
        "pareto_round": "Round",
        "pareto_error": "⚠️ Could not render Pareto Frontier due to data inconsistency",
        "pareto_missing": "⚠️ Not enough data. Run this factor at least two times to generate a Pareto Frontier.",
        "asset_zoo_title": "🦁 The Asset Zoo: Market Microstructure Clustering",
        "asset_zoo_subtitle": "Gaussian Mixture Model (GMM) clustering of asset micro-personalities to determine "
        "strategy targeting.",
        "asset_zoo_missing": "❌ Profile data not found. Please run `data_engine/asset_profiler.py` first.",
        "asset_cluster_dist": "### 🧬 Cluster Distribution",
        "asset_tier1": "Tier 1 (Trend / Breakout)",
        "asset_tier2": "Tier 2 (Mean-Reverting)",
        "asset_tier3": "Tier 3 (Toxic / Illiquid)",
        "asset_radar": "### 🔭 Market Radar (Risk vs. Trendiness)",
        "asset_target_board": "### 🎯 Strategy Targeting Board",
        "asset_select_target": "Select Strategy Target:",
        "asset_copy_list": "**Copy-Paste Target Universe (Python List):**",
        "exec_page_title": "🏛️ Executive Dashboard",
        "exec_page_subtitle": "Conditional Strategy Performance",
        "regime_page_title": "🏛️ Market Regime Characterisation",
        "regime_page_subtitle": "Visualizing HMM Continuous Probabilities over Asset Prices.",
        "regime_select_asset": "Select Asset to Overlay:",
        "dna_gmm_caption": "🧠 GMM Regime Analysis: Tracking the AI's defensive cash shielding.",
        "dna_gmm_regime_alloc": "### Regime Allocation",
        "dna_gmm_active": "Active Market Exposure",
        "dna_gmm_cash": "Defensive Cash (Panic)",
        "dna_gmm_cum_pnl": "### Cumulative PnL: Active vs Flatline",
        "dna_gmm_port_val": "Portfolio Net Value",
        "dna_gmm_cash_zones": "Cash Ejection Zones",
        "dna_gmm_total_days": "Total Days",
        "dna_gmm_days_cash": "Days in Cash",
        "dna_detected_factor": "Detected Factor",
        "dna_architecture": "Architecture",
        "exec_dash_title": "🏛️ Institutional Alpha Architecture",
        "exec_dash_evo_title": "### 🧠 Evolution of the Alpha Engine",
        "exec_dash_evo_desc": "Our pipeline systematically evolved from static trend-following baselines to an "
        "AI-driven, regime-aware ensemble. We proved that predictive power alone is useless "
        "without execution shielding, ultimately culminating in a Walk-Forward Microstructure "
        "model.",
        "exec_v1_title": "#### V1: Baselines",
        "exec_v1_desc": "**SMA (fac_051)** uses a classic 50/200-day moving average crossover to capture macro trends. "
        "**Bollinger (fac_040)** uses a 20-day mean with 2-standard deviation bands to trade "
        "mean-reversion. Both proved that static linear logic is too slow for modern market chop, "
        "yielding unacceptable risk-adjusted returns.",
        "exec_v2_title": "#### V2: Raw ML",
        "exec_v2_desc": "**Raw ML (fac_054)** replaced linear math with an XGBoost non-linear decision tree, trained "
        "on 100+ price/volume features to predict 1-day forward returns. While it successfully found "
        "high-win-rate micro-reversions, the AI traded too frequently ('chattering'). The unshielded "
        "frictional costs (Square-Root TCA slippage) completely destroyed the gross alpha.",
        "exec_v3_title": "#### V3: HMM Router",
        "exec_v3_desc": "**HMM Router (fac_056)** introduced a Hidden Markov Model to act as a traffic controller. It "
        "tracks historical variance to classify the market into 3 hidden states: Trend, Chop, and "
        "Crisis. It routes capital to the SMA during Trends, to XGBoost during Chop, and physicalizes "
        "a 63% probability threshold to trigger a 100% Cash Panic Shield during Crises.",
        "exec_v4_title": "#### V4: GMM Microstructure",
        "exec_v4_desc": "**GMM Microstructure (fac_057)** upgraded the brain to a Gaussian Mixture Model trained "
        "walk-forward to eliminate lookahead bias. Instead of just price, it clusters Market "
        "Microstructure (Amihud Illiquidity & Garman-Klass Volatility) to mathematically predict "
        "fat-tail crashes before price drops. Capital is scaled to a strict 12% Volatility mandate.",
        "exec_metrics_ret": "Ret",
        "exec_metrics_vol": "Vol",
        "exec_metrics_shp": "Sharpe",
        "exec_metrics_mdd": "MDD",
        "exec_chart_cum_title": "### 📈 Cumulative Performance & Risk",
        "exec_chart_cum_sub": "**The Alpha Mountain (Cumulative Equity)**",
        "exec_chart_dd_sub": "**The Drawdown Cavern**",
        "exec_chart_y_eq": "Cumulative Equity (Base 1.0)",
        "exec_chart_y_dd": "Drawdown (%)",
        "exec_xray_title": "### 🧬 Strategy X-Ray: V4 Asset Preferences & Trade Profile",
        "exec_xray_desc": "Looking under the hood of the V4 GMM Microstructure engine to analyze where the AI deployed "
        "capital and how long it held positions under the 8% lazy execution mandate.",
        "exec_xray_pnl_title": "**V4: Top 15 Most Profitable Assets**",
        "exec_xray_hold_title": "**V4: Trade Holding Period Distribution (Hours)**",
        "exec_layer4_title": "### 🛡️ Layer 4: Deep Execution & Risk Shields",
        "exec_layer4_desc": "A raw signal is un-tradeable without execution architecture. We deployed **Bayesian "
        "Optimization (Optuna)** utilizing an Out-Of-Sample penalty wall to mathematically derive "
        "survival parameters.\n"
        "\n"
        "* **Target Volatility Multiplier (3.2x):** Pure normalization logic that safely scales "
        "portfolio capital to meet an institutional 12% Volatility mandate.\n"
        "* **Lazy Execution Discretization (8%):** Solves the 'Quantization Trap'. At 3.2x "
        "leverage, minor signal noise causes deadly turnover. Optuna proved we must force the "
        "engine to only trade in massive 8% blocks, letting the Prime Broker deadband handle the "
        "noise.\n"
        "* **EWMA Shock Absorbers:** Optuna dialed the memory alpha down to 0.20 to slow the AI's "
        "execution reflexes, protecting it from Square-Root TCA slippage.\n"
        "* **Microstructure Ejection:** When the GMM detects a Fat-Tail Crash, the router "
        "violently cuts gross exposure to 0.0 (Cash).\n"
        "\n"
        "**🔬 Microstructure Definitions:**\n"
        "* **Garman-Klass Volatility:** Unlike simple close-to-close volatility, GK incorporates "
        "Intraday High, Low, Open, and Close prices to capture hidden intraday variance and "
        "structural market stress.\n"
        "* **Amihud Illiquidity:** Calculates the absolute daily return divided by daily dollar "
        "volume. It measures 'Price Impact'—how much the price moves per dollar traded. Spikes in "
        "Amihud indicate a severe liquidity vacuum, often preceding massive market crashes.",
        "regime_title": "🏛️ Market Regime Characterisation",
        "regime_subtitle": "Visualizing Walk-Forward GMM Probabilities and Microstructure Regime Detection.",
        "regime_insight_title": "💡 How to read the Engine Room",
        "regime_stress_title": "Microstructure Stress (Z-Scores)",
        "regime_feat_gk": "GK Volatility (Intraday Variance)",
        "regime_feat_amihud": "Amihud (Liquidity Vacuum)",
        "regime_phase_title": "Covariance Phase Space (AI Brain)",
        "regime_phase_desc": "Scatter plot of historical Market Microstructure. The AI only panics when data crosses "
        "diagonally into the high-covariance Red Zone.",
        "regime_boundary": "Theoretical Panic Boundary",
        "regime_insight_desc": "### 💡 How to read the Engine Room\n"
        "**1. The Brain (GMM-HMM):** The AI does not look at price drops. It looks at Market "
        "Microstructure (Amihud Illiquidity & GK Volatility).\n"
        "**2. The Scatter Plot (Right):** Think of this as a radar screen. \n"
        "* **Bottom-Left:** Safe zone. Normal liquidity and volatility (Green dots).\n"
        "* **Top-Right:** The Danger Zone. When order book depth evaporates AND intraday "
        "variance expands, the dot moves to the top right. The AI mathematically classifies "
        "this as a 'Fat-Tail Crash' (Red dots).\n"
        "**3. The Ejection Seat (Left):** When the Red Zone (Panic) crosses the **75.9%** "
        "dashed line, the Meta-Router physically ejects the portfolio to 100% Cash.",
        "regime_state0": "State 0: Quiet / Trend",
        "regime_state1": "State 1: Chop / Reversion",
        "regime_state2": "State 2: Panic / Fat-Tail Crash",
        "regime_panic_thresh": "Panic Threshold (75.9%)",
        "regime_ejection": "Cash Ejection",
        "regime_close_price": "Close Price",
        "regime_y_price": "Price",
        "regime_y_prob": "Probability",
        "regime_feat_ker": "Trend Efficiency (KER)",
        "regime_profiler_title": "📊 The State Profiler: Economic Translation",
        "regime_profiler_desc": "Quantifying the physical realities of the AI's unsupervised clusters.",
        "regime_dna_matrix": "**Regime DNA Matrix**",
        "regime_stress_profile": "**Microstructure Stress Profile**",
        "col_state": "State",
        "col_time": "% of Time",
        "col_return": "Avg Daily Return",
        "col_vol": "Volatility (Z)",
        "col_liq": "Illiquidity (Z)",
        "col_ker": "Trend Efficiency",
        "regime_gmm_title": "🌌 The GMM Probability Density (Visualizing Fat Tails)",
        "expander_ts_title": "💡 How to read the Time Series & Z-Score Patterns",
        "expander_ts_text": "**1. Understanding the Z-Scores**\n"
        "- **Volatility (GK Z):** >0 means higher than average.\n"
        "- **Illiquidity (Amihud Z):** >0 means thin markets (high slippage).\n"
        "- **Trend Efficiency (KER Z):** >0 means clean trends; <0 means choppy.\n"
        "\n"
        "**2. Pattern Recognition**\n"
        "- **State 0 (Trend):** Vol/Liq flatline near 0. KER is positive.\n"
        "- **State 1 (Chop):** KER dips deep negative. Price goes nowhere.\n"
        "- **State 2 (Panic):** Sudden positive spikes in Vol & Liq.",
        "expander_phase_title": "💡 How to read the 3D Phase Space",
        "expander_phase_text": "This 3D scatter plot proves why we added the 3rd dimension (KER). \n"
        "\n"
        "If you rotate the chart to look purely at the X/Y axes (Volatility/Illiquidity), the "
        "Green (Trend) and Orange (Chop) dots overlap completely. But when you look at the "
        "Z-axis (Trend Efficiency), the AI perfectly separates them by altitude. This "
        "completely solves the 'Semantic Gap'.",
        "expander_gmm_title": "💡 How to interpret the 3D Probability Distributions (Fat Tails)",
        "expander_gmm_text": "**What are you looking at?**\n"
        "This shows the Bivariate Normal Probability Density Functions (PDF).\n"
        "\n"
        "- **The Green Spike (State 0):** Tall and narrow. 'Quiet' days are highly predictable.\n"
        "- **The Red Puddle (State 2):** Flat and spreading into high Z-scores. This is a **Fat "
        "Tail**.\n"
        "- **The Spatial Shift (Location):** Why is the Red distribution shifted far to the "
        "right? Because the axes are Z-Scores. 0.0 is the historical average. Green/Orange are "
        "clustered near the left (< 0) because they represent calmer-than-average days. Red is "
        "shifted right (> 2) because crises physically manifest as massive positive deviations "
        "from the mean.",
        "expander_profiler_title": "💡 How to read the Regime DNA & Ranges",
        "expander_profiler_text": "**Expected Mathematical Ranges:**\n"
        "- **State 0 (Trend):** Vol/Liq between [-1.0 to 0.0]. KER > 0.5. The market is "
        "liquid and trending. Maximum leverage deployed.\n"
        "- **State 1 (Chop):** Vol/Liq between [-0.5 to 1.0]. KER < 0.0. The market is "
        "stuck. Alpha shifts to XGBoost mean-reversion.\n"
        "- **State 2 (Panic):** Vol/Liq > 2.0. The plumbing is breaking. Strict risk-off.",
    },
    "zh": {
        "title": "研究首页",
        "ledger": "## 📜 运行记录 (Ledger)",
        "ledger_desc": "历史回测迭代记录。",
        "ledger_filter": "运行记录视图",
        "ledger_completed": "✅ 已完成回测（{count}）",
        "ledger_blocked": "⛔ 回测前已阻断（{count}）",
        "ledger_missing": "⚠️ 缺少收益文件（{count}）",
        "ledger_all": "全部运行记录（{count}）",
        "ledger_not_backtested": "未回测——预检阶段已阻断",
        "ledger_artifact_unavailable": "收益文件不可用",
        "db_empty": "未找到数据库。请先运行评估器 (Evaluator)！",
        "metrics": ["实验总数", "最佳样本外 IC", "候选因子数量", "研究警告率"],
        "matrix_title": "### 🗄️ 候选因子矩阵",
        "cols": {
            "Factor": "因子名称",
            "Round": "迭代轮次",
            "Val_IC": "样本内 IC",
            "Holdout_IC": "样本外 IC",
            "Diagnostics": "诊断结果",
            "Next_Step": "下一步建议",
        },
        "chart_title": "### 📈 机构级风控分析 (Tear Sheet)",
        "chart_desc": "受回撤约束的样本外投资组合模拟。",
        "trace_strat": "因子投资组合",
        "trace_bench": "基准 (沪深300 / SPY)",
        "risk_metrics": [
            "年化收益率",
            "年化波动率",
            "夏普比率",
            "最大回撤",
            "卡玛比率",
        ],
        "glossary_title": "📖 指标定义与机构级标准",
        "glossary": {
            "信息系数 (IC)": "衡量因子得分与未来收益的相关性。<br>• <b>< 2%:</b> 噪音<br>• <b>2% - 5%:</b> 优质可交易 Alpha<br>• <b>> "
            "10%:</b> 警告 (极可能存在未来函数)",
            "夏普比率 (Sharpe)": "每单位波动率带来的超额收益。<br>• <b>< 1.0:</b> 较差<br>• <b>1.0 - 2.0:</b> 优秀<br>• <b>> "
            "3.0:</b> 警告 (需检查是否遗漏手续费/滑点)",
            "最大回撤 (MDD)": "资产从最高点到最低点的最大跌幅。CTA 基金因杠杆原因对此要求极严。<br>• <b>< 10%:</b> 极佳<br>• <b>> 20%:</b> "
            "无法在高杠杆下交易",
            "卡玛比率 (Calmar)": "年化收益率除以最大回撤 (每承担一单位回撤带来的收益)。<br>• <b>< 1.0:</b> 风险过高<br>• <b>> 2.0:</b> 表现优异",
        },
        "tab_tearsheet": "📊 风控分析 (Tear Sheet)",
        "tab_correlation": "🔗 相关性矩阵 (Correlation Matrix)",
        "corr_title": "因子正交化 (收益率相关性检测)",
        "corr_desc": "检测历史因子收益率之间的共线性。高相关性 (>0.7) 意味着重叠的风险敞口。",
        "heatmap_title": "🗓️ 月度收益率热力图",
        "vol_title": "📉 滚动 30 天年化波动率",
        "heatmap_help_title": "💡 如何解读热力图",
        "heatmap_help": "寻找**一致性**而非**高收益**。一个稳健的因子应该在不同年份均匀地分布着绿色的盈利月份。如果所有的利润都集中在某一年或某个月（例如只在2020年3月赚钱），模型很可能是对特定宏观事件的过度拟合，而不是捕捉到了真正的 "
        "Alpha。",
        "vol_help_title": "💡 如何解读滚动波动率",
        "vol_help": "这追踪了策略风险随时间的波动情况。因为我们的执行模块使用了**波动率缩放 (Volatility "
        "Scaling)**，这条线理想情况下应该保持相对平稳。巨大的尖峰表明策略在市场冲击期间失去了对风险敞口的控制。",
        "ic_decay_title": "📉 IC 衰减曲线",
        "ic_decay_desc": "检验所选特征的横截面预测力，是否能随着未来收益周期拉长而继续存在。未来收益使用“下一期开盘到未来收盘”计算，避免同一收盘价成交的泄漏。",
        "ic_decay_select_feature": "选择要检验的特征",
        "ic_decay_missing": "IC 衰减暂不可用",
        "ic_decay_no_features": "未找到可用于 IC 衰减的工程化特征列。",
        "ic_decay_no_data": "该特征没有有效的 IC 衰减样本。",
        "ic_decay_1d": "1日 IC",
        "ic_decay_peak": "峰值 |IC|",
        "ic_decay_days": "有效天数",
        "ic_decay_xaxis": "未来收益周期（天）",
        "ic_decay_yaxis": "日均 Spearman IC",
        "ic_decay_help_title": "💡 如何解读 IC 衰减",
        "ic_decay_help": "IC 是每日横截面 Spearman 相关系数：今日特征排序 vs 未来收益排序。国内期货可粗略参考：**<0.01 噪声**，**0.01-0.03 偏弱**，**0.03-0.05 "
        "有价值**，**>0.05 较强**。曲线很快回到 0 表示 alpha 生命周期很短；曲线稳定表示信号可持续更久。负 IC 表示该特征在相反方向上有预测力。",
        "dna_pnl_title": "💡 如何解读盈亏分布与价位散点",
        "dna_pnl": "**盈亏分布 (左上):** 关注尾部特征。如果绿色（亏损）柱形向左延伸的距离，远大于红色（盈利）柱形向右延伸的距离，说明单笔亏损远大于单笔盈利（负偏度）。<br><br>**开平仓价位 "
        "(左下):** 虚线是盈亏平衡点。如果你看到大量散点远离这条虚线，说明你的持仓周期或止损设置过于宽松。",
        "dna_time_title": "💡 如何解读持仓时间分布",
        "dna_time": "**持仓时间 (右上):** 统计套利模型应该具有紧凑、一致的持仓时间（例如 24-48 "
        "小时）。如果你发现有一个长尾分布，包含持仓数百小时的交易，说明模型被困在了一个不回归的价差中。<br><br>**盈亏 vs 时间 (右下):** "
        "交易的圣杯是“截断亏损，让利润奔跑”。如果你最大的红色气泡（亏损）集中在图表的最右侧（长持仓时间），说明你的模型在顽固地死扛亏损单。",
        "dna_total_trades": "交易笔数",
        "dna_win_rate": "胜率",
        "dna_profit_factor": "盈亏因子",
        "dna_median_hold": "中位持仓",
        "dna_payoff_ratio": "盈亏比",
        "dna_avg_win": "平均盈利",
        "dna_avg_loss": "平均亏损",
        "dna_profit_concentration": "80%盈利标的",
        "dna_expectancy": "单笔期望",
        "tab_dna": "🧬 策略 DNA",
        "tab_pareto": "📈 帕累托前沿 (Pareto)",
        "tab_ml": "🧠 机器学习特征重要性",
        "tab_assumptions": "🧾 假设清单",
        "assumptions_title": "🧾 因子假设清单",
        "assumptions_select_run": "查看运行",
        "assumptions_missing": "未找到该次运行的假设清单。",
        "assumptions_reconstructed": "该旧运行没有保存的 JSON 假设清单，因此此视图由研究运行记录重建。",
        "assumptions_expected_path": "预期 JSON 路径",
        "assumptions_not_saved": "未保存；当前为内存重建",
        "assumptions_reconstruction": "重建说明",
        "assumptions_available": "可用假设清单",
        "assumptions_manifest_path": "清单文件",
        "assumptions_data": "数据",
        "assumptions_data_health": "数据健康",
        "assumptions_data_health_note": "会计视图可以前向填充陈旧价格，但 Alpha 视图必须阻止合成输入。",
        "assumptions_data_health_missing": "数据健康快照不可用。",
        "assumptions_health_status": "状态",
        "assumptions_health_asset": "资产",
        "assumptions_health_timeframe": "周期",
        "assumptions_health_file": "样本文件",
        "assumptions_health_fresh": "新鲜占比",
        "assumptions_health_synthetic": "合成占比",
        "assumptions_health_expired": "过期行数",
        "assumptions_health_policy": "填充策略",
        "assumptions_signal": "信号与执行",
        "assumptions_engine": "执行引擎",
        "assumptions_costs": "成本与滑点",
        "assumptions_cost_readiness": "交易成本就绪状态",
        "assumptions_cost_field": "字段",
        "assumptions_cost_value": "当前值",
        "assumptions_cost_market": "市场",
        "assumptions_cost_profile": "成本配置",
        "assumptions_cost_source": "来源",
        "assumptions_cost_source_frozen": "已随本次运行冻结",
        "assumptions_cost_source_current": "当前默认值；未随历史运行冻结",
        "assumptions_cost_use_case": "结果口径",
        "assumptions_cost_schedule": "费率状态",
        "assumptions_cost_completeness": "完整度",
        "assumptions_cost_engine_support": "引擎支持",
        "assumptions_cost_net_ready": "研究净收益可用",
        "assumptions_cost_production_ready": "生产可用",
        "assumptions_cost_ready_production": "本次运行使用的成本配置可支持研究净收益与生产口径。",
        "assumptions_cost_ready_research": "该成本配置可支持研究净收益，但尚未获准用于生产口径。",
        "assumptions_cost_gross_only": "本次运行仅为毛收益研究，不得表述为净收益或生产可用。",
        "assumptions_cost_not_net_ready": "该费率尚未完整接入回测引擎，因此净收益结果尚未验证。",
        "assumptions_cost_historical_unfrozen": "该历史运行没有冻结交易成本配置。此处仅显示该市场当前默认值，不能据此验证历史净收益结果。",
        "assumptions_cost_required_work": "升级前仍需完成",
        "assumptions_cost_fingerprint": "冻结成本配置指纹",
        "assumptions_cost_current_fingerprint": "当前默认成本配置指纹",
        "assumptions_cost_yes": "是",
        "assumptions_cost_no": "否",
        "assumptions_liquidity": "流动性规则",
        "assumptions_option_selection": "期权合约选择",
        "assumptions_realized": "实际结果摘要",
        "assumptions_raw_json": "原始 JSON",
        "assumptions_download": "下载清单",
        "assumptions_asset": "资产",
        "assumptions_mode": "模式",
        "assumptions_alpha_col": "Alpha",
        "assumptions_trades": "交易数",
        "assumptions_no_values": "没有记录值。",
        "tab_oracle": "🔮 Oracle 状态验证",
        "oracle_title": "🔮 Oracle 市场状态验证",
        "oracle_caption": "将 GMM/HMM 状态概率与事后趋势、震荡、恐慌标签进行对比。该模块验证状态引擎，而不是普通因子收益。",
        "oracle_not_applicable": "Oracle 验证仅适用于具备状态感知的因子。",
        "oracle_missing": "缺少 Oracle 输入文件。",
        "oracle_scope": "验证样本范围",
        "oracle_scope_run": "当前回测实际交易资产池",
        "oracle_scope_all": "全部 GMM 资产",
        "oracle_loading": "正在运行 Oracle 状态验证...",
        "oracle_error": "Oracle 验证失败",
        "oracle_accuracy": "Oracle 准确率",
        "oracle_panic_auc": "恐慌 ROC-AUC",
        "oracle_samples": "标注样本数",
        "oracle_panic_rate": "Oracle / AI 恐慌占比",
        "oracle_panic_box": "按 Oracle 状态分组的 AI 恐慌概率",
        "oracle_state_mix": "各 Oracle 状态下的 AI 预测分布",
        "oracle_select_asset": "选择资产查看",
        "oracle_asset_drilldown": "单资产状态概率钻取",
        "oracle_state_map": "状态映射",
        "oracle_manual_title": "如何使用 Oracle 页面",
        "oracle_manual": "**这个页面回答的问题：** 无监督状态引擎是否识别了未来的趋势、震荡、恐慌结构，而不是策略本身是否赚钱。\n"
        "\n"
        "**使用流程：**\n"
        "1. 先选择验证样本范围。如果想看当前因子实际交易过的资产，就选当前回测实际交易资产池；如果想做整体模型体检，就选全部 GMM 资产。\n"
        "2. 先读顶部指标。Accuracy 是整体状态匹配度。Panic ROC-AUC 更重要，因为它衡量 AI 的恐慌概率是否能把未来崩盘窗口和普通窗口区分开。\n"
        "3. 再看混淆矩阵，判断模型错在哪里。把恐慌误判成震荡，比把趋势误判成震荡危险得多。\n"
        "4. 最后用资产下拉框做单合约局部检查。切换资产时，单资产指标和下方图表会更新。",
        "oracle_metrics_help_title": "如何解读顶部指标",
        "oracle_metrics_help": "**Oracle 准确率** 是基于上方选择的验证样本范围计算的。切换下方单资产下拉框时，它不会改变。\n"
        "\n"
        "**如何判断准确率：** 三分类问题中，约 33% 接近平衡随机猜测。但由于震荡状态通常占样本多数，也要与“永远预测多数类”的基准比较。粗略区间：**<40% "
        "偏弱**，**40-50% 有轻微信号**，**50-60% 有使用价值**，**>60% 较强**。所以 8 万样本下 **33.8%** "
        "的准确率表示当前状态语义仍然偏弱，模型还没有稳定匹配 Oracle 标签。\n"
        "\n"
        "**恐慌占比** 不是准确率，而是未来恐慌窗口的基础频率。例如 **19.2% Oracle Panic** 表示每 100 个 20 日窗口中，大约 19 个之后出现了 "
        "Oracle 定义的恐慌回撤。将它与 AI Panic Rate 比较：AI 明显更高可能过度保守、误报较多；AI 明显更低可能对尾部风险不敏感。\n"
        "\n"
        "**恐慌 ROC-AUC** "
        "通常比总体准确率更重要。**0.50=随机**，**0.55=非常弱的边际优势**，**0.60-0.65=中等偏弱**，**0.65-0.70=较有价值**，**>0.70=较强**，**<0.50=信号方向可能反了**。类似 "
        "**0.549** 的值意味着 AI 恐慌概率仅略好于随机排序，只能轻微地区分未来恐慌窗口和普通窗口。",
        "oracle_confusion_help_title": "如何解读混淆矩阵",
        "oracle_confusion_help": "行是事后 Oracle 标签，列是 AI 预测标签。对角线代表预测正确，非对角线代表误判。\n"
        "\n"
        "**数字含义：** Count 模式下，每个格子是该真实状态/预测状态组合的样本数量。Row % 模式下，每一行加总为 100%，可以直接看每个真实状态的召回率。例如 "
        "True Panic -> Pred Panic 就是 Oracle 恐慌窗口中被 AI 成功识别为恐慌的比例。\n"
        "\n"
        "**颜色含义：** 颜色越亮/越深，表示该格子的样本数量或行内比例越高，取决于当前模式。对角线颜色亮是好事；True Panic -> Pred Trend/Chop "
        "颜色亮则很危险，因为这表示未来恐慌窗口中模型仍然暴露在趋势或震荡状态。",
        "oracle_confusion_mode": "混淆矩阵显示方式",
        "oracle_confusion_counts": "样本数",
        "oracle_confusion_row_pct": "行百分比",
        "oracle_confusion_title_counts": "混淆矩阵（样本数）",
        "oracle_confusion_title_pct": "混淆矩阵（行百分比）",
        "oracle_classification_report": "分类报告",
        "oracle_probability_help_title": "如何解读概率诊断图",
        "oracle_probability_help": "**按 Oracle 状态分组的 AI 恐慌概率：** Oracle Trend/Chop 下恐慌概率应较低，Oracle Panic "
        "下应明显更高。如果三个箱体高度重叠，说明模型没有有效分离危机结构。\n"
        "\n"
        "**AI 预测分布：** 用来观察模型是否偏向某一个状态。如果所有 Oracle 标签下都被同一种颜色主导，GMM 可能退化成了几乎静态的分类器。",
        "oracle_asset_help_title": "如何使用单资产钻取图",
        "oracle_asset_help": "资产下拉框只改变下方钻取区域，以及图表上方的单资产指标。顶部汇总指标仍然属于所选验证样本范围。\n"
        "\n"
        "图中灰线是价格，背景色表示 AI 的主导状态：绿色 = 平稳/趋势，橙色 = 震荡，红色 = 恐慌。红色 x 标记表示 Oracle "
        "事后认定的未来恐慌窗口。一个有用的护盾应该在红色 x 聚集之前或期间，切换到橙色/红色防御区域。将鼠标悬停在价格线上，可以看到底层 Quiet/Chop/Panic "
        "概率，不需要再用多条线挤满图表。",
        "oracle_asset_scope_note": "切换资产下拉框时，单资产指标会更新；顶部汇总指标保持为所选样本范围。",
        "oracle_asset_accuracy": "单资产准确率",
        "oracle_asset_panic_auc": "单资产恐慌 ROC-AUC",
        "oracle_asset_samples": "单资产样本数",
        "oracle_asset_panic_rate": "单资产 Oracle / AI 恐慌占比",
        "oracle_asset_no_data": "该资产没有可用的价格/状态数据。",
        "tab_conditional_perf": "🏛️ 条件策略绩效",
        "exec_dd_title": "### 📉 状态模型带来的回撤节省",
        "exec_dd_desc": "对比基线 SMA/Bollinger 与最终 Layer 4 Router 的最大回撤。",
        "exec_to_title": "### 🔁 换手率演化（二值 → 连续 → 离散）",
        "exec_to_desc": "量化连续化与离散化对执行摩擦损耗的改善。",
        "exec_missing": "缺少关键回测记录。请先运行基线与 Layer 4 回测。",
        "exec_overlay_title": "### 📈 叠加净值曲线（基准 vs SMA vs Layer 4）",
        "exec_overlay_desc": "对比累计净值路径，直观看到崩盘期回撤与保本能力差异。",
        "exec_crisis_title": "### 🚨 危机窗口放大（2024-01-01 至 2024-02-28）",
        "exec_crisis_desc": "聚焦指定危机区间，验证防御路由是否有效。",
        "exec_pareto2_title": "### 🎯 消融帕累托图（年化波动 vs 年化收益）",
        "exec_pareto2_desc": "五步消融路径：X轴风险，Y轴收益。",
        "exec_overlay_missing": "无法生成叠加净值曲线。请确保至少存在 SMA 基线与 Layer 4 的收益日志。",
        "select_run_hint": "👈 请先从左侧运行记录中选择一个已完成的回测以查看详细分析。",
        "blocked_run_title": "⛔ 尚未回测——该试验在预检阶段被阻断。",
        "blocked_run_reason": "**阻断原因：** {reason}",
        "blocked_run_reason_missing": "未记录具体阻断原因。",
        "blocked_run_explainer": "这是一条计划中的因子与持仓规则试验审计记录，并非执行失败的回测。解决阻断项之前，不应存在收益序列。",
        "missing_artifact_title": "⚠️ 该运行记录没有可读取的收益文件，因此无法显示分析。",
        "ml_caption": "可视化机器学习模型内部决策权重。",
        "ml_missing": "ℹ️ 未找到特征重要性数据。选择由机器学习模型生成的回测后将自动显示。",
        "ts_no_returns": "⚠️ 未找到该次运行的收益数据，可能执行失败。",
        "strategy_net": "#### 🟢 策略（净值，含费用）",
        "benchmark_eq": "#### 🟡 基准",
        "avg_turnover": "平均换手率",
        "holdout_ic": "样本外 IC",
        "total_trades": "总交易笔数",
        "test_scope": "运行：{run_id} | 下方指标/图表使用的回测结果区间：{start} 至 {end} | 收益行数：{rows:,}（约 {years:.2f} 年化）| 准备后数据：{prepared_window} | 请求过滤：{requested_window} | 频率：{frequency} | 数据：{role} | 收益口径：{return_clock} | 来源：{source}",
        "raw_entries": "原始开仓信号",
        "raw_exits": "原始平仓信号",
        "target_weight_changes": "目标权重变化次数",
        "active_signal_rows": "活跃信号行数",
        "context_asset_class": "资产类别",
        "context_traded_universe": "交易池",
        "context_run": "运行",
        "context_window": "结果区间",
        "context_rows": "收益行数",
        "context_prepared_data": "准备后数据",
        "context_requested_filter": "请求过滤",
        "context_frequency": "频率",
        "context_data": "数据",
        "context_return_clock": "收益口径",
        "context_source": "来源",
        "tearsheet_benchmark_guide": "这些基准是什么意思",
        "tearsheet_benchmark_guide_intro": "上方基准指标卡使用主基准 `benchmark_return`。如果存在额外基准列，它们会在累计收益图中以虚线显示。",
        "tearsheet_benchmark_guide_headers": ["位置", "基准", "收益列", "含义"],
        "benchmark_slot_primary": "主基准",
        "benchmark_slot_secondary": "辅助基准",
        "benchmark_slot_control": "控制组",
        "benchmark_slot_additional": "额外基准",
        "benchmark_mode_label_same_horizon": "同周期",
        "benchmark_mode_label_passive_close_to_close": "被动收盘到收盘",
        "benchmark_mode_label_unknown": "口径未记录",
        "benchmark_role_active_universe_futures_basket": "因子有效合约池的动态等权篮子。它适合作为市场环境参考，但不一定和日内信号使用同一个收益时钟。",
        "benchmark_role_same_horizon_universe_control": "在与策略相同信号/执行周期上度量的活跃合约池等权基准。对 next-bar 或日内因子来说，这通常是最公平的比较对象。",
        "benchmark_role_broad_commodity_index": "宽基商品指数参考，例如有可用数据时的南华商品指数。",
        "benchmark_role_asset_class_market_index": "资产类别市场指数的买入持有参考。",
        "benchmark_role_style_context_index": "辅助风格或市场指数参考。",
        "benchmark_role_full_universe_equal_weight": "完整可交易池的等权篮子，用来区分选品能力和大盘/全市场漂移。",
        "benchmark_role_active_crypto_universe": "活跃加密货币池的买入持有篮子。",
        "benchmark_mode_same_horizon": "使用与策略相同的收益时钟构造，因此可以审计 alpha 是否真的跑赢了同一测试周期下的朴素活跃池敞口。",
        "benchmark_mode_passive_close_to_close": "被动收盘到收盘持有收益。它适合观察市场状态，但不是日内信号的严格同周期执行基准。",
        "benchmark_mode_unknown": "运行 manifest 没有记录该收益构造；只能作为上下文参考，并应审计对应收益列来源。",
        "tearsheet_quality_guide": "如何解读指标标签",
        "tearsheet_quality_guide_text": """
这些标签是**诊断性启发式提示，不是晋级门槛**。如果存在同周期基准，策略收益和波动率会与当前显示的基准比较。夏普、卡玛、回撤、换手率和样本外 IC 使用绝对研究区间。交易数和活跃信号行反映样本覆盖度，但不能证明观测相互独立或策略稳健。开仓、平仓和目标权重变化次数只表示执行诊断已被记录。

关键区间：夏普 `<0.5` 偏弱，`1-2` 良好，`2-3` 较强，`>=3` 需审计；样本外 IC `<1%` 偏弱，`1-3%` 中等，`3-5%` 良好，`5-10%` 较强，`>=10%` 触发完整性审计；平均每日换手率 `<5%` 较低，`5-20%` 中等，`20-50%` 较高，`>=50%` 极高。还应检查成本、年化方式、重叠收益、试验次数和跨市场状态稳定性。
""",
        "tearsheet_quality_labels": {
            "not_available": "暂无数据",
            "not_applicable": "不适用",
            "negative": "负值",
            "low": "较低",
            "positive": "正收益",
            "below_benchmark": "低于基准",
            "ahead": "领先基准",
            "in_line": "接近基准",
            "lower_risk": "风险较低",
            "very_low": "极低",
            "controlled": "受控",
            "moderate": "中等",
            "elevated": "偏高",
            "high": "高",
            "very_high": "极高",
            "weak": "偏弱",
            "marginal": "临界",
            "modest": "中等",
            "good": "良好",
            "strong": "较强",
            "audit": "需审计",
            "high_audit": "偏高 - 需审计",
            "extreme_audit": "极高 - 需审计",
            "severe": "严重",
            "none": "无",
            "sparse": "稀疏",
            "limited": "有限",
            "adequate": "充足",
            "broad": "样本较广",
            "missing": "未记录",
            "recorded": "已记录",
            "reference": "基准参考",
        },
        "tearsheet_quality_help": {
            "return_unavailable": "没有可用的有限年化收益率。",
            "return_negative": "评估区间内的年化收益率非正。",
            "return_below_benchmark": "收益为正，但比当前同周期基准低超过 20 个基点。",
            "return_ahead": "年化收益率比当前同周期基准高超过 20 个基点。",
            "return_in_line": "年化收益率与当前同周期基准相差不超过 20 个基点。",
            "return_low": "年化收益为正但低于 2%；需要结合风险和成本判断。",
            "return_positive": "年化收益为正常正值，但仍需结合风险调整后证据。",
            "return_high_audit": "年化收益超过 30%；应审计年化方式、杠杆、成本、数据泄漏和测试窗口。",
            "volatility_unavailable": "没有可用的有限年化波动率。",
            "volatility_flat_audit": "接近零的波动率可能意味着无敞口、陈旧价格或收益构造问题。",
            "volatility_lower": "年化波动率比同周期基准低至少 25%；需确认并非仅因敞口过低。",
            "volatility_in_line": "年化波动率与同周期基准相差不超过 25%。",
            "volatility_very_low": "年化波动率低于 2%；需确认策略确实处于活跃状态。",
            "volatility_controlled": "年化波动率处于 2% 至 10%。",
            "volatility_moderate": "年化波动率处于 10% 至 20%。",
            "volatility_elevated": "波动率相对基准偏高，或绝对年化波动率超过 20%。",
            "volatility_high": "波动率相对基准很高，或绝对年化波动率超过 35%。",
            "sharpe_unavailable": "没有可用的有限夏普比率。",
            "sharpe_negative": "风险调整后收益非正。",
            "sharpe_weak": "夏普低于 0.5，风险调整后证据偏弱。",
            "sharpe_marginal": "夏普处于 0.5 至 1.0，可能可用但仍较脆弱。",
            "sharpe_good": "夏普处于 1.0 至 2.0；若确为样本外且计入成本，则较可信。",
            "sharpe_strong": "夏普处于 2.0 至 3.0；证据较强，但仍需检查稳定性与试验筛选。",
            "sharpe_high_audit": "夏普处于 3.0 至 5.0，异常偏高，需要完整性审计。",
            "sharpe_extreme_audit": "夏普至少为 5.0；应审计泄漏、年化方式、重叠收益、成本和曲线拟合。",
            "drawdown_unavailable": "没有可用的有限最大回撤。",
            "drawdown_none_audit": "接近零的回撤可能来自短样本、低敞口、陈旧价格或构造问题。",
            "drawdown_controlled": "最大回撤幅度不超过 5%。",
            "drawdown_moderate": "最大回撤幅度处于 5% 至 10%。",
            "drawdown_elevated": "最大回撤幅度处于 10% 至 20%。",
            "drawdown_severe": "最大回撤幅度超过 20%。",
            "calmar_unavailable": "没有可用的有限卡玛比率。",
            "calmar_negative": "相对于回撤，年化收益非正。",
            "calmar_weak": "卡玛低于 0.5，收益相对峰谷损失偏弱。",
            "calmar_modest": "卡玛处于 0.5 至 1.0。",
            "calmar_good": "卡玛处于 1.0 至 2.0。",
            "calmar_strong": "卡玛处于 2.0 至 5.0。",
            "calmar_high_audit": "卡玛至少为 5.0；需核查回撤样本和收益构造。",
            "turnover_unavailable": "没有可用的平均每日换手率。",
            "turnover_low": "平均每日资本换手低于 5%，执行摩擦通常较易控制。",
            "turnover_moderate": "平均每日资本换手处于 5% 至 20%。",
            "turnover_high": "平均每日资本换手处于 20% 至 50%；需重点检查成本和容量。",
            "turnover_very_high": "平均每日资本换手至少为 50%；策略高度暴露于成本、滑点和容量约束。",
            "ic_unavailable": "没有可用的未触碰样本外 IC。",
            "ic_negative": "样本外 IC 非正。",
            "ic_weak": "样本外 IC 为正但低于 1%。",
            "ic_modest": "样本外 IC 处于 1% 至 3%。",
            "ic_good": "样本外 IC 处于 3% 至 5%。",
            "ic_strong": "样本外 IC 处于 5% 至 10%。",
            "ic_extreme_audit": "样本外 IC 至少为 10%；应审计前视偏差、目标泄漏和试验筛选。",
            "trades_unavailable": "没有记录离散交易笔数。",
            "trades_none": "没有记录离散交易。",
            "trades_sparse": "少于 30 笔交易，推断对样本非常敏感。",
            "trades_limited": "30 至 99 笔交易；可供研究但样本仍有限。",
            "trades_adequate": "100 至 299 笔交易；数量较充足，但仍需考虑独立性。",
            "trades_broad": "至少 300 笔交易；数量覆盖较广，但不代表观测相互独立。",
            "diagnostic_missing": "本次运行没有记录该执行诊断。",
            "diagnostic_none": "该诊断已记录，但没有事件。",
            "diagnostic_recorded": "执行事件已记录。该数量仅用于描述，不是质量分数。",
            "signal_rows_missing": "没有记录活跃信号行覆盖度。",
            "signal_rows_limited": "活跃信号行少于 100，覆盖度有限。",
            "signal_rows_moderate": "活跃信号行处于 100 至 499。",
            "signal_rows_broad": "活跃信号行至少为 500；覆盖较广，但不一定是独立证据。",
            "reference_control": "被动基准仅用于参考，不适用换手率和交易数评级。",
            "not_applicable": "该指标不适用于被动基准。",
        },
        "axis_cum_return": "累计收益 (%)",
        "axis_drawdown": "回撤 (%)",
        "axis_gross_leverage": "总风险敞口 (x)",
        "axis_date": "日期",
        "leaderboard_title": "### 🏆 Alpha 排行榜",
        "corr_select_factors": "🔍 选择要对比的因子",
        "corr_select_min": "👆 请至少选择两个因子以查看相关性矩阵。",
        "corr_waiting": "⏳ 等待更多因子... 至少需要 2 个成功回测才能生成相关性矩阵。",
        "dna_trade_caption": "交易级别分析（统计套利 / 配对）",
        "dna_portfolio_caption": "横截面日度净值盈亏频率（组合级）",
        "dna_error": "⚠️ 交易分析渲染失败",
        "dna_no_returns": "⚠️ 未找到该次运行的收益数据。",
        "pareto_caption": "可视化该因子在历史迭代中的执行边界。",
        "pareto_help": "💡 解读：优先关注高收益、低回撤幅度、低换手率，并且样本外 IC 颜色更强的迭代。气泡大小代表样本外 IC 的绝对值。",
        "pareto_axis_drawdown": "回撤幅度 (%)",
        "pareto_axis_return": "收益率 (%)",
        "pareto_axis_turnover": "换手率 (%)",
        "pareto_color_ic": "样本外 IC",
        "pareto_size_abs_ic": "IC绝对值",
        "pareto_round": "轮次",
        "pareto_error": "⚠️ 帕累托前沿渲染失败，数据可能不一致",
        "pareto_missing": "⚠️ 数据不足。请至少运行该因子两次以生成帕累托前沿。",
        "asset_zoo_title": "🦁 资产动物园：市场微观结构聚类",
        "asset_zoo_subtitle": "使用高斯混合模型（GMM）对资产微观特征聚类，用于策略定向。",
        "asset_zoo_missing": "❌ 未找到画像数据。请先运行 `data_engine/asset_profiler.py`。",
        "asset_cluster_dist": "### 🧬 聚类分布",
        "asset_tier1": "第一层（趋势 / 突破）",
        "asset_tier2": "第二层（均值回归）",
        "asset_tier3": "第三层（高毒性 / 低流动性）",
        "asset_radar": "### 🔭 市场雷达（风险 vs 趋势效率）",
        "asset_target_board": "### 🎯 策略定向看板",
        "asset_select_target": "选择策略目标：",
        "asset_copy_list": "**可复制目标池（Python 列表）：**",
        "exec_page_title": "🏛️ 高管看板",
        "exec_page_subtitle": "条件策略绩效",
        "regime_page_title": "🏛️ 市场状态刻画与分析",
        "regime_page_subtitle": "可视化 HMM 连续状态概率与价格叠加。",
        "regime_select_asset": "选择要叠加的资产：",
        "dna_gmm_caption": "🧠 GMM 状态分析：追踪 AI 的防御性现金保护（熔断机制）。",
        "dna_gmm_regime_alloc": "### 市场状态时间分配",
        "dna_gmm_active": "活跃交易敞口",
        "dna_gmm_cash": "防御性空仓 (恐慌熔断)",
        "dna_gmm_cum_pnl": "### 累计盈亏：活跃交易 vs 空仓平滑",
        "dna_gmm_port_val": "投资组合净值",
        "dna_gmm_cash_zones": "现金避险区间",
        "dna_gmm_total_days": "总交易日",
        "dna_gmm_days_cash": "空仓避险天数",
        "dna_detected_factor": "检测到因子",
        "dna_architecture": "策略架构",
        "exec_dash_title": "🏛️ 机构级量化架构",
        "exec_dash_evo_title": "### 🧠 策略演进史",
        "exec_dash_evo_desc": "我们的系统从静态趋势跟踪基线演变为AI驱动、具备状态感知的组合策略。我们证明了单纯的预测胜率若无执行风控将毫无意义，最终迭代出滚动前向微观结构模型。",
        "exec_v1_title": "#### V1: 静态技术指标基线",
        "exec_v1_desc": "**SMA (fac_051)** 使用经典的 50/200 日均线金叉/死叉来捕捉宏观趋势。**Bollinger (fac_040)** 使用 20 日均线及 2 "
        "倍标准差轨来进行均值回归交易。两者都证明了静态的线性逻辑在现代震荡市中反应过慢，风险调整后收益极差。",
        "exec_v2_title": "#### V2: 纯机器学习",
        "exec_v2_desc": "**纯机器学习 (fac_054)** 使用 XGBoost 非线性决策树取代线性数学，基于 100+ 量价特征预测 T+1 日收益。虽然它成功找到了高胜率的微观均值回归特征，但 AI "
        "交易过于频繁（信号闪烁）。在缺乏执行护盾的情况下，换手率带来的摩擦成本（平方根 TCA 滑点）完全摧毁了超额收益。",
        "exec_v3_title": "#### V3: 隐马尔可夫路由",
        "exec_v3_desc": "**隐马尔可夫路由 (fac_056)** 引入了 HMM 作为中央流量控制器。它通过追踪历史方差将市场划分为 3 个隐状态：趋势、震荡和危机。在趋势中资金路由至 SMA，在震荡中路由至 "
        "XGBoost，并在检测到危机概率超过 63% 时，触发 100% 强制空仓的恐慌熔断机制。",
        "exec_v4_title": "#### V4: 微观肥尾防御",
        "exec_v4_desc": "**微观肥尾防御 (fac_057)** 将引擎升级为高斯混合模型 (GMM)，并采用滚动前向训练以消除未来函数。它不再仅依赖价格，而是对市场微观结构（Amihud 非流动性与 GK "
        "波动率）进行聚类，在价格实质下跌前从数学上预测肥尾崩盘。资金通过惰性执行安全杠杆化至 12% 目标波动率。",
        "exec_metrics_ret": "收益",
        "exec_metrics_vol": "波动",
        "exec_metrics_shp": "夏普",
        "exec_metrics_mdd": "回撤",
        "exec_chart_cum_title": "### 📈 累计收益与风险敞口",
        "exec_chart_cum_sub": "**Alpha 雪山 (累计净值)**",
        "exec_chart_dd_sub": "**历史回撤**",
        "exec_chart_y_eq": "累计净值 (基准 1.0)",
        "exec_chart_y_dd": "回撤幅度 (%)",
        "exec_xray_title": "### 🧬 策略 X 光透视：V4 资产偏好与交易画像",
        "exec_xray_desc": "深入剖析 V4 GMM 微观结构引擎，分析在 8% 惰性执行机制下，AI 将资金部署在了哪些资产，以及持仓周期的分布情况。",
        "exec_xray_pnl_title": "**V4: 最赚钱的 15 个资产**",
        "exec_xray_hold_title": "**V4: 单笔交易持仓时间分布 (小时)**",
        "exec_layer4_title": "### 🛡️ 第四层：深度执行与风控护盾",
        "exec_layer4_desc": "未经处理的原始信号在真实市场中是无法交易的。我们部署了**贝叶斯优化器（Optuna）**，利用严格的样本外惩罚墙从数学层面推导出极限生存参数。\n"
        "\n"
        "* **目标波动率杠杆 (3.2x):** 纯粹的标准化逻辑，安全放大投资组合资金，以满足机构 12% 目标波动率的要求。\n"
        "* **惰性交易离散化 (8%):** 解决“量化陷阱”。在 3.2 倍杠杆下，微小的信号噪音会导致致命的换手率。Optuna 证明我们必须强制引擎仅以 8% "
        "的大区块进行交易，让执行网关的死区处理噪音。\n"
        "* **指数平滑避震器 (EWMA):** Optuna 将记忆因子降低至 0.20，减缓 AI 的执行反射，保护其免受平方根交易成本滑点的影响。\n"
        "* **微观肥尾熔断:** 当 GMM 检测到肥尾崩盘且确信度极高时，路由引擎会强制将总敞口降至 0.0（全现金）。\n"
        "\n"
        "**🔬 微观结构指标定义:**\n"
        "* **Garman-Klass 波动率 (GK):** 与简单的收盘价波动率不同，GK "
        "综合了日内最高价、最低价、开盘价和收盘价，能够精准捕捉隐藏的日内剧烈波动与市场结构性压力。\n"
        "* **Amihud 非流动性指标:** 计算每日绝对收益率与每日成交额的比值。它衡量的是“价格冲击”——即每交易一美元带来的价格变动幅度。Amihud "
        "的急剧飙升意味着市场流动性真空，通常是系统性崩盘的前兆。",
        "regime_title": "🏛️ 市场状态刻画与分析",
        "regime_subtitle": "基于微观结构与滚动前向 GMM 的市场状态概率可视化。",
        "regime_insight_title": "💡 如何解读引擎室图表",
        "regime_stress_title": "微观结构压力测试 (Z-Score 标准化)",
        "regime_feat_gk": "GK 波动率 (日内方差)",
        "regime_feat_amihud": "Amihud (流动性真空)",
        "regime_phase_title": "协方差相空间图 (AI 决策大脑)",
        "regime_phase_desc": "历史微观结构散点图。只有当数据沿对角线进入高协方差的红色象限时，AI 才会触发熔断。",
        "regime_boundary": "理论熔断边界",
        "regime_insight_desc": "### 💡 如何解读引擎室图表\n"
        "**1. 引擎大脑 (GMM-HMM):** AI 并不依赖价格下跌来判断风险，而是实时监控市场微观结构（Amihud 非流动性与 GK 波动率）。\n"
        "**2. 相空间散点图 (右侧):** 可以将其视为 AI 的风险雷达。\n"
        "* **左下角:** 安全区。流动性与波动率均处于正常水平（绿点）。\n"
        "* **右上角:** 危险区。当订单薄深度蒸发且日内方差同时扩大时，数据点会移向右上角。AI 会从数学上将其归类为“肥尾崩盘”（红点）。\n"
        "**3. 熔断弹射 (左侧):** 当红色区域（恐慌概率）越过 **75.9%** 的虚线时，动态路由将断开 AI 连接，并将投资组合强制转换为 100% 现金避险。",
        "regime_state0": "状态 0: 平稳 / 趋势",
        "regime_state1": "状态 1: 震荡 / 均值回归",
        "regime_state2": "状态 2: 恐慌 / 肥尾崩盘",
        "regime_panic_thresh": "恐慌熔断阈值 (75.9%)",
        "regime_ejection": "现金熔断",
        "regime_close_price": "收盘价",
        "regime_y_price": "价格",
        "regime_y_prob": "概率",
        "regime_feat_ker": "趋势效率 (KER)",
        "regime_profiler_title": "📊 市场状态剖析：经济学翻译",
        "regime_profiler_desc": "量化AI无监督聚类下的物理市场现实。",
        "regime_dna_matrix": "**市场状态 DNA 矩阵**",
        "regime_stress_profile": "**微观结构压力分布**",
        "col_state": "市场状态",
        "col_time": "时间占比",
        "col_return": "平均日收益",
        "col_vol": "波动率 (Z)",
        "col_liq": "非流动性 (Z)",
        "col_ker": "趋势效率",
        "regime_gmm_title": "🌌 GMM 概率密度分布 (肥尾效应可视化)",
        "expander_ts_title": "💡 如何解读时间序列与 Z-Score 模式",
        "expander_ts_text": "**1. 理解 Z-Scores**\n"
        "- **波动率 (GK Z):** >0 表示高于历史均值。\n"
        "- **非流动性 (Amihud Z):** >0 表示市场流动性稀薄（高滑点）。\n"
        "- **趋势效率 (KER Z):** >0 代表趋势清晰；<0 代表震荡洗盘。\n"
        "\n"
        "**2. 模式识别**\n"
        "- **State 0 (趋势):** 波动率/非流动性在 0 轴附近平稳。KER 保持正值。\n"
        "- **State 1 (震荡):** KER 深度跌入负值区域。价格停滞不前。\n"
        "- **State 2 (恐慌):** 波动率与非流动性同时出现向上尖峰。",
        "expander_phase_title": "💡 如何解读 3D 相空间",
        "expander_phase_text": "这张 3D 散点图证明了为何我们必须引入第三维度 (KER)。\n"
        "\n"
        "如果您将图表旋转至仅观察 X/Y 轴（波动率/非流动性），绿色（趋势）与橙色（震荡）的数据点会完全重叠。但当我们观察 Z 轴（趋势效率）时，AI "
        "通过“海拔高度”将它们完美分离。这彻底解决了“语义鸿沟”问题。",
        "expander_gmm_title": "💡 如何解读 3D 概率分布 (肥尾效应)",
        "expander_gmm_text": "**您在看什么？**\n"
        "这展示了二元正态概率密度函数 (PDF)。\n"
        "\n"
        "- **绿色尖峰 (State 0):** 高耸且狭窄。“平静”的日子高度可预测。\n"
        "- **红色泥潭 (State 2):** 扁平并向高 Z-Score 区域延伸。这就是**肥尾 (Fat Tail)**。\n"
        "- **空间位移 (位置):** 为什么红色分布大幅偏右？因为坐标轴是 Z-Score。0.0 代表历史均值。绿/橙聚集在左侧 (< 0)，代表低于平均的平静日。红色右移 (> "
        "2)，因为危机在物理上表现为对均值的巨大正向偏离。",
        "expander_profiler_title": "💡 如何解读状态 DNA 与特征区间",
        "expander_profiler_text": "**预期数学区间:**\n"
        "- **State 0 (趋势):** Vol/Liq 介于 [-1.0 到 0.0]。KER > 0.5。市场流动性充足且趋势良好。部署最大杠杆。\n"
        "- **State 1 (震荡):** Vol/Liq 介于 [-0.5 到 1.0]。KER < 0.0。市场陷入停滞。Alpha 转移至 XGBoost "
        "均值回归引擎。\n"
        "- **State 2 (恐慌):** Vol/Liq > 2.0。市场微观结构正在崩溃。严格规避风险。",
    },
}

RESEARCH_PAGE_TEXT: dict[str, dict[str, dict[str, Any]]] = {
    "discovery_lab": {
        "en": {
            "title": "Discovery Lab",
            "subtitle": "Find recurring behavior, test defined events, and investigate cross-asset relationships without mixing their evidence contracts.",
            "mode_label": "Discovery workflow",
            "pattern": "Pattern Scan",
            "event_study": "Event Study",
            "relationships": "Relationship Scan",
        },
        "zh": {
            "title": "发现研究",
            "subtitle": "在不混淆证据合约的前提下，寻找重复行为、检验明确事件，并研究跨资产关系。",
            "mode_label": "发现研究流程",
            "pattern": "模式扫描",
            "event_study": "事件研究",
            "relationships": "关系扫描",
        },
    },
    "strategy_construction": {
        "en": {
            "title": "Strategy Construction",
            "subtitle": "Build a strategy by combining factor scores, then test whether completed strategy sleeves should be routed by market state.",
            "boundary": "Factor Blend combines comparable predictive scores before execution. Strategy Router allocates between already completed and independently tested sleeves. These are consecutive stages, not interchangeable weighting methods.",
            "mode_label": "Construction stage",
            "factor_blend": "Factor Blend",
            "strategy_router": "Strategy Router",
        },
        "zh": {
            "title": "策略构建",
            "subtitle": "先组合因子分数形成策略，再检验是否应按市场状态在已完成策略模块之间进行路由。",
            "boundary": "因子组合在执行前合并可比的预测分数；策略路由则在已经独立测试的完整策略模块之间配置资金。两者是前后相接的阶段，不是可互换的权重方法。",
            "mode_label": "构建阶段",
            "factor_blend": "因子组合",
            "strategy_router": "策略路由",
        },
    },
    "factor_portfolio_lab": {
        "en": {
            "title": "Factor Blend",
            "subtitle": "Compose compatible factor scores into one auditable strategy before shared execution and backtesting.",
            "boundary": "Factor weights combine comparable predictive scores. Asset weights are created afterward by the execution stage. Regime routers operate later across completed sleeves.",
            "market": "Market vertical",
            "router_registry": "Router Registry",
            "router_id": "Router ID",
            "router_name": "Router",
            "router_status": "Status",
            "router_frequency": "Frequency",
            "router_state": "State column",
            "router_lag": "Decision lag",
            "include_advanced": "Include states, sleeves, and routers",
            "include_advanced_help": "Normally keep this off. Raw factor blending is for comparable scores; states and complete sleeves belong in routing or capital allocation.",
            "select_factors": "Select factors",
            "need_factors": "Select at least two compatible factors.",
            "data_file": "Research data file",
            "no_data": "No local parquet or CSV files were found for this market.",
            "weighting": "Factor weighting",
            "equal": "Equal",
            "static": "Static",
            "normalization": "Normalization",
            "cross_sectional_zscore": "Cross-sectional z-score",
            "cross_sectional_rank": "Cross-sectional rank",
            "raw": "Raw (already comparable)",
            "missing_policy": "Missing-factor policy",
            "renormalize_available": "Renormalize available",
            "complete_case": "Require every factor",
            "zero": "Treat missing as zero",
            "min_available": "Minimum available factors",
            "winsor_limit": "Z-score cap",
            "strategy_id": "Strategy ID",
            "strategy_name": "Strategy name",
            "weight": "Weight",
            "orientation": "Orientation",
            "long_orientation": "Normal",
            "inverse_orientation": "Invert",
            "build": "Build composite preview",
            "building": "Computing factor scores and validating contracts...",
            "build_success": "Composite signal built successfully.",
            "build_stale": "Construction settings changed. Build a new preview before running the backtest.",
            "download": "Download strategy YAML",
            "tabs": ["Construction", "Diagnostics", "Backtest", "Attribution & Robustness"],
            "compatibility": "Factor Contract Compatibility",
            "weights_title": "Declared Factor Weights",
            "factor": "Factor",
            "category": "Category",
            "factor_family": "Factor family",
            "data_frequency": "Data frequency",
            "governance_status": "Registry status",
            "component_type": "Component type",
            "supported_markets": "Supported markets",
            "geometry": "Geometry",
            "execution_mode": "Execution mode",
            "execution_lag": "Execution lag",
            "return_assumption": "Return assumption",
            "source": "Source",
            "preview_required": "Build the composite preview to inspect signal overlap and coverage.",
            "rows": "Rows",
            "valid_scores": "Valid scores",
            "assets": "Assets",
            "date_range": "Date range",
            "factor_correlation": "Normalized Factor Correlation",
            "factor_correlation_help": "High absolute correlation means two factors may be paying for similar information. It is not automatically bad, but it reduces diversification.",
            "coverage_title": "Factor Signal Coverage",
            "coverage": "Coverage",
            "contribution_title": "Composite Signal Contribution",
            "mean_abs_contribution": "Mean absolute contribution",
            "correlation_to_composite": "Correlation to composite",
            "backtest_title": "Shared Engine Backtest",
            "backtest_desc": "The composite score now enters the same execution mode and AlphaEvaluator used by single-factor research. This creates one netted position book and one cost model.",
            "split_mode": "Split mode",
            "validation_fraction": "Training/validation fraction",
            "purge_periods": "Purge periods",
            "embargo_periods": "Embargo periods",
            "run_backtest": "Run factor-portfolio backtest",
            "running_backtest": "Running shared execution and evaluation...",
            "run_success": "Backtest completed: {run_id}",
            "latest_run": "Latest factor-portfolio run",
            "annual_return": "Annualized Return",
            "sharpe": "Sharpe Ratio",
            "drawdown": "Max Drawdown",
            "holdout_ic": "Holdout IC",
            "turnover": "Average Turnover",
            "total_trades": "Total Trades",
            "command": "Equivalent terminal command",
            "attribution_title": "Factor Contribution Through Time",
            "attribution_help": "This is contribution to the composite score, not realized P&L attribution. P&L attribution is calculated from the completed backtest artifacts.",
            "date": "Date",
            "contribution": "Composite-score contribution",
            "robustness_title": "Leave-One-Factor-Out Robustness",
            "robustness_help": "Remove one factor at a time. A large signal change means the strategy depends heavily on that factor; near-perfect similarity across every removal may indicate redundant factors.",
            "omitted_factor": "Omitted factor",
            "signal_change": "Mean absolute signal change",
            "correlation_to_full": "Correlation to full blend",
            "advanced_warning": "The selection contains a state, router, or sleeve. Review its output contract carefully; completed sleeves should usually be allocated after their own backtests instead of blended as raw factors.",
            "options_warning": "Options use the event-driven lifecycle engine and are not yet supported by this factor-level tabular composer.",
            "config_error": "Strategy configuration error: {error}",
            "build_error": "Composite build failed: {error}",
            "run_error": "Backtest failed: {error}",
        },
        "zh": {
            "title": "因子组合",
            "subtitle": "在共用执行与回测之前，将兼容的因子分数组合成一套可审计策略。",
            "boundary": "因子权重用于组合可比的预测分数；资产权重随后由执行阶段生成；市场状态路由器则在更后端对完整策略模块进行配置。",
            "market": "市场类别",
            "router_registry": "路由器目录",
            "router_id": "路由器 ID",
            "router_name": "路由器",
            "router_status": "状态",
            "router_frequency": "频率",
            "router_state": "状态变量列",
            "router_lag": "决策滞后",
            "include_advanced": "包括状态、策略模块与路由器",
            "include_advanced_help": "通常保持关闭。原始因子组合应只处理可比分数；状态变量和完整策略模块应进入路由或资金配置阶段。",
            "select_factors": "选择因子",
            "need_factors": "请至少选择两个兼容因子。",
            "data_file": "研究数据文件",
            "no_data": "该市场下没有发现本地 parquet 或 CSV 文件。",
            "weighting": "因子权重方法",
            "equal": "等权",
            "static": "固定权重",
            "normalization": "标准化方式",
            "cross_sectional_zscore": "横截面 Z 分数",
            "cross_sectional_rank": "横截面排名",
            "raw": "原始值（已具可比性）",
            "missing_policy": "缺失因子处理",
            "renormalize_available": "按可用因子重新归一",
            "complete_case": "要求所有因子齐全",
            "zero": "缺失值视为零",
            "min_available": "最少可用因子数",
            "winsor_limit": "Z 分数上限",
            "strategy_id": "策略 ID",
            "strategy_name": "策略名称",
            "weight": "权重",
            "orientation": "方向",
            "long_orientation": "正常",
            "inverse_orientation": "反向",
            "build": "构建组合预览",
            "building": "正在计算因子分数并验证合约……",
            "build_success": "组合信号构建成功。",
            "build_stale": "构建设置已改变。运行回测前请重新构建预览。",
            "download": "下载策略 YAML",
            "tabs": ["组合构建", "诊断", "回测", "归因与稳健性"],
            "compatibility": "因子合约兼容性",
            "weights_title": "已声明因子权重",
            "factor": "因子",
            "category": "类别",
            "factor_family": "因子类别",
            "data_frequency": "数据频率",
            "governance_status": "目录规范状态",
            "component_type": "组件类型",
            "supported_markets": "支持市场",
            "geometry": "评估结构",
            "execution_mode": "执行模式",
            "execution_lag": "执行滞后",
            "return_assumption": "收益假设",
            "source": "来源",
            "preview_required": "请先构建组合预览，以检查信号重叠与覆盖率。",
            "rows": "数据行数",
            "valid_scores": "有效分数",
            "assets": "资产数",
            "date_range": "日期范围",
            "factor_correlation": "标准化因子相关性",
            "factor_correlation_help": "高绝对相关性意味着两个因子可能使用相似信息。这并非必然错误，但会降低分散化。",
            "coverage_title": "因子信号覆盖率",
            "coverage": "覆盖率",
            "contribution_title": "组合信号贡献",
            "mean_abs_contribution": "平均绝对贡献",
            "correlation_to_composite": "与组合信号的相关性",
            "backtest_title": "共用引擎回测",
            "backtest_desc": "组合分数将进入单因子研究所用的同一执行模式与 AlphaEvaluator，生成一个净额化持仓簿和统一成本模型。",
            "split_mode": "样本切分方式",
            "validation_fraction": "训练/验证样本比例",
            "purge_periods": "清除期数",
            "embargo_periods": "隔离期数",
            "run_backtest": "运行多因子组合回测",
            "running_backtest": "正在运行共用执行与评估……",
            "run_success": "回测完成：{run_id}",
            "latest_run": "最新多因子组合运行",
            "annual_return": "年化收益",
            "sharpe": "夏普比率",
            "drawdown": "最大回撤",
            "holdout_ic": "留出样本 IC",
            "turnover": "平均换手率",
            "total_trades": "总交易笔数",
            "command": "等效终端命令",
            "attribution_title": "因子贡献时间序列",
            "attribution_help": "这里展示的是对组合分数的贡献，不是已实现盈亏归因；盈亏归因将在完整回测产物中计算。",
            "date": "日期",
            "contribution": "组合分数贡献",
            "robustness_title": "逐一剔除因子的稳健性",
            "robustness_help": "每次移除一个因子。信号变化很大说明策略高度依赖该因子；移除任何因子后都几乎不变，则可能存在因子冗余。",
            "omitted_factor": "被剔除因子",
            "signal_change": "平均绝对信号变化",
            "correlation_to_full": "与完整组合的相关性",
            "advanced_warning": "当前选择包含状态变量、路由器或策略模块。请仔细检查输出合约；完整策略模块通常应在各自回测后配置资金，而不是当作原始因子混合。",
            "options_warning": "期权使用事件驱动生命周期引擎，目前尚未接入本多因子表格型组合器。",
            "config_error": "策略配置错误：{error}",
            "build_error": "组合构建失败：{error}",
            "run_error": "回测失败：{error}",
        },
    },
    "adaptive_relationship_lab": {
        "en": {
            "title": "Relationship Scan",
            "subtitle": "Pair, calendar-spread, cross-product, and statistical-arbitrage relationship analysis.",
            "source": "Daily market file",
            "min_obs": "Minimum observations",
            "lookback": "Scan lookback",
            "z_window": "Z-score window",
            "max_assets": "Max assets in scan",
            "min_corr": "Minimum |correlation|",
            "run_note": "The scanner ranks research candidates, not trade recommendations.",
            "tabs": ["Radar", "Workspace", "Maps", "Audit"],
            "empty": "No usable daily market files were found. Add a public/demo parquet "
            "with at least date, ticker, and close columns.",
            "no_candidates": "No candidates passed the current filters. Loosen the "
            "correlation/min-observation filters or use a broader source "
            "file.",
            "manual_title": "How to use this upgraded lab",
            "manual": "\n"
            "This page works in two layers.\n"
            "\n"
            "First, the **Opportunity Radar** scans many possible spreads and "
            "ranks them by current dislocation, relationship stability, "
            "mean-reversion evidence, liquidity, and estimated cost drag.\n"
            "\n"
            "Second, the **Workspace** inspects one candidate in detail: adaptive "
            "hedge ratio, spread construction, and a lightweight backtest preview. "
            'A high score means "worth researching", not "place a trade". A '
            "candidate still needs cost-aware backtesting, execution checks, and "
            "out-of-sample validation.\n",
            "workspace_help_title": "How to use the Workspace",
            "workspace_help": "\n"
            "Use the Workspace when a candidate from the Radar looks "
            "interesting.\n"
            "\n"
            "**Scanner candidate** keeps you in the ranked workflow: "
            "choose one opportunity and inspect it.\n"
            "\n"
            "**Manual pair** is for your own ideas. It uses the same "
            "analytics, but it is not pre-ranked by the scanner.\n"
            "\n"
            "Read it top to bottom: first check whether the adaptive "
            "relationship is stable, then check whether the spread "
            "construction makes economic sense, then treat the backtest "
            "preview as a quick sanity check.\n",
            "maps_help_title": "How to use Opportunity Maps",
            "maps_help": "\n"
            "This tab groups the specialized arbitrage views without forcing "
            "you through three separate top-level tabs.\n"
            "\n"
            "Use **Calendar** for same-product different-expiry spreads, "
            "**Cross-product** for economically related instruments, and **Stat "
            "arb** for data-discovered pairs that may not have an obvious "
            "economic story.\n",
            "controls_help_title": "How to interpret the controls",
            "controls_help": "\n"
            "**Minimum observations** protects against short histories. For "
            "daily futures, 252 is roughly one trading year.\n"
            "\n"
            "**Scan lookback** is the recent history used to estimate "
            "correlation, beta, residual z-score, and half-life.\n"
            "\n"
            "**Z-score window** controls how local the dislocation estimate "
            "is. Shorter windows react faster but create more false "
            "alarms.\n"
            "\n"
            "**Max assets in scan** keeps the pair count manageable. Public "
            "users with small demo files can lower it.\n"
            "\n"
            "**Minimum |correlation|** filters out pairs with almost no "
            "return relationship before scoring.\n",
            "radar_help_title": "How to read the Opportunity Radar",
            "radar_help": "\n"
            "The radar is a triage board. Look for candidates with both high "
            "**dislocation score** and high **stability score**.\n"
            "\n"
            "**Promising** means the current z-score is meaningfully "
            "stretched, the relationship has not obviously broken, and "
            "half-life is in a plausible mean-reversion range. It is a "
            "research lead, not a trading signal.\n"
            "\n"
            "**Fragile** usually means the spread is stretched but either beta "
            "is moving too much or mean reversion is weak.\n"
            "\n"
            "**Watchlist** means the relationship may be valid, but current "
            "dislocation is not large enough to prioritize.\n",
            "metric_help_title": "Metric guide",
            "metric_help": "\n"
            "| Metric | What it means | Rough interpretation |\n"
            "|---|---|---|\n"
            "| **Opportunity score** | Weighted scanner score from "
            "dislocation, stability, half-life, liquidity, and cost. | `>70` "
            "promising for research, `50-70` worth reviewing, `<50` low "
            "priority. |\n"
            "| **Latest z** | Current spread residual versus its recent "
            "mean/std. | `|z| > 2` is stretched; `|z| > 3` is extreme but may "
            "also mean breakage. |\n"
            "| **Correlation** | Recent return co-movement. | Higher helps "
            "hedge quality, but very high alone is not enough. |\n"
            "| **Half-life** | Estimated observations for spread shock to "
            "decay by half. | `2-60` daily bars is usually researchable; very "
            "long/infinite is fragile. |\n"
            "| **Beta drift** | Difference between early and late hedge ratio "
            "estimates. | Lower is better. High drift means the relationship "
            "is changing. |\n"
            "| **Estimated cost bps** | Rough two-leg round-turn fee proxy. | "
            "High costs make small spreads untradeable. |\n",
            "drill_help_title": "How to read the Dual Kalman drilldown",
            "drill_help": "\n"
            "Dual Kalman regression updates the hedge ratio one observation at "
            "a time:\n"
            "\n"
            "`Y_return_t = alpha_t + beta_t * X_return_t + residual_t`\n"
            "\n"
            "**Dynamic beta** is the adaptive hedge ratio. A smooth beta "
            "supports a stable relationship. A jumping beta is a warning.\n"
            "\n"
            "**Residual z-score** is the abnormal move after hedging. Extreme "
            "z can be an opportunity if the relationship is stable, or a break "
            "if uncertainty and beta drift rise too.\n"
            "\n"
            "**State uncertainty** is the model's uncertainty about "
            'alpha/beta. Rising uncertainty says "trust this relationship '
            'less".\n',
            "builder_help_title": "How to choose a spread construction",
            "builder_help": "\n"
            "**Return residual** is best for relationship monitoring and "
            "statistical pairs. It is the current model's default.\n"
            "\n"
            "**Price ratio** is simple and intuitive, useful when both legs "
            "are economically similar.\n"
            "\n"
            "**Linear price spread** uses `Y - beta * X`. It is closer to "
            "classic pair spread charts.\n"
            "\n"
            "**Contract-value spread** multiplies futures prices by contract "
            "multipliers before forming the spread. Use it when comparing "
            "futures with different point values.\n",
            "calendar_help_title": "How to read calendar arbitrage",
            "calendar_help": "\n"
            "Calendar spreads require multiple contracts of the same base "
            "product, such as `au2608` vs `au2610`. Index-level files "
            "usually cannot support this because each product appears only "
            "once.\n"
            "\n"
            "When contract-level data is available, focus on term-structure "
            "shape, spread percentile, expiry liquidity, and roll/carry "
            "logic.\n",
            "cross_help_title": "How to read cross-product arbitrage",
            "cross_help": "\n"
            "Cross-product spreads compare economically related instruments, "
            "often inside one sector or production chain.\n"
            "\n"
            "These are stronger when there is both a statistical relationship "
            "and an economic reason for convergence, such as substitutes, "
            "inputs/outputs, or shared macro drivers.\n",
            "stat_help_title": "How to read statistical arbitrage",
            "stat_help": "\n"
            "Statistical arbitrage candidates are data-discovered pairs. They "
            "may have no obvious economic story.\n"
            "\n"
            "Treat them more skeptically: require stronger stability, cleaner "
            "half-life, robust out-of-sample behavior, and stricter cost "
            "checks.\n",
            "backtest_help_title": "How to read the backtest preview",
            "backtest_help": "\n"
            "This is a lightweight sanity check, not the production "
            "backtester.\n"
            "\n"
            "The rule enters when z-score is stretched, exits near zero, "
            "and stops if the spread gets more extreme. A promising preview "
            "has multiple trades, tolerable drawdown, and results that do "
            "not disappear after costs.\n",
            "audit_help_title": "How to read the audit",
            "audit_help": "\n"
            "The public version of this repo should work without private data. "
            "This tab shows what the selected file contains and whether the "
            "lab has enough observations to produce valid research "
            "diagnostics.\n"
            "\n"
            "Required columns are `date`, `ticker`, and `close`. Volume, open "
            "interest, sector, multiplier, and fees improve scoring but are "
            "optional.\n",
            "full_universe_note": "Showing the full candidate universe: {count:,} candidates.",
            "workspace_mode": "Workspace mode",
            "workspace_mode_options": {
                "scanner": "Scanner candidate",
                "manual": "Manual pair",
            },
            "y_asset": "Y asset",
            "x_asset": "X asset / hedge leg",
            "dynamic_relationship": "Dynamic Relationship",
            "process_noise": "Process noise",
            "observation_noise": "Observation noise",
            "initial_uncertainty": "Initial uncertainty",
            "recent_relationship_rows": "Recent relationship rows",
            "spread_construction": "Spread Construction",
            "spread_method": "Spread method",
            "hedge_method": "Hedge method",
            "hedge_lookback": "Hedge lookback",
            "spread_z_window": "Z-score window",
            "spread_method_labels": {
                "Return residual": "Return residual",
                "Price ratio": "Price ratio",
                "Linear price spread": "Linear price spread",
                "Contract-value spread": "Contract-value spread",
            },
            "hedge_method_labels": {
                "ols": "OLS",
                "fixed": "Fixed",
            },
            "backtest_preview": "Backtest Preview",
            "entry_z": "Entry z",
            "exit_z": "Exit z",
            "stop_z": "Stop z",
            "cost_bps": "Cost bps",
            "trades": "Trades",
            "win_rate": "Win rate",
            "net_pnl": "Net PnL",
            "max_dd": "Max DD",
            "map_view": "Opportunity family",
            "map_view_options": {
                "calendar": "Calendar",
                "cross_product": "Cross-product",
                "statistical": "Stat arb",
            },
            "cross_empty": "No cross-product candidates were found in the current universe.",
            "stat_empty": "No statistical arbitrage candidates were found in the current universe.",
            "selected": "Selected candidate",
            "score": "Opportunity Score",
            "latest_z": "Latest Z",
            "half_life": "Half-Life",
            "cost": "Est. Cost",
            "rows": "Rows",
            "assets": "Assets",
            "eligible_assets": "Eligible assets",
            "contract_level": "Contract-level",
            "yes": "Yes",
            "no": "No",
            "date_range": "{file}: {start} to {end}",
            "schema": "Schema",
            "asset_coverage": "Asset Coverage",
            "beta": "Beta",
            "beta_change": "Beta Change",
            "extreme_rate": "|Z| > 2 Rate",
            "run_error": "Analysis failed",
            "calendar_empty": "No calendar candidates were found. The selected file likely "
            "has index-level data rather than multiple expiries per "
            "product.",
            "read_only": "Manager demo mode is read-only: saving feature artifacts is "
            "disabled.",
        },
        "zh": {
            "title": "关系扫描",
            "subtitle": "用于配对、跨期、跨品种与统计套利关系的偏离分析。",
            "source": "日频市场文件",
            "min_obs": "最少观测数",
            "lookback": "扫描回看窗口",
            "z_window": "Z-score 窗口",
            "max_assets": "扫描资产上限",
            "min_corr": "最低 |相关性|",
            "run_note": "扫描结果是研究候选，不是交易建议。",
            "tabs": ["雷达", "工作台", "地图", "审计"],
            "empty": "未找到可用日频市场文件。请加入至少包含 date、ticker、close 的公开/示例 parquet。",
            "no_candidates": "当前筛选条件下没有候选。可以放宽相关性/观测数过滤，或使用更广的市场文件。",
            "manual_title": "如何使用升级后的页面",
            "manual": "\n"
            "这个页面分两层。\n"
            "\n"
            "第一层，**机会雷达** 扫描大量可能价差，并按当前偏离、关系稳定性、均值回复证据、流动性、成本拖累排序。\n"
            "\n"
            "第二层，**工作台** "
            "深入检查某一个候选：自适应对冲比率、价差构建、轻量回测预览。高分代表“值得研究”，不代表“可以直接下单”。候选仍需扣费回测、执行检查、样本外验证。\n",
            "workspace_help_title": "如何使用工作台",
            "workspace_help": "\n"
            "当雷达中某个候选看起来有意思时，用工作台继续检查。\n"
            "\n"
            "**扫描候选** 会沿用雷达排序结果：选择一个机会并钻取。\n"
            "\n"
            "**手动配对** 用于你自己的想法。它使用同样的分析方法，但没有经过雷达预排序。\n"
            "\n"
            "阅读顺序建议：先看自适应关系是否稳定，再看价差构建是否有经济意义，最后把回测预览当作快速 sanity check。\n",
            "maps_help_title": "如何使用机会地图",
            "maps_help": "\n"
            "这个标签把几个专门套利视图合并在一起，避免顶层出现太多标签。\n"
            "\n"
            "用 **跨期** 看同品种不同到期的价差；用 **跨品种** 看经济相关品种；用 **统计套利** "
            "看数据发现、但未必有明显经济故事的配对。\n",
            "controls_help_title": "参数如何理解",
            "controls_help": "\n"
            "**最少观测数** 用来避免短历史误判。日频期货中，252 大约是一年交易日。\n"
            "\n"
            "**扫描回看窗口** 用来估计相关性、beta、残差 z-score、半衰期。\n"
            "\n"
            "**Z-score 窗口** 决定偏离判断有多局部。窗口越短越敏感，但误报更多。\n"
            "\n"
            "**扫描资产上限** 控制配对数量。公开示例数据较小时可以调低。\n"
            "\n"
            "**最低 |相关性|** 会先剔除几乎没有收益关系的组合。\n",
            "radar_help_title": "如何阅读机会雷达",
            "radar_help": "\n"
            "雷达是研究线索筛选器。优先看 **偏离分数** 和 **稳定性分数** 都高的候选。\n"
            "\n"
            "**Promising** 表示当前 z-score "
            "有明显偏离、关系没有明显失效、半衰期处于可研究范围。这仍只是研究线索，不是交易信号。\n"
            "\n"
            "**Fragile** 通常表示价差虽然偏离，但 beta 漂移过大，或均值回复证据不足。\n"
            "\n"
            "**Watchlist** 表示关系可能有效，但当前偏离还不够大。\n",
            "metric_help_title": "指标说明",
            "metric_help": "\n"
            "| 指标 | 含义 | 粗略解读 |\n"
            "|---|---|---|\n"
            "| **Opportunity score** | 由偏离、稳定性、半衰期、流动性、成本加权得到的扫描分。 | `>70` "
            "值得重点研究，`50-70` 可复核，`<50` 优先级低。 |\n"
            "| **Latest z** | 当前价差残差相对近期均值/标准差的偏离。 | `|z| > 2` 属于拉开；`|z| > 3` "
            "很极端，但也可能是关系失效。 |\n"
            "| **Correlation** | 近期收益共动。 | 越高越利于对冲，但只有高相关还不够。 |\n"
            "| **Half-life** | 冲击衰减一半所需观测数。 | 日频 `2-60` 通常可研究；很长或无限说明脆弱。 |\n"
            "| **Beta drift** | 前半段与后半段 hedge ratio 的差异。 | 越低越好。高漂移说明关系在变。 |\n"
            "| **Estimated cost bps** | 两腿粗略 round-turn 手续费估计。 | 成本高会吞掉小价差机会。 "
            "|\n",
            "drill_help_title": "如何阅读 Dual Kalman 钻取",
            "drill_help": "\n"
            "Dual Kalman 回归会逐条更新 hedge ratio：\n"
            "\n"
            "`Y_return_t = alpha_t + beta_t * X_return_t + residual_t`\n"
            "\n"
            "**动态 beta** 是自适应对冲比率。平滑说明关系较稳定，跳变是风险信号。\n"
            "\n"
            "**残差 z-score** 是对冲后的异常波动。极端 z 可能是机会，也可能是关系断裂，需要结合 uncertainty 与 "
            "beta drift。\n"
            "\n"
            "**状态不确定性** 表示模型对 alpha/beta 的不确定。上升时要降低对关系的信任。\n",
            "builder_help_title": "如何选择价差构建方式",
            "builder_help": "\n"
            "**Return residual** 适合关系监控与统计套利，也是当前模型默认方式。\n"
            "\n"
            "**Price ratio** 简单直观，适合同类资产。\n"
            "\n"
            "**Linear price spread** 使用 `Y - beta * X`，更接近经典配对价差图。\n"
            "\n"
            "**Contract-value spread** 会先乘以期货合约乘数，适合点值不同的期货比较。\n",
            "calendar_help_title": "如何阅读跨期套利",
            "calendar_help": "\n"
            "跨期需要同一品种的多个到期合约，例如 `au2608` vs "
            "`au2610`。指数级文件通常不支持，因为每个品种只出现一次。\n"
            "\n"
            "有合约级数据时，重点看期限结构、价差分位数、到期流动性、roll/carry 逻辑。\n",
            "cross_help_title": "如何阅读跨品种套利",
            "cross_help": "\n"
            "跨品种比较经济上相关的工具，通常在同一板块或产业链内。\n"
            "\n"
            "当统计关系和经济收敛逻辑同时存在时，候选更可靠，例如替代品、上下游、共同宏观驱动。\n",
            "stat_help_title": "如何阅读统计套利",
            "stat_help": "\n"
            "统计套利候选是数据发现的配对，可能没有明显经济故事。\n"
            "\n"
            "需要更谨慎：要求更稳定的关系、更干净的半衰期、样本外稳健性、以及更严格的成本检查。\n",
            "backtest_help_title": "如何阅读回测预览",
            "backtest_help": "\n"
            "这是轻量 sanity check，不是生产级回测。\n"
            "\n"
            "规则在 z-score "
            "拉开时入场，回到零附近离场，继续拉开则止损。较好的预览应有足够交易次数、可接受回撤，并且扣成本后不消失。\n",
            "audit_help_title": "如何阅读审计",
            "audit_help": "\n"
            "公开版仓库不应依赖私有数据。本页展示所选文件包含什么，以及是否有足够观测来生成有效研究诊断。\n"
            "\n"
            "必需列是 `date`、`ticker`、`close`。volume、open "
            "interest、sector、multiplier、fees 会改善评分，但不是必需。\n",
            "full_universe_note": "当前显示完整候选池：{count:,} 个候选。",
            "workspace_mode": "工作台模式",
            "workspace_mode_options": {
                "scanner": "扫描候选",
                "manual": "手动配对",
            },
            "y_asset": "Y 资产",
            "x_asset": "X 资产 / 对冲腿",
            "dynamic_relationship": "动态关系",
            "process_noise": "过程噪声",
            "observation_noise": "观测噪声",
            "initial_uncertainty": "初始不确定性",
            "recent_relationship_rows": "近期关系数据",
            "spread_construction": "价差构建",
            "spread_method": "价差方法",
            "hedge_method": "对冲方法",
            "hedge_lookback": "对冲回看窗口",
            "spread_z_window": "Z-score 窗口",
            "spread_method_labels": {
                "Return residual": "收益残差",
                "Price ratio": "价格比值",
                "Linear price spread": "线性价格价差",
                "Contract-value spread": "合约价值价差",
            },
            "hedge_method_labels": {
                "ols": "OLS",
                "fixed": "固定",
            },
            "backtest_preview": "回测预览",
            "entry_z": "入场 z",
            "exit_z": "离场 z",
            "stop_z": "止损 z",
            "cost_bps": "成本 bps",
            "trades": "交易数",
            "win_rate": "胜率",
            "net_pnl": "净盈亏",
            "max_dd": "最大回撤",
            "map_view": "机会类别",
            "map_view_options": {
                "calendar": "跨期",
                "cross_product": "跨品种",
                "statistical": "统计套利",
            },
            "cross_empty": "当前候选池中未发现跨品种候选。",
            "stat_empty": "当前候选池中未发现统计套利候选。",
            "selected": "选中候选",
            "score": "机会分数",
            "latest_z": "最新 Z",
            "half_life": "半衰期",
            "cost": "估计成本",
            "rows": "样本行数",
            "assets": "资产数",
            "eligible_assets": "可用资产",
            "contract_level": "合约级数据",
            "yes": "是",
            "no": "否",
            "date_range": "{file}: {start} 至 {end}",
            "schema": "数据结构",
            "asset_coverage": "资产覆盖",
            "beta": "Beta",
            "beta_change": "Beta 变化",
            "extreme_rate": "|Z| > 2 比例",
            "run_error": "分析失败",
            "calendar_empty": "未发现跨期候选。当前文件大概率是指数级数据，而不是每个品种多个到期合约。",
            "read_only": "管理层演示模式为只读：已禁用保存特征产物。",
        },
    },
    "tick_pulse_lab": {
        "en": {
            "title": "Intraday Event Study",
            "subtitle": "Intraday event tests for market microstructure hypotheses.",
            "file": "Tick parquet file",
            "nav_label": "Intraday Event Study navigation",
            "loading_file": "Loading tick parquet...",
            "loading_summary": "Building contract universe...",
            "loading_features": "Building selected contract features...",
            "loading_thresholds": "Calibrating thresholds...",
            "loading_events": "Evaluating hypothesis events...",
            "loading_done": "Ready.",
            "loading_complete": "Complete",
            "math_cache_hit": "Loaded horizon sweep from SQLite cache",
            "math_cache_miss": "Computing horizon sweep and saving to SQLite cache...",
            "math_cache_store": "Saving horizon sweep cache...",
            "math_cache_backend": "Math sweep cache",
            "no_tick_files": "No tick_all_data parquet files found in "
            "runtime/data/futures_cn/tick.",
            "product": "Product",
            "symbol": "Contract",
            "hypothesis": "Hypothesis",
            "hypothesis_source": "Hypothesis source",
            "hypothesis_source_options": {
                "built_in": "Built-in rule",
                "saved_seed": "Saved Discovery seed",
            },
            "saved_seed_select": "Saved seed",
            "saved_seed_empty": "No saved discovery seeds yet. Open Discovery Lab, "
            "inspect a pulse, then save it as a hypothesis seed.",
            "saved_seed_scope_warning": "This seed was created from `{seed_file}` / `{seed_symbol}`. "
            "You are currently testing `{current_file}` / "
            "`{current_symbol}`.",
            "saved_seed_context": "Seed rule context",
            "saved_seed_math_note": "This is a saved Discovery seed. The exact rule is shown below; use "
            "Hypothesis Test for event outcomes and horizon sweeps.",
            "saved_seed_cross_asset_note": "Cross-asset sweeps currently support built-in hypotheses "
            "only. Saved seeds are evaluated on the selected contract "
            "first.",
            "threshold_mode": "Threshold mode",
            "threshold_modes": {
                "adaptive": "Asset-adaptive percentiles (recommended)",
                "fixed": "Fixed gold-style constants (debug)",
            },
            "threshold_mode_advanced_title": "Advanced / debug threshold mode",
            "threshold_mode_active": "Active threshold mode: **{mode}**",
            "threshold_mode_debug_note": "Use fixed constants only when you deliberately want to "
            "reproduce/debug the original gold rule. For cross-asset "
            "research, leave this on adaptive.",
            "threshold_mode_note": "\n"
            "For cross-asset work, prefer **asset-adaptive percentiles**. It "
            "keeps the hypothesis structure the same, but maps each criterion to "
            "that contract's own distribution.\n"
            "\n"
            "Example: instead of saying every asset must use `volume_intensity "
            ">= 3.0`, we ask for that asset's own top decile of volume "
            "intensity. This is probability matching, not ML.\n",
            "base_rate_visual_title": "Base Rate vs Hypothesis Accuracy: Event Microscope",
            "base_rate_visual_caption": "This section uses real triggered events to explain the "
            "mechanics before the sweep tables judge the hypothesis "
            "statistically.",
            "base_rate_visual_empty": "Not enough valid rows to draw the base-rate explainer.",
            "base_rate_mechanics": "\n"
            "**Read this first.** A tick-data row is a market **snapshot** from "
            "the parquet file. So `t + {horizon}` means the `{horizon}`-th later "
            "snapshot inside the same contract and same trading session, not "
            "`{horizon}` seconds later and not `{horizon}` consecutive price "
            "changes.\n"
            "\n"
            "`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / "
            "tick_size`. If this shows `+28`, it means the future mid-price is "
            "28 minimum price increments above the event price. It does **not** "
            "mean the price rose 28 rows in a row.\n"
            "\n"
            "Base Rate uses all rows that have a valid future snapshot and an "
            "expected direction. Hypothesis Accuracy uses only the stricter "
            "subset where the event criteria also fired.\n",
            "base_rate_metric": "Base Rate",
            "event_accuracy_metric": "Event Accuracy",
            "base_rate_valid_rows": "Base-rate rows",
            "base_rate_event_rows": "Triggered events",
            "base_rate_criteria_title": "Base-rate criteria",
            "event_criteria_title": "Hypothesis event criteria",
            "criteria_valid_future": "Has row t + {horizon} inside the same contract/session",
            "criteria_expected_direction": "Expected direction is defined: {direction}",
            "criteria_future_success": "Future move {move} price ticks reaches {direction} by at least "
            "{success} tick(s)",
            "criteria_hypothesis_signal": "Hypothesis event fires: every event criterion above is true",
            "criteria_rtv_direction": "Fast-window direction exists: burst={direction}, "
            "expected={expected}",
            "criteria_rtv_percentile": "abs fast move {value} ticks >= prior rolling {percentile} "
            "threshold {threshold} ticks",
            "criteria_rule": "{name} {op} {threshold}{unit}; current={value}{unit}",
            "criteria_current_row_note": "These checkboxes show whether the selected example row passes "
            "each rule. They are not controls.",
            "criteria_feature_names": {
                "flow_imbalance": "Flow imbalance",
                "book_imbalance": "Book imbalance",
                "volume_intensity": "Volume intensity",
                "rolling_mid_move_ticks": "Rolling mid move",
                "price_shock": "Price shock",
                "rtv_abs_move_ticks": "Absolute fast move",
            },
            "event_examples_title": "Two Triggered-Event Examples",
            "event_examples_caption": "Both charts are hypothesis events. The left example succeeded; "
            "the right example failed. Hover over any point to see the "
            "row-level variables used by the event logic.",
            "success_example_title": "Success event",
            "failure_example_title": "Failure event",
            "missing_success_example": "No successful triggered event exists under the current "
            "settings.",
            "missing_failure_example": "No failed triggered event exists under the current settings.",
            "event_point_label": "event t",
            "outcome_point_label": "horizon outcome",
            "entry_price_label": "event price",
            "snapshot_role_fast": "past fast-window snapshot",
            "snapshot_role_event": "event row t",
            "snapshot_role_horizon": "future outcome row",
            "snapshot_role_forward": "forward waiting snapshot",
            "event_datetime_label": "Event snapshot",
            "future_datetime_label": "Future snapshot",
            "expected_direction_label": "Expected direction",
            "future_move_formula_label": "Future move formula",
            "future_move_formula": "({future} - {current}) / {tick_size} = {move} price ticks",
            "outcome_label": "Outcome",
            "base_checks_title": "Base-rate checks",
            "event_checks_title": "Event trigger checks",
            "event_chart_hover": "<b>%{customdata[0]}</b><br>Role: %{customdata[2]}<br>Snapshot offset "
            "from event t: %{x}<br>Mid price: %{y:.3f}<br>Move from event t: "
            "%{customdata[1]:+.2f} price ticks<br>Flow imbalance: "
            "%{customdata[3]:.3f}<br>Book imbalance: "
            "%{customdata[4]:.3f}<br>Volume intensity: "
            "%{customdata[5]:.3f}<br>Rolling mid move: %{customdata[6]:+.2f} price "
            "ticks<br>Price shock: %{customdata[7]:.3f}<br>RTV abs move: "
            "%{customdata[8]:.2f}<br>RTV threshold: %{customdata[9]:.2f}<br>RTV "
            "ratio: %{customdata[10]:.2f}<extra></extra>",
            "outcome_hover": "Outcome: {outcome}<br>Future move: {move} price ticks<br>Expected: "
            "{expected}<br>Success threshold: {success} price tick(s)<extra></extra>",
            "show_fast_window": "Show 6-snapshot fast window",
            "show_horizon": "Show forward horizon",
            "show_success_threshold": "Show success threshold",
            "show_event_threshold": "Show event threshold",
            "show_base_population": "Show base-rate population note",
            "base_rate_plain_language": "\n"
            "**Two meanings of tick:**\n"
            "\n"
            "`{fast_window}-tick move` = price difference across "
            "`{fast_window}` tick-data snapshots.\n"
            "\n"
            "`{success} price tick success` = price moved by at least "
            "`{success}` minimum price increments.\n"
            "\n"
            "`horizon={horizon}` = compare current snapshot `t` with "
            "snapshot `t+{horizon}`, not the immediate next snapshot.\n",
            "chart_mid_price": "Mid price",
            "snapshot_axis": "Snapshot offset from current t",
            "price_axis": "Price",
            "fast_window_label": "past {fast_window} snapshots",
            "horizon_label": "t + {horizon}",
            "success_price_label": "success if {direction} by {success} price ticks",
            "event_threshold_annotation": "past 6-snapshot move={past_move} ticks<br>RTV "
            "threshold={threshold}<br>ratio={ratio}",
            "concept_col": "Concept",
            "definition_col": "Definition",
            "example_col": "Selected-row value",
            "fast_move_concept": "Fast move",
            "fast_move_definition": "mid_price[t] - mid_price[t - {fast_window}], divided by price tick "
            "size.",
            "future_move_concept": "Future move",
            "future_move_definition": "mid_price[t + {horizon}] - mid_price[t], divided by price tick "
            "size.",
            "success_label_concept": "Success label",
            "success_label_definition": "Boolean True/False: did the future move reach the expected "
            "direction by the minimum success move?",
            "price_ticks_unit": "price ticks",
            "correct_label": "Correct",
            "failed_label": "Failed",
            "base_rate_population_note": "Base Rate uses all rows with a valid future label. Accuracy "
            "uses only rows where the hypothesis/event rule fired. For "
            "relative velocity, rows without a clear 6-snapshot direction "
            "count as not successful in the background base-rate "
            "calculation.",
            "hypothesis_labels": {
                "bullish": "Bullish absorption: sell pressure absorbed -> up",
                "bearish": "Bearish impulse: sell pressure confirms -> down",
                "bearish_breakdown": "Bearish breakdown: sell shock already breaking "
                "price -> down",
                "relative_velocity": "Relative tick velocity: 99th percentile "
                "breakout",
                "relative_velocity_fade": "Relative tick velocity exhaustion: fade "
                "the spike",
            },
            "window": "Rolling tick window",
            "horizon": "Forward test horizon",
            "success_move": "Min success move",
            "top": "Max event markers on chart",
            "calculation_audit": "Calculation Audit",
            "audit_note": "Feature calculations use cleaned tradable ticks only. Rows with zero/invalid "
            "bid, ask, or last price are excluded, and rolling/future windows reset at "
            "session gaps.",
            "raw_rows": "Raw rows",
            "valid_feature_rows": "Valid feature rows",
            "dropped_rows": "Dropped invalid rows",
            "session_count": "Detected sessions",
            "marker_caption": "Chart markers: showing {shown:,} of {total:,} events. If capped, events "
            "are evenly sampled across time rather than taking only the first events.",
            "plot_downsample_note": "Large line charts are downsampled to {points:,} points for browser "
            "speed. Event counts and ML training still use the full event "
            "table.",
            "event_episode_note": "Consecutive signal ticks are collapsed into one event episode, using "
            "the first tick of the episode. This prevents one market burst from "
            "appearing as many duplicate events.",
            "event_distribution": "Event Distribution By Date",
            "audit_date": "Date",
            "audit_first_event": "First event",
            "audit_last_event": "Last event",
            "schema": "Data Schema",
            "row_viewer": "Raw Tick Row Viewer",
            "row_mode": "Which row?",
            "row_modes": ["First", "Middle", "Last", "Custom index"],
            "row_index": "Row index inside selected contract",
            "row_position": "{symbol} contract row {row:,} of {last:,} | global file row {source:,}",
            "row_empty": "No raw rows found for {symbol}.",
            "field": "Field",
            "value": "Value",
            "dtype": "Dtype",
            "meaning": "Meaning",
            "row_meanings": {
                "symbol": "Contract code. For example, au2608 is the August 2026 Shanghai "
                "gold futures contract.",
                "datetime": "Exact tick timestamp.",
                "last_price": "Most recent traded price at this tick.",
                "volume": "Cumulative daily traded volume reported by the exchange/feed.",
                "bid_price_1": "Best visible bid price.",
                "bid_volume_1": "Visible size resting at the best bid.",
                "ask_price_1": "Best visible ask price.",
                "ask_volume_1": "Visible size resting at the best ask.",
                "oi": "Open interest, usually cumulative open contracts.",
            },
            "contracts": "Contract Universe",
            "asset_ranker_tab": "Pattern Scope",
            "asset_ranker_title": "Pattern Lens Scope",
            "asset_ranker_intro": "Choose a lens before ranking assets. Daily, 1-minute, and tick data "
            "answer different research questions, so their purpose is shown "
            "explicitly before the metrics.",
            "asset_ranker_manual_title": "Why use different lenses?",
            "asset_ranker_manual": "\n"
            "Different timeframes expose different market behavior.\n"
            "\n"
            "1. **Daily Scope** finds broad asset candidates: which markets move "
            "enough and have enough volume/coverage to deserve deeper work.\n"
            "2. **1m Intraday** checks whether movement survives inside the day: "
            "session effects, trendiness, choppiness, and time-of-day clues.\n"
            "3. **Tick Microstructure** checks execution reality: pulse density, "
            "spread/noise, liquidity bursts, and whether a pattern can survive "
            "costs.\n"
            "\n"
            "`Download Priority = 65% volatility percentile + 20% volume "
            "percentile + 10% open-interest percentile + 5% data coverage`.\n",
            "pattern_lens": "Pattern lens",
            "pattern_lens_options": {
                "daily": "Daily Scope",
                "minute": "1m Intraday",
                "tick": "Tick Microstructure",
            },
            "pattern_lens_purpose": {
                "daily": "Daily lens: broad triage. Use it to decide which assets are worth deeper research before spending time on intraday or tick data.",
                "minute": "1m lens: intraday behavior. Use it to compare realized volatility, trendiness, and session effects inside the trading day.",
                "tick": "Tick lens: execution reality. Use it to inspect pulse density, spread/noise, liquidity bursts, and microstructure feasibility.",
            },
            "pattern_lens_data_note": {
                "daily": "Daily ranking uses close-to-close returns and daily liquidity. It is a broad asset-selection lens, not proof that minute/tick strategies will work.",
                "minute": "1m ranking uses bar-to-bar returns from intraday parquet. It is closer to strategy behavior, but still ignores order-book execution costs.",
            },
            "asset_source_file": {
                "daily": "Whole-market daily parquet",
                "minute": "1-minute parquet",
            },
            "asset_lookback_by_lens": {
                "daily": "Volatility lookback (days)",
                "minute": "Volatility lookback (1m bars)",
            },
            "asset_min_obs_by_lens": {
                "daily": "Minimum valid days",
                "minute": "Minimum valid 1m bars",
            },
            "asset_daily_file": "Whole-market daily parquet",
            "asset_lookback": "Volatility lookback",
            "asset_min_obs": "Minimum valid days",
            "asset_top_n": "Rows to show",
            "asset_sort": "Sort by",
            "asset_sort_options": {
                "download_priority_score": "Download priority",
                "recent_ann_vol": "Recent volatility",
                "avg_daily_volume": "Average daily volume",
                "avg_oi": "Average open interest",
                "avg_intraday_range_pct": "Average intraday range",
            },
            "asset_sector_filter": "Sector filter",
            "asset_min_volume": "Minimum average volume",
            "asset_rank_empty": "No assets passed the current ranking filters.",
            "asset_rank_download": "Download ranked candidates CSV",
            "manager_demo_download_hidden": "Download disabled in manager demo mode.",
            "asset_rank_chart": "Top Volatility Candidates",
            "asset_rank_table": "Candidate Pattern List",
            "asset_rank_chart_by_lens": {
                "daily": "Daily Volatility Candidates",
                "minute": "1m Intraday Volatility Candidates",
            },
            "asset_rank_table_by_lens": {
                "daily": "Daily Scope Candidate List",
                "minute": "1m Pattern Candidate List",
            },
            "asset_rank_note": "High volatility is useful, but each lens answers a different question. "
            "Prefer names that combine movement, enough liquidity, and healthy coverage.",
            "dataset_timeframe": "Data timeframe",
            "dataset_start": "Start",
            "dataset_end": "End",
            "dataset_assets": "Assets",
            "dataset_rows": "Rows",
            "dataset_date_range": "Date range",
            "tick_lens_title": "Available Tick Files",
            "tick_lens_caption": "Tick data is not ranked with daily-volatility logic. Use it for pulse density, spread/noise, liquidity bursts, and event-level validation.",
            "tick_lens_route": "Next step for this lens: open Pulse Workspace for a single contract, or Cross-Asset Map for pulse behavior across downloaded tick files.",
            "tick_file_count": "Tick files",
            "tick_file": "Tick file",
            "tick_size_mb": "Size MB",
            "tick_modified": "Modified",
            "asset_rank_metrics": {
                "ranked_assets": "Ranked assets",
                "top_vol": "Top vol",
                "median_vol": "Median vol",
                "top_priority": "Top priority",
            },
            "detected": "Detected Products",
            "liquid": "Most Liquid Contract",
            "rows": "Rows",
            "events": "Hypothesis Events",
            "event_rate": "Event Rate",
            "correct_moves": "Correct Moves",
            "accuracy": "Accuracy",
            "avg_move": "Avg Future Move",
            "evidence_ticket_button": "Save Evidence Ticket",
            "evidence_ticket_saved": "Evidence ticket saved: {ticket_id}",
            "evidence_ticket_no_events": "No hypothesis events to save under the current settings.",
            "evidence_ticket_error": "Could not save evidence ticket: {error}",
            "contract_health": "Contract Health",
            "event_map": "Hypothesis Test",
            "trading_hours_compressed_note": "The x-axis hides standard non-trading gaps for "
            "readability: 02:00-09:30, 10:15-10:30, 11:30-13:30, and "
            "15:00-21:00. Tick order and calculations are unchanged.",
            "event_horizon_section_title": "Payoff And Selected-Event Forward Move",
            "payoff_chart_title": "Payoff Ratio Snapshot",
            "payoff_chart_caption": "This uses all events under the currently selected hypothesis and "
            "horizon. The bars compare average favorable move versus average "
            "adverse move.",
            "payoff_empty": "Not enough valid event moves to calculate payoff asymmetry.",
            "payoff_ratio": "Payoff Ratio",
            "payoff_health_note": "Healthy rough guide: Accuracy should beat the Base Rate in the sweep "
            "table, and Payoff Ratio should be above 1.0x.",
            "payoff_avg_favorable": "Avg Favorable",
            "payoff_avg_adverse": "Avg Adverse",
            "payoff_yaxis": "Expected-direction move (price ticks)",
            "payoff_hover": "Metric: %{x}<br>Value: %{customdata[0]:+.2f} ticks<br>Type: "
            "%{customdata[1]}<extra></extra>",
            "event_horizon_title": "Selected Event Forward Move",
            "event_horizon_caption": "Pick one triggered event and inspect how far price moved after "
            "t+30, t+60, t+120, t+240, and t+480 snapshots. Bars are measured "
            "in the event's expected direction, so green means the move helped "
            "the hypothesis.",
            "event_horizon_empty": "Could not build the forward-horizon chart for this event.",
            "event_horizon_yaxis": "Move in expected direction ({expected}, price ticks)",
            "event_horizon_success_line": "success threshold = {success} tick(s)",
            "event_horizon_result_labels": {
                "Profitable": "Profitable direction",
                "Loss": "Wrong direction",
                "Flat": "Flat",
                "Missing": "Missing future row",
            },
            "event_horizon_hover": "Future snapshot: %{customdata[0]}<br>Future price: "
            "%{customdata[1]:.3f}<br>Raw future move: %{customdata[2]:+.2f} "
            "ticks<br>Move in expected direction: %{customdata[3]:+.2f} "
            "ticks<br>Result: %{customdata[4]}<extra></extra>",
            "anatomy": "Criteria And Outcomes",
            "microscope": "Event Microscope",
            "outcome_overview": "Outcome Overview",
            "event_ledger": "All Event Ledger",
            "event_ledger_note": "Every row below is one detected hypothesis event. Green means the "
            "later price move matched the hypothesis; red means it did not.",
            "event_ledger_download": "Download event ledger CSV",
            "events_by_day_chart": "Events By Day",
            "future_move_chart": "Future Move Distribution",
            "ledger_date": "Date",
            "ledger_time": "Event Time",
            "ledger_session": "Session",
            "ledger_outcome": "Outcome",
            "ledger_expected": "Expected",
            "ledger_last_price": "Last Price",
            "ledger_future_time": "Future Time",
            "ledger_future_move": "Future Move",
            "ledger_flow": "Flow Imbalance",
            "ledger_book": "Book Imbalance",
            "ledger_volume_x": "Volume x",
            "ledger_rolling_mid": "Rolling Mid Move",
            "ledger_price_shock": "Price Shock",
            "ledger_rtv_fast": "6-Tick Move",
            "ledger_rtv_pct": "Velocity / Threshold",
            "ledger_rtv_threshold": "99% Threshold",
            "ledger_count": "Events",
            "chart_rtv_fast": "6-tick move",
            "chart_rtv_abs": "Absolute 6-tick move",
            "chart_rtv_threshold": "Rolling 99% threshold",
            "chart_rtv_percentile": "Velocity / threshold",
            "chart_rtv_direction": "Burst direction",
            "select_event": "Select event",
            "no_events": "No events under the current hypothesis criteria.",
            "math_title": "Hypothesis Definition And Math",
            "why_title": "How to read the accuracy",
            "research_title": "Current Evidence And Bearish Redesign Plan",
            "sweep_title": "Live Horizon Sweep",
            "redesign_title": "How we redesign the bearish event",
            "sweep_hypothesis": "Hypothesis",
            "sweep_horizon": "Horizon",
            "sweep_events": "Events",
            "sweep_accuracy": "Accuracy",
            "sweep_base_rate": "Base Rate",
            "sweep_lift": "Lift",
            "sweep_ci": "95% CI",
            "sweep_avg_move": "Avg Move (ticks)",
            "sweep_expected_avg": "Avg Expected Move (ticks)",
            "cross_asset_title": "Cross-Asset Main Contract Sweep",
            "cross_asset_intro": "Compare the selected hypothesis across each downloaded asset's main "
            "contract. Main contract = highest positive volume delta inside that "
            "asset file.",
            "cross_asset_run": "Run / Refresh Cross-Asset Sweep",
            "cross_asset_loading": "Scanning downloaded assets...",
            "cross_asset_empty": "No product-specific tick files were found for cross-asset comparison.",
            "cross_asset_not_run": "Click Run / Refresh to compute the cross-asset table for the "
            "selected hypothesis.",
            "cross_asset_note": "This table fixes the selected hypothesis and compares assets across "
            "the same horizons. Base rate is the unconditional probability of the "
            "expected move; hypothesis accuracy is the success rate after the event "
            "rule fires.",
            "cross_asset_asset": "Asset",
            "cross_asset_contract": "Main Contract",
            "cross_asset_lift_stability": "Lift Stability",
            "cross_asset_peak_lift": "Peak Lift",
            "cross_asset_peak_lift_label": "Peak",
            "cross_asset_stability_labels": {
                "stable_positive": "Stable positive lift",
                "stable_negative": "Stable negative lift",
                "unstable": "Mixed / unstable lift",
            },
            "cross_asset_cache_hit": "Loaded cached cross-asset sweep",
            "cross_asset_cache_saved": "Saved cross-asset sweep cache",
            "ml_prob": "ML probability threshold",
            "ml_cache_title": "ML Study Cache",
            "ml_train_button": "Train / Refresh ML Study",
            "ml_calibrate_button": "Bayesian Calibrate + Train",
            "ml_calibration_trials": "Bayesian trials",
            "ml_training": "Training XGBoost and saving the study to SQLite...",
            "ml_calibrating": "Running Optuna Bayesian calibration, then training the best XGBoost "
            "model...",
            "ml_train_success": "ML study saved to SQLite.",
            "ml_calibration_success": "Bayesian calibration complete. Best validation ROC-AUC: "
            "{score:.3f}.",
            "ml_train_error": "ML training failed",
            "ml_cache_hit": "Loaded cached ML study from SQLite. Updated {updated_at}; {rows:,} "
            "prediction rows; key {key}.",
            "ml_cache_miss": "No cached ML study for these settings yet. Click Train / Refresh ML Study "
            "when you want to run XGBoost.",
            "ml_hyperparams": "Saved hyperparameters",
            "ml_calibration_summary": "Bayesian calibration summary",
            "heuristic_tab": "Pure Heuristic Optimizer",
            "heuristic_title": "Pure-Speed Heuristic Optimizer",
            "heuristic_intro": "Optuna tunes only the raw C++-portable math: fast window, slow window, "
            "percentile threshold, minimum burst size, horizon, and success "
            "threshold. No supervised ML filter is used.",
            "heuristic_train_button": "Optimize Raw Math",
            "heuristic_training": "Running Optuna on the raw pulse thresholds...",
            "heuristic_trials": "Optuna trials",
            "heuristic_folds": "Day folds",
            "heuristic_disabled": "This optimizer is currently for relative tick velocity hypotheses "
            "only. Select the continuation or exhaustion velocity hypothesis "
            "first.",
            "heuristic_success": "Pure heuristic calibration complete.",
            "heuristic_best_score": "Robust Score",
            "heuristic_calibration": "Calibration Window",
            "heuristic_holdout": "Untouched Holdout",
            "heuristic_folds_table": "Walk-forward day folds",
            "heuristic_best_params": "Best C++-portable parameters",
            "heuristic_trial_leaderboard": "Trial leaderboard",
            "heuristic_objective": "Objective",
            "heuristic_ci": "Wilson 95% CI",
            "heuristic_avg_expected": "Avg expected move",
            "ml_title": "ML Feature Discovery",
            "compare_title": "Heuristic vs ML Comparison",
            "ml_intro": "XGBoost is trained chronologically on the selected tick feature frame. The "
            "charts below are out-of-sample on the final time slice.",
            "ml_how_title": "How to read this ML tab",
            "ml_how": "\n"
            "**Feature importance** answers: which variables did XGBoost rely on most?\n"
            "\n"
            "**Tree split thresholds** answers: at roughly what values did the model keep "
            "splitting the data? Treat these as discovered zones, not final trading rules.\n"
            "\n"
            "**Model response by feature** answers: when one feature moves from low to high, "
            "does the model probability and actual success rate move with it?\n"
            "\n"
            "**Split discipline:** normal training uses a chronological holdout with a "
            "horizon-sized embargo. Bayesian calibration tunes on a separate embargoed "
            "validation slice, then reports the final saved model on the untouched holdout.\n"
            "\n"
            "Useful pattern: if ML ranks the same variables as our heuristic rule, and its "
            "split zones sit near our thresholds, the hand-built logic has support. If ML "
            "prefers different variables, the heuristic may be missing part of the story.\n",
            "ml_unavailable": "Tick ML analysis is unavailable",
            "ml_no_thresholds": "No tree split thresholds found.",
            "ml_no_response": "Not enough unique feature values for a response curve.",
            "ml_train_rows": "Train Rows",
            "ml_test_rows": "Test Rows",
            "ml_auc": "Test ROC-AUC",
            "ml_acc50": "Accuracy @ 0.50",
            "ml_signal_count": "ML Signals",
            "ml_signal_acc": "ML Signal Accuracy",
            "ml_target_rate": "Base Target Rate",
            "ml_importance": "XGBoost Feature Importance",
            "ml_thresholds": "Tree Split Thresholds",
            "ml_response": "Model Response By Feature",
            "ml_feature": "Feature",
            "ml_probability": "ML probability",
            "ml_target_rate_axis": "Observed success rate",
            "ml_threshold_zone": "ML split zone",
            "compare_agreement": "Signal Agreement",
            "compare_both": "Both Fire",
            "compare_heur_only": "Heuristic Only",
            "compare_ml_only": "ML Only",
            "compare_silent": "Both Silent",
            "compare_bucket": "Bucket",
            "compare_accuracy": "Success Rate",
            "compare_rows": "Rows",
            "compare_avg_move": "Avg Future Move",
            "compare_threshold_table": "Heuristic Rules vs ML Split Zones",
            "heuristic_rule": "Heuristic Rule",
            "ml_importance_col": "ML Importance",
            "compare_how_title": "How to read this comparison",
            "compare_how": "\n"
            "`Both Fire` means the heuristic rule and XGBoost agree that this row is a "
            "signal.\n"
            "\n"
            "`Heuristic Only` means our hand-written rule fired, but the model "
            "probability stayed below the ML threshold.\n"
            "\n"
            "`ML Only` means XGBoost found a signal outside the hand-written rule. This "
            "is the discovery bucket: inspect these rows for new rule ideas.\n"
            "\n"
            "`Both Silent` is the background sample. It should usually be large, because "
            "most ticks should not be events.\n",
            "raw_title": "Nearby Tick Rows",
            "chart_time": "Time",
            "chart_price": "Price",
            "chart_last_price": "Last price",
            "chart_future_move": "Future move",
            "chart_flow": "Flow",
            "chart_book": "Book",
            "chart_flow_imbalance": "Flow imbalance",
            "chart_book_imbalance": "Book imbalance",
            "chart_volume_intensity": "Volume intensity",
            "chart_rolling_mid_move": "Rolling mid move",
            "chart_volume_delta": "Volume delta",
            "chart_bid": "Bid",
            "chart_ask": "Ask",
            "chart_last": "Last",
            "chart_flow_book": "Flow / Book",
            "chart_volume_x": "Volume x",
            "chart_future_outcome": "Future outcome",
            "chart_count": "Count",
            "chart_positive_volume_delta": "Positive volume delta",
            "chart_median_spread": "Median spread",
            "flow_rule": "Flow imbalance rule: {rule}",
            "book_rule": "Book imbalance rule: {rule}",
            "outcome": "Outcome",
            "expected": "Expected",
            "ticks": "ticks",
            "outcome_labels": {"Correct": "Correct", "Failed": "Failed"},
            "direction_labels": {"Up": "Up", "Down": "Down"},
            "column_labels": {
                "datetime": "datetime",
                "last_price": "last_price",
                "volume_delta": "volume_delta",
                "bid_price_1": "bid_price_1",
                "bid_volume_1": "bid_volume_1",
                "ask_price_1": "ask_price_1",
                "ask_volume_1": "ask_volume_1",
                "flow_imbalance": "flow_imbalance",
                "book_imbalance": "book_imbalance",
                "volume_intensity": "volume_intensity",
                "rolling_mid_move_ticks": "rolling_mid_move_ticks",
                "price_shock": "price_shock",
                "future_move_ticks": "future_move_ticks",
                "rtv_fast_move_ticks": "rtv_fast_move_ticks",
                "rtv_abs_move_ticks": "rtv_abs_move_ticks",
                "rtv_threshold_ticks": "rtv_threshold_ticks",
                "rtv_threshold_ratio": "rtv_threshold_ratio",
                "rtv_direction": "rtv_direction",
            },
            "manual_title": "Beginner Manual: How to read this page",
            "manual": "\n"
            "### What this page is for\n"
            "This page is a tick-level event lab. It tests one selected market hypothesis at "
            "a time, then asks whether XGBoost discovers similar feature logic from the same "
            "data.\n"
            "\n"
            "### The beginner workflow\n"
            "**Step 1: Start with Contract Universe.**  \n"
            "Use the most liquid contract first. For gold this is usually `au2608` in the "
            "current files. Thin contracts can create fake-looking microstructure events "
            "because the spread and queues are unstable.\n"
            "\n"
            "**Step 2: Pick a hypothesis.**  \n"
            "`Bullish absorption` asks whether strong sell flow plus a resilient bid predicts "
            "future upside. `Bearish impulse` asks whether sell flow plus weak/resistant "
            "price action predicts future downside.\n"
            "\n"
            "**Step 3: Read the event study.**  \n"
            "`Events` is the count of rows passing the hand-written rule. `Accuracy` is the "
            "percentage that later moved in the expected direction by at least the selected "
            "minimum tick move.\n"
            "\n"
            "**Step 4: Read the ML tab.**  \n"
            "XGBoost is trained on the same feature frame, but without being told our exact "
            "rule. Feature importance shows what it found useful. Split thresholds show the "
            "value zones it repeatedly used.\n"
            "\n"
            "**Step 5: Compare the two.**  \n"
            "If ML and the heuristic fire together, the idea is more internally consistent. "
            "If ML fires alone, inspect those rows for possible new rules. If the heuristic "
            "fires alone and accuracy is weak, the hand-built thresholds may be too rigid.\n"
            "\n"
            "This is still research, not a tradable backtest. Fees, spread crossing, queue "
            "priority, and position sizing are not included here.\n",
            "formula": "\n"
            "**Research question:** when a bullish absorption event appears, does price go "
            "up after the selected forward horizon?\n"
            "\n"
            "We are testing one event family only: **bullish absorption**. There is no "
            "shared generic `pulse_score` here.\n"
            "\n"
            "Core quantities:\n"
            "\n"
            "`mid = (bid_price_1 + ask_price_1) / 2`\n"
            "\n"
            "`volume_delta = max(volume_t - volume_{t-1}, 0)` because this feed stores "
            "cumulative daily volume.\n"
            "\n"
            "`book_imbalance = (bid_volume_1 - ask_volume_1) / (bid_volume_1 + "
            "ask_volume_1)`\n"
            "\n"
            "`flow_imbalance = rolling_sum(trade_sign * volume_delta) / "
            "rolling_sum(volume_delta)`\n"
            "\n"
            "`volume_intensity = volume_delta / rolling_median(positive volume_delta)`\n"
            "\n"
            "`rolling_mid_move_ticks = rolling_sum(mid_price_change / tick_size)`\n"
            "\n"
            "Bullish absorption event:\n"
            "\n"
            "`flow_imbalance <= -0.35`\n"
            "\n"
            "and\n"
            "\n"
            "`book_imbalance >= 0.10`\n"
            "\n"
            "and\n"
            "\n"
            "`volume_intensity >= 3.0`\n"
            "\n"
            "and\n"
            "\n"
            "`rolling_mid_move_ticks >= 0`\n"
            "\n"
            "Plain English: sellers are hitting the market, the visible bid side is still "
            "stronger, volume is abnormal, and price has held flat/up over the rolling "
            "window. That is the bullish absorption hypothesis.\n"
            "\n"
            "Outcome test:\n"
            "\n"
            "`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / tick_size`\n"
            "\n"
            "`correct = future_move_ticks > 0`\n"
            "\n"
            "`accuracy = correct_absorption_events / total_absorption_events`\n",
            "why": "\n"
            "**What accuracy means:** if the page says `60%`, then 60% of detected bullish "
            "absorption events were followed by a higher mid-price after the selected horizon.\n"
            "\n"
            "**What it does not mean:** it does not include fees, spread crossing, fill quality, "
            "stop losses, or position sizing. This is a clean first-pass event study, not a "
            "trading backtest.\n"
            "\n"
            "**Why the horizon matters:** a pattern may fail over 30 ticks but work over 240 "
            "ticks. The horizon slider is therefore part of the hypothesis, not a cosmetic "
            "control.\n"
            "\n"
            "**How to use this page:** start with `au2608`, keep the rolling window at `120`, "
            "then test horizons such as `60`, `120`, and `240` ticks. Look for enough events, a "
            "clear accuracy edge above 50%, and positive average future ticks.\n",
        },
        "zh": {
            "title": "日内事件研究",
            "subtitle": "用日内数据检验市场微观结构假设。",
            "file": "Tick parquet 文件",
            "nav_label": "日内事件研究导航",
            "loading_file": "正在读取 tick parquet...",
            "loading_summary": "正在构建合约池...",
            "loading_features": "正在计算当前合约特征...",
            "loading_thresholds": "正在校准阈值...",
            "loading_events": "正在评估假设事件...",
            "loading_done": "已就绪。",
            "loading_complete": "完成",
            "math_cache_hit": "已从 SQLite 缓存读取 horizon 扫描",
            "math_cache_miss": "正在计算 horizon 扫描并写入 SQLite 缓存...",
            "math_cache_store": "正在保存 horizon 扫描缓存...",
            "math_cache_backend": "Math 扫描缓存",
            "no_tick_files": "在 runtime/data/futures_cn/tick 中没有找到 tick_all_data parquet 文件。",
            "product": "品种",
            "symbol": "合约",
            "hypothesis": "假设",
            "hypothesis_source": "假设来源",
            "hypothesis_source_options": {
                "built_in": "内置规则",
                "saved_seed": "已保存发现种子",
            },
            "saved_seed_select": "已保存种子",
            "saved_seed_empty": "还没有保存的发现种子。先打开模式实验室，检查一个脉冲，然后保存为假设种子。",
            "saved_seed_scope_warning": "这个种子来自 `{seed_file}` / `{seed_symbol}`。你当前正在测试 "
            "`{current_file}` / `{current_symbol}`。",
            "saved_seed_context": "种子规则上下文",
            "saved_seed_math_note": "这是已保存的发现种子。下面展示具体规则；请在假设验证页观察事件结果和 horizon 扫描。",
            "saved_seed_cross_asset_note": "跨资产扫描目前只支持内置假设。已保存种子先在当前选择合约上评估。",
            "threshold_mode": "阈值模式",
            "threshold_modes": {
                "adaptive": "按资产分位数自适应（推荐）",
                "fixed": "固定黄金参数（调试）",
            },
            "threshold_mode_advanced_title": "高级 / 调试阈值模式",
            "threshold_mode_active": "当前阈值模式：**{mode}**",
            "threshold_mode_debug_note": "只有在你想复现/调试最早的黄金规则时，才使用固定参数。做跨资产研究时，保持自适应即可。",
            "threshold_mode_note": "\n"
            "跨资产研究时，建议使用 **按资产分位数自适应**。它保持假设结构不变，但把每个条件映射到该合约自己的分布。\n"
            "\n"
            "例如：不再要求所有品种都用 `volume_intensity >= 3.0`，而是要求该品种自己的成交强度前 "
            "10%。这是概率匹配，不是机器学习。\n",
            "base_rate_visual_title": "Base Rate 与假设准确率：事件显微镜",
            "base_rate_visual_caption": "这一段用真实触发的事件先解释机制，再让后面的扫描表格判断假设在统计上是否站得住。",
            "base_rate_visual_empty": "有效行数不足，无法绘制 Base Rate 解释图。",
            "base_rate_mechanics": "\n"
            "**先看这里。** tick 数据的一行是 parquet 文件里的一个市场**快照**。所以 `t + {horizon}` "
            "的意思是：在同一个合约、同一个交易 session 内，当前事件行之后第 `{horizon}` 行快照；不是 `{horizon}` "
            "秒以后，也不是连续 `{horizon}` 次价格上涨/下跌。\n"
            "\n"
            "`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / "
            "tick_size`。如果这里显示 `+28`，意思是未来中间价比事件时刻高了 28 个最小报价单位。它不是说价格连续涨了 28 "
            "行。\n"
            "\n"
            "Base Rate 使用所有拥有有效未来快照、且预期方向已定义的行。Hypothesis Accuracy "
            "只使用更严格的子集：事件条件也触发的行。\n",
            "base_rate_metric": "Base Rate",
            "event_accuracy_metric": "事件准确率",
            "base_rate_valid_rows": "Base Rate 行数",
            "base_rate_event_rows": "触发事件数",
            "base_rate_criteria_title": "Base Rate 条件",
            "event_criteria_title": "假设事件条件",
            "criteria_valid_future": "同一合约/session 内存在第 t + {horizon} 行快照",
            "criteria_expected_direction": "预期方向已定义：{direction}",
            "criteria_future_success": "未来变动 {move} 个价格 tick，是否按{direction}至少达到 {success} tick",
            "criteria_hypothesis_signal": "假设事件触发：上方所有事件条件都成立",
            "criteria_rtv_direction": "快窗口方向存在：脉冲={direction}，预期={expected}",
            "criteria_rtv_percentile": "快窗口绝对变动 {value} tick >= 过去滚动 {percentile} 阈值 {threshold} tick",
            "criteria_rule": "{name} {op} {threshold}{unit}；当前={value}{unit}",
            "criteria_current_row_note": "这些勾选框只显示当前示例行是否满足规则，不是控制按钮。",
            "criteria_feature_names": {
                "flow_imbalance": "成交流失衡",
                "book_imbalance": "盘口失衡",
                "volume_intensity": "成交量强度",
                "rolling_mid_move_ticks": "滚动中间价变动",
                "price_shock": "价格冲击",
                "rtv_abs_move_ticks": "快窗口绝对变动",
            },
            "event_examples_title": "两个真实触发事件示例",
            "event_examples_caption": "两个图都是真实触发的假设事件。左边是成功事件，右边是失败事件。鼠标悬停任意点，可以看到该行参与事件判断的变量。",
            "success_example_title": "成功事件",
            "failure_example_title": "失败事件",
            "missing_success_example": "当前设置下没有成功的触发事件。",
            "missing_failure_example": "当前设置下没有失败的触发事件。",
            "event_point_label": "事件 t",
            "outcome_point_label": "horizon 结果",
            "entry_price_label": "事件价格",
            "snapshot_role_fast": "过去快窗口快照",
            "snapshot_role_event": "事件行 t",
            "snapshot_role_horizon": "未来结果行",
            "snapshot_role_forward": "等待中的未来快照",
            "event_datetime_label": "事件快照",
            "future_datetime_label": "未来快照",
            "expected_direction_label": "预期方向",
            "future_move_formula_label": "未来变动公式",
            "future_move_formula": "({future} - {current}) / {tick_size} = {move} 个价格 tick",
            "outcome_label": "结果",
            "base_checks_title": "Base Rate 检查",
            "event_checks_title": "事件触发检查",
            "event_chart_hover": "<b>%{customdata[0]}</b><br>角色：%{customdata[2]}<br>相对事件 t "
            "的快照偏移：%{x}<br>中间价：%{y:.3f}<br>相对事件价变动：%{customdata[1]:+.2f} 个价格 "
            "tick<br>成交流失衡：%{customdata[3]:.3f}<br>盘口失衡：%{customdata[4]:.3f}<br>成交量强度：%{customdata[5]:.3f}<br>滚动中间价变动：%{customdata[6]:+.2f} "
            "个价格 tick<br>价格冲击：%{customdata[7]:.3f}<br>RTV "
            "绝对变动：%{customdata[8]:.2f}<br>RTV 阈值：%{customdata[9]:.2f}<br>RTV "
            "比例：%{customdata[10]:.2f}<extra></extra>",
            "outcome_hover": "结果：{outcome}<br>未来变动：{move} 个价格 tick<br>预期：{expected}<br>成功阈值：{success} "
            "个价格 tick<extra></extra>",
            "show_fast_window": "显示 6 个快窗口快照",
            "show_horizon": "显示向前 horizon",
            "show_success_threshold": "显示成功阈值",
            "show_event_threshold": "显示事件阈值",
            "show_base_population": "显示 Base Rate 样本说明",
            "base_rate_plain_language": "\n"
            "**tick 有两个意思：**\n"
            "\n"
            "`{fast_window}-tick move` = 跨 `{fast_window}` 个 tick "
            "数据快照的价格差。\n"
            "\n"
            "`{success} price tick success` = 价格至少移动 `{success}` 个最小报价单位。\n"
            "\n"
            "`horizon={horizon}` = 比较当前快照 `t` 和 `t+{horizon}` "
            "的价格，不是只看下一行快照。\n",
            "chart_mid_price": "中间价",
            "snapshot_axis": "相对当前 t 的快照偏移",
            "price_axis": "价格",
            "fast_window_label": "过去 {fast_window} 个快照",
            "horizon_label": "t + {horizon}",
            "success_price_label": "若未来向{direction}至少 {success} 个价格 tick，则成功",
            "event_threshold_annotation": "过去 6 快照变动={past_move} 个价格 tick<br>RTV "
            "阈值={threshold}<br>比例={ratio}",
            "concept_col": "概念",
            "definition_col": "定义",
            "example_col": "当前示例行数值",
            "fast_move_concept": "快速变动",
            "fast_move_definition": "mid_price[t] - mid_price[t - {fast_window}]，再除以最小价格 tick。",
            "future_move_concept": "未来变动",
            "future_move_definition": "mid_price[t + {horizon}] - mid_price[t]，再除以最小价格 tick。",
            "success_label_concept": "成功标签",
            "success_label_definition": "布尔值 True/False：未来价格是否按预期方向达到最小成功变动？",
            "price_ticks_unit": "价格 tick",
            "correct_label": "正确",
            "failed_label": "失败",
            "base_rate_population_note": "Base Rate 使用所有拥有有效未来标签的行。Accuracy 只使用假设/事件触发的行。对相对速度假设来说，没有明确 "
            "6 快照方向的行，在背景 Base Rate 计算里不算成功。",
            "hypothesis_labels": {
                "bullish": "看涨吸收：卖压被承接 -> 上涨",
                "bearish": "看跌冲击：卖压确认 -> 下跌",
                "bearish_breakdown": "看跌破位：卖压冲击已经压破价格 -> 下跌",
                "relative_velocity": "相对Tick速度：99分位突破",
                "relative_velocity_fade": "相对Tick速度衰竭：反向做均值回归",
            },
            "window": "滚动 tick 窗口",
            "horizon": "向前验证窗口",
            "success_move": "最小成功变动",
            "top": "图上最多事件标记数",
            "calculation_audit": "计算审计",
            "audit_note": "特征计算只使用清洗后的可交易 tick。bid、ask 或 last price 为 "
            "0/无效的行会被剔除，并且滚动窗口与向前验证窗口会在交易时段断点处重置。",
            "raw_rows": "原始行数",
            "valid_feature_rows": "有效特征行数",
            "dropped_rows": "剔除无效行",
            "session_count": "识别到的交易时段",
            "marker_caption": "图上事件标记：显示 {shown:,} / {total:,} 个事件。若数量被限制，会按时间均匀抽样，而不是只取最早的事件。",
            "plot_downsample_note": "大规模折线图会降采样到 {points:,} 个点以保证浏览器速度。事件统计和 ML 训练仍使用完整事件表。",
            "event_episode_note": "连续触发信号的 tick 会被合并为一个事件 episode，并取该 episode 的第一个 "
            "tick。这样可以避免同一次市场脉冲被重复显示成很多事件。",
            "event_distribution": "按日期统计事件",
            "audit_date": "日期",
            "audit_first_event": "第一个事件",
            "audit_last_event": "最后一个事件",
            "schema": "数据字段",
            "row_viewer": "原始 Tick 行查看器",
            "row_mode": "查看哪一行？",
            "row_modes": ["第一行", "中间行", "最后一行", "自定义序号"],
            "row_index": "当前合约内的行序号",
            "row_position": "{symbol} 合约内第 {row:,} 行 / 共 {last:,} 行 | 全文件第 {source:,} 行",
            "row_empty": "{symbol} 没有找到原始行。",
            "field": "字段",
            "value": "数值",
            "dtype": "类型",
            "meaning": "含义",
            "row_meanings": {
                "symbol": "合约代码。例如 au2608 表示 2026 年 8 月交割的上海黄金期货合约。",
                "datetime": "该 tick 的精确时间戳。",
                "last_price": "该 tick 时最近一笔成交价。",
                "volume": "交易所/数据源报告的日内累计成交量。",
                "bid_price_1": "买一价，即当前最优可见买价。",
                "bid_volume_1": "买一价位上的可见挂单量。",
                "ask_price_1": "卖一价，即当前最优可见卖价。",
                "ask_volume_1": "卖一价位上的可见挂单量。",
                "oi": "持仓量，通常表示当前未平仓合约数量。",
            },
            "contracts": "合约池",
            "asset_ranker_tab": "模式范围",
            "asset_ranker_title": "模式镜头范围",
            "asset_ranker_intro": "先选择研究镜头，再排序资产。日频、1分钟和 tick 数据回答的是不同问题，所以这里会先把每个镜头的用途说明清楚。",
            "asset_ranker_manual_title": "为什么要用不同镜头？",
            "asset_ranker_manual": "\n"
            "不同时间尺度会暴露不同的市场行为。\n"
            "\n"
            "1. **日频范围**：先找大方向候选，判断哪些市场有足够波动、成交和覆盖度，值得深入研究。\n"
            "2. **1分钟日内**：检查波动是否存在于日内，观察交易时段、趋势性、震荡和特定时间线索。\n"
            "3. **Tick 微观结构**：检查执行现实，包括脉冲密度、价差/噪声、流动性爆发，以及模式是否可能覆盖成本。\n"
            "\n"
            "`下载优先级 = 65% 波动率分位 + 20% 成交量分位 + 10% 持仓量分位 + 5% 数据覆盖率`。\n",
            "pattern_lens": "模式镜头",
            "pattern_lens_options": {
                "daily": "日频范围",
                "minute": "1分钟日内",
                "tick": "Tick 微观结构",
            },
            "pattern_lens_purpose": {
                "daily": "日频镜头：用于大范围初筛。先判断哪些资产值得进一步研究，再投入时间看日内或 tick 数据。",
                "minute": "1分钟镜头：用于观察日内行为。比较日内实现波动、趋势性、震荡性和交易时段效应。",
                "tick": "Tick 镜头：用于检查执行现实。观察脉冲密度、价差/噪声、流动性爆发和微观结构可交易性。",
            },
            "pattern_lens_data_note": {
                "daily": "日频排序使用收盘到收盘收益和日频流动性。它是资产初筛镜头，不证明分钟级或 tick 级策略一定有效。",
                "minute": "1分钟排序使用日内 parquet 的 bar-to-bar 收益。它更接近策略行为，但仍未纳入盘口执行成本。",
            },
            "asset_source_file": {
                "daily": "全市场日频 parquet",
                "minute": "1分钟 parquet",
            },
            "asset_lookback_by_lens": {
                "daily": "波动率回看窗口（日）",
                "minute": "波动率回看窗口（1分钟 bar）",
            },
            "asset_min_obs_by_lens": {
                "daily": "最少有效天数",
                "minute": "最少有效1分钟 bar",
            },
            "asset_daily_file": "全市场日频 parquet",
            "asset_lookback": "波动率回看窗口",
            "asset_min_obs": "最少有效天数",
            "asset_top_n": "展示行数",
            "asset_sort": "排序方式",
            "asset_sort_options": {
                "download_priority_score": "下载优先级",
                "recent_ann_vol": "近期波动率",
                "avg_daily_volume": "平均成交量",
                "avg_oi": "平均持仓量",
                "avg_intraday_range_pct": "平均日内振幅",
            },
            "asset_sector_filter": "板块筛选",
            "asset_min_volume": "最低平均成交量",
            "asset_rank_empty": "当前筛选条件下没有资产。",
            "asset_rank_download": "下载候选列表 CSV",
            "manager_demo_download_hidden": "管理层演示模式下已禁用下载。",
            "asset_rank_chart": "高波动候选品种",
            "asset_rank_table": "模式候选列表",
            "asset_rank_chart_by_lens": {
                "daily": "日频波动候选",
                "minute": "1分钟日内波动候选",
            },
            "asset_rank_table_by_lens": {
                "daily": "日频范围候选列表",
                "minute": "1分钟模式候选列表",
            },
            "asset_rank_note": "高波动有用，但每个镜头回答的是不同问题。优先选择有价格运动、流动性足够、数据覆盖健康的品种。",
            "dataset_timeframe": "数据周期",
            "dataset_start": "起始时间",
            "dataset_end": "结束时间",
            "dataset_assets": "资产数",
            "dataset_rows": "行数",
            "dataset_date_range": "日期区间",
            "tick_lens_title": "可用 Tick 文件",
            "tick_lens_caption": "Tick 数据不应套用日频波动排序。它用于观察脉冲密度、价差/噪声、流动性爆发和事件级验证。",
            "tick_lens_route": "这个镜头的下一步：进入“脉冲工作台”检查单合约，或进入“跨资产地图”比较已下载 tick 文件的脉冲行为。",
            "tick_file_count": "Tick 文件数",
            "tick_file": "Tick 文件",
            "tick_size_mb": "大小 MB",
            "tick_modified": "修改时间",
            "asset_rank_metrics": {
                "ranked_assets": "入榜资产数",
                "top_vol": "最高波动率",
                "median_vol": "中位波动率",
                "top_priority": "最高优先级",
            },
            "detected": "识别到的品种",
            "liquid": "最活跃合约",
            "rows": "行数",
            "events": "假设事件",
            "event_rate": "事件占比",
            "correct_moves": "正确次数",
            "accuracy": "准确率",
            "avg_move": "平均未来变动",
            "evidence_ticket_button": "保存证据票据",
            "evidence_ticket_saved": "证据票据已保存：{ticket_id}",
            "evidence_ticket_no_events": "当前设置下没有可保存的假设事件。",
            "evidence_ticket_error": "无法保存证据票据：{error}",
            "contract_health": "合约健康度",
            "event_map": "假设验证",
            "trading_hours_compressed_note": "为了更容易阅读，x "
            "轴会隐藏标准非交易时间段：02:00-09:30、10:15-10:30、11:30-13:30、15:00-21:00。tick "
            "顺序和计算本身不变。",
            "event_horizon_section_title": "盈亏比与单个事件未来变动",
            "payoff_chart_title": "盈亏比快照",
            "payoff_chart_caption": "这里使用当前假设和当前 horizon 下的全部事件。柱子比较平均有利变动和平均不利变动。",
            "payoff_empty": "有效事件变动不足，无法计算盈亏不对称。",
            "payoff_ratio": "盈亏比",
            "payoff_health_note": "粗略健康标准：Accuracy 应该高于扫描表里的 Base Rate；盈亏比应高于 1.0x。",
            "payoff_avg_favorable": "平均有利变动",
            "payoff_avg_adverse": "平均不利变动",
            "payoff_yaxis": "按预期方向衡量的变动（价格 tick）",
            "payoff_hover": "指标：%{x}<br>数值：%{customdata[0]:+.2f} "
            "tick<br>类型：%{customdata[1]}<extra></extra>",
            "event_horizon_title": "单个事件的未来变动",
            "event_horizon_caption": "选择一个触发事件，观察价格在 t+30、t+60、t+120、t+240、t+480 "
            "个快照后的变动。柱子按该事件的预期方向衡量，所以绿色表示走势帮助假设。",
            "event_horizon_empty": "无法为这个事件生成未来 horizon 柱状图。",
            "event_horizon_yaxis": "按预期方向衡量的变动（{expected}，价格 tick）",
            "event_horizon_success_line": "成功阈值 = {success} tick",
            "event_horizon_result_labels": {
                "Profitable": "方向正确",
                "Loss": "方向错误",
                "Flat": "持平",
                "Missing": "缺少未来行",
            },
            "event_horizon_hover": "未来快照：%{customdata[0]}<br>未来价格：%{customdata[1]:.3f}<br>原始未来变动：%{customdata[2]:+.2f} "
            "tick<br>按预期方向衡量的变动：%{customdata[3]:+.2f} "
            "tick<br>结果：%{customdata[4]}<extra></extra>",
            "anatomy": "条件与结果",
            "microscope": "事件显微镜",
            "outcome_overview": "结果总览",
            "event_ledger": "全部事件表",
            "event_ledger_note": "下表每一行都是一个被识别出的假设事件。绿色表示之后价格走势符合假设，红色表示没有符合。",
            "event_ledger_download": "下载事件表 CSV",
            "events_by_day_chart": "按日期统计事件",
            "future_move_chart": "未来变动分布",
            "ledger_date": "日期",
            "ledger_time": "事件时间",
            "ledger_session": "交易时段",
            "ledger_outcome": "结果",
            "ledger_expected": "预期方向",
            "ledger_last_price": "最新价",
            "ledger_future_time": "验证时间",
            "ledger_future_move": "未来变动",
            "ledger_flow": "成交流失衡",
            "ledger_book": "盘口失衡",
            "ledger_volume_x": "成交量倍数",
            "ledger_rolling_mid": "滚动中间价变动",
            "ledger_price_shock": "价格冲击",
            "ledger_rtv_fast": "6 Tick 变动",
            "ledger_rtv_pct": "速度/阈值",
            "ledger_rtv_threshold": "99% 阈值",
            "ledger_count": "事件数",
            "chart_rtv_fast": "6 tick 变动",
            "chart_rtv_abs": "6 tick 绝对变动",
            "chart_rtv_threshold": "滚动99%阈值",
            "chart_rtv_percentile": "速度/阈值",
            "chart_rtv_direction": "速度方向",
            "select_event": "选择事件",
            "no_events": "当前假设条件下没有事件。",
            "math_title": "假设定义与数学公式",
            "why_title": "如何阅读准确率",
            "research_title": "当前证据与看跌事件重设计方案",
            "sweep_title": "实时 Horizon 扫描",
            "redesign_title": "如何重设计看跌事件",
            "sweep_hypothesis": "假设",
            "sweep_horizon": "向前窗口",
            "sweep_events": "事件数",
            "sweep_accuracy": "准确率",
            "sweep_base_rate": "基础成功率",
            "sweep_lift": "超额",
            "sweep_ci": "95% 置信区间",
            "sweep_avg_move": "平均变动 (tick)",
            "sweep_expected_avg": "预期方向平均变动 (tick)",
            "cross_asset_title": "跨资产主力合约扫描",
            "cross_asset_intro": "把当前选择的假设，应用到每个已下载品种的主力合约上做横向比较。主力合约 = 该品种文件中 positive volume delta "
            "最大的合约。",
            "cross_asset_run": "运行 / 刷新跨资产扫描",
            "cross_asset_loading": "正在扫描已下载品种...",
            "cross_asset_empty": "没有找到可用于跨资产比较的品种级 tick 文件。",
            "cross_asset_not_run": "点击运行 / 刷新，即可按当前假设计算跨资产表格。",
            "cross_asset_note": "这个表固定当前选择的假设，然后比较不同资产在同一组 horizon 下的表现。Base Rate 是无条件未来走势概率；Hypothesis "
            "Accuracy 是事件触发后的成功率。",
            "cross_asset_asset": "资产",
            "cross_asset_contract": "主力合约",
            "cross_asset_lift_stability": "Lift 稳定性",
            "cross_asset_peak_lift": "最高 Lift",
            "cross_asset_peak_lift_label": "最高",
            "cross_asset_stability_labels": {
                "stable_positive": "稳定正 Lift",
                "stable_negative": "稳定负 Lift",
                "unstable": "方向混杂 / 不稳定 Lift",
            },
            "cross_asset_cache_hit": "已读取跨资产扫描缓存",
            "cross_asset_cache_saved": "已保存跨资产扫描缓存",
            "ml_prob": "ML 概率阈值",
            "ml_cache_title": "ML 研究缓存",
            "ml_train_button": "训练 / 刷新 ML 研究",
            "ml_calibrate_button": "贝叶斯校准 + 训练",
            "ml_calibration_trials": "贝叶斯试验次数",
            "ml_training": "正在训练 XGBoost，并把研究结果保存到 SQLite...",
            "ml_calibrating": "正在运行 Optuna 贝叶斯校准，然后训练最佳 XGBoost 模型...",
            "ml_train_success": "ML 研究结果已保存到 SQLite。",
            "ml_calibration_success": "贝叶斯校准完成。最佳验证 ROC-AUC：{score:.3f}。",
            "ml_train_error": "ML 训练失败",
            "ml_cache_hit": "已从 SQLite 读取缓存的 ML 研究。更新时间 {updated_at}；预测行数 {rows:,}；key {key}。",
            "ml_cache_miss": "当前参数还没有缓存的 ML 研究结果。需要运行 XGBoost 时，请点击训练 / 刷新 ML 研究。",
            "ml_hyperparams": "已保存的超参数",
            "ml_calibration_summary": "贝叶斯校准摘要",
            "heuristic_tab": "纯规则优化器",
            "heuristic_title": "纯速度规则优化器",
            "heuristic_intro": "Optuna 只优化可以迁移到 C++ 的原始数学参数：快速窗口、慢速窗口、分位数阈值、最小速度冲刺、horizon 和成功 tick "
            "门槛。这里不使用监督学习过滤器。",
            "heuristic_train_button": "优化原始数学规则",
            "heuristic_training": "正在用 Optuna 优化原始脉冲阈值...",
            "heuristic_trials": "Optuna 试验次数",
            "heuristic_folds": "按日期折数",
            "heuristic_disabled": "这个优化器目前只支持相对 tick 速度假设。请先选择速度延续或速度衰竭假设。",
            "heuristic_success": "纯规则校准完成。",
            "heuristic_best_score": "稳健分数",
            "heuristic_calibration": "校准窗口",
            "heuristic_holdout": "未参与优化的样本外",
            "heuristic_folds_table": "Walk-forward 日期折表现",
            "heuristic_best_params": "最佳 C++ 可移植参数",
            "heuristic_trial_leaderboard": "试验排行榜",
            "heuristic_objective": "目标函数",
            "heuristic_ci": "Wilson 95% 区间",
            "heuristic_avg_expected": "预期方向平均变动",
            "ml_title": "机器学习特征发现",
            "compare_title": "人工规则 vs 机器学习比较",
            "ml_intro": "XGBoost 按时间顺序训练在当前选择的 tick 特征上。下面结果是最后一段时间切片的样本外表现。",
            "ml_how_title": "如何阅读这个 ML 页面",
            "ml_how": "\n"
            "**特征重要性**回答：XGBoost 最依赖哪些变量？\n"
            "\n"
            "**树模型切分阈值**回答：模型大概在哪些数值附近反复把样本切开？这些是机器发现的区间，不是最终交易规则。\n"
            "\n"
            "**模型对特征的响应**回答：当某个特征从低到高变化时，模型概率和真实成功率是否跟着变化？\n"
            "\n"
            "**切分纪律：** 普通训练使用时间顺序 holdout，并加入等于预测窗口的 embargo。贝叶斯校准只在单独的、带 embargo "
            "的验证切片上调参，最后保存的模型仍然在未触碰的最终 holdout 上报告结果。\n"
            "\n"
            "有用的观察方式：如果 ML 排名前几的变量和人工规则一致，而且切分区间接近我们的阈值，说明手写逻辑有一定支持。如果 ML "
            "偏好完全不同的变量，人工规则可能漏掉了市场结构的一部分。\n",
            "ml_unavailable": "Tick 机器学习分析不可用",
            "ml_no_thresholds": "没有找到树模型切分阈值。",
            "ml_no_response": "该特征的不同取值太少，无法画响应曲线。",
            "ml_train_rows": "训练行数",
            "ml_test_rows": "测试行数",
            "ml_auc": "测试 ROC-AUC",
            "ml_acc50": "0.50 准确率",
            "ml_signal_count": "ML 信号数",
            "ml_signal_acc": "ML 信号准确率",
            "ml_target_rate": "基础成功率",
            "ml_importance": "XGBoost 特征重要性",
            "ml_thresholds": "树模型切分阈值",
            "ml_response": "模型对特征的响应",
            "ml_feature": "特征",
            "ml_probability": "ML 概率",
            "ml_target_rate_axis": "实际成功率",
            "ml_threshold_zone": "ML 切分区间",
            "compare_agreement": "信号一致率",
            "compare_both": "两者都触发",
            "compare_heur_only": "仅人工规则",
            "compare_ml_only": "仅 ML",
            "compare_silent": "两者都沉默",
            "compare_bucket": "分组",
            "compare_accuracy": "成功率",
            "compare_rows": "行数",
            "compare_avg_move": "平均未来变动",
            "compare_threshold_table": "人工规则 vs ML 切分区间",
            "heuristic_rule": "人工规则",
            "ml_importance_col": "ML 重要性",
            "compare_how_title": "如何阅读这个比较",
            "compare_how": "\n"
            "`两者都触发` 表示人工规则和 XGBoost 都认为这一行是信号。\n"
            "\n"
            "`仅人工规则` 表示手写规则触发了，但 ML 概率低于阈值。\n"
            "\n"
            "`仅 ML` 表示 XGBoost 在人工规则之外发现了信号。这是最值得研究的发现区：可以去看这些行是否暗示新的规则。\n"
            "\n"
            "`两者都沉默` 是背景样本。它通常应该最大，因为大多数 tick 不应该是事件。\n",
            "raw_title": "事件附近 Tick 明细",
            "chart_time": "时间",
            "chart_price": "价格",
            "chart_last_price": "最新成交价",
            "chart_future_move": "未来变动",
            "chart_flow": "成交流",
            "chart_book": "盘口",
            "chart_flow_imbalance": "成交流失衡",
            "chart_book_imbalance": "盘口失衡",
            "chart_volume_intensity": "成交量强度",
            "chart_rolling_mid_move": "滚动中间价变动",
            "chart_volume_delta": "本 tick 新增成交量",
            "chart_bid": "买一",
            "chart_ask": "卖一",
            "chart_last": "最新价",
            "chart_flow_book": "成交流 / 盘口",
            "chart_volume_x": "成交量倍数",
            "chart_future_outcome": "未来结果",
            "chart_count": "次数",
            "chart_positive_volume_delta": "正向新增成交量",
            "chart_median_spread": "中位买卖价差",
            "flow_rule": "成交流失衡规则：{rule}",
            "book_rule": "盘口失衡规则：{rule}",
            "outcome": "结果",
            "expected": "预期方向",
            "ticks": "跳",
            "outcome_labels": {"Correct": "正确", "Failed": "失败"},
            "direction_labels": {"Up": "上涨", "Down": "下跌"},
            "column_labels": {
                "datetime": "时间",
                "last_price": "最新价",
                "volume_delta": "新增成交量",
                "bid_price_1": "买一价",
                "bid_volume_1": "买一量",
                "ask_price_1": "卖一价",
                "ask_volume_1": "卖一量",
                "flow_imbalance": "成交流失衡",
                "book_imbalance": "盘口失衡",
                "volume_intensity": "成交量强度",
                "rolling_mid_move_ticks": "滚动中间价变动(tick)",
                "price_shock": "价格冲击",
                "future_move_ticks": "未来变动(tick)",
                "rtv_fast_move_ticks": "6 tick 变动",
                "rtv_abs_move_ticks": "6 tick 绝对变动",
                "rtv_threshold_ticks": "99%速度阈值",
                "rtv_threshold_ratio": "速度/阈值",
                "rtv_direction": "速度方向",
            },
            "manual_title": "新手使用手册：如何读这个页面",
            "manual": "\n"
            "### 这个页面是做什么的？\n"
            "这是一个 tick 级别事件研究实验室。它一次只测试一个市场假设，然后让 XGBoost 在同一批数据上学习，看看机器发现的特征逻辑是否和人工规则相似。\n"
            "\n"
            "### 新手阅读顺序\n"
            "**第一步：先看合约池。**  \n"
            "先使用最活跃合约。当前黄金文件里通常是 `au2608`。不活跃合约的价差和盘口更不稳定，容易产生假的微观结构信号。\n"
            "\n"
            "**第二步：选择假设。**  \n"
            "`看涨吸收` 问的是：强卖压出现，但买盘承接住，之后是否上涨？`看跌冲击` 问的是：卖压出现且价格已经被压住，之后是否继续下跌？\n"
            "\n"
            "**第三步：读事件研究。**  \n"
            "`事件` 是通过人工规则的行数。`准确率` 是这些事件之后，价格是否按预期方向移动，并且至少达到你选择的最小 tick 变动。\n"
            "\n"
            "**第四步：读 ML 页面。**  \n"
            "XGBoost 使用同一批特征训练，但不会被告知我们的人工规则。特征重要性显示机器觉得哪些变量有用；切分阈值显示机器在哪些数值区间反复做判断。\n"
            "\n"
            "**第五步：比较两者。**  \n"
            "如果 ML 和人工规则同时触发，说明这个想法内部更一致。如果只有 ML "
            "触发，可以去研究这些行，看看是否能形成新的规则。如果只有人工规则触发但准确率弱，说明手写阈值可能太僵硬。\n"
            "\n"
            "这仍然是研究工具，不是可交易回测。手续费、买卖价差、排队成交、仓位大小都还没有纳入。\n",
            "formula": "\n"
            "**研究问题：** 当看涨吸收事件出现后，价格在所选向前窗口后是否上涨？\n"
            "\n"
            "我们现在只研究一个事件家族：**看涨吸收**。这里不再使用通用 `pulse_score`。\n"
            "\n"
            "核心变量：\n"
            "\n"
            "`mid = (bid_price_1 + ask_price_1) / 2`\n"
            "\n"
            "`volume_delta = max(volume_t - volume_{t-1}, 0)`，因为该数据里的 `volume` 是日内累计成交量。\n"
            "\n"
            "`book_imbalance = (bid_volume_1 - ask_volume_1) / (bid_volume_1 + "
            "ask_volume_1)`\n"
            "\n"
            "`flow_imbalance = rolling_sum(trade_sign * volume_delta) / "
            "rolling_sum(volume_delta)`\n"
            "\n"
            "`volume_intensity = volume_delta / rolling_median(正 volume_delta)`\n"
            "\n"
            "`rolling_mid_move_ticks = rolling_sum(mid_price_change / tick_size)`\n"
            "\n"
            "看涨吸收事件：\n"
            "\n"
            "`flow_imbalance <= -0.35`\n"
            "\n"
            "并且\n"
            "\n"
            "`book_imbalance >= 0.10`\n"
            "\n"
            "并且\n"
            "\n"
            "`volume_intensity >= 3.0`\n"
            "\n"
            "并且\n"
            "\n"
            "`rolling_mid_move_ticks >= 0`\n"
            "\n"
            "直觉解释：卖方正在主动砸盘，但可见买盘仍然更强，成交量异常放大，同时价格在滚动窗口内没有下跌而是守住/上移。这就是看涨吸收假设。\n"
            "\n"
            "结果验证：\n"
            "\n"
            "`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / tick_size`\n"
            "\n"
            "`correct = future_move_ticks > 0`\n"
            "\n"
            "`accuracy = 正确的看涨吸收事件数 / 全部看涨吸收事件数`\n",
            "why": "\n"
            "**准确率是什么意思：** 如果页面显示 `60%`，表示被检测到的看涨吸收事件中，有 60% 在所选向前窗口后 mid price 更高。\n"
            "\n"
            "**它不代表什么：** 它还没有考虑手续费、买卖价差、成交质量、止损、仓位大小。这是事件研究，不是完整回测。\n"
            "\n"
            "**为什么窗口重要：** 一个结构可能在 30 个 tick 后失败，但在 240 个 tick 后有效。所以 horizon 不是装饰参数，而是假设本身的一部分。\n"
            "\n"
            "**如何使用这个页面：** 先看 `au2608`，rolling window 保持 `120`，然后分别测试 `60`、`120`、`240` tick "
            "的向前窗口。重点看事件数量是否足够、准确率是否明显高于 50%、平均未来 tick 变动是否为正。\n",
        },
    },
    "pulse_discovery_lab": {
        "en": {
            "_lang": "EN",
            "title": "Pattern Scan",
            "subtitle": "Discover daily, intraday, and tick-level patterns before formal validation.",
            "file": "Tick parquet file",
            "symbol": "Contract",
            "window": "Rolling window",
            "percentile": "Pulse percentile",
            "top": "Events to show",
            "seconds": "seconds",
            "rows": "Rows",
            "events": "Collapsed pulses",
            "threshold": "Pulse threshold",
            "p99": "p99 move",
            "cache": "Cache",
            "cache_hit": "hit",
            "cache_saved": "saved",
            "cache_computed": "computed",
            "cache_frame": "frame",
            "cache_events": "events",
            "rows_label": "rows",
            "elapsed": "elapsed",
            "axis_percentile": "Percentile of rolling windows",
            "axis_pulses_per_hour": "Pulses / trading hour",
            "hover_percentile": "Percentile",
            "hover_move": "Move",
            "hover_selected_pulse": "Selected pulse",
            "hover_price": "Price",
            "last_price": "Last price",
            "window_start": "Window start",
            "pulse_peak": "Pulse peak",
            "p99_ticks": "p99 ticks",
            "main_short": "Main",
            "pulses_hour_short": "Pulses/hour",
            "top20_avg": "Top20 avg",
            "tab_single": "Single Contract Explorer",
            "tab_cross_summary": "Cross-Asset Pulse Summary",
            "workflow_mode_label": "Pattern workflow",
            "workflow_mode_discovery": "Pattern Scan",
            "workflow_mode_event_study": "Event Study",
            "nav_label": "Pattern Scan navigation",
            "section_contract_universe": "Contract Health",
            "section_clock_edge": "Clock Edge",
            "section_pulse_discovery": "Pulse Workspace",
            "section_cross_summary": "Cross-Asset Map",
            "clock_title": "Clock Edge",
            "clock_caption": "Selected contract `{symbol}` is resampled to 1-minute moves. Use this "
            "as a pattern clue before testing execution-aware strategy PnL.",
            "clock_min_bars": "Minimum 1m bars per bucket",
            "clock_loading": "Building clock-edge profile...",
            "clock_empty": "Not enough timestamped price rows to estimate clock edge for the "
            "selected contract.",
            "clock_best_hour": "Best hour",
            "clock_weak_hour": "Weak hour",
            "clock_active_hour": "Most active hour",
            "clock_bars": "1m bars",
            "clock_heatmap": "Average 1m Move by Session and Hour",
            "clock_table": "Clock Edge Summary",
            "clock_session": "Session",
            "clock_hour": "Hour",
            "clock_rows": "1m bars",
            "clock_avg_move": "Avg 1m move",
            "clock_avg_move_short": "Avg move",
            "clock_median_move": "Median 1m move",
            "clock_positive_share": "Positive share",
            "clock_avg_abs_move": "Avg abs 1m move",
            "clock_abs_p95": "p95 abs move",
            "clock_net_move": "Net move",
            "clock_session_morning": "Morning",
            "clock_session_afternoon": "Afternoon",
            "clock_session_night": "Night",
            "clock_session_other": "Other",
            "scope_story": "Current scope: `{symbol}` from `{file}` | window `{window}` seconds | "
            "pulse cutoff `p{percentile}`. The scan is directionless: it finds "
            "abnormal movement first, then labels Up/Down afterward.",
            "metric_help_title": "How to interpret these metrics",
            "metric_help": "\n"
            "- **Rows**: sample size after building rolling pulse windows. Too few "
            "rows means unstable evidence.\n"
            "- **Collapsed pulses**: independent pulse events after merging "
            "adjacent duplicate flags.\n"
            "- **Pulse threshold**: minimum move size needed to enter the selected "
            "percentile tail.\n"
            "- **p99 move**: normal high-tail reference for this contract.\n"
            "\n"
            "Promising discovery usually has enough rows, at least several dozen "
            "independent pulses, and a threshold that is economically meaningful "
            "relative to spread and tick size.\n",
            "validity_no_events": "No candidate pulses under the current threshold. Lower the "
            "percentile or choose a more active contract before researching "
            "behavior.",
            "validity_sparse": "Weak sample: the rolling frame has too few rows for a robust pulse "
            "study. Treat any event as anecdotal.",
            "validity_few_events": "Early-stage sample: only a small number of independent pulses "
            "were found. Inspect visually, but do not promote without "
            "broader testing.",
            "validity_tiny_threshold": "Weak threshold: the selected tail is less than one tick. "
            "This is usually too small to be economically meaningful.",
            "validity_promising": "Promising discovery sample: enough independent pulses and a "
            "meaningful threshold. Next step is visual inspection and Intraday "
            "Event Study testing.",
            "validity_ok": "Usable discovery sample: there is enough structure to inspect events, "
            "but treat it as research evidence rather than a trading signal.",
            "no_tick_files": "No tick_all_data parquet files found in "
            "runtime/data/futures_cn/tick.",
            "contract_universe_title": "Contract Universe",
            "contract_universe_help": "Use this before pulse research to verify what contracts "
            "exist in the selected tick file, which contract is most "
            "liquid, and what raw fields are available.",
            "loading_summary": "Loading contract summary...",
            "loading_frame": "Computing rolling pulse frame...",
            "loading_events": "Collapsing pulse events...",
            "loading_done": "Ready",
            "distribution": "Pulse Severity Ladder",
            "distribution_help": "Read left to right: calm windows, active windows, then the "
            "violent right tail. The red Pulse zone is the selected top "
            "percentile.",
            "zone": "Zone",
            "zone_range": "Percentile range",
            "zone_windows": "Windows",
            "zone_share": "Share",
            "zone_avg": "Avg abs distance",
            "zone_max": "Max abs distance",
            "zone_note": "Distance is measured in price ticks. It does not mean consecutive "
            "same-direction ticks; price may wiggle inside the window.",
            "zone_calm": "Normal",
            "zone_active": "Active",
            "zone_watch": "Watch",
            "zone_pulse": "Pulse",
            "zone_extreme": "Extreme",
            "selected_pulse": "Selected pulse",
            "cross_zone_title": "Cross-Asset Severity Zone Summary",
            "cross_zone_help": "This repeats the same severity table for every downloaded asset's "
            "main contract. Use it to compare which assets spend more time in "
            "Pulse/Extreme zones and how large those windows are.",
            "cross_zone_run": "Run / Refresh Cross-Asset Severity Summary",
            "cross_zone_loading": "Building cross-asset severity summary...",
            "cross_zone_empty": "No cross-asset severity summary has been computed yet.",
            "asset": "Asset",
            "asset_name": "Product name",
            "main_contract": "Main contract",
            "event_table": "Largest Collapsed Pulse Events",
            "event_select": "Inspect pulse event",
            "inspector": "Selected Event Inspector",
            "fingerprint": "Pulse Fingerprint vs Background",
            "cross_asset": "Cross-Asset Pulse Map",
            "cross_asset_help": "Ranks downloaded product tick files by main-contract pulse "
            "frequency and severity. This is descriptive: it tells us where "
            "pulses are common, not whether they are profitable.",
            "cross_asset_run": "Run / Refresh Cross-Asset Pulse Map",
            "cross_asset_empty": "No product-specific tick files found.",
            "cross_asset_loading": "Scanning downloaded tick files...",
            "source": "Source file",
            "direction": "Direction",
            "time": "Time",
            "abs_move": "Abs move (ticks)",
            "net_move": "Signed move (ticks)",
            "velocity": "Velocity (ticks/sec)",
            "volume": "Volume delta",
            "spread": "Mean spread",
            "book": "Mean book imbalance",
            "flow": "Flow imbalance",
            "range": "Path range (ticks)",
            "cluster": "Cluster rows",
            "no_events": "No pulses passed the current percentile threshold.",
            "metric_move": "Move size",
            "metric_velocity": "Speed (ticks/sec)",
            "metric_volume": "Volume burst",
            "metric_spread": "Spread widening",
            "metric_book": "Book imbalance",
            "metric_flow": "Flow imbalance",
            "speed_context": "Speed context: this pulse window spans {elapsed:.6f}s across "
            "{snapshots} snapshots. Raw speed = absolute move / elapsed seconds.",
            "speed_caution": "Speed caution: this pulse spans only {elapsed:.6f}s across "
            "{snapshots} snapshots, so ticks/sec can look enormous. Treat speed "
            "as a timestamp-intensity flag and confirm it visually in the chart.",
            "quality_good": "Clean research pulse",
            "quality_review": "Mixed pulse: inspect manually",
            "quality_danger": "Dangerous/noisy pulse",
            "quality_spread": "Spread",
            "quality_volume": "Volume z-score",
            "quality_elapsed": "Elapsed",
            "quality_session": "Session age",
            "quality_reason_prefix": "Why",
            "quality_good_help": "Large move with real volume and controlled spread. Useful for "
            "research classification.",
            "quality_review_help": "Some conditions are useful, but one or more diagnostics need "
            "manual inspection.",
            "quality_danger_help": "Pulse may be dominated by opening mechanics, timestamp "
            "compression, or expensive/wide spread.",
            "quality_reason_wide_spread": "spread is wide: {spread_ticks:.2f} ticks",
            "quality_reason_extreme_spread": "spread is abnormally wide vs background: "
            "{spread_z:+.2f}σ",
            "quality_reason_tiny_elapsed": "timestamp span is tiny: {elapsed:.6f}s",
            "quality_reason_open": "near session open/reset: {session_age:.1f}s after session "
            "start",
            "quality_reason_low_volume": "volume confirmation is weak: {volume_z:+.2f}σ",
            "quality_reason_good_move": "move is clearly abnormal: {move_z:+.2f}σ",
            "quality_reason_good_volume": "volume confirms participation: {volume_z:+.2f}σ",
            "quality_reason_good_spread": "spread is controlled: {spread_ticks:.2f} ticks",
            "event_reading_help_title": "How to read this event",
            "behavior_preview_title": "After-Pulse Behavior Preview",
            "behavior_preview_caption": "This intentionally looks after the pulse, so it is "
            "hindsight-only. The first table uses all collapsed pulses "
            "under the current window and percentile; the "
            "selected-event rows are only a visual sanity check.",
            "behavior_asset_recommendation": "Across all collapsed pulses under the current "
            "settings, behavior leans toward **{label}**. Average "
            "directional follow-through score: `{score}` ticks.",
            "behavior_asset_recommendation_unclear": "Across all collapsed pulses under the "
            "current settings, behavior is mixed/unclear. "
            "The population does not strongly support "
            "continuation or fade yet.",
            "behavior_recommendation": "This example leans toward **{label}**. Average directional "
            "follow-through score: `{score}` ticks.",
            "behavior_recommendation_unclear": "This example is mixed/unclear after the pulse. "
            "Save either behavior only if the visual story "
            "still makes sense.",
            "behavior_recommendation_labels": {
                "continuation": "continuation",
                "fade": "fade / reversal",
                "unclear": "mixed / unclear",
            },
            "behavior_horizon": "After pulse",
            "behavior_events": "Events",
            "behavior_continuation_rate": "Continuation rate",
            "behavior_fade_rate": "Fade rate",
            "behavior_future_time": "Future time",
            "behavior_raw_move": "Raw future move (ticks)",
            "behavior_directional_move": "Move in pulse direction (ticks)",
            "behavior_label": "Observed behavior",
            "behavior_selected_caption": "Selected-event case study. This is useful for human "
            "inspection, but the save default uses the broader "
            "asset-level summary when it is clear.",
            "behavior_outcome_labels": {
                "continuation": "Continuation",
                "fade": "Fade / reversal",
                "flat": "Flat",
                "missing": "Missing",
            },
            "seed_promote_title": "Promote this pulse into a hypothesis seed",
            "seed_promote_help": "\n"
            "Use this when a pulse looks interesting enough to test later. "
            "The saved seed is a reproducible research handoff: source file, "
            "contract, timestamp, pulse quality, and a first-pass "
            "deterministic rule.\n"
            "\n"
            "This does not create a production factor yet. It simply makes "
            'the Intraday Event Study able to ask: "If we saw this type of pulse '
            'again, did the future price move as expected?"\n'
            "\n"
            "Choose `Continuation` if the after-pulse preview shows price "
            "kept moving in the pulse direction. Choose `Fade / reversal` if "
            "price snapped back against the pulse direction.\n",
            "seed_behavior": "Hypothesis behavior",
            "seed_behavior_continuation": "Continuation",
            "seed_behavior_fade": "Fade / reversal",
            "seed_name": "Seed name",
            "seed_save": "Save hypothesis seed",
            "seed_saved": "Saved hypothesis seed `{seed_id}`. Open Intraday Event Study and choose Saved "
            "Discovery Seed as the hypothesis source.",
            "manager_demo_read_only": "Manager demo mode is read-only: saving hypothesis seeds is "
            "disabled.",
            "how": "Definition",
            "how_text": "\n"
            "For each snapshot `t`, the page looks back by real clock time, not by a "
            "fixed number of rows:\n"
            "\n"
            "`pulse_net_move_ticks = (last_price_t - last_price_at_start_of_window) / "
            "tick_size`\n"
            "\n"
            "`pulse_size = abs(pulse_net_move_ticks)`\n"
            "\n"
            "A pulse is a row whose `pulse_size` is in the selected contract's top "
            "percentile. Neighboring flagged rows are collapsed into one event, "
            "keeping the strongest row, so one violent move does not appear as "
            "hundreds of duplicate pulses.\n",
            "tick_size_help_title": "What does tick size mean?",
            "tick_size_help": "\n"
            "`Tick size` is the smallest legal price step for the contract. In "
            "our InstrumentMaster, Chinese gold futures `au` has `tick_size = "
            "0.02`, so valid prices move like `900.00 -> 900.02 -> 900.04`.\n"
            "\n"
            "So a `4 tick` pulse in gold means the price moved by `4 * 0.02 = "
            "0.08` price units over the selected time window. It does **not** "
            "mean four tick-data rows, four trades, or four consecutive "
            "upward/downward jumps.\n"
            "\n"
            "Healthy reading: use ticks to compare movement in the contract's "
            "own natural unit. A 4-tick pulse in a 1-tick-spread liquid contract "
            "is meaningful; a 4-tick move in a very wide-spread/thin contract "
            "needs caution.\n",
            "severity_help_title": "How to read the severity ladder and zone table",
            "severity_help": "\n"
            "- `Zone`: a descriptive bucket of rolling-window move sizes. Normal "
            "is the quiet body of the distribution; Pulse/Extreme is the violent "
            "right tail.\n"
            "- `Percentile range`: where that bucket sits in the asset's own "
            "history. `p99-p99.5` means larger than 99% of windows but below the "
            "top 0.5%.\n"
            "- `Windows`: number of rolling windows in that bucket.\n"
            "- `Share`: percentage of non-zero rolling windows in that bucket. "
            "This should roughly match the percentile width; big deviations "
            "usually mean many tied values.\n"
            "- `Avg abs distance`: average absolute price distance in ticks. Low "
            "values are normal; Pulse/Extreme values are the ones worth "
            "inspecting.\n"
            "- `Max abs distance`: largest absolute price distance in that "
            "bucket.\n"
            "\n"
            "Healthy reading: the Pulse and Extreme rows should be rare but "
            "visibly larger than Normal/Active. If every zone is almost the same "
            "size, the asset may not have clean pulse behavior. If Extreme is "
            "huge but spread is also huge, inspect for illiquidity or bad "
            "ticks.\n",
            "event_columns_help_title": "How to read the pulse event table",
            "event_columns_help": "\n"
            "- `event_rank`: strongest collapsed pulse is ranked `1`.\n"
            "- `Time`: timestamp of the strongest row inside the pulse "
            "cluster.\n"
            "- `Direction`: `Up` means end price > window-start price; "
            "`Down` means end price < window-start price.\n"
            "- `Abs move`: absolute price distance in ticks. Higher means a "
            "more violent pulse.\n"
            "- `Signed move`: same move with direction. Positive = up pulse, "
            "negative = down pulse.\n"
            "- `Velocity`: signed ticks per second. Large positive = fast "
            "upward pulse; large negative = fast downward pulse.\n"
            "- `Volume delta`: total new traded volume inside the window. "
            "High volume confirms the move had real trading behind it; "
            "near-zero volume can mean a thin/quote-driven move.\n"
            "- `Mean spread`: average ask-bid spread in raw price units. "
            "Convert to ticks by dividing by `tick_size`. A spread close to "
            "one tick size is usually cleaner; above 2-3 ticks is a caution "
            "flag for slippage or thin liquidity.\n"
            "- `Mean book imbalance`: ranges from `-1` to `+1`. Positive = "
            "bid queue heavier; negative = ask queue heavier; near zero = "
            "balanced.\n"
            "- `Flow imbalance`: ranges from `-1` to `+1`. Positive = "
            "buyer-initiated flow dominates; negative = seller-initiated "
            "flow dominates.\n"
            "- `Path range`: high-low range inside the window, in ticks. If "
            "path range is much larger than abs move, price whipped around "
            "rather than moved cleanly.\n"
            "- `Cluster rows`: number of adjacent flagged snapshots merged "
            "into one event. Very large clusters mean the pulse lasted "
            "longer; very small clusters are sharper.\n"
            "\n"
            "Conditional reading: price direction plus flow in the same "
            "direction looks like momentum pressure. Price direction "
            "opposite to book/flow pressure can look like absorption. Large "
            "move plus wide spread is tradable only with caution.\n",
            "inspector_help_title": "How to inspect one pulse visually",
            "inspector_help": "\n"
            "The yellow band is the exact pulse window. The orange marker is the "
            "window start; the red X is the strongest pulse row.\n"
            "\n"
            "Read it in three passes:\n"
            "1. Price: did price really travel quickly, or is it just one bad "
            "print?\n"
            "2. Volume: did volume rise during the pulse? Real pulses usually "
            "have visible volume.\n"
            "3. Spread/book: did liquidity stay healthy, or did the spread blow "
            "out? Tight spread + high volume is cleaner; wide spread + low "
            "volume is suspicious.\n",
            "fingerprint_help_title": "How to read the pulse fingerprint",
            "fingerprint_help": "\n"
            "Each card compares the selected pulse with the normal background. "
            "The delta is a z-score: `+2σ` means about two standard deviations "
            "above normal.\n"
            "\n"
            "Widget meanings and healthy ranges:\n"
            "\n"
            "- `Move size`: absolute price distance from window start to pulse "
            "point, in ticks. A useful pulse should be in the selected tail, "
            "usually `p99+`, and often above `+2σ` versus background.\n"
            "- `Speed`: `abs move / actual elapsed seconds`. This is a raw "
            "timestamp speed. If elapsed time is tiny, for example "
            "microseconds, the number can explode into millions. In that case, "
            "focus more on the z-score and the chart than the raw value.\n"
            "- `Volume burst`: total new traded volume inside the pulse "
            "window. `+1σ` helps confirm participation; `+2σ` or higher is "
            "stronger.\n"
            "- `Spread widening`: average ask-bid spread in raw price units. "
            "Convert to ticks by dividing by `tick_size`. Around one tick is "
            "cleaner; several ticks wide or `+2σ` means the pulse may be hard "
            "to trade.\n"
            "- `Book imbalance`: visible order-book pressure, range `-1` to "
            "`+1`. Near `0` is balanced. Above `+0.3` means bid-heavy; below "
            "`-0.3` means ask-heavy; beyond `±0.7` is extreme.\n"
            "- `Flow imbalance`: estimated aggressive trade pressure, range "
            "`-1` to `+1`. `+1` means mostly buyer-initiated flow; `-1` means "
            "mostly seller-initiated flow. It is most meaningful when volume "
            "is also high.\n"
            "\n"
            "The quality badge uses practical first-pass rules:\n"
            "\n"
            "- Green: abnormal move, volume confirmation, and spread not wider "
            "than about `2` ticks.\n"
            "- Amber: mixed evidence. Useful to inspect, but not clean enough "
            "to trust quickly.\n"
            "- Red: spread wider than about `3` ticks, extremely compressed "
            "timestamp span, or session-open/reset behavior.\n"
            "\n"
            "How we know spread is wide: `spread in ticks = average spread / "
            "tick_size`. If this is `1-2`, liquidity is usually acceptable. "
            "Around `3+` is expensive; `5+` is very wide for a short-horizon "
            "pulse.\n",
            "cross_zone_columns_help_title": "How to read the cross-asset severity table",
            "cross_zone_columns_help": "\n"
            "This is the same severity-zone table, repeated across each "
            "downloaded asset's main contract.\n"
            "\n"
            "Use it to answer: which assets have a fatter pulse tail? "
            "Look for Pulse/Extreme rows with larger `Avg abs "
            "distance`, larger `Max abs distance`, and enough `Windows` "
            "to inspect. A single extreme row can be interesting, but "
            "repeated Pulse/Extreme windows are more robust.\n",
            "cross_asset_columns_help_title": "How to read the cross-asset pulse map table",
            "cross_asset_columns_help": "\n"
            "- `Collapsed pulses`: independent pulse events after "
            "merging duplicates.\n"
            "- `Pulses/hour`: pulse frequency normalized by trading "
            "hours. Higher means more opportunities to inspect.\n"
            "- `p95`, `p99`, `p99.5`: absolute move thresholds in "
            "ticks. Higher means the asset has larger short-window "
            "price jumps.\n"
            "- `Pulse threshold`: the active percentile cutoff used to "
            "flag events.\n"
            "- `Top20 avg`: average size of the 20 largest pulses. "
            "This is a quick severity score.\n"
            "- `Rows`: median snapshots inside each rolling time "
            "window. Higher means denser tick data; very low values "
            "mean sparse observations.\n"
            "\n"
            "Healthy reading: good candidates combine high "
            "`Pulses/hour`, high `p99/p99.5`, enough rows per window, "
            "and reasonable spread when inspected manually.\n",
        },
        "zh": {
            "_lang": "ZH",
            "title": "模式扫描",
            "subtitle": "在正式验证前，发现日频、日内和 tick 级模式。",
            "file": "Tick parquet 文件",
            "symbol": "合约",
            "window": "滚动时间窗口",
            "percentile": "脉冲分位数",
            "top": "展示事件数",
            "seconds": "秒",
            "rows": "行数",
            "events": "合并后脉冲数",
            "threshold": "脉冲阈值",
            "p99": "p99 跳动幅度",
            "cache": "缓存",
            "cache_hit": "命中",
            "cache_saved": "已保存",
            "cache_computed": "已计算",
            "cache_frame": "特征表",
            "cache_events": "事件表",
            "rows_label": "行数",
            "elapsed": "耗时",
            "axis_percentile": "滚动窗口分位数",
            "axis_pulses_per_hour": "每交易小时脉冲数",
            "hover_percentile": "分位数",
            "hover_move": "跳动",
            "hover_selected_pulse": "当前选择脉冲",
            "hover_price": "价格",
            "last_price": "最新价",
            "window_start": "窗口起点",
            "pulse_peak": "脉冲点",
            "p99_ticks": "p99 tick",
            "main_short": "主力",
            "pulses_hour_short": "脉冲/小时",
            "top20_avg": "Top20 平均",
            "tab_single": "单合约观察",
            "tab_cross_summary": "跨资产脉冲汇总",
            "workflow_mode_label": "模式研究流程",
            "workflow_mode_discovery": "模式扫描",
            "workflow_mode_event_study": "事件研究",
            "nav_label": "模式扫描导航",
            "section_contract_universe": "合约健康",
            "section_clock_edge": "时段边际",
            "section_pulse_discovery": "脉冲工作台",
            "section_cross_summary": "跨资产地图",
            "clock_title": "时段边际",
            "clock_caption": "当前合约 `{symbol}` 会被重采样为 1 分钟价格变动。这里先找模式线索，正式结论仍要用考虑执行成本的策略 PnL 验证。",
            "clock_min_bars": "每组最少 1分钟 bar",
            "clock_loading": "正在生成时段画像...",
            "clock_empty": "当前合约没有足够带时间戳的价格行，无法估计时段边际。",
            "clock_best_hour": "最强小时",
            "clock_weak_hour": "最弱小时",
            "clock_active_hour": "最活跃小时",
            "clock_bars": "1分钟 bar",
            "clock_heatmap": "按交易时段和小时统计的平均1分钟变动",
            "clock_table": "时段边际汇总",
            "clock_session": "交易时段",
            "clock_hour": "小时",
            "clock_rows": "1分钟 bar",
            "clock_avg_move": "平均1分钟变动",
            "clock_avg_move_short": "平均变动",
            "clock_median_move": "中位1分钟变动",
            "clock_positive_share": "上涨占比",
            "clock_avg_abs_move": "平均绝对1分钟变动",
            "clock_abs_p95": "p95绝对变动",
            "clock_net_move": "净变动",
            "clock_session_morning": "早盘",
            "clock_session_afternoon": "午后",
            "clock_session_night": "夜盘",
            "clock_session_other": "其他",
            "scope_story": "当前范围：`{symbol}` 来自 `{file}` | 窗口 `{window}` 秒 | 脉冲阈值 "
            "`p{percentile}`。扫描本身是无方向的：先找异常运动，再事后标记 Up/Down。",
            "metric_help_title": "如何解读这些指标",
            "metric_help": "\n"
            "- **行数**：构造滚动脉冲窗口后的样本量。行数太少，证据不稳定。\n"
            "- **合并后脉冲数**：相邻重复触发合并后的独立脉冲事件。\n"
            "- **脉冲阈值**：进入当前分位尾部所需的最小跳动幅度。\n"
            "- **p99 跳动幅度**：当前合约高尾部的参考值。\n"
            "\n"
            "有潜力的发现通常需要足够行数、几十个以上独立脉冲，并且阈值相对于价差和 tick size 有经济意义。\n",
            "validity_no_events": "当前阈值下没有候选脉冲。降低分位数或选择更活跃合约后再研究。",
            "validity_sparse": "样本偏弱：滚动窗口行数太少，脉冲研究不够稳健。任何事件都只能当作个案。",
            "validity_few_events": "早期样本：独立脉冲数量偏少。可以人工观察，但不要在未做更广泛检验前推广。",
            "validity_tiny_threshold": "阈值偏弱：尾部分位对应的跳动小于一个 tick，通常缺少经济意义。",
            "validity_promising": "有潜力的发现样本：独立脉冲数量足够，阈值也有意义。下一步是视觉检查和日内事件研究检验。",
            "validity_ok": "可用的发现样本：已经有足够结构可以检查事件，但仍应视为研究证据，而不是交易信号。",
            "no_tick_files": "在 runtime/data/futures_cn/tick 中没有找到 tick_all_data "
            "parquet 文件。",
            "contract_universe_title": "合约池",
            "contract_universe_help": "在研究脉冲之前，先用这里确认当前 tick 文件里有哪些合约、哪个合约最活跃，以及原始字段长什么样。",
            "loading_summary": "正在读取合约摘要...",
            "loading_frame": "正在计算滚动脉冲...",
            "loading_events": "正在合并脉冲事件...",
            "loading_done": "就绪",
            "distribution": "脉冲强度阶梯图",
            "distribution_help": "从左到右读：平静窗口、活跃窗口、再到最右侧的剧烈尾部。红色 Pulse 区域就是当前选择的顶部百分位。",
            "zone": "区域",
            "zone_range": "分位区间",
            "zone_windows": "窗口数",
            "zone_share": "占比",
            "zone_avg": "平均绝对距离",
            "zone_max": "最大绝对距离",
            "zone_note": "距离的单位是价格 tick。它不是连续同方向跳了几次；窗口内部价格可以来回波动。",
            "zone_calm": "普通",
            "zone_active": "活跃",
            "zone_watch": "观察",
            "zone_pulse": "脉冲",
            "zone_extreme": "极端",
            "selected_pulse": "当前选择脉冲",
            "cross_zone_title": "跨资产脉冲强度区域汇总",
            "cross_zone_help": "把同一张强度区域表应用到每个已下载品种的主力合约。用它比较哪些品种更常进入 Pulse/Extreme "
            "区域，以及这些窗口的净距离有多大。",
            "cross_zone_run": "运行 / 刷新跨资产强度汇总",
            "cross_zone_loading": "正在生成跨资产强度汇总...",
            "cross_zone_empty": "尚未计算跨资产强度汇总。",
            "asset": "品种",
            "asset_name": "品种名称",
            "main_contract": "主力合约",
            "event_table": "最大合并脉冲事件",
            "event_select": "查看脉冲事件",
            "inspector": "单个脉冲事件检查",
            "fingerprint": "脉冲特征画像 vs 背景",
            "cross_asset": "跨资产脉冲地图",
            "cross_asset_help": "按主力合约比较已下载品种的脉冲频率和强度。这是描述性统计：告诉我们哪里更容易出现脉冲，不代表一定赚钱。",
            "cross_asset_run": "运行 / 刷新跨资产脉冲地图",
            "cross_asset_empty": "没有找到品种级 tick 文件。",
            "cross_asset_loading": "正在扫描已下载 tick 文件...",
            "source": "数据文件",
            "direction": "方向",
            "time": "时间",
            "abs_move": "绝对跳动 (tick)",
            "net_move": "有方向跳动 (tick)",
            "velocity": "速度 (tick/秒)",
            "volume": "成交量增量",
            "spread": "平均价差",
            "book": "平均盘口不平衡",
            "flow": "主动流不平衡",
            "range": "路径振幅 (tick)",
            "cluster": "簇内行数",
            "no_events": "当前分位阈值下没有脉冲事件。",
            "metric_move": "跳动幅度",
            "metric_velocity": "速度 (tick/秒)",
            "metric_volume": "成交量爆发",
            "metric_spread": "价差扩大",
            "metric_book": "盘口不平衡",
            "metric_flow": "主动流不平衡",
            "speed_context": "速度背景：这个脉冲窗口实际跨度 {elapsed:.6f} 秒，共 {snapshots} 个快照。原始速度 = 绝对跳动 / "
            "实际秒数。",
            "speed_caution": "速度提醒：这个脉冲实际只跨了 {elapsed:.6f} 秒，共 {snapshots} 个快照，所以 tick/秒 "
            "会显得特别大。请把速度当成时间戳强度提示，并结合下方图形人工确认。",
            "quality_good": "干净研究脉冲",
            "quality_review": "混合脉冲：需要人工检查",
            "quality_danger": "危险/噪声脉冲",
            "quality_spread": "价差",
            "quality_volume": "成交量 z-score",
            "quality_elapsed": "经过时间",
            "quality_session": "距开盘/重置",
            "quality_reason_prefix": "原因",
            "quality_good_help": "跳动异常、成交量确认、价差可控。适合作为研究分类样本。",
            "quality_review_help": "有些条件有研究价值，但存在需要人工检查的诊断项。",
            "quality_danger_help": "脉冲可能主要来自开盘机制、时间戳压缩，或价差过宽导致交易成本太高。",
            "quality_reason_wide_spread": "价差偏宽：{spread_ticks:.2f} tick",
            "quality_reason_extreme_spread": "价差相对背景异常偏宽：{spread_z:+.2f}σ",
            "quality_reason_tiny_elapsed": "时间戳跨度极短：{elapsed:.6f} 秒",
            "quality_reason_open": "接近开盘/交易段重置：距交易段开始 {session_age:.1f} 秒",
            "quality_reason_low_volume": "成交量确认偏弱：{volume_z:+.2f}σ",
            "quality_reason_good_move": "跳动明显异常：{move_z:+.2f}σ",
            "quality_reason_good_volume": "成交量确认参与度：{volume_z:+.2f}σ",
            "quality_reason_good_spread": "价差可控：{spread_ticks:.2f} tick",
            "event_reading_help_title": "如何阅读这个事件",
            "behavior_preview_title": "脉冲之后走势预览",
            "behavior_preview_caption": "这里是故意往脉冲之后看，所以它是 "
            "hindsight-only（事后观察）。第一张表基于当前窗口和分位数下的全部合并脉冲；单个事件表只是人工视觉检查。",
            "behavior_asset_recommendation": "在当前设置下，全部合并脉冲整体更偏向 "
            "**{label}**。平均顺着脉冲方向的延续分数：`{score}` tick。",
            "behavior_asset_recommendation_unclear": "在当前设置下，全部合并脉冲的后续走势比较混合/不清晰。暂时不能强烈支持延续或反转。",
            "behavior_recommendation": "这个样本更偏向 **{label}**。平均顺着脉冲方向的延续分数：`{score}` tick。",
            "behavior_recommendation_unclear": "这个样本在脉冲之后表现混合/不清晰。如果视觉逻辑仍然成立，再选择保存某一种行为。",
            "behavior_recommendation_labels": {
                "continuation": "延续",
                "fade": "反转 / fade",
                "unclear": "混合 / 不清晰",
            },
            "behavior_horizon": "脉冲后",
            "behavior_events": "事件数",
            "behavior_continuation_rate": "延续比例",
            "behavior_fade_rate": "反转比例",
            "behavior_future_time": "未来时间",
            "behavior_raw_move": "原始未来变动 (tick)",
            "behavior_directional_move": "顺着脉冲方向的变动 (tick)",
            "behavior_label": "观察到的行为",
            "behavior_selected_caption": "单个事件案例检查。它适合人工看图确认，但保存种子的默认选择会优先使用更广泛的资产级汇总结果。",
            "behavior_outcome_labels": {
                "continuation": "延续",
                "fade": "反转 / fade",
                "flat": "持平",
                "missing": "缺少未来数据",
            },
            "seed_promote_title": "把这个脉冲提升为假设种子",
            "seed_promote_help": "\n"
            "当你觉得某个脉冲值得后续验证，就用这个功能保存。这个“假设种子”是一个可复现的研究交接：数据文件、合约、时间戳、脉冲质量，以及第一版确定性规则。\n"
            "\n"
            "它还不是生产因子，也不是已经验证的策略。它只是让日内事件研究 "
            "可以继续问：如果未来再次出现这种脉冲，之后价格有没有按我们预期走？\n"
            "\n"
            "如果“脉冲之后走势预览”显示价格继续顺着脉冲方向走，选择 `延续`；如果价格很快反向回撤，选择 `反转 / fade`。\n",
            "seed_behavior": "假设行为",
            "seed_behavior_continuation": "延续",
            "seed_behavior_fade": "反转 / fade",
            "seed_name": "种子名称",
            "seed_save": "保存假设种子",
            "seed_saved": "已保存假设种子 `{seed_id}`。打开日内事件研究，并选择“已保存发现种子”作为假设来源。",
            "manager_demo_read_only": "管理层演示模式为只读：已禁用保存假设种子。",
            "how": "定义",
            "how_text": "\n"
            "对每一个快照 `t`，页面按真实时间往前看，而不是固定行数：\n"
            "\n"
            "`pulse_net_move_ticks = (当前 last_price - 窗口起点 last_price) / tick_size`\n"
            "\n"
            "`pulse_size = abs(pulse_net_move_ticks)`\n"
            "\n"
            "如果 `pulse_size` "
            "进入该合约自身的顶部百分位，就标记为脉冲。相邻的标记行会合并成一个事件，只保留最强的一行，避免一次剧烈波动被重复记录成上百个脉冲。\n",
            "tick_size_help_title": "tick size 到底是什么意思？",
            "tick_size_help": "\n"
            "`tick size` 是这个合约允许的最小价格跳动单位。在我们的 InstrumentMaster 里，中国黄金期货 `au` 的 "
            "`tick_size = 0.02`，所以合法价格大概是 `900.00 -> 900.02 -> 900.04`。\n"
            "\n"
            "所以黄金里 `4 tick` 的脉冲，意思是价格在这个时间窗口里移动了 `4 * 0.02 = 0.08` 个价格单位。它不是 4 行 "
            "tick 数据，不是 4 笔成交，也不是连续上涨/下跌了 4 次。\n"
            "\n"
            "健康解读：tick 是把价格变动换算成该合约自己的自然单位。一个流动性好、价差约 1 tick 的合约里出现 4 tick "
            "变动比较有意义；如果合约很薄、价差很宽，就要小心是假脉冲。\n",
            "severity_help_title": "如何阅读强度阶梯图和区域表",
            "severity_help": "\n"
            "- `区域`：滚动窗口价格跳动强度的分组。普通区是大部分平静窗口，脉冲/极端区是右尾剧烈窗口。\n"
            "- `分位区间`：这个区域在该品种自身历史分布中的位置。`p99-p99.5` 表示比 99% 的窗口都大，但还不属于最极端的 "
            "0.5%。\n"
            "- `窗口数`：落在该区域的滚动窗口数量。\n"
            "- `占比`：该区域占所有非零窗口的比例。通常应接近分位宽度；如果偏差很大，可能是很多窗口数值相同。\n"
            "- `平均绝对距离`：该区域平均价格移动距离，单位 tick。越高代表越剧烈。\n"
            "- `最大绝对距离`：该区域最大的价格移动距离，单位 tick。\n"
            "\n"
            "健康解读：Pulse/Extreme 应该很少，但明显比 Normal/Active "
            "大。如果所有区域都差不多，说明这个品种脉冲特征不明显。如果 Extreme 很大但价差也很大，要检查是否是流动性差或脏数据。\n",
            "event_columns_help_title": "如何阅读脉冲事件表",
            "event_columns_help": "\n"
            "- `event_rank`：脉冲强度排名，`1` 是最强事件。\n"
            "- `时间`：该脉冲簇中最强一行的时间戳。\n"
            "- `方向`：`Up` 表示窗口终点价高于起点价；`Down` 表示低于起点价。\n"
            "- `绝对跳动`：绝对价格移动距离，单位 tick。越大代表越剧烈。\n"
            "- `有方向跳动`：带方向的移动。正数 = 上行脉冲，负数 = 下行脉冲。\n"
            "- `速度`：每秒移动多少 tick。大正数 = 快速上冲；大负数 = 快速下杀。\n"
            "- `成交量增量`：窗口内新增成交量。高成交量说明脉冲背后有真实交易参与；接近 0 可能只是很薄的报价变化。\n"
            "- `平均价差`：平均买卖价差，显示的是原始价格单位。除以 `tick_size` 后才是 tick 数。接近 1 个 "
            "tick size 通常较干净；超过 2-3 tick 要小心滑点和流动性。\n"
            "- `平均盘口不平衡`：范围 `-1` 到 `+1`。正数 = 买盘更厚；负数 = 卖盘更厚；接近 0 = 相对平衡。\n"
            "- `主动流不平衡`：范围 `-1` 到 `+1`。正数 = 主动买占优；负数 = 主动卖占优。\n"
            "- `路径振幅`：窗口内最高价 - 最低价，单位 tick。如果路径振幅远大于绝对跳动，说明价格来回甩动，不是干净单边。\n"
            "- `簇内行数`：多少个相邻触发快照被合并成一个事件。很大表示脉冲持续较久；很小说明更尖锐。\n"
            "\n"
            "条件解读：价格方向和主动流方向一致，更像动量冲击；价格方向和盘口/主动流相反，可能是吸收。大跳动但价差很宽，交易上要谨慎。\n",
            "inspector_help_title": "如何人工检查单个脉冲",
            "inspector_help": "\n"
            "黄色区域是脉冲窗口。橙色点是窗口起点，红色 X 是最强脉冲点。\n"
            "\n"
            "按三步看：\n"
            "1. 价格：它是真的快速移动，还是一个异常脏点？\n"
            "2. 成交量：脉冲期间成交量有没有放大？真实脉冲通常有明显成交。\n"
            "3. 价差/盘口：流动性是否健康？价差窄 + 成交量大更干净；价差宽 + 成交量低要谨慎。\n",
            "fingerprint_help_title": "如何阅读脉冲画像",
            "fingerprint_help": "\n"
            "每张卡片把当前脉冲和正常背景对比。旁边的变化值是 z-score：`+2σ` 表示比平时高大约 2 个标准差。\n"
            "\n"
            "每个 widget 的含义和健康区间：\n"
            "\n"
            "- `跳动幅度`：从窗口起点到脉冲点的绝对价格距离，单位 tick。值得看的脉冲通常应进入所选尾部分位，比如 "
            "`p99+`，并且经常会高于背景 `+2σ`。\n"
            "- `速度`：`绝对跳动 / "
            "实际经过秒数`。这是原始时间戳速度。如果实际经过时间非常短，比如微秒级，数字可能会膨胀到几百万。这种情况下不要只看原始数值，要结合 "
            "z-score 和图形。\n"
            "- `成交量爆发`：窗口内新增成交量。超过 `+1σ` 说明有参与度，超过 `+2σ` 更强。\n"
            "- `价差扩大`：平均买卖价差，显示的是原始价格单位。除以 `tick_size` 后才是 tick 数。接近 1 个 tick "
            "更干净；如果宽到好几个 tick 或超过 `+2σ`，交易难度更高。\n"
            "- `盘口不平衡`：可见订单簿压力，范围 `-1` 到 `+1`。接近 `0` 表示平衡；大于 `+0.3` 买盘更厚；小于 "
            "`-0.3` 卖盘更厚；超过 `±0.7` 属于极端。\n"
            "- `主动流不平衡`：估算主动买卖压力，范围 `-1` 到 `+1`。`+1` 基本是主动买占优；`-1` "
            "基本是主动卖占优。它在成交量也很高时才最有意义。\n"
            "\n"
            "质量标签使用一套实用的一阶规则：\n"
            "\n"
            "- 绿色：跳动异常、成交量确认、价差不超过约 `2` tick。\n"
            "- 黄色：证据混合。值得看，但不能快速认为是干净脉冲。\n"
            "- 红色：价差超过约 `3` tick、时间戳极度压缩，或接近开盘/交易段重置。\n"
            "\n"
            "如何判断价差宽不宽：`价差 tick 数 = 平均价差 / tick_size`。如果是 `1-2`，通常还可以；`3+` "
            "已经偏贵；`5+` 对短周期脉冲来说非常宽。\n",
            "cross_zone_columns_help_title": "如何阅读跨资产强度表",
            "cross_zone_columns_help": "\n"
            "这是把同一张强度区域表，重复应用到每个已下载品种的主力合约。\n"
            "\n"
            "用它回答：哪个品种的脉冲尾部更厚？重点看 Pulse/Extreme 行里的 `平均绝对距离`、`最大绝对距离` 和 "
            "`窗口数`。单个极端点有参考价值，但重复出现的 Pulse/Extreme 更稳。\n",
            "cross_asset_columns_help_title": "如何阅读跨资产脉冲地图表",
            "cross_asset_columns_help": "\n"
            "- `合并后脉冲数`：去重合并后的独立脉冲事件数。\n"
            "- `脉冲/小时`：按交易小时标准化后的脉冲频率。越高代表越常出现可观察事件。\n"
            "- `p95`, `p99`, `p99.5`：绝对跳动阈值，单位 tick。越高说明短窗口价格跳动越大。\n"
            "- `脉冲阈值`：当前分位数设置下的触发阈值。\n"
            "- `Top20 平均`：最大的 20 个脉冲的平均大小，是一个快速强度指标。\n"
            "- `行数`：每个滚动时间窗口里的中位快照数。越高表示 tick 数据更密；太低表示观测稀疏。\n"
            "\n"
            "健康解读：好的候选品种通常同时具备较高 `脉冲/小时`、较高 "
            "`p99/p99.5`、足够窗口行数，并且人工检查时价差不能太离谱。\n",
        },
    },
    "regime_characterisation_lab": {
        "en": {
            "title": "Regime Analysis",
            "subtitle": "Market-state analysis with GMM diagnostics and latent-state cross-checks.",
            "workflow_title": "How this page works",
            "workflow_text": "\n"
            "Use this page to understand the market state, not to approve "
            "a trade by itself.\n"
            "\n"
            "1. **Current State** shows the latest regime, confidence, "
            "Panic probability, and the price/probability timeline.\n"
            "2. **Regime Map** shows where Quiet, Chop, and Panic live in "
            "volatility/liquidity/trend space.\n"
            "3. **Diagnostics** checks whether the feature distributions "
            "and state profiles look stable enough to study.\n"
            "4. **Validation** checks whether independent latent codes "
            "agree with the GMM and whether Panic historically preceded "
            "future stress.\n"
            "\n"
            "Healthy workflow: confirm the selected asset lane, check "
            "that the current regime has clear confidence, verify that "
            "the stress features and diagnostics match the label, then "
            "use Validation before treating the result as a research "
            "candidate.\n",
            "scope_note": "This page now reads the project asset taxonomy. The current "
            "bundled regime matrices are local Chinese futures/index data; "
            "US equities and US options can be selected as taxonomy lanes, "
            "but they need vendor-backed regime matrices before the charts "
            "become active.",
            "scope_help_title": "Data scope",
            "scope_stats": "Sample: {assets} assets | feature dates {feature_start} to "
            "{feature_end} | GMM dates {regime_start} to {regime_end}",
            "tab_timeline": "Current State",
            "tab_geometry": "Regime Map",
            "tab_cross": "Validation",
            "tab_diag": "Diagnostics",
            "asset_class": "Asset lane",
            "taxonomy_title": "Asset taxonomy context",
            "taxonomy_help_title": "How to read the asset lane",
            "taxonomy_help": "\n"
            "The asset lane tells the page what market rules it should "
            "assume before interpreting regimes.\n"
            "\n"
            "- **FUTURES_CN** is the current local/static research lane, "
            "so the bundled parquet matrices can render immediately.\n"
            "- **EQUITY_US** is taxonomy-ready but should use FMP-backed "
            "equity data once the adapter is wired.\n"
            "- **OPTIONS_US** is taxonomy-ready but non-vectorizable, so "
            "future options regimes should use event-driven option-chain "
            "logic rather than reusing futures matrix assumptions.\n"
            "\n"
            "If a lane has no local regime rows, the page stops before "
            "showing charts. That is intentional: it prevents Chinese "
            "futures regimes from being accidentally interpreted as US "
            "equity or options regimes.\n",
            "taxonomy_status_ready": "Selected lane has local regime rows and can render "
            "now.",
            "taxonomy_status_empty": "Selected lane is known to the taxonomy, but no local "
            "feature/regime rows are available yet.",
            "taxonomy_status_nonvector": "Taxonomy marks this lane as non-vectorizable. "
            "Treat current GMM matrix logic as a research "
            "placeholder until an event-driven version is "
            "wired.",
            "taxonomy_description": "Description",
            "taxonomy_region": "Region",
            "taxonomy_settlement": "Settlement",
            "taxonomy_price_limit": "Price limit",
            "taxonomy_vectorizable": "Vectorizable",
            "taxonomy_provider": "Provider",
            "taxonomy_data_mode": "Data mode",
            "taxonomy_local_rows": "Local rows",
            "taxonomy_local_assets": "Local assets",
            "lane_summary": "Region {region} · Settlement {settlement} · Price limit "
            "{price_limit} · Vectorizable {vectorizable} · {rows} local rows · "
            "{assets} local assets",
            "yes": "Yes",
            "no": "No",
            "asset": "Select asset",
            "current_state": "Current State",
            "panic_prob": "Panic Prob.",
            "confidence": "Confidence",
            "hurst": "Hurst",
            "vq_code": "VQ Code",
            "latest_date": "Latest Date",
            "sample_rows": "Rows",
            "panic_flag": "Panic flag",
            "dynamic_threshold": "Dynamic Panic Threshold",
            "quality_title": "Regime signal quality",
            "quality_stale": "Diagnostic only: the latest row is more than 30 days old, so "
            "the current-state readout is stale.",
            "quality_sparse": "Weak evidence: this asset has too few overlapping "
            "feature/regime rows for a stable regime read.",
            "quality_unclear": "Weak evidence: the model is split across states, so treat "
            "the label as ambiguous.",
            "quality_risk": "Risk-off candidate: Panic is the dominant state with strong "
            "confidence. Validate against price action and downstream risk "
            "rules before acting.",
            "quality_promising": "Promising research context: the state is clear and the "
            "sample is large enough for further validation.",
            "quality_ok": "Usable context: the regime read is interpretable, but it "
            "remains research evidence rather than a standalone trading "
            "signal.",
            "current_context": "Start here. This tab answers: what state is the selected "
            "asset in, how confident is the GMM, and do the stress "
            "features agree with that story?",
            "geometry_context": "Use this to see whether the model's states occupy "
            "distinct regions of feature space. Cleaner separation "
            "means the labels are easier to trust.",
            "validation_context": "Use this before believing the regime label. Agreement "
            "with VQ codes and useful hindsight stress diagnostics "
            "make the signal more credible.",
            "diagnostic_context": "Use this when the page feels surprising. It checks "
            "whether the inputs and state profiles look "
            "mathematically sensible.",
            "current_help_title": "How to interpret the current-state metrics",
            "current_help": "\n"
            "- **Current State** is the highest smoothed GMM probability "
            "after mapping raw states into Quiet, Chop, and Panic.\n"
            "- **Panic Prob.** is the smoothed probability of the "
            "highest-stress state. High values are a risk warning, not an "
            "automatic trade.\n"
            "- **Confidence** is the largest state probability. Below "
            "roughly 50% means the model is unsure.\n"
            "- **Hurst** below 0.5 usually points to noisier or "
            "mean-reverting behavior; above 0.5 usually points to smoother "
            "trend behavior.\n"
            "- **VQ Code** is an independent latent bucket if VQ-VAE "
            "artifacts exist. Missing VQ does not break GMM, but removes "
            "one cross-check.\n",
            "timeline_help_title": "How to read the timeline",
            "timeline_help": "\n"
            "The top panel is price. The middle panel stacks the smoothed "
            "GMM probabilities. The bottom panel shows stress features.\n"
            "\n"
            "A convincing Panic regime usually has high Panic probability "
            "together with rising volatility and illiquidity. A "
            "convincing Quiet regime usually has low stress and stable "
            "probabilities. Fast state-flipping is a caution sign.\n",
            "phase_help_title": "How to read the regime map",
            "phase_help": "\n"
            "Each dot is one asset-date observation. The axes are "
            "illiquidity, volatility, and trend efficiency.\n"
            "\n"
            "Good regime geometry means Quiet, Chop, and Panic are not "
            "randomly mixed. If the colors heavily overlap, the model may "
            "still be descriptive, but the labels are weaker.\n",
            "density_help_title": "How to read density surfaces",
            "density_help": "\n"
            "These surfaces approximate where each state normally lives in "
            "volatility/liquidity space.\n"
            "\n"
            "Narrow, separated surfaces are easier to interpret. Flat or "
            "overlapping surfaces mean the GMM states are less clean and "
            "need more validation.\n",
            "profiler_help_title": "How to read the state profiler",
            "profiler_help": "\n"
            "The profiler translates labels into averages: how often each "
            "state appears, average return, volatility, illiquidity, and "
            "trend efficiency.\n"
            "\n"
            "A useful state map should have intuitive profiles. Panic "
            "should generally show higher stress than Quiet. If the "
            "profiles look inverted, inspect the state mapping and source "
            "data.\n",
            "no_vq": "No saved VQ-VAE artifacts found. Train and save the VQ codebook from "
            "Feature Governance first.",
            "vq_title": "VQ-VAE Cross-Check",
            "vq_usage": "Global Codebook Usage",
            "vq_timeline": "Selected Asset VQ Code Timeline",
            "vq_gmm": "VQ Code vs GMM Dominant State",
            "vq_missing_asset": "Saved VQ artifacts exist, but no VQ rows match this "
            "selected asset.",
            "vq_collapse": "Codebook collapse warning: one VQ code owns more than 80% of "
            "samples.",
            "meta_title": "Meta-Labelling Diagnostic",
            "meta_caption": "Hindsight-only validation. This is not a tradable model and "
            "does not train or save a classifier.",
            "horizon": "Future horizon",
            "stress_base": "Stress Base Rate",
            "panic_precision": "Panic Precision",
            "stress_recall": "Stress Recall",
            "false_alarm": "False Alarm Rate",
            "meta_box": "Future Return by Dominant GMM State",
            "meta_rates": "Stress Rates by Dominant State",
            "meta_metric_help_title": "How to interpret validation metrics",
            "meta_metric_help": "\n"
            "- **Stress Base Rate** is how often future stress happens "
            "under this asset-specific definition. It is the "
            "benchmark.\n"
            "- **Panic Precision** asks: when the model called Panic, "
            "how often did stress follow?\n"
            "- **Stress Recall** asks: of all future stress events, "
            "how many were caught by Panic?\n"
            "- **False Alarm Rate** asks: how often did Panic fire "
            "without later stress?\n"
            "\n"
            "Promising evidence means Panic precision is meaningfully "
            "above the base rate, recall is not trivial, and false "
            "alarms are not extreme. This still only justifies further "
            "research.\n",
            "pipeline": "Data Pipeline Audit",
            "hist": "Anomaly Detection: Z-Score Distributions",
            "density": "GMM Probability Density",
            "profiler": "State Profiler",
            "density_empty": "Not enough state observations to render density surfaces.",
            "meta_help": "\n"
            "Meta-labelling here asks: when the GMM said Panic, did stress "
            "actually arrive later?\n"
            "\n"
            "Stress is defined adaptively for the selected asset:\n"
            "\n"
            "- Downside stress: the worst future move over the horizon is in "
            "the bottom 20%.\n"
            "- Turbulence: the largest absolute future move over the horizon "
            "is in the top 20%.\n"
            "- Stress event: either downside stress or turbulence.\n"
            "\n"
            "Precision answers how many Panic calls were followed by stress. "
            "Recall answers how many stress events were caught by Panic "
            "calls.\n",
            "vq_help": "\n"
            "VQ-VAE is a discrete latent-state cross-check. It should not "
            "replace GMM yet.\n"
            "\n"
            "Useful signs:\n"
            "\n"
            "- many active VQ codes,\n"
            "- no single code dominates the whole sample,\n"
            "- VQ codes overlap with GMM states in interpretable ways,\n"
            "- or VQ codes disagree with GMM before important market "
            "transitions.\n",
        },
        "zh": {
            "title": "市场状态分析",
            "subtitle": "基于 GMM 诊断和潜在状态交叉验证的市场状态分析。",
            "workflow_title": "这个页面如何工作",
            "workflow_text": "\n"
            "这个页面用于理解市场状态，不是单独批准交易的工具。\n"
            "\n"
            "1. **当前状态** 展示最新状态、置信度、恐慌概率，以及价格/概率时间线。\n"
            "2. **状态地图** 展示 Quiet、Chop、Panic 在波动率、流动性、趋势效率空间里的位置。\n"
            "3. **诊断** 检查特征分布与状态画像是否足够稳定，值得继续研究。\n"
            "4. **验证** 检查 VQ 潜在 code 是否与 GMM 一致，以及 Panic 是否在历史上领先未来压力。\n"
            "\n"
            "健康流程：先确认所选资产线，再看当前状态是否清晰，然后检查压力特征和诊断是否支持这个标签，最后用验证页判断是否值得作为研究候选。\n",
            "scope_note": "此页面现在读取项目资产分类。当前内置状态矩阵仍是本地中国期货/指数数据；美股和美股期权可以作为资产线选择，但需要接入供应商支持的状态矩阵后图表才会激活。",
            "scope_help_title": "数据范围",
            "scope_stats": "样本：{assets} 个资产 | 特征日期 {feature_start} 至 {feature_end} | GMM "
            "日期 {regime_start} 至 {regime_end}",
            "tab_timeline": "当前状态",
            "tab_geometry": "状态地图",
            "tab_cross": "验证",
            "tab_diag": "诊断",
            "asset_class": "资产线",
            "taxonomy_title": "资产分类上下文",
            "taxonomy_help_title": "如何阅读资产线",
            "taxonomy_help": "\n"
            "资产线告诉页面在解释状态前应使用哪套市场规则。\n"
            "\n"
            "- **FUTURES_CN** 是当前本地/静态研究线，因此内置 parquet 矩阵可以直接渲染。\n"
            "- **EQUITY_US** 已在分类中准备好，但应在 FMP 美股适配器接入后使用。\n"
            "- **OPTIONS_US** "
            "已在分类中准备好，但属于非向量化市场，未来期权状态应使用事件驱动的期权链逻辑，而不是直接套用期货矩阵假设。\n"
            "\n"
            "如果某条资产线没有本地状态行，页面会在展示图表前停止。这是故意的：防止把中国期货状态误解释成美股或期权状态。\n",
            "taxonomy_status_ready": "所选资产线已有本地状态行，可以渲染。",
            "taxonomy_status_empty": "所选资产线存在于分类中，但当前还没有本地特征/状态行。",
            "taxonomy_status_nonvector": "分类标记该资产线为非向量化。在接入事件驱动版本前，应把当前 GMM 矩阵逻辑视作研究占位。",
            "taxonomy_description": "描述",
            "taxonomy_region": "地区",
            "taxonomy_settlement": "交割",
            "taxonomy_price_limit": "涨跌停",
            "taxonomy_vectorizable": "可向量化",
            "taxonomy_provider": "供应商",
            "taxonomy_data_mode": "数据模式",
            "taxonomy_local_rows": "本地行数",
            "taxonomy_local_assets": "本地资产数",
            "lane_summary": "地区 {region} · 交割 {settlement} · 涨跌停 {price_limit} · "
            "可向量化 {vectorizable} · 本地 {rows} 行 · 本地 {assets} 个资产",
            "yes": "是",
            "no": "否",
            "asset": "选择资产",
            "current_state": "当前状态",
            "panic_prob": "恐慌概率",
            "confidence": "置信度",
            "hurst": "Hurst",
            "vq_code": "VQ Code",
            "latest_date": "最新日期",
            "sample_rows": "行数",
            "panic_flag": "Panic 标记",
            "dynamic_threshold": "动态恐慌阈值",
            "quality_title": "状态信号质量",
            "quality_stale": "仅作诊断：最新数据已经超过 30 天，当前状态读数偏旧。",
            "quality_sparse": "证据偏弱：该资产的特征/状态重叠样本太少，状态判断不够稳定。",
            "quality_unclear": "证据偏弱：模型概率分散在多个状态之间，当前标签偏模糊。",
            "quality_risk": "风险规避候选：Panic 是强置信度主导状态。行动前仍需结合价格行为与下游风控规则验证。",
            "quality_promising": "有研究价值：当前状态较清晰，样本量也足够进入下一步验证。",
            "quality_ok": "可用背景信息：状态读数可解释，但仍是研究证据，不是独立交易信号。",
            "current_context": "从这里开始。这个页签回答：所选资产当前处于什么状态，GMM 有多确定，压力特征是否支持这个判断。",
            "geometry_context": "用这里观察模型状态是否在特征空间中分得开。分离越清晰，标签越容易被信任。",
            "validation_context": "相信状态标签前先看这里。VQ code 一致性和事后压力诊断越好，信号越可信。",
            "diagnostic_context": "当页面结果让人意外时看这里。它检查输入与状态画像是否在数学上合理。",
            "current_help_title": "如何解读当前状态指标",
            "current_help": "\n"
            "- **当前状态** 是原始 GMM 状态映射为 Quiet、Chop、Panic 后，平滑概率最高的状态。\n"
            "- **恐慌概率** 是最高压力状态的平滑概率。高值是风险提示，不是自动交易指令。\n"
            "- **置信度** 是最大状态概率。低于约 50% 时，模型本身并不确定。\n"
            "- **Hurst** 低于 0.5 通常偏噪声/均值回复；高于 0.5 通常偏趋势更顺滑。\n"
            "- **VQ Code** 是 VQ-VAE 存在时的独立潜在桶。没有 VQ 不会破坏 GMM，但少了一层交叉验证。\n",
            "timeline_help_title": "如何阅读时间线",
            "timeline_help": "\n"
            "上方是价格，中间是平滑后的 GMM 状态概率，下方是压力特征。\n"
            "\n"
            "可信的 Panic 通常伴随较高恐慌概率，以及波动率和非流动性上升。可信的 Quiet "
            "通常压力较低且概率稳定。频繁跳状态是谨慎信号。\n",
            "phase_help_title": "如何阅读状态地图",
            "phase_help": "\n"
            "每个点代表一个资产-日期观测。坐标轴是非流动性、波动率和趋势效率。\n"
            "\n"
            "好的状态几何应让 Quiet、Chop、Panic "
            "不至于完全混在一起。如果颜色高度重叠，模型仍可描述市场，但标签可信度会下降。\n",
            "density_help_title": "如何阅读密度曲面",
            "density_help": "\n"
            "这些曲面近似展示每个状态通常位于波动率/流动性空间的什么位置。\n"
            "\n"
            "曲面窄且分离清楚更容易解释。曲面很平或高度重叠，说明 GMM 状态不够干净，需要更多验证。\n",
            "profiler_help_title": "如何阅读状态画像",
            "profiler_help": "\n"
            "状态画像把标签翻译成平均特征：每个状态出现频率、平均收益、波动率、非流动性和趋势效率。\n"
            "\n"
            "有用的状态映射应该符合直觉。Panic 一般应比 Quiet 压力更高。如果画像反过来，要检查状态映射和源数据。\n",
            "no_vq": "未找到已保存的 VQ-VAE 结果。请先在 Feature Governance 页面训练并保存 VQ codebook。",
            "vq_title": "VQ-VAE 交叉验证",
            "vq_usage": "全局 Codebook 使用情况",
            "vq_timeline": "所选资产 VQ Code 时间线",
            "vq_gmm": "VQ Code vs GMM 主导状态",
            "vq_missing_asset": "已找到 VQ 结果，但没有匹配当前资产的 VQ 行。",
            "vq_collapse": "Codebook 塌缩警告：单个 VQ code 占据超过 80% 样本。",
            "meta_title": "Meta-Label 诊断",
            "meta_caption": "仅用于事后验证。这不是可交易模型，也不会训练或保存分类器。",
            "horizon": "未来窗口",
            "stress_base": "压力基础概率",
            "panic_precision": "恐慌精准率",
            "stress_recall": "压力召回率",
            "false_alarm": "误报率",
            "meta_box": "不同 GMM 主导状态下的未来收益",
            "meta_rates": "不同主导状态下的压力事件比例",
            "meta_metric_help_title": "如何解读验证指标",
            "meta_metric_help": "\n"
            "- **压力基础概率** 是在当前资产定义下，未来压力自然出现的频率，是基准线。\n"
            "- **恐慌精准率** 回答：模型叫 Panic 后，未来真的出现压力的比例。\n"
            "- **压力召回率** 回答：所有未来压力事件中，有多少被 Panic 捕捉到。\n"
            "- **误报率** 回答：Panic 触发后没有出现未来压力的比例。\n"
            "\n"
            "有潜力的证据通常意味着 Panic "
            "精准率明显高于基础概率，召回率不是接近零，误报率也不能过高。即便如此，也只能支持进一步研究。\n",
            "pipeline": "数据管道审计",
            "hist": "异常检测：Z-Score 分布",
            "density": "GMM 概率密度",
            "profiler": "状态画像",
            "density_empty": "状态样本不足，无法渲染密度曲面。",
            "meta_help": "\n"
            "这里的 meta-label 诊断回答：当 GMM 判断为恐慌后，未来是否真的出现了压力？\n"
            "\n"
            "压力标签会针对当前资产自适应定义：\n"
            "\n"
            "- 下行压力：未来窗口内最差价格变动位于历史最低 20%。\n"
            "- 剧烈波动：未来窗口内最大绝对价格变动位于历史最高 20%。\n"
            "- 压力事件：满足下行压力或剧烈波动任一条件。\n"
            "\n"
            "精准率表示 Panic 调用后有多少真的出现压力。召回率表示压力事件中有多少被 Panic 捕捉到。\n",
            "vq_help": "\n"
            "VQ-VAE 是离散潜在状态的交叉验证工具，目前不替代 GMM。\n"
            "\n"
            "有价值的迹象包括：\n"
            "\n"
            "- 多个 VQ code 被使用；\n"
            "- 没有单一 code 统治全部样本；\n"
            "- VQ code 与 GMM 状态有可解释的重叠；\n"
            "- 或者 VQ code 在关键市场切换前，比 GMM 更早给出不同信息。\n",
        },
    },
    "alpha_feature_governance": {
        "en": {
            "page_title": "ML Hub",
            "subtitle": "Audit feature data, run reproducible model experiments, test predictive evidence, explain fitted models, and register reusable artifacts.",
            "pipeline": "Pipeline: Data & Features -> Models & Experiments -> Feature Evidence -> Explainability -> Registry -> Strategy Construction",
            "matrix": "Feature matrix",
            "asset_class": "Asset taxonomy lane",
            "target": "Target column",
            "corr_threshold": "Correlation cluster threshold",
            "min_assets": "Minimum assets per day for IC",
            "overview": "Overview",
            "corr": "Correlation Clusters",
            "stability": "IC and Stability",
            "mda": "OOS MDA",
            "missing": "Missingness",
            "pca": "PCA Baseline",
            "latent": "Latent Cross-Check",
            "protocol": "Shortlist and Protocol",
            "tab_data": "Data & Features",
            "tab_evidence": "Feature Evidence",
            "tab_models": "Models & Experiments",
            "tab_explainability": "Explainability",
            "tab_registry": "Registry",
            "data_stage_title": "Data and feature readiness",
            "data_stage_caption": "Question answered: is the scoped feature matrix complete, correctly classified, and usable for modelling? This tab makes no performance claim.",
            "evidence_stage_title": "Out-of-sample feature evidence",
            "evidence_stage_caption": "Question answered: do individual inputs contain stable out-of-sample predictive information before portfolio construction and costs?",
            "evidence_scope_title": "Evidence scope",
            "evidence_scope_caption": "This view starts from the {feature_count} engineered feature columns in `{matrix_name}`. It does not automatically include the separate factor library.",
            "evidence_all_families": "All feature families",
            "evidence_family": "Feature family",
            "evidence_set": "Evidence set",
            "evidence_all_features": "All scoped features",
            "evidence_representatives": "Correlation representatives",
            "evidence_feature": "Feature focus",
            "visible_features": "Visible features",
            "visible_families": "Visible families",
            "mean_abs_rank_ic": "Mean |Rank IC|",
            "median_coverage": "Median coverage",
            "evidence_inventory_title": "Filtered feature inventory",
            "evidence_empty_scope": "No features match the current scope. Broaden one of the filters above.",
            "predictiveness_title": "1. Does the feature predict the forward target?",
            "predictiveness_caption": "Bars show mean daily cross-sectional Spearman Rank IC. A negative value can still be informative if the economic orientation is expected to be negative.",
            "mean_rank_ic_axis": "Mean daily Rank IC",
            "feature_axis": "Feature",
            "implementation_friction_title": "Implementation friction: predictive strength versus rank turnover",
            "implementation_friction_caption": "Rank turnover is a signal-churn proxy, not realised portfolio turnover. Use it to spot features that may require frequent position changes.",
            "implementation_friction_empty": "The current scope has no usable Rank IC and rank-turnover observations.",
            "rank_turnover_axis": "Rank turnover proxy",
            "stability_title": "2. Is the relationship stable through time?",
            "stability_caption": "The 60-day rolling Rank IC shows whether the sign and strength persist or come from a short episode.",
            "daily_ic_empty": "No daily Rank IC observations are available for this matrix and target.",
            "stability_features": "Features to compare",
            "stability_select_feature": "Select at least one visible feature.",
            "date_axis": "Date",
            "rolling_rank_ic_axis": "60-day rolling Rank IC",
            "distinctiveness_title": "3. Is the information distinct?",
            "distinctiveness_caption": "The heatmap and cluster table reveal features that carry nearly the same cross-sectional information. Representatives are candidates for a simpler, less duplicated input set.",
            "distinctiveness_single_feature": "Select at least two features to inspect cross-feature redundancy.",
            "models_stage_title": "Models and experiment ledger",
            "models_stage_caption": "Question answered: which model implementation was run, against which target and frozen validation contract, and what was the recorded outcome?",
            "model_library_title": "Research model library",
            "model_library_caption": "This inventory says which implementations exist. It does not claim that a model has been fitted, validated, or promoted.",
            "model_library_implementations": "Implementations",
            "model_library_supervised": "Supervised predictors",
            "model_library_regimes": "Regime estimators",
            "model_library_latent": "Latent models",
            "model_library_state_space": "State-space models",
            "model_library_filter": "ML task family",
            "model_library_all": "All ML task families",
            "model_taxonomy_title": "How the ML library is organised",
            "model_taxonomy_help": "`oqp.research.ml` is the common umbrella. `ml.tree_based` identifies an algorithm family (LightGBM and XGBoost); `ml.regression` defines the supervised prediction task and its validation contract; `ml.regimes` contains unsupervised sequential state models; `ml.latent` contains self-supervised representation learning; and `ml.state_space` contains online adaptive estimators such as Dual Kalman regression. They are all ML, but a task category and an algorithm category are not the same thing, so unrelated models do not inherit a target-dependent base class.",
            "experiment_designs_title": "Registered research designs",
            "experiment_designs_caption": "A registered design fixes the intended comparisons. It remains separate from the executed experiment ledger below.",
            "experiment_empirical_status": "Empirical status",
            "experiment_primary_metric": "Primary metric",
            "experiment_comparisons": "Controlled comparisons",
            "explain_stage_title": "Fitted-model explainability",
            "explain_stage_caption": "Question answered: which inputs did a completed model rely on, and how concentrated was that reliance? Importance diagnoses model behaviour; it does not establish causality or trading profitability.",
            "registry_stage_title": "Reusable model registry",
            "registry_stage_caption": "Question answered: which trained artifact is frozen, fingerprinted, and available for reuse by Strategy Construction? Registration is a reproducibility record, not an approval decision.",
            "runtime_readiness": "Supervised booster runtime readiness",
            "check_runtimes": "Check runtimes",
            "runtime_checking": "Testing native model runtimes...",
            "runtime_not_checked": "Run the isolated checks before retraining; each adapter is tested in a subprocess so a native crash cannot take down the dashboard.",
            "ready": "Ready",
            "unavailable": "Unavailable",
            "experiments": "Experiments",
            "completed": "Completed",
            "failed": "Failed",
            "latest_oos_ic": "Latest OOS IC",
            "experiment_ledger": "Experiment ledger",
            "registry_error": "Research registry unavailable",
            "no_experiments": "No governed ML experiments have been recorded yet. Use the training entrypoint below; completed and failed attempts will appear here.",
            "all_models": "All models",
            "all_assets": "All asset classes",
            "model_filter": "Model adapter",
            "asset_filter": "Asset class",
            "inspect_experiment": "Inspect experiment",
            "validation_policy": "Validation policy",
            "experiment_metrics": "Out-of-sample metrics",
            "reproducibility_record": "Reproducibility record",
            "training_entrypoint": "Training entrypoint",
            "training_entrypoint_caption": "This command is for supervised LightGBM/XGBoost runs. The regime-study executor will be exposed after its final point-in-time panel adapter is connected.",
            "model_adapter": "Model adapter",
            "validation_mode": "Validation mode",
            "factor_module": "Factor module",
            "model_registry": "Registered model artifacts",
            "no_registered_models": "No trained model artifact has been registered yet.",
            "no_explainable_experiments": "No completed experiment with a fitted model is available for explanation.",
            "explain_experiment": "Completed experiment",
            "oos_rank_ic": "OOS Rank IC",
            "explained_features": "Features measured",
            "top_feature_share": "Top feature share",
            "top_five_share": "Top-five share",
            "registered_artifacts": "Artifacts",
            "registered_models": "Distinct models",
            "available_artifacts": "Files available",
            "latest_registration": "Latest registration",
            "all_factors": "All factors",
            "factor_filter": "Factor",
            "inspect_artifact": "Inspect registered artifact",
            "validation_contract": "Frozen validation contract",
            "artifact_fingerprints": "Artifact and data fingerprints",
            "artifact_configuration": "Stored configuration and metrics",
            "saved_importance": "Saved model feature importance",
            "importance_unavailable": "The selected experiment has no readable feature-importance artifact.",
            "importance_error": "Feature-importance artifact could not be read",
            "pca_diagnostic": "Linear redundancy baseline",
            "rows": "Rows",
            "features": "Features",
            "assets": "Assets",
            "dates": "Date Range",
            "taxonomy_inferred": "This legacy matrix has no asset_class column. The page assigns `{asset_class}` from its governed matrix path. New matrices should declare asset_class on every row.",
            "taxonomy_empty": "The selected feature matrix contains no rows for a recognized asset taxonomy lane.",
            "manual_title": "How to read this page",
            "manual": "\n"
            "This page is the feature governance layer between "
            "`feature_engineering.py` and the ML engines.\n"
            "\n"
            "1. Correlation clusters: features with high absolute Spearman "
            "correlation are grouped. Inside each group, the suggested "
            "representative is the feature with stronger walk-forward IC and "
            "cleaner coverage. This keeps interpretability while reducing duplicate "
            "risk.\n"
            "2. Feature stability: IC is measured day by day across the "
            "cross-section. Stable features have a useful mean IC, decent ICIR, and "
            "a positive-day rate above random.\n"
            "3. PCA baseline: PCA is shown only as a diagnostic linear compression "
            "baseline. If PCA needs many components, the feature space is not "
            "simply linear. If PCA explains everything with a few components, there "
            "may be redundant engineered features.\n"
            "4. VQ-VAE path: the reusable immutable core lives in "
            "`oqp.research.ml.latent.vqvae`. The interactive panel below still uses "
            "the historical dashboard adapter and is labelled accordingly.\n"
            "5. Model comparison: raw features, correlation-cluster "
            "representatives, PCA components, and VQ-VAE latents should be judged "
            "by the same OOS IC, Sharpe, drawdown, and turnover after costs.\n"
            "\n"
            "Turnover proxy here is not actual portfolio turnover. It is the "
            "average day-to-day change in a feature's cross-sectional rank. High "
            "values mean a model using that feature may chatter more.\n",
            "family_title": "Feature family map",
            "summary_title": "Feature quality table",
            "heatmap_title": "Spearman feature correlation heatmap",
            "pairs_title": "High-correlation pairs",
            "clusters_title": "Correlation cluster representatives",
            "no_pairs": "No feature pairs breach the selected correlation threshold.",
            "ic_bar": "Mean daily IC by feature",
            "turnover_scatter": "Predictive power vs rank turnover proxy",
            "daily_ic": "Rolling daily IC stability",
            "mda_title": "4. Does a model actually need the feature?",
            "mda_section_caption": "This is the expensive confirmation step. It retrains a model across purged time folds and asks whether shuffling each scoped feature damages out-of-sample Rank IC.",
            "mda_setup": "Configure and run the OOS MDA test",
            "mda_help": "\n"
            "Tree gain is in-sample: it tells us which features the model used "
            "while fitting. It can reward noise.\n"
            "\n"
            "OOS MDA is stricter. For each purged time fold, we train on the "
            "allowed past/future rows, score the untouched test fold, then "
            "shuffle one feature inside that test fold. If shuffling a feature "
            "damages OOS Rank IC, the feature contains real information. If MDA "
            "is near zero or negative, the feature is probably redundant, noisy, "
            "or overfit.\n"
            "\n"
            "Reading guide:\n"
            "\n"
            "- Positive and stable MDA: useful feature.\n"
            "- MDA near zero: model can live without it.\n"
            "- Negative MDA: shuffling helped, so the feature may be harmful.\n"
            "- High tree gain but flat/negative MDA: classic in-sample overfit "
            "suspect.\n",
            "mda_run": "Run OOS MDA",
            "mda_running": "Running purged-k-fold MDA...",
            "mda_features": "Max features",
            "mda_folds": "Folds",
            "mda_embargo": "Embargo days",
            "mda_rows": "Max rows",
            "mda_repeats": "Permutation repeats",
            "mda_score": "Mean OOS Rank IC",
            "mda_plot": "OOS MDA by feature",
            "mda_gain_scatter": "Tree gain vs OOS MDA",
            "mda_table": "MDA feature audit",
            "mda_not_run": "Run OOS MDA to audit whether feature importance survives "
            "out-of-sample permutation.",
            "mda_gain_unavailable": "Tree gain is unavailable for this estimator.",
            "mda_fold_scores": "Fold scores",
            "missing_bar": "Missingness by feature",
            "pca_variance": "PCA explained variance",
            "pca_loadings": "Component loadings",
            "pca_market_note": "Feature PCA here diagnoses engineered-feature redundancy. Use "
            "the Market Breadth Lab for market covariance PCA and "
            "achievable breadth.",
            "component": "Component",
            "shortlist": "Suggested representative features",
            "comparison": "Research comparison protocol",
            "latent_title": "Legacy STORM-style VQ-VAE Cross-Check",
            "latent_legacy_warning": "This interactive panel still uses the historical joblib-based dashboard encoder. The reusable immutable VQ-VAE core is listed in the model library, but it is not yet connected to this Train button.",
            "latent_help": "\n"
            "This tab trains or loads a compact temporal VQ-VAE. The model "
            "reads rolling windows of manual features and snaps each window to "
            "one discrete codebook state.\n"
            "\n"
            "Use it as a cross-check, not an oracle. A useful codebook "
            "should:\n"
            "\n"
            "- use many codes without collapsing into one state,\n"
            "- map to sensible manual-feature profiles,\n"
            "- show some IC against future target ranks,\n"
            "- and have interpretable overlap or disagreement with GMM/HMM "
            "states.\n"
            "\n"
            "If a VQ code consistently appears before GMM panic states, or "
            "captures useful IC while staying weakly correlated with manual "
            "features, it becomes a serious candidate latent factor.\n",
            "train_latent": "Train / refresh VQ codebook",
            "load_latent": "Load saved VQ artifacts",
            "latent_unavailable": "No saved VQ artifacts yet. Configure the small model and "
            "train it from this tab.",
            "latent_training": "Training compact VQ-VAE latent engine...",
            "latent_saved": "Saved latent artifacts",
            "window": "Window",
            "samples": "Max samples",
            "codes": "Codes",
            "latent_dim": "Latent dim",
            "epochs": "Epochs",
            "active_codes": "Active Codes",
            "perplexity": "Perplexity",
            "largest_code": "Largest Code",
            "code_ic": "Code IC",
            "usage_title": "Codebook usage",
            "loss_title": "Training loss",
            "manual_profile": "Manual feature meaning by code",
            "gmm_overlap": "VQ code vs GMM dominant state",
            "gmm_missing": "GMM probability file not found or no overlap available.",
            "latent_table": "Latent samples",
            "collapse_warning": "Codebook collapse warning: one code dominates more than 80% "
            "of samples. Treat this run as a failed latent-factor "
            "candidate unless a larger model/training run fixes it.",
            "comparison_text": "\n"
            "Current status:\n"
            "\n"
            "- Raw engineered features: available now from "
            "`runtime/data/feature_store/ML_Feature_Matrix.parquet`.\n"
            "- Correlation-cluster representatives: available now from "
            "this page's shortlist.\n"
            "- PCA components: available now as a linear compression "
            "baseline.\n"
            "- VQ-VAE latent variables: scaffolded and available in the "
            "Latent Cross-Check tab, but should only be promoted after "
            "purged walk-forward validation.\n"
            "\n"
            "The next clean experiment is to train identical ML models on "
            "each feature set and compare OOS IC, Sharpe, max drawdown, "
            "average turnover, and transaction-cost-adjusted return.\n",
            "no_file": "No ML feature parquet file was found in "
            "runtime/data/feature_store.",
            "load_error": "Feature governance failed",
        },
        "zh": {
            "page_title": "机器学习中心",
            "subtitle": "审计特征数据、执行可复现实验、检验预测证据、解释拟合模型，并登记可复用模型产物。",
            "pipeline": "流程：数据与特征 -> 模型与实验 -> 特征证据 -> 模型解释 -> 注册库 -> 策略构建",
            "matrix": "特征矩阵文件",
            "asset_class": "资产分类线",
            "target": "目标列",
            "corr_threshold": "相关性聚类阈值",
            "min_assets": "计算 IC 的每日最少资产数",
            "overview": "总览",
            "corr": "相关性聚类",
            "stability": "IC 与稳定性",
            "mda": "样本外 MDA",
            "missing": "缺失值",
            "pca": "PCA 基线",
            "latent": "潜变量交叉验证",
            "protocol": "候选特征与研究协议",
            "tab_data": "数据与特征",
            "tab_evidence": "特征证据",
            "tab_models": "模型与实验",
            "tab_explainability": "模型解释",
            "tab_registry": "注册库",
            "data_stage_title": "数据与特征就绪度",
            "data_stage_caption": "本页回答：当前范围内的特征矩阵是否完整、分类正确并可用于建模？这里不作任何绩效结论。",
            "evidence_stage_title": "特征样本外证据",
            "evidence_stage_caption": "本页回答：在构建组合和计算交易成本前，各输入特征是否具有稳定的样本外预测信息？",
            "evidence_scope_title": "证据范围",
            "evidence_scope_caption": "当前视图从 `{matrix_name}` 中的 {feature_count} 个工程化特征列开始，并不会自动包含独立维护的因子库。",
            "evidence_all_families": "全部特征家族",
            "evidence_family": "特征家族",
            "evidence_set": "证据集合",
            "evidence_all_features": "全部当前特征",
            "evidence_representatives": "相关性簇代表特征",
            "evidence_feature": "聚焦特征",
            "visible_features": "当前特征数",
            "visible_families": "当前家族数",
            "mean_abs_rank_ic": "平均 |Rank IC|",
            "median_coverage": "覆盖率中位数",
            "evidence_inventory_title": "筛选后特征清单",
            "evidence_empty_scope": "当前筛选条件下没有特征，请放宽上方任一筛选项。",
            "predictiveness_title": "1. 特征能否预测未来目标？",
            "predictiveness_caption": "柱状图展示每日横截面 Spearman Rank IC 的平均值。如果经济方向本应为负，负 IC 同样可能包含有效信息。",
            "mean_rank_ic_axis": "日均 Rank IC",
            "feature_axis": "特征",
            "implementation_friction_title": "实施摩擦：预测力与排名换手",
            "implementation_friction_caption": "排名换手只是信号频繁变化的代理，并非真实组合换手率；它用于识别可能需要频繁调仓的特征。",
            "implementation_friction_empty": "当前范围内没有可用的 Rank IC 与排名换手观测。",
            "rank_turnover_axis": "排名换手代理",
            "stability_title": "2. 预测关系能否跨时间保持？",
            "stability_caption": "60 日滚动 Rank IC 用于判断预测方向和强度是否持续，还是只来自某个短暂时期。",
            "daily_ic_empty": "当前特征矩阵与目标没有可用的日度 Rank IC 观测。",
            "stability_features": "对比特征",
            "stability_select_feature": "请至少选择一个当前可见特征。",
            "date_axis": "日期",
            "rolling_rank_ic_axis": "60 日滚动 Rank IC",
            "distinctiveness_title": "3. 信息是否具有独立性？",
            "distinctiveness_caption": "热力图与聚类表用于识别横截面信息几乎相同的特征。代表特征可用于构建更简洁、重复度更低的输入集合。",
            "distinctiveness_single_feature": "请至少选择两个特征，才能检查特征间冗余。",
            "models_stage_title": "模型与实验账本",
            "models_stage_caption": "本页回答：使用了哪个模型、预测哪个目标、采用什么冻结验证方案，以及实验实际得到了什么结果？",
            "model_library_title": "研究模型库",
            "model_library_caption": "这里展示已经存在的模型实现，但不代表模型已经训练、验证或获准上线。",
            "model_library_implementations": "模型实现数",
            "model_library_supervised": "监督式预测模型",
            "model_library_regimes": "状态推断模型",
            "model_library_latent": "潜变量模型",
            "model_library_state_space": "状态空间模型",
            "model_library_filter": "机器学习任务类别",
            "model_library_all": "全部机器学习任务",
            "model_taxonomy_title": "机器学习模型库如何组织",
            "model_taxonomy_help": "`oqp.research.ml` 是统一入口。`ml.tree_based` 表示算法家族（LightGBM、XGBoost）；`ml.regression` 定义监督式回归任务及其验证契约；`ml.regimes` 保存无监督序列状态模型；`ml.latent` 保存自监督表征学习；`ml.state_space` 保存 Dual Kalman 回归等在线自适应估计器。它们都属于机器学习，但“任务类别”和“算法类别”不是同一层概念，因此不应让无关模型强行继承依赖目标变量的同一个基类。",
            "experiment_designs_title": "已注册研究设计",
            "experiment_designs_caption": "注册设计用于固定计划比较的内容；它与下方真正执行过的实验账本保持分离。",
            "experiment_empirical_status": "实证状态",
            "experiment_primary_metric": "主要评价指标",
            "experiment_comparisons": "受控比较",
            "explain_stage_title": "拟合模型解释",
            "explain_stage_caption": "本页回答：已完成模型依赖了哪些输入，这种依赖是否过度集中？重要性只用于诊断模型行为，不证明因果关系或可交易盈利。",
            "registry_stage_title": "可复用模型注册库",
            "registry_stage_caption": "本页回答：哪些训练产物已经冻结、留有指纹并可被策略构建模块复用？登记代表可复现记录，不代表获准上线。",
            "runtime_readiness": "监督式树模型运行环境就绪度",
            "check_runtimes": "检查运行环境",
            "runtime_checking": "正在测试原生模型运行环境...",
            "runtime_not_checked": "重新训练前请运行隔离检查；每个适配器都在子进程中测试，原生库崩溃不会拖垮仪表盘。",
            "ready": "就绪",
            "unavailable": "不可用",
            "experiments": "实验数",
            "completed": "已完成",
            "failed": "失败",
            "latest_oos_ic": "最新样本外 IC",
            "experiment_ledger": "实验账本",
            "registry_error": "研究注册库不可用",
            "no_experiments": "尚未记录受治理的机器学习实验。请使用下方训练入口；成功与失败的尝试都会显示在这里。",
            "all_models": "全部模型",
            "all_assets": "全部资产分类",
            "model_filter": "模型适配器",
            "asset_filter": "资产分类",
            "inspect_experiment": "查看实验",
            "validation_policy": "验证政策",
            "experiment_metrics": "样本外指标",
            "reproducibility_record": "可复现记录",
            "training_entrypoint": "训练入口",
            "training_entrypoint_caption": "该命令仅用于监督式 LightGBM/XGBoost 实验。状态模型研究将在最终时点数据面板适配器接通后提供执行入口。",
            "model_adapter": "模型适配器",
            "validation_mode": "验证模式",
            "factor_module": "因子模块",
            "model_registry": "已注册模型产物",
            "no_registered_models": "尚未注册训练完成的模型产物。",
            "no_explainable_experiments": "尚无已完成并可解释的拟合模型实验。",
            "explain_experiment": "已完成实验",
            "oos_rank_ic": "样本外 Rank IC",
            "explained_features": "已计算特征数",
            "top_feature_share": "第一特征占比",
            "top_five_share": "前五特征占比",
            "registered_artifacts": "模型产物数",
            "registered_models": "不同模型数",
            "available_artifacts": "文件可用数",
            "latest_registration": "最近登记日期",
            "all_factors": "全部因子",
            "factor_filter": "因子",
            "inspect_artifact": "查看已注册产物",
            "validation_contract": "冻结验证方案",
            "artifact_fingerprints": "模型与数据指纹",
            "artifact_configuration": "已保存配置与指标",
            "saved_importance": "已保存的模型特征重要性",
            "importance_unavailable": "所选实验没有可读取的特征重要性产物。",
            "importance_error": "无法读取特征重要性产物",
            "pca_diagnostic": "线性冗余基线",
            "rows": "样本行数",
            "features": "特征数",
            "assets": "资产数",
            "dates": "日期范围",
            "taxonomy_inferred": "该旧版矩阵没有 asset_class 列。页面根据受管矩阵路径将其归类为 `{asset_class}`。新矩阵应在每一行明确声明 asset_class。",
            "taxonomy_empty": "所选特征矩阵在已识别的资产分类线中没有可用行。",
            "manual_title": "如何阅读本页面",
            "manual": "\n"
            "这个页面是 `feature_engineering.py` 和机器学习引擎之间的特征治理层。\n"
            "\n"
            "1. 相关性聚类：如果两个特征的 Spearman 绝对相关性很高，就把它们放进同一个簇。每个簇中建议保留的代表特征，会优先选择 IC "
            "更稳定、覆盖率更好的特征。这样可以减少重复风险，同时保留可解释性。\n"
            "2. 特征稳定性：IC 会按交易日横截面计算。一个稳定特征通常需要有可用的平均 IC、较好的 ICIR，以及高于随机水平的正 IC "
            "天数比例。\n"
            "3. PCA 基线：PCA 这里只作为线性压缩诊断，不直接代表最终建模方式。如果 PCA "
            "需要很多主成分才解释大部分方差，说明特征空间不是简单线性的；如果几个主成分就解释了绝大部分方差，说明工程化特征可能有明显冗余。\n"
            "4. VQ-VAE 路径：可复用、不可变的核心位于 `oqp.research.ml.latent.vqvae`。下方互动面板目前仍使用历史仪表盘适配器，并已明确标注。\n"
            "5. 模型比较：原始特征、相关性簇代表特征、PCA 主成分、VQ-VAE 潜变量，都必须用同一套样本外 "
            "IC、夏普、最大回撤、换手率、扣费后收益来比较。\n"
            "\n"
            "这里的换手率代理不是真实组合换手率。它衡量的是某个特征横截面排名的日度变化幅度。数值越高，代表使用该特征的模型越可能产生频繁信号闪烁。\n",
            "family_title": "特征家族地图",
            "summary_title": "特征质量表",
            "heatmap_title": "Spearman 特征相关性热力图",
            "pairs_title": "高相关特征对",
            "clusters_title": "相关性簇代表特征",
            "no_pairs": "当前阈值下没有高相关特征对。",
            "ic_bar": "各特征日均 IC",
            "turnover_scatter": "预测力 vs 排名换手代理",
            "daily_ic": "滚动日度 IC 稳定性",
            "mda_title": "4. 模型是否真的需要这个特征？",
            "mda_section_caption": "这是计算成本较高的确认步骤。系统会在经过净化的时间折中重复训练模型，并检验打乱当前范围内某个特征后，样本外 Rank IC 是否下降。",
            "mda_setup": "配置并运行样本外 MDA 检验",
            "mda_help": "\n"
            "树模型的 Gain 是样本内指标：它告诉我们模型训练时用了哪些特征切分。问题是，噪音特征也可能被树拿来过拟合，所以 Gain 会骗人。\n"
            "\n"
            "样本外 MDA 更严格。每个 purged "
            "时间折中，我们只用允许的训练行训练模型，在未见过的测试折上打分，然后只在测试折里打乱某一个特征。如果打乱后样本外 Rank IC "
            "明显下降，说明这个特征真的有信息。如果 MDA 接近 0 或为负，说明它可能只是重复、噪音或过拟合产物。\n"
            "\n"
            "阅读规则：\n"
            "\n"
            "- MDA 为正且稳定：特征有样本外价值。\n"
            "- MDA 接近 0：模型没有它也差不多。\n"
            "- MDA 为负：打乱反而更好，特征可能有害。\n"
            "- Gain 很高但 MDA 很低/为负：典型样本内过拟合嫌疑。\n",
            "mda_run": "运行样本外 MDA",
            "mda_running": "正在运行 purged-k-fold MDA...",
            "mda_features": "最大特征数",
            "mda_folds": "折数",
            "mda_embargo": "Embargo 天数",
            "mda_rows": "最大样本行数",
            "mda_repeats": "打乱重复次数",
            "mda_score": "平均样本外 Rank IC",
            "mda_plot": "各特征样本外 MDA",
            "mda_gain_scatter": "树 Gain vs 样本外 MDA",
            "mda_table": "MDA 特征审计表",
            "mda_not_run": "运行样本外 MDA，检查特征重要性是否经得起样本外打乱检验。",
            "mda_gain_unavailable": "当前模型无法提供树 Gain。",
            "mda_fold_scores": "各折分数",
            "missing_bar": "各特征缺失率",
            "pca_variance": "PCA 解释方差",
            "pca_loadings": "主成分载荷",
            "pca_market_note": "这里的 PCA 用来诊断工程化特征冗余。市场协方差 PCA 与可实现广度，请看市场广度实验室。",
            "component": "主成分",
            "shortlist": "建议保留的代表特征",
            "comparison": "研究比较协议",
            "latent_title": "旧版 STORM 风格 VQ-VAE 交叉验证",
            "latent_legacy_warning": "该互动面板目前仍使用基于 joblib 的历史仪表盘编码器。模型库已列出新的不可变 VQ-VAE 核心，但当前“训练”按钮尚未接入新核心。",
            "latent_help": "\n"
            "这个标签页会训练或读取一个轻量级 temporal VQ-VAE。模型读取手工特征的滚动窗口，并将每个窗口压缩到一个离散 "
            "codebook 状态。\n"
            "\n"
            "它不是神谕，而是交叉验证工具。一个有价值的 codebook 应该：\n"
            "\n"
            "- 使用多个 code，而不是塌缩成单一状态；\n"
            "- 每个 code 对应的手工特征画像有经济含义；\n"
            "- 对未来 target rank 有一定 IC；\n"
            "- 与 GMM/HMM 状态既能解释重叠，也能暴露有价值的差异。\n"
            "\n"
            "如果某个 VQ code 总是在 GMM 恐慌状态之前出现，或者能提供与手工特征低相关但有样本外 IC "
            "的信息，它就可能成为新的潜在因子候选。\n",
            "train_latent": "训练 / 刷新 VQ codebook",
            "load_latent": "读取已保存 VQ 结果",
            "latent_unavailable": "尚未保存 VQ 结果。请在本标签页配置小模型并训练。",
            "latent_training": "正在训练轻量级 VQ-VAE 潜变量引擎...",
            "latent_saved": "已保存潜变量结果",
            "window": "窗口",
            "samples": "最大样本数",
            "codes": "Code 数量",
            "latent_dim": "潜变量维度",
            "epochs": "训练轮数",
            "active_codes": "活跃 Code",
            "perplexity": "困惑度",
            "largest_code": "最大 Code 占比",
            "code_ic": "Code IC",
            "usage_title": "Codebook 使用情况",
            "loss_title": "训练损失",
            "manual_profile": "各 Code 的手工特征含义",
            "gmm_overlap": "VQ Code vs GMM 主导状态",
            "gmm_missing": "未找到 GMM 概率文件，或无法计算重叠关系。",
            "latent_table": "潜变量样本",
            "collapse_warning": "Codebook 塌缩警告：单个 code 占据超过 80% "
            "样本。除非更大的模型或更长训练能修复，否则这次潜变量实验应视为失败候选。",
            "comparison_text": "\n"
            "当前状态：\n"
            "\n"
            "- 原始工程化特征：已经可从 "
            "`runtime/data/feature_store/ML_Feature_Matrix.parquet` "
            "读取。\n"
            "- 相关性簇代表特征：本页面的候选表已经给出。\n"
            "- PCA 主成分：已经作为线性压缩基线展示。\n"
            "- VQ-VAE 潜变量：已在“潜变量交叉验证”标签页提供轻量级实验入口，但必须经过 purged "
            "walk-forward 验证后才能正式使用。\n"
            "\n"
            "下一步最干净的实验，是用完全相同的机器学习模型分别训练这些特征集合，并比较样本外 "
            "IC、夏普、最大回撤、平均换手率、以及扣除交易成本后的收益。\n",
            "no_file": "runtime/data/feature_store 下未找到 ML 特征 parquet 文件。",
            "load_error": "特征治理计算失败",
        },
    },
    "risk_factor_breadth_lab": {
        "en": {
            "title": "Market Breadth Lab",
            "subtitle": "Directional participation, volatility, capital concentration, and independent risk dimensions across taxonomy-aware asset universes.",
            "asset_class": "Asset class",
            "source": "Daily market file",
            "volatility_lookback": "Volatility lookback",
            "variance": "Variance threshold",
            "window": "Rolling window",
            "max_assets": "Max assets",
            "risk_view": "Risk data lens",
            "risk_ffill": "Forward-fill",
            "risk_bridge": "Brownian Bridge",
            "tab_overview": "Overview",
            "tab_directional": "Directional Breadth",
            "tab_volatility": "Volatility Map",
            "tab_concentration": "Concentration Breadth",
            "tab_risk": "Risk Breadth",
            "tab_windows": "Research Windows",
            "directional_now": "Directional breadth",
            "concentration_now": "Effective assets",
            "risk_now": "Independent risk dimensions",
            "volatility_now": "Median realized volatility",
            "overview_note": "Each card answers a different market-structure question; the four lenses are complementary, not interchangeable.",
            "lens_col": "Lens",
            "question_col": "Question answered",
            "method_col": "Method",
            "directional_question": "How widely are assets advancing or declining?",
            "concentration_question": "How many economically weighted assets are present?",
            "risk_question": "How many statistically independent covariance dimensions exist?",
            "volatility_question": "Where is realized movement concentrated by asset and industry?",
            "realized_vol_method": "Rolling realized volatility",
            "history_percentile": "Historical percentile",
            "directional_empty": "Directional breadth is unavailable for this source.",
            "daily_breadth": "Daily breadth",
            "advancers": "Advancers",
            "decliners": "Decliners",
            "active_assets": "Active assets",
            "rolling_21d": "21-day average",
            "directional_axis": "Advance / decline breadth",
            "month_col": "Month",
            "industry_col": "Industry / sector",
            "industry_directional": "Directional Breadth by Industry",
            "volatility_empty": "Volatility diagnostics are unavailable for this source and lookback.",
            "annualized_vol": "Annualized volatility",
            "asset_col": "Asset",
            "vol_percentile": "Vol percentile",
            "name_col": "Name",
            "observations_col": "Observations",
            "asset_volatility": "Highest-Volatility Assets",
            "median_vol": "Median volatility",
            "high_vol_share": "High-vol share",
            "industry_volatility": "Volatility by Industry",
            "industry_volatility_history": "Industry Volatility Through Time",
            "weight_market_cap": "Market capitalization",
            "weight_open_interest": "Open-interest notional proxy",
            "weight_turnover": "Traded value",
            "weight_volume": "Price x volume proxy",
            "weight_equal": "Equal-weight fallback",
            "weight_source": "Weight source",
            "weight_source_note": "The source is shown explicitly because HHI changes meaning with the selected weights.",
            "unavailable": "Unavailable",
            "concentration_empty": "Concentration breadth is unavailable for this source.",
            "effective_assets": "Effective assets",
            "effective_industries": "Effective industries",
            "top5_share": "Top 5 share",
            "largest_weight": "Largest weight",
            "effective_count": "Effective count",
            "metric_col": "Metric",
            "weight_col": "Weight",
            "industry_weight": "Latest Industry Weight",
            "largest_assets": "Largest Weighted Assets",
            "windows_empty": "Research-window guidance is unavailable for this source.",
            "windows_note": "Use these monthly structural states to cover different market conditions across train, validation, and out-of-sample periods. They diagnose dates; they do not prescribe one universal window length.",
            "state_broad_advance": "Broad advance",
            "state_broad_decline": "Broad decline",
            "state_mixed": "Mixed",
            "state_high_concentration": "High concentration",
            "state_normal_concentration": "Normal concentration",
            "state_low_concentration": "Low concentration",
            "state_high_volatility": "High volatility",
            "state_normal_volatility": "Normal volatility",
            "state_low_volatility": "Low volatility",
            "unsupported_asset": "This asset class is not vectorizable in the taxonomy yet, so covariance PCA is disabled.",
            "options_note": "Options market breadth needs a separate options risk engine using underlying returns, IV moves, Greeks, expiry buckets, and liquidity. Plain option price PCA is deliberately disabled here.",
            "no_source": "no daily parquet files found for this asset class.",
            "manual_title": "How to read this lab",
            "manual": "\n"
            "This page asks: **how many independent market dimensions does the selected "
            "asset universe really contain?**\n"
            "\n"
            "If hundreds of assets mostly move through a small number of common covariance "
            "drivers, treating every ticker as an independent bet overstates diversification. "
            "PCA gives a conservative linear estimate of the market's dimensional rank.\n"
            "\n"
            "Workflow:\n"
            "\n"
            "1. Read the breadth cards first. `BR95` is the number of principal "
            "components needed to explain 95% of market variance.\n"
            "2. Use the eigen spectrum to see whether variance is concentrated in a "
            "few giant common drivers.\n"
            "3. Use the sector/family map to understand what each PC physically means: "
            "industrials, banks, commodities, rates, growth baskets, or other common risk groups.\n"
            "4. Use rolling breadth to see whether the market becomes more "
            "one-dimensional during stress.\n",
            "cards": "Market Breadth Snapshot",
            "naive": "Naive Breadth",
            "valid": "Valid Assets",
            "br95": "BR95",
            "eff_rank": "Effective Rank",
            "participation": "Participation Ratio",
            "haircut": "Breadth Haircut",
            "spectrum": "Eigen Spectrum",
            "sector": "Sector / Family Map",
            "rolling": "Rolling Market Breadth",
            "absolute": "Absolute sector contribution",
            "signed": "Signed mean loading",
            "component": "Component",
            "top_loadings": "Top Asset Loadings",
            "sector_help_title": "How to read the sector map",
            "sector_help": "\n"
            "PCA signs are arbitrary: PC1 can be multiplied by -1 and still "
            "mean the same mathematical factor.\n"
            "\n"
            "Use **absolute contribution** to ask: “which sectors define this "
            "component?”\n"
            "\n"
            "Use **signed loading** to ask: “which sectors oppose each other "
            "inside this component?”\n",
            "low_assets": "Fewer than 20 valid assets are available. Results still render, but "
            "breadth interpretation is less stable.",
            "rolling_empty": "Rolling breadth unavailable. The selected history may be shorter "
            "than the rolling window or too sparse.",
            "skipped": "Skipped rolling windows",
            "error": "Market breadth analysis failed",
            "feature_link": "Note: Feature Governance PCA analyzes feature redundancy. This "
            "page analyzes market covariance breadth.",
            "metric_toggle": "Metric guide: what is healthy?",
            "metric_guide": "\n"
            "| Metric | Meaning | Rough reading guide |\n"
            "|---|---|---|\n"
            "| **PC1 variance** | Share of total market variance explained by "
            "the first principal component. | `<15%` fragmented market; "
            "`15-35%` normal/common macro driver; `35-50%` concentrated "
            "risk-on/risk-off market; `>50%` crisis-like one-factor market. |\n"
            "| **BR95** | Minimum number of PCs needed to explain 95% of "
            "covariance variance. | Higher means more independent market "
            "dimensions. For a 60-asset universe, `<15` is very compressed, "
            "`15-30` moderate, `30-45` broad, `>45` very diversified/noisy. |\n"
            "| **Effective Rank** | Entropy-based breadth of the eigenvalue "
            "spectrum. It asks: “if variance were spread smoothly, how many "
            "equal factors would this look like?” | Closer to asset count is "
            "healthier for diversification. For 60 assets, `<10` compressed, "
            "`10-25` moderate, `>25` broad. |\n"
            "| **Participation Ratio** | Concentration-sensitive effective "
            "dimension: `1 / sum(weight_i^2)`. | Usually lower than effective "
            "rank. For 60 assets, `<5` concentrated, `5-15` moderate, `>15` "
            "broad. |\n"
            "| **Breadth Haircut** | `BR95 / naive asset count` in this lab. | "
            "`1.0` means almost no haircut; `0.3-0.7` is common; `<0.3` means "
            "the universe behaves like a small number of common bets. |\n"
            "\n"
            "These are research heuristics, not laws. The “healthy” value "
            "depends on whether you want diversified alpha capacity or a "
            "focused macro risk book.\n",
            "pc_label_toggle": "How should I label PC1?",
            "pc_label_prefix": "Suggested PC1 label",
            "pc1_top_sectors": "Top sectors",
            "pc_label_note": "\n"
            "PCA components are mathematical directions, so the sign is "
            "arbitrary. A negative loading does **not** mean the factor is "
            "bearish; the entire vector could be multiplied by `-1`.\n"
            "\n"
            "To label a PC, look at **absolute sector contribution** and the "
            "**largest absolute asset loadings**. If the same economic family "
            "dominates, we can give the component an economic name.\n",
            "pc1_valid": "\n"
            "For this default all-market file, PC1 explains about 29% of total "
            "variance. That is a valid and plausible number: large enough to show "
            "a real common macro driver, but not so high that the whole Chinese "
            "futures market is behaving as one single trade.\n",
            "component_interpreter": "Component Diagnostics",
            "component_col": "PC",
            "variance_col": "Variance",
            "cumulative_col": "Cumulative",
            "label_col": "Label",
            "positive_basket": "Positive Basket",
            "negative_basket": "Negative Basket",
            "label_confidence": "Label Confidence",
            "interpretation_col": "Interpretation",
            "confidence_note": "Label Confidence is a heuristic semantic score based on loading concentration and signed-basket readability. It is not a statistical confidence interval.",
            "component_stability": "Component Stability",
            "component_stability_note": "Rolling PCA stability compares recent-window component loadings with the full-sample component. It is robustness evidence, not a statistical confidence interval.",
            "component_stability_empty": "Component stability is unavailable for this source/window.",
            "date_col": "Date",
            "loading_similarity": "Loading Similarity",
            "windows_col": "Windows",
            "avg_similarity": "Avg Similarity",
            "min_similarity": "Min Similarity",
            "label_match_rate": "Label Match",
            "sector_match_rate": "Sector Match",
            "avg_label_confidence": "Avg Label Confidence",
            "latest_label": "Latest Label",
            "latest_sector": "Latest Sector",
            "regime_periods": "Breadth Regime Periods",
            "regime_periods_caption": "Low/Normal/High breadth blocks are adaptive quantiles of the rolling breadth haircut. Red is compressed breadth; green is expanded breadth.",
            "regime_start": "Start",
            "regime_end": "End",
            "regime_col": "Regime",
            "regime_low": "Low / Compressed",
            "regime_normal": "Normal",
            "regime_high": "High / Expanded",
            "avg_br95": "Avg BR95",
            "avg_effective_rank": "Avg Effective Rank",
            "avg_haircut": "Avg Haircut",
            "research_use": "Research Use",
        },
        "zh": {
            "title": "市场广度实验室",
            "subtitle": "基于资产分类，从涨跌参与度、波动率、资本集中度和独立风险维度观察市场结构。",
            "asset_class": "资产类别",
            "source": "日频市场文件",
            "volatility_lookback": "波动率回看期",
            "variance": "解释方差阈值",
            "window": "滚动窗口",
            "max_assets": "最大资产数",
            "risk_view": "风险数据视角",
            "risk_ffill": "前向填充",
            "risk_bridge": "Brownian Bridge",
            "tab_overview": "概览",
            "tab_directional": "方向广度",
            "tab_volatility": "波动率地图",
            "tab_concentration": "集中度广度",
            "tab_risk": "风险广度",
            "tab_windows": "研究窗口",
            "directional_now": "方向广度",
            "concentration_now": "有效资产数",
            "risk_now": "独立风险维度",
            "volatility_now": "中位已实现波动率",
            "overview_note": "每张卡片回答不同的市场结构问题；四种视角相互补充，不能互相替代。",
            "lens_col": "视角",
            "question_col": "回答的问题",
            "method_col": "方法",
            "directional_question": "上涨或下跌在多少资产中得到广泛参与？",
            "concentration_question": "按经济权重计算，市场相当于多少个有效资产？",
            "risk_question": "协方差结构中存在多少个统计独立维度？",
            "volatility_question": "已实现波动集中在哪些资产和行业？",
            "realized_vol_method": "滚动已实现波动率",
            "history_percentile": "历史分位数",
            "directional_empty": "当前数据源无法计算方向广度。",
            "daily_breadth": "日度广度",
            "advancers": "上涨资产",
            "decliners": "下跌资产",
            "active_assets": "活跃资产",
            "rolling_21d": "21 日平均",
            "directional_axis": "涨跌方向广度",
            "month_col": "月份",
            "industry_col": "行业 / 板块",
            "industry_directional": "分行业方向广度",
            "volatility_empty": "当前数据源和回看期无法计算波动率诊断。",
            "annualized_vol": "年化波动率",
            "asset_col": "资产",
            "vol_percentile": "波动率分位数",
            "name_col": "名称",
            "observations_col": "观测数",
            "asset_volatility": "最高波动率资产",
            "median_vol": "中位波动率",
            "high_vol_share": "高波动资产占比",
            "industry_volatility": "分行业波动率",
            "industry_volatility_history": "行业波动率历史",
            "weight_market_cap": "总市值",
            "weight_open_interest": "持仓量名义价值代理",
            "weight_turnover": "成交金额",
            "weight_volume": "价格乘成交量代理",
            "weight_equal": "等权回退",
            "weight_source": "权重来源",
            "weight_source_note": "由于不同权重会改变 HHI 的经济含义，因此这里明确展示权重来源。",
            "unavailable": "不可用",
            "concentration_empty": "当前数据源无法计算集中度广度。",
            "effective_assets": "有效资产数",
            "effective_industries": "有效行业数",
            "top5_share": "前五大权重",
            "largest_weight": "最大单一权重",
            "effective_count": "有效数量",
            "metric_col": "指标",
            "weight_col": "权重",
            "industry_weight": "最新行业权重",
            "largest_assets": "最大权重资产",
            "windows_empty": "当前数据源无法生成研究窗口建议。",
            "windows_note": "用这些月度结构状态，让训练、验证和样本外区间覆盖不同市场环境。它们用于诊断具体日期，不规定唯一通用的窗口长度。",
            "state_broad_advance": "广泛上涨",
            "state_broad_decline": "广泛下跌",
            "state_mixed": "涨跌混合",
            "state_high_concentration": "高集中度",
            "state_normal_concentration": "正常集中度",
            "state_low_concentration": "低集中度",
            "state_high_volatility": "高波动",
            "state_normal_volatility": "正常波动",
            "state_low_volatility": "低波动",
            "unsupported_asset": "该资产类别在资产分类中尚不可向量化，因此不启用协方差 PCA。",
            "options_note": "期权市场广度需要独立的期权风险引擎，使用标的收益、IV 变动、Greeks、到期分组和流动性。这里有意禁用简单期权价格 PCA。",
            "no_source": "该资产类别未找到日频 parquet 文件。",
            "manual_title": "如何阅读本页面",
            "manual": "\n"
            "这个页面回答：**所选资产宇宙到底有多少个真正独立的市场维度？**\n"
            "\n"
            "如果几百甚至几千个资产主要由少数共同协方差驱动控制，那么把每个 ticker 都当成独立下注会高估分散化。PCA "
            "给出一个保守的线性市场维度估计。\n"
            "\n"
            "阅读顺序：\n"
            "\n"
            "1. 先看顶部广度卡片。`BR95` 表示解释 95% 市场方差需要多少个主成分。\n"
            "2. 看特征值谱，判断市场方差是否集中在少数几个共同驱动里。\n"
            "3. 看板块/资产族地图，理解每个 PC 的物理含义：工业品、银行、商品、利率、成长篮子或其他共同风险组。\n"
            "4. 看滚动广度，观察市场在压力期是否变得更加“一维化”。\n",
            "cards": "市场广度快照",
            "naive": "名义广度",
            "valid": "有效资产数",
            "br95": "BR95",
            "eff_rank": "有效秩",
            "participation": "参与率",
            "haircut": "广度折扣",
            "spectrum": "特征值谱",
            "sector": "板块 / 资产族地图",
            "rolling": "滚动市场广度",
            "absolute": "板块绝对贡献",
            "signed": "带符号平均载荷",
            "component": "主成分",
            "top_loadings": "资产载荷排名",
            "sector_help_title": "如何阅读板块地图",
            "sector_help": "\n"
            "PCA 的符号是任意的：PC1 整体乘以 -1，数学含义仍然完全一样。\n"
            "\n"
            "用 **绝对贡献** 回答：“这个主成分主要由哪些板块定义？”\n"
            "\n"
            "用 **带符号载荷** 回答：“这个主成分内部哪些板块彼此对冲或对立？”\n",
            "low_assets": "有效资产少于 20 个。结果仍会展示，但广度解释稳定性会下降。",
            "rolling_empty": "无法计算滚动广度。所选历史可能短于滚动窗口，或资产覆盖太稀疏。",
            "skipped": "跳过的滚动窗口",
            "error": "市场广度分析失败",
            "feature_link": "提示：Feature Governance 的 PCA 分析特征冗余；本页面分析市场协方差广度。",
            "metric_toggle": "指标说明：什么范围比较健康？",
            "metric_guide": "\n"
            "| 指标 | 含义 | 粗略阅读范围 |\n"
            "|---|---|---|\n"
            "| **PC1 解释方差** | 第一个主成分解释了整个市场协方差方差的比例。 | `<15%` 市场较分散；`15-35%` "
            "常见宏观共同驱动；`35-50%` 风险较集中；`>50%` 接近危机式“一因子市场”。 |\n"
            "| **BR95** | 解释 95% 市场方差所需的最少主成分数量。 | 越高代表独立风险维度越多。以 60 "
            "个资产为例，`<15` 很压缩，`15-30` 中等，`30-45` 较宽，`>45` 非常分散或偏噪声。 |\n"
            "| **有效秩** | 基于熵的有效维度，回答：“如果方差平滑分布，这像多少个等权因子？” | 越接近资产数量，分散度越好。以 "
            "60 个资产为例，`<10` 压缩，`10-25` 中等，`>25` 较宽。 |\n"
            "| **参与率** | 对集中度更敏感的有效维度：`1 / sum(weight_i^2)`。 | 通常低于有效秩。以 60 "
            "个资产为例，`<5` 集中，`5-15` 中等，`>15` 较宽。 |\n"
            "| **广度折扣** | 本页面定义为 `BR95 / 名义资产数`。 | `1.0` 表示几乎不打折；`0.3-0.7` "
            "常见；`<0.3` 说明很多合约本质上只是少数共同风险下注。 |\n"
            "\n"
            "这些是研究经验范围，不是铁律。“健康”取决于你想要的是分散 alpha 容量，还是集中宏观风险暴露。\n",
            "pc_label_toggle": "PC1 应该怎么命名？",
            "pc_label_prefix": "PC1 建议标签",
            "pc1_top_sectors": "主要板块",
            "pc_label_note": "\n"
            "PCA 主成分是数学方向，所以符号本身是任意的。负载荷为负 **不代表** 这个因子看空；整个向量乘以 `-1` "
            "后数学含义完全不变。\n"
            "\n"
            "给 PC 命名时，主要看 **板块绝对贡献** 和 "
            "**绝对载荷最大的资产**。如果同一经济家族明显占主导，我们就能给它一个经济解释。\n",
            "pc1_valid": "\n"
            "在默认全市场文件中，PC1 解释约 29% "
            "的总方差。这个数字是有效且合理的：它足够大，说明存在真实的共同宏观驱动；但又没有大到说明整个中国期货市场只剩下一笔单一交易。\n",
            "component_interpreter": "主成分诊断",
            "component_col": "PC",
            "variance_col": "解释方差",
            "cumulative_col": "累计方差",
            "label_col": "标签",
            "positive_basket": "正载荷篮子",
            "negative_basket": "负载荷篮子",
            "label_confidence": "标签可信度",
            "interpretation_col": "解释",
            "confidence_note": "标签可信度是基于载荷集中度和正负篮子可读性的语义启发式分数，不是统计置信区间。",
            "component_stability": "主成分稳定性",
            "component_stability_note": "滚动 PCA 稳定性会比较近期窗口的主成分载荷与全样本主成分。它是稳健性证据，不是统计置信区间。",
            "component_stability_empty": "当前数据源或窗口无法计算主成分稳定性。",
            "date_col": "日期",
            "loading_similarity": "载荷相似度",
            "windows_col": "窗口数",
            "avg_similarity": "平均相似度",
            "min_similarity": "最低相似度",
            "label_match_rate": "标签匹配率",
            "sector_match_rate": "板块匹配率",
            "avg_label_confidence": "平均标签可信度",
            "latest_label": "最新标签",
            "latest_sector": "最新板块",
            "regime_periods": "市场广度阶段",
            "regime_periods_caption": "低/正常/高广度阶段由滚动广度折扣的自适应分位数划分。红色表示广度压缩，绿色表示广度扩张。",
            "regime_start": "开始",
            "regime_end": "结束",
            "regime_col": "阶段",
            "regime_low": "低广度 / 压缩",
            "regime_normal": "正常",
            "regime_high": "高广度 / 扩张",
            "avg_br95": "平均 BR95",
            "avg_effective_rank": "平均有效秩",
            "avg_haircut": "平均广度折扣",
            "research_use": "研究用途",
        },
    },
}


def normalize_language(language: str | None) -> str:
    """Normalize dashboard language aliases to ``en`` or ``zh``."""

    if language is None:
        return "en"
    return LANGUAGE_ALIASES.get(str(language).strip(), "en")


def translate(
    catalog: Mapping[str, Mapping[str, Any]],
    language: str | None,
    key: str,
    default: Any | None = None,
) -> Any:
    """Look up a localized value with English/default fallback."""

    normalized = normalize_language(language)
    language_values = catalog.get(normalized, {})
    if key in language_values:
        return language_values[key]
    english_values = catalog.get("en", {})
    if key in english_values:
        return english_values[key]
    return key if default is None else default


def ops_text(
    language: str | None, key: str, default: str | None = None, **format_values: Any
) -> str:
    """Return a localized Ops dashboard string."""

    value = translate(OPS_TEXT, language, key, default)
    if isinstance(value, list):
        return str(value)
    text = str(value)
    return text.format(**format_values) if format_values else text


def ops_tabs(language: str | None, key: str) -> list[str]:
    """Return a localized list of tab labels."""

    value = translate(OPS_TEXT, language, key, [])
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def research_text(
    language: str | None, key: str, default: str | None = None, **format_values: Any
) -> str:
    """Return a localized Research dashboard string."""

    value = translate(RESEARCH_TEXT, language, key, default)
    if isinstance(value, list):
        return str(value)
    text = str(value)
    return text.format(**format_values) if format_values else text


def research_tabs(language: str | None, key: str) -> list[str]:
    """Return a localized Research dashboard list of labels."""

    value = translate(RESEARCH_TEXT, language, key, [])
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]


def research_page_catalog(page_key: str) -> Mapping[str, Mapping[str, Any]]:
    """Return the normalized bilingual catalog for a specialized Research dashboard page."""

    if page_key not in RESEARCH_PAGE_TEXT:
        raise KeyError(f"Unknown research page text catalog: {page_key}")
    return RESEARCH_PAGE_TEXT[page_key]


def research_page_legacy_catalog(page_key: str) -> dict[str, dict[str, Any]]:
    """Return an ``EN``/``ZH`` catalog for legacy Streamlit pages."""

    catalog = research_page_catalog(page_key)
    return {"EN": dict(catalog["en"]), "ZH": dict(catalog["zh"])}


def research_page_text(
    page_key: str,
    language: str | None,
    key: str,
    default: str | None = None,
    **format_values: Any,
) -> str:
    """Return a localized string for a specialized Research dashboard page."""

    value = translate(research_page_catalog(page_key), language, key, default)
    if isinstance(value, list):
        return str(value)
    text = str(value)
    return text.format(**format_values) if format_values else text


def research_page_tabs(page_key: str, language: str | None, key: str) -> list[str]:
    """Return localized tab labels for a specialized Research dashboard page."""

    value = translate(research_page_catalog(page_key), language, key, [])
    return [str(item) for item in value] if isinstance(value, list) else [str(value)]
