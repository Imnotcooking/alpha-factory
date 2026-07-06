from __future__ import annotations

from .constants import (
    RTV_FAST_WINDOW_TICKS,
    RTV_MIN_FAST_MOVE_TICKS,
    RTV_PERCENTILE,
    RTV_SLOW_WINDOW_TICKS,
)
from .engine import (
    _is_bearish_hypothesis,
    _is_relative_velocity_fade_hypothesis,
    _is_relative_velocity_hypothesis,
)

def _hypothesis_formula(hypothesis: str, lang: str, min_success_ticks: float) -> str:
    threshold = f"{min_success_ticks:g}"
    if lang == "ZH":
        if hypothesis == "relative_velocity_fade":
            return f"""
**研究问题：** 当价格出现 99 分位级别的极端短线速度突破后，价格是否会反向回补，而不是继续突破？

事件检测和相对速度突破完全一样：

`fast_move_ticks = (mid_price[t] - mid_price[t - {RTV_FAST_WINDOW_TICKS}]) / tick_size`

并且

`abs_fast_move_ticks = abs(fast_move_ticks)`

慢速基准只看过去，不看当前 tick：

`threshold_99 = rolling_quantile(abs_fast_move_ticks.shift(1), {RTV_PERCENTILE:.0%}, window={RTV_SLOW_WINDOW_TICKS:,})`

事件触发条件：

`abs_fast_move_ticks >= threshold_99`

并且

`abs_fast_move_ticks >= {RTV_MIN_FAST_MOVE_TICKS:g}`

但预期方向反过来：

`fast_move_ticks > 0` -> 价格向上暴冲，预期随后下跌回补

`fast_move_ticks < 0` -> 价格向下暴跌，预期随后上涨反弹

直觉解释：这是“流动性真空 / 橡皮筋拉太远”假设。极端速度突破可能不是趋势开始，而是主动单短时间吃穿盘口后留下的空洞。做市商看到价格被拉离短期公平值后，可能反向报价并推动价格回补。

结果验证：

`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / tick_size`

**正确 / Correct：**

如果原始速度向上：`future_move_ticks <= -{threshold}`

如果原始速度向下：`future_move_ticks >= {threshold}`

**失败 / Failed：**

未来价格没有反向回补至少 `{threshold}` 跳。

`accuracy = 正确的速度衰竭事件数 / 全部速度衰竭事件数`
"""
        if hypothesis == "relative_velocity":
            return f"""
**研究问题：** 当价格在极短窗口内出现相对自身历史非常罕见的速度突破后，价格是否会沿同方向继续？

相对 Tick 速度事件：

`fast_move_ticks = (mid_price[t] - mid_price[t - {RTV_FAST_WINDOW_TICKS}]) / tick_size`

并且

`abs_fast_move_ticks = abs(fast_move_ticks)`

慢速基准只看过去，不看当前 tick：

`threshold_99 = rolling_quantile(abs_fast_move_ticks.shift(1), {RTV_PERCENTILE:.0%}, window={RTV_SLOW_WINDOW_TICKS:,})`

事件触发条件：

`abs_fast_move_ticks >= threshold_99`

并且

`abs_fast_move_ticks >= {RTV_MIN_FAST_MOVE_TICKS:g}`

方向定义：

`fast_move_ticks > 0` -> 预期继续上涨

`fast_move_ticks < 0` -> 预期继续下跌

直觉解释：我们不是问“价格动了 5 跳大不大”，而是问“在今天/当前交易时段的最近 {RTV_SLOW_WINDOW_TICKS:,} 个 tick 里，这种 {RTV_FAST_WINDOW_TICKS} tick 速度是否已经超过过去 99% 的速度”。这是一种纯数学、低延迟、可移植到 C++ 的异常检测。

结果验证：

`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / tick_size`

**正确 / Correct：**

如果事件方向是上涨：`future_move_ticks >= {threshold}`

如果事件方向是下跌：`future_move_ticks <= -{threshold}`

**失败 / Failed：**

未来价格没有沿事件方向继续至少 `{threshold}` 跳。

`accuracy = 正确的速度突破事件数 / 全部速度突破事件数`
"""
        if hypothesis == "bearish_breakdown":
            return f"""
**研究问题：** 当看跌破位事件出现后，价格在所选向前窗口后是否继续下跌？

看跌破位事件：

`flow_imbalance <= -0.35`

并且

`book_imbalance <= +0.10`

并且

`volume_intensity >= 2.0`

并且

`rolling_mid_move_ticks <= -5.0`

并且

`price_shock >= 10.0`

直觉解释：这里不再要求盘口必须明显偏空到 `-0.10` 以下，因为 Level-1 可见盘口可能有噪声。我们要求的是：卖方主动成交占优，成交量至少是平时的 2 倍，价格在滚动窗口里已经下跌至少 5 跳，而且这个下跌相对近期噪声足够剧烈。换句话说，它不是“吸收”，而是“价格已经被打穿”。

结果验证：

`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / tick_size`

**正确 / Correct：**

`future_move_ticks <= -{threshold}`

也就是未来中间价至少继续下跌 `{threshold}` 跳。

**失败 / Failed：**

`future_move_ticks > -{threshold}`

包括：价格上涨、价格不动、或虽然下跌但没有达到 `{threshold}` 跳。

`accuracy = 正确的看跌破位事件数 / 全部看跌破位事件数`
"""
        if hypothesis == "bearish":
            return f"""
**研究问题：** 当看跌冲击事件出现后，价格在所选向前窗口后是否下跌？

看跌冲击事件：

`flow_imbalance <= -0.35`

并且

`book_imbalance <= -0.10`

并且

`volume_intensity >= 3.0`

并且

`rolling_mid_move_ticks <= 0`

直觉解释：卖方正在主动砸盘，可见卖盘仍然更强，成交量异常放大，同时价格在滚动窗口内已经被压住/下移。这就是看跌冲击假设。

结果验证：

`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / tick_size`

**正确 / Correct：**

`future_move_ticks <= -{threshold}`

也就是未来中间价至少下跌 `{threshold}` 跳。

**失败 / Failed：**

`future_move_ticks > -{threshold}`

包括：价格上涨、价格不动、或虽然下跌但没有达到 `{threshold}` 跳。

`accuracy = 正确的看跌冲击事件数 / 全部看跌冲击事件数`
"""
        return f"""
**研究问题：** 当看涨吸收事件出现后，价格在所选向前窗口后是否上涨？

看涨吸收事件：

`flow_imbalance <= -0.35`

并且

`book_imbalance >= 0.10`

并且

`volume_intensity >= 3.0`

并且

`rolling_mid_move_ticks >= 0`

直觉解释：卖方正在主动砸盘，但可见买盘仍然更强，成交量异常放大，同时价格在滚动窗口内没有下跌而是守住/上移。这就是看涨吸收假设。

结果验证：

`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / tick_size`

**正确 / Correct：**

`future_move_ticks >= {threshold}`

也就是未来中间价至少上涨 `{threshold}` 跳。

**失败 / Failed：**

`future_move_ticks < {threshold}`

包括：价格下跌、价格不动、或虽然上涨但没有达到 `{threshold}` 跳。

`accuracy = 正确的看涨吸收事件数 / 全部看涨吸收事件数`
"""

    if hypothesis == "bearish_breakdown":
        return f"""
**Research question:** when a bearish breakdown event appears, does price continue down after the selected forward horizon?

Bearish breakdown event:

`flow_imbalance <= -0.35`

and

`book_imbalance <= +0.10`

and

`volume_intensity >= 2.0`

and

`rolling_mid_move_ticks <= -5.0`

and

`price_shock >= 10.0`

Plain English: this rule does not require the visible Level-1 book to be deeply bearish, because the visible book can be noisy. Instead it asks for aggressive sell flow, at least 2x abnormal volume, a rolling mid-price drop of at least 5 ticks, and a large price shock relative to recent noise. This is not “sell pressure got absorbed”; it is “price has already broken.”

Outcome test:

`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / tick_size`

**Correct:**

`future_move_ticks <= -{threshold}`

The future mid-price must continue falling by at least `{threshold}` tick(s).

**Failed:**

`future_move_ticks > -{threshold}`

This includes price going up, staying flat, or going down by less than `{threshold}` tick(s).

`accuracy = correct_bearish_breakdown_events / total_bearish_breakdown_events`
"""

    if hypothesis == "relative_velocity_fade":
        return f"""
**Research question:** when price makes a 99th-percentile short-window velocity burst, does it snap back instead of continuing?

The event detector is exactly the same as relative velocity breakout:

`fast_move_ticks = (mid_price[t] - mid_price[t - {RTV_FAST_WINDOW_TICKS}]) / tick_size`

and

`abs_fast_move_ticks = abs(fast_move_ticks)`

The slow baseline uses only prior ticks:

`threshold_99 = rolling_quantile(abs_fast_move_ticks.shift(1), {RTV_PERCENTILE:.0%}, window={RTV_SLOW_WINDOW_TICKS:,})`

Event trigger:

`abs_fast_move_ticks >= threshold_99`

and

`abs_fast_move_ticks >= {RTV_MIN_FAST_MOVE_TICKS:g}`

But the expected direction is flipped:

`fast_move_ticks > 0` -> price spiked up, expect snap-back down

`fast_move_ticks < 0` -> price crashed down, expect bounce up

Plain English: this is the liquidity-vacuum / stretched-rubber-band hypothesis. The burst may not be the start of a trend; it may be an aggressive order temporarily eating through the book and leaving price away from short-term fair value. Market makers can then fade the extreme and push price back into the gap.

Outcome test:

`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / tick_size`

**Correct:**

If the original velocity burst was up: `future_move_ticks <= -{threshold}`

If the original velocity burst was down: `future_move_ticks >= {threshold}`

**Failed:**

The future price does not snap back by at least `{threshold}` tick(s).

`accuracy = correct_relative_velocity_exhaustion_events / total_relative_velocity_exhaustion_events`
"""

    if hypothesis == "relative_velocity":
        return f"""
**Research question:** when price makes an unusually fast short-window move relative to its own recent baseline, does it continue in that same direction?

Relative tick velocity event:

`fast_move_ticks = (mid_price[t] - mid_price[t - {RTV_FAST_WINDOW_TICKS}]) / tick_size`

and

`abs_fast_move_ticks = abs(fast_move_ticks)`

The slow baseline uses only prior ticks:

`threshold_99 = rolling_quantile(abs_fast_move_ticks.shift(1), {RTV_PERCENTILE:.0%}, window={RTV_SLOW_WINDOW_TICKS:,})`

Event trigger:

`abs_fast_move_ticks >= threshold_99`

and

`abs_fast_move_ticks >= {RTV_MIN_FAST_MOVE_TICKS:g}`

Direction:

`fast_move_ticks > 0` -> expect continuation up

`fast_move_ticks < 0` -> expect continuation down

Plain English: we are not asking whether a 5-tick move is large in absolute terms. We are asking whether this {RTV_FAST_WINDOW_TICKS}-tick move is larger than 99% of comparable {RTV_FAST_WINDOW_TICKS}-tick moves in the recent {RTV_SLOW_WINDOW_TICKS:,}-tick baseline. This is a pure heuristic anomaly detector designed to be portable into C++.

Outcome test:

`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / tick_size`

**Correct:**

If the event direction is up: `future_move_ticks >= {threshold}`

If the event direction is down: `future_move_ticks <= -{threshold}`

**Failed:**

The future price does not continue in the event direction by at least `{threshold}` tick(s).

`accuracy = correct_relative_velocity_events / total_relative_velocity_events`
"""

    if hypothesis == "bearish":
        return f"""
**Research question:** when a bearish impulse event appears, does price go down after the selected forward horizon?

Bearish impulse event:

`flow_imbalance <= -0.35`

and

`book_imbalance <= -0.10`

and

`volume_intensity >= 3.0`

and

`rolling_mid_move_ticks <= 0`

Plain English: sellers are hitting the market, visible ask-side supply is still stronger, volume is abnormal, and price has already been held down/falling over the rolling window. That is the bearish impulse hypothesis.

Outcome test:

`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / tick_size`

**Correct:**

`future_move_ticks <= -{threshold}`

The future mid-price must fall by at least `{threshold}` tick(s).

**Failed:**

`future_move_ticks > -{threshold}`

This includes price going up, staying flat, or going down by less than `{threshold}` tick(s).

`accuracy = correct_bearish_impulse_events / total_bearish_impulse_events`
"""

    return f"""
**Research question:** when a bullish absorption event appears, does price go up after the selected forward horizon?

Bullish absorption event:

`flow_imbalance <= -0.35`

and

`book_imbalance >= 0.10`

and

`volume_intensity >= 3.0`

and

`rolling_mid_move_ticks >= 0`

Plain English: sellers are hitting the market, the visible bid side is still stronger, volume is abnormal, and price has held flat/up over the rolling window. That is the bullish absorption hypothesis.

Outcome test:

`future_move_ticks = (mid_price[t + horizon] - mid_price[t]) / tick_size`

**Correct:**

`future_move_ticks >= {threshold}`

The future mid-price must rise by at least `{threshold}` tick(s).

**Failed:**

`future_move_ticks < {threshold}`

This includes price going down, staying flat, or going up by less than `{threshold}` tick(s).

`accuracy = correct_bullish_absorption_events / total_bullish_absorption_events`
"""


def _hypothesis_notes(hypothesis: str, lang: str, min_success_ticks: float) -> str:
    expected = "下跌" if _is_bearish_hypothesis(hypothesis) else "上涨"
    expected_en = "down" if _is_bearish_hypothesis(hypothesis) else "up"
    threshold = f"{min_success_ticks:g}"
    if _is_relative_velocity_fade_hypothesis(hypothesis):
        if lang == "ZH":
            return f"""
**准确率是什么意思：** 如果页面显示 `60%`，表示所有极端速度衰竭事件中，有 60% 在所选向前窗口后发生了反向回补，而且幅度至少达到 `{threshold}` 跳。

**正确/失败如何定义：** 如果 6 tick 速度暴涨向上，未来价格下跌至少 `{threshold}` 跳才算正确；如果 6 tick 速度暴跌向下，未来价格上涨至少 `{threshold}` 跳才算正确。也就是说，这个假设专门在测试“剧烈冲刺后是否反转”。

**一个关键提醒：** 衰竭准确率不等于 `1 - 突破延续准确率`。突破延续失败可能只是未来没动、只动了半跳、或者噪声震荡；只有真的反向移动至少 `{threshold}` 跳，才算本假设正确。

**为什么它有金融直觉：** 99 分位速度冲刺经常代表一边主动单瞬间吃掉盘口流动性。价格跳到新位置后，后方可能出现流动性真空；如果没有新的追单接力，做市商会把价格压回或拉回更合理的位置。

**为什么它适合低延迟：** 这个规则只需要最近 {RTV_FAST_WINDOW_TICKS} tick 的价格变动，以及过去 {RTV_SLOW_WINDOW_TICKS:,} tick 的滚动 99% 阈值。它可以先作为 Python 研究信号，之后很自然地翻译成 C++ 原生规则。

**它不代表什么：** `60%` 不是最终胜率。它还没有扣除点差、滑点、排队位置、成交概率和撤单速度。它只是告诉我们：“这个事件之后，价格反向移动的频率是否明显高于基础概率？”
"""
        return f"""
**What accuracy means:** if the page says `60%`, then 60% of extreme velocity exhaustion events snapped back after the selected horizon, with magnitude of at least `{threshold}` tick(s).

**Correct vs failed:** if the 6-tick burst was upward, future price must fall by at least `{threshold}` tick(s). If the 6-tick burst was downward, future price must rise by at least `{threshold}` tick(s). This hypothesis is testing reversal after a violent sprint, not continuation.

**Key caution:** exhaustion accuracy is not simply `1 - continuation accuracy`. A failed continuation can be flat, too small, or noisy; it only counts as a correct fade if price actually reverses by at least `{threshold}` tick(s).

**Why it has market intuition:** a 99th-percentile velocity burst can mean one side consumed several layers of resting liquidity almost instantly. After the jump, there may be a liquidity vacuum behind the move; if follow-through orders do not arrive, market makers can push price back toward fair value.

**Why this is latency-friendly:** the rule needs only the latest {RTV_FAST_WINDOW_TICKS}-tick price move and a rolling 99% threshold from the prior {RTV_SLOW_WINDOW_TICKS:,} ticks. It can start as a Python research signal and later translate cleanly into a native C++ rule.

**What it does not mean:** `60%` is not final trading win rate. It does not yet include spread, slippage, queue position, fill probability, or cancel speed. It only answers: “after this event, does price reverse more often than the base rate?”
"""
    if _is_relative_velocity_hypothesis(hypothesis):
        if lang == "ZH":
            return f"""
**准确率是什么意思：** 如果页面显示 `60%`，表示所有相对速度突破事件中，有 60% 在所选向前窗口后继续沿突破方向移动，而且幅度至少达到 `{threshold}` 跳。

**正确/失败如何定义：** 向上速度突破后，未来上涨至少 `{threshold}` 跳才算正确；向下速度突破后，未来下跌至少 `{threshold}` 跳才算正确。方向不同，但评分规则是同一个。

**为什么它适合低延迟：** 这个规则只需要最近 {RTV_FAST_WINDOW_TICKS} tick 的价格变动，以及一个过去 {RTV_SLOW_WINDOW_TICKS:,} tick 的滚动 99% 阈值。没有模型推理，没有树模型，没有 Python-only 依赖，之后很容易写进 C++。

**它不代表什么：** 它仍然只是事件研究。它还没有考虑盘口排队位置、滑点、手续费、撤单速度和真实成交概率。

**如何使用：** 先看事件数量是否足够，再看准确率是否高于基础成功率，最后看 `Avg In Expected Direction` 是否为正。如果准确率一般但平均方向变动很大，它可能更像“少数大行情捕捉器”。
"""
        return f"""
**What accuracy means:** if the page says `60%`, then 60% of relative velocity breakout events continued in the breakout direction after the selected horizon, with magnitude of at least `{threshold}` tick(s).

**Correct vs failed:** after an upward velocity breakout, future price must rise by at least `{threshold}` tick(s). After a downward velocity breakout, future price must fall by at least `{threshold}` tick(s). The direction changes, but the scoring rule is the same.

**Why this is latency-friendly:** the rule needs only the latest {RTV_FAST_WINDOW_TICKS}-tick price move and a rolling 99% threshold from the prior {RTV_SLOW_WINDOW_TICKS:,} ticks. No model inference, no tree traversal, no Python-only dependency. It is a natural C++ candidate.

**What it does not mean:** it is still an event study. It does not include queue position, slippage, fees, cancel speed, or actual fill probability.

**How to use it:** first check event count, then compare accuracy against base rate, then check `Avg In Expected Direction`. If accuracy is only moderate but expected-direction move is large, this may be a rare big-move detector rather than a high-hit-rate signal.
"""
    if lang == "ZH":
        return f"""
**准确率是什么意思：** 如果页面显示 `60%`，表示被检测到的假设事件中，有 60% 在所选向前窗口后价格确实{expected}，而且幅度至少达到 `{threshold}` 跳。

**为什么需要最小成功变动：** 如果只要求 `> 0` 或 `< 0`，很小的噪声也会被算作正确。用 `{threshold}` 跳作为门槛，可以避免把几乎没有意义的微小移动当成有效预测。

**为什么要有假设下拉框：** 有些交易日整体偏空，只有看涨吸收可能样本很少。切换到看跌冲击，可以测试“卖压继续推动价格下行”的结构。

**它不代表什么：** 它还没有考虑手续费、买卖价差、成交质量、止损、仓位大小。这是事件研究，不是完整回测。

**如何使用：** 比较 bullish 和 bearish 两个假设在同一窗口下的事件数量、准确率、平均未来 tick 变动。如果某个方向事件更多、准确率更高、平均变动方向也一致，它更可能贴近当天的市场状态。
"""
    return f"""
**What accuracy means:** if the page says `60%`, then 60% of detected hypothesis events were followed by a price move {expected_en} after the selected horizon, with magnitude of at least `{threshold}` tick(s).

**Why the minimum success move exists:** if we only require `> 0` or `< 0`, tiny noise can count as correct. A `{threshold}` tick threshold avoids treating nearly flat movement as a useful prediction.

**Why the dropdown matters:** some sessions are directionally bearish or bullish. Testing only bullish absorption can hide the useful mirror pattern, so the dropdown lets you compare both sides cleanly.

**What it does not mean:** it does not include fees, spread crossing, fill quality, stop losses, or position sizing. This is an event study, not a full trading backtest.

**How to use it:** compare bullish and bearish hypotheses at the same rolling window and horizon. Prefer the side with enough events, accuracy above 50%, and average future ticks in the expected direction.
"""


def _research_readout_text(lang: str, min_success_ticks: float) -> str:
    threshold = f"{min_success_ticks:g}"
    if lang == "ZH":
        return f"""
### 这张表要回答什么？

我们现在不再只问“某一个公式是否漂亮”，而是问：**在不同向前窗口下，哪个市场故事真的更像有边际预测力？**

每一行都是一次事件研究：

`事件数` = 人工规则触发了多少次。

`准确率` = 事件出现后，未来价格是否按假设方向移动，并且至少达到 `{threshold}` 跳。

`基础成功率` = 不看任何信号、随机选一个 tick 时，未来达到同样方向和幅度的概率。

`超额` = `准确率 - 基础成功率`。这比单看准确率更重要，因为当天市场本身可能偏多或偏空。

`95% 置信区间` = 样本太小时要特别看它。区间很宽，说明我们不能过度相信这个准确率。

### 当前研究结论

目前的四天黄金样本里，**看涨吸收不是优先方向**。它在默认 `120` tick 附近表现明显偏弱：卖压出现、买盘似乎承接住，但价格之后并没有稳定上涨。这说明我们的“吸收”条件可能把真正的承接和仍在恶化的卖压混在一起了。

更值得先研究的是 **看跌冲击 / 看跌延续**。它的直觉是：卖方主动砸盘、盘口也偏弱、价格已经被压住，这种结构不是反转，而可能是下跌趋势仍在释放。前面计算里，较长 horizon，特别是 `240-480` tick，更像有边际优势。

我新增的第三个假设是 **看跌破位**。它不是把原来的看跌规则简单放松，而是测试另一个 bearish 子类型：价格已经出现明显下跌冲击，成交量放大，卖方主动流占优，同时盘口没有强到足以说明买盘在稳定承接。它更像“破位延续”，不是“盘口确认卖压”。

### 如何重设计看跌事件

第一步：先比较 `看跌冲击` 和 `看跌破位`，然后重点测试 `240` 和 `480` tick，而不是只看默认 `120` tick。

第二步：如果要尝试 ML，不要在这个页面堆一套额外研究流程。更清晰的做法是把 ML 逻辑封装成一个新的 hypothesis，例如 `ML-filtered bearish breakdown`，让它和其他假设一样进入事件表、准确率、样本外验证。

第三步：只有当同一个规则在不同 horizon、不同日期文件、不同市场状态下反复出现，我们才把它升级成正式规则。否则它只是样本内噪声。
"""
    return f"""
### What is this table answering?

We are no longer asking whether one formula looks elegant. We are asking: **across forward horizons, which market story actually shows marginal predictive power?**

Each row is an event study:

`Events` = how many times the hand-written rule fired.

`Accuracy` = after the event, did price move in the hypothesized direction by at least `{threshold}` tick(s)?

`Base Rate` = if we ignored the signal and randomly picked a tick, how often would the same directional move happen anyway?

`Lift` = `Accuracy - Base Rate`. This matters more than raw accuracy because the day itself may already be bullish or bearish.

`95% CI` = uncertainty from small sample size. If this interval is wide, we should not worship the point estimate.

### Current research conclusion

In the current four-day gold sample, **bullish absorption is not the priority**. Around the default `120`-tick horizon it is weak: sell pressure appears, the bid seems to absorb it, but price does not reliably rise afterward. That means our absorption rule is probably mixing true absorption with still-dangerous selling pressure.

The better first target is **bearish impulse / bearish continuation**. The story is: sellers are hitting the market, the visible book is also weak, and price has already been held down. That looks less like reversal and more like downside continuation. In the prior readout, the longer `240-480` tick horizons were the only ones that looked meaningfully interesting.

The new third hypothesis is **bearish breakdown**. It is not simply a looser bearish rule. It tests a different bearish subtype: price has already suffered a meaningful downward shock, volume is abnormal, sell flow dominates, and the visible book is not strong enough to suggest stable absorption. This is closer to “breakdown continuation” than “book-confirmed sell pressure.”

### How we redesign the bearish event

Step 1: compare `Bearish impulse` with `Bearish breakdown`, then focus on `240` and `480` ticks instead of only the default `120` ticks.

Step 2: if we want ML, do not bolt a separate research workflow onto this page. The cleaner design is to package it as another hypothesis, for example `ML-filtered bearish breakdown`, so it uses the same event table, accuracy scoring, and holdout validation.

Step 3: only promote a threshold into the formal rule if it repeats across horizons, different date files, and different market states. Otherwise it is probably sample noise wearing a convincing hat.
"""

