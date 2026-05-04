# 在 TFB + Optuna 中实现严格 Train/Val Split 的改造说明

> 本文档解释：我们如何在**不破坏原有 Benchmark 评测流程**的前提下，把 Optuna 调参目标从 test 指标改为 val 指标，并补齐「HPO(Val) → Final Test」的流程。

---

## 1. 背景问题（改造前）

在原流程中，Optuna 每个 trial 都会调用统一 pipeline 跑一遍评测，然后从评测日志里读出 `mse/mae/rmse...` 作为 objective。  
问题是：默认 `FixedForecast` 策略会把序列最后 `horizon` 段当作 test，并在这个 test 上打分。

这意味着：

- trial objective 实际来自 **test**；
- 会造成 test 信息泄漏进超参选择；
- 不符合科研上常见的两阶段协议（先在 val 上选参，再在 test 上一次性汇报）。

---

## 2. 改造目标

实现以下协议：

1. **HPO 阶段（Val）**：每个 trial 只在验证集上评估，Optuna 最小化 val metric；
2. **最终评估（Test）**：best params 选出后，再单独跑一次 test 指标。

同时要求：

- 尽量复用 TFB 原 pipeline；
- 不影响非 HPO 的默认评测逻辑。

---

## 3. 代码改动总览（忽略 search_space）

本次只动了两处核心逻辑：

- `ts_benchmark/evaluation/strategy/fixed_forecast.py`
- `ts_benchmark/hpo/optuna_search.py`

### 3.1 `FixedForecast`：新增可切换的 `hpo_eval_mode="val"`

在 `FixedForecast._execute()` 中增加分支：

- 当 `strategy_args["hpo_eval_mode"] == "val"`：
  - 先按原逻辑得到 `train_valid_data` 与 `test_data`；
  - 再把 `train_valid_data` 按 `train_ratio_in_tv` 切成 `train_data` / `val_data`；
  - 用 `train_data` 训练，用 `val_data` 计算评估指标；
  - `eval_horizon` 动态等于 `len(val_data)`。

- 否则（默认行为）：
  - 与原来一致：训练用 `train_valid_data`，评估用 `test_data`。

这样做的好处是：

- HPO 时不碰 test（仅 val）；
- 普通 benchmark 完全保持原语义。

### 3.2 `evaluate_params`：增加 `eval_mode` 并向 strategy 透传

在 `evaluate_params(...)` 增加参数：

- `eval_mode="val" | "test"`。

执行每个 horizon 时：

- 构造 `strategy_args`；
- 总是设置 horizon；
- 若 `eval_mode == "val"`，写入 `strategy_args["hpo_eval_mode"] = "val"`；
- 若 `eval_mode == "test"`，不写该字段，走策略默认 test 评估。

注意：

- 这里有一个关键点是要把 `hpo_eval_mode` 写在**最终传给 pipeline 的 strategy_args 对象**上，避免被后续覆盖。

### 3.3 `run_optuna_search`：HPO 后增加一次 final test

在 `run_optuna_search(...)` 中：

- baseline 和 trial 优化沿用 `evaluate_params(..., eval_mode="val")`；
- 选出 best params 后，额外调用一次 `evaluate_params(..., eval_mode="test")`；
- 将最终 test 结果写入：
  - `final_test_value`
  - `final_test_per_horizon`

这让输出结果天然分离了：

- `best_value` / `best_value_rechecked`（val 体系）
- `final_test_value`（test 体系）

---

## 4. 数据切分语义（以单序列为例）

假设序列长度 `N`，预测长度 `horizon = H`，`train_ratio_in_tv = r`。

### 默认 test 评估模式（旧逻辑 / eval_mode="test"）

1. `train_length = N - H`
2. `train_valid = series[:train_length]`
3. `test = series[train_length:]`
4. 模型在 `train_valid`（内部可再做训练早停验证）上 fit
5. 在 `test` 上算 metric

### HPO val 评估模式（新逻辑 / eval_mode="val"）

1. 先同上得到 `train_valid` / `test`
2. 在 `train_valid` 内再切：
   - `train = train_valid[:int(len(train_valid) * r)]`
   - `val = train_valid[int(len(train_valid) * r):]`
3. 模型在 `train` 上 fit
4. 在 `val` 上算 metric（trial objective）
5. `test` 仅保留给最后单次评估

---

## 5. 为什么这种改造是“最小侵入”的

- 不改 pipeline 主流程；
- 不改 evaluator/collector/reporting 框架；
- 只在 strategy 层新增一个可选开关；
- HPO 通过 `strategy_args` 注入开关，和原有配置机制一致；
- 非 HPO 任务不设置该开关，行为保持不变。

---

## 6. 兼容性与边界条件

### 6.1 兼容性

- 如果没有传 `hpo_eval_mode`，`FixedForecast` 完全按旧路径走。
- 因此旧脚本、旧配置不需要修改即可继续运行。

### 6.2 边界条件

在 val 模式下，如果 `train_ratio_in_tv` 导致 train 或 val 为空，会抛错（例如数据过短或比例极端）。  
这是故意的 fail-fast：避免 silent bug 影响 HPO 结果可靠性。

---

## 7. 推荐使用方式

1. 用 `run_optuna_search(...)` 做超参搜索（内部已按 val 目标优化）。
2. 读取输出 JSON：
   - 用 `best_params` 作为最佳超参；
   - 看 `final_test_value` 作为最终 test 汇报指标。
3. 若需要更严格科研流程，可再加一步显式「refit(train+val) + 保存权重」。

---

## 8. 一句话总结

这次改造把 Optuna 从“在 test 上选参”纠正为“在 val 上选参、在 test 上最终汇报”，并且尽量复用了 TFB 原有执行框架，做到低风险、可回滚、对旧流程兼容。
