# Innovation-1 oracle-relative decision：实现与真实模型验证

本次使用本地 MiniCPM5-2B、84个 q/v 目标模块。未修改官方 RaLoRA 副本、原始模型文件和 benchmark。

## 修改文件与实现位置

| 文件（相对 finetune/） | 修改内容 |
|---|---|
| adaptive_lora/tangent_metrics.py（新增） | tangent_metrics：C_T、top-2r oracle、direction_efficiency、side oracle 和数值检查；residual_energy_rank：累计 sigma² 最小 k |
| adaptive_lora/decision.py | 主判据改为联合 efficiency/oracle；CA/CB 仅用于侧别；按整数 k_star 限制扩 rank |
| adaptive_lora/residual_gradient.py | 保留无单位矩阵的 tangent residual；能量改为 FP64 归约和小 epsilon |
| adaptive_lora/rank_expansion.py | 默认 zero_b_residual_v；旧 symmetric_residual 保留为 ablation；原分支/scaling 不变 |
| adaptive_lora/direction_correction.py | 互补 A/B target 接口；空 target 不修改参数；blend 仅为 ablation |
| adaptive_lora/adaptation.py | 接入新指标与策略；保存小 target 因子；diagnostic_only 禁止实际扩 rank/校正 |
| adaptive_lora/diagnostics.py | 扩展 modules/summary 字段；区分建议 new_rank 与实际 applied_new_rank |
| train_adaptive_lora.py | 新 CLI；连续 warm-up 的 checkpoint probe；轨迹不重置 optimizer/scheduler |
| run_minicpm5_oracle_diagnostic.sh（新增） | smoke/trajectory 可复现入口，既有输出目录拒绝覆盖 |
| adaptive_lora/tests/test_tangent_oracle.py（新增） | oracle、互补覆盖、能量选 rank、zero-B、诊断不变性测试 |
| adaptive_lora/tests/test_adaptive.py | 更新 decision 用例；显式测试旧 symmetric ablation，保留非零 adapter 保存/恢复测试 |
| adaptive_lora/README.md | 更新接口、配置与数学适用范围 |

本报告及 outputs 下的验证文件是新增实验记录。已有 validation.json、test_training_data.py 等先前工作未重写。

## 判定与边界

- 默认 direction efficiency 阈值0.8、oracle coverage阈值0.9、residual阈值0.1。
- residual已足够小则keep；否则 efficiency低时先诊断方向问题。
- oracle和efficiency同时低时，记录 mixed_direction_and_capacity_direction_first，不把混合问题宣称为纯方向问题。
- efficiency高且oracle低时按 residual energy rank扩展；rank上限阻止扩展时记录具体reason。
- oracle/efficiency高但residual仍超阈值时保留参数并记录deferred reason。
- energy target默认0.90；residual_rank_k90始终表示90%，另存配置对应k_star。
- 零梯度约定CT/oracle/efficiency=1、rho=0，避免0/0；一般梯度同时验证两种CT计算方式。
- residual GID只记录复杂度，不再作为ceil(GID)扩展数量。
- 新分支A=残余Vh的top-k行、B=0，确定性forward零扰动；旧scaling完整保留。

## 测试

27/27 unittest通过。包含用户指定的全部数学 sanity tests：正交能量分解、oracle上界、互补top-2r最优布局、
同rank错方向、平坦谱容量不足、[5,4,1,0]的90%最小k=2、zero-B扩展输出逐值一致。
另外验证互补target、零/微小梯度、GID不控制扩展数量、诊断前后参数/RNG/optimizer state不变、保存恢复与合并。
日志：outputs/oracle_unit_tests.log。静态编译和git diff --check通过。

## 真实模型4步 smoke

保持旧smoke的初始化、数据和warm-up配置；本次direction correction为none。
下表均为84个模块的均值。

| 指标 | 新结果 |
|---|---:|
| coverage_A | 0.00381062 |
| coverage_B | 0.09846220 |
| tangent_coverage | 0.10175125 |
| oracle_tangent_coverage | 0.94922387 |
| direction_efficiency | 0.10563149 |
| residual_energy_ratio | 0.89824874 |
| gid_residual | 44.68137554 |
| residual_rank_k90 | 6.97619048 |

keep/direction_mismatch/expand = 0/84/0。旧规则也是0/84/0，CA/CB与旧smoke逐均值一致。
新规则的依据是实际联合利用率相对oracle不足；不是CA/CB的绝对阈值。
其中71层为direction_gap、13层为mixed方向优先；侧别69 both、15 A。
最大CT+rho误差2.1141e-7，无oracle超界；probe逐层断言base weight不变。
phase1与phase2 initial因子完全相同，最终adapter全部finite。
见outputs/minicpm5_oracle_smoke/{adaptive_stats/summary.json,verification.json}。

## 连续warm-up trajectory（已完成）

500个training updates完成，2/20/100/500各诊断84层。每次使用同一训练calibration batch。
全程480256个可训练LoRA参数，rank分布不变，不执行校正/扩展，不重置optimizer/scheduler。

| step | CA | CB | CT | oracle | direction efficiency | rho | GID_R | k90 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.003806 | 0.103678 | 0.106932 | 0.949779 | 0.110947 | 0.893068 | 44.215402 | 6.880952 |
| 20 | 0.003871 | 0.069745 | 0.073319 | 0.944598 | 0.077466 | 0.926681 | 46.869878 | 7.142857 |
| 100 | 0.004078 | 0.059367 | 0.063155 | 0.918949 | 0.067774 | 0.936845 | 53.439045 | 10.011905 |
| 500 | 0.004411 | 0.045142 | 0.049270 | 0.932370 | 0.051677 | 0.950730 | 48.960634 | 8.630952 |

四个checkpoint的建议判定均为keep=0、direction_mismatch=84、expand=0；没有强迫结果变成预设分类。
四次共336条逐层记录全部finite，CT/rho分解检查通过，最大误差1.9586e-7，无oracle超界。
所有probe的base weight保持不变；最终84层adapter全部finite。
实测这些checkpoint的平均direction efficiency下降，并非逐步逼近；GID_R和k90也不单调。

原始轨迹：outputs/minicpm5_oracle_trajectory/warmup_trajectory.jsonl。
逐层记录：同目录diagnostics/step_{2,20,100,500}/modules.jsonl。
完成核查：同目录verification.json；训练记录：train.jsonl。

## 数学与工程限制

1. **所有84层都是n_split>1**。按请求的稠密BA公式计算的tangent及top-2r oracle，
   是官方block LoRA可达空间的放宽，不是固定分块参数的精确oracle。不能据此证明纯方向问题。
   日志明确标记ambient_dense_relaxation_of_block_factors，并保留真实block residual。
   smoke的真实block平均residual=0.974435；trajectory依次为0.973112、0.981599、0.984587、0.987369。
   后续需要block-constrained oracle来区分分块结构限制与分块内方向利用不足。
2. 训练数据只有9条演示样本，calibration只有1 batch；结果是工程/局部诊断证据，不是泛化结论。
3. 每次probe的G随训练改变。coverage是相对能量；下降不等于绝对residual增大或loss变坏。
   当前日志没有保存G/R绝对范数，尚不能对此下结论。
4. 四步smoke与500步trajectory的线性学习率计划长度不同，step2并非同一训练checkpoint。
5. 新zero-B默认只保证确定性forward零扰动；训练时额外dropout分支会消耗随机数。
   symmetric residual仍是会产生正梯度方向函数扰动的消融，不是默认方法。
6. correction默认none；未实现创新点3。blend不是正式的basis-invariant子空间优化。
7. 保持现有单GPU、一次自适应事件限制；未扩展为多GPU/重复动态rank重构。

## 重跑命令

```bash
# 使用新目录，避免覆盖已完成结果
bash finetune/run_minicpm5_oracle_diagnostic.sh smoke \
  --output_dir finetune/adaptive_lora/outputs/minicpm5_oracle_smoke_rerun
bash finetune/run_minicpm5_oracle_diagnostic.sh trajectory \
  --output_dir finetune/adaptive_lora/outputs/minicpm5_oracle_trajectory_rerun
```
