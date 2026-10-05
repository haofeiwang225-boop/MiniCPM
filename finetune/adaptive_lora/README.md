# Gradient Subspace Matching + Residual-GID Adaptive LoRA

只实现创新点1；没有 Stiefel、Oblique、SPD 或 manifold optimization。
先阅读 [审计结论](AUDIT.md)，特别是 MELoRA 分块 tangent 的限制。


## 当前默认：GID + Dense LoRA（2026-10-04）

Adaptive 和诊断路径默认 `--lora_structure dense`。使用官方普通 `LinearWithLoRA`，
不调用 RaLoRA/MELoRA dynamic_init，不使用 n_split/mini_rank/block_diag。
`dense_init.py` 逐层做第一次 full-gradient probe、SVD/GID，再按实际参数预算分配 Dense rank。
原分块代码仅保留给显式 `--lora_structure block` 消融和官方 baseline wrapper；下面旧分块结果为历史记录。

令每层成本 c_l=d_in+d_out，选择连续 rank≈lambda*GID_l，施加 dense_rank_min/max，
再做整数预算内舍入；保证 sum(r_l*c_l)<=dense_lora_param_budget。
没有显式预算时使用 lora_rank 对应的普通 Dense 总预算，绝不将旧 block 的总代数rank当作Dense rank。
scaling=alpha/新分配rank。所有扩展仍保留旧scaling，默认新B=0。
第二次probe、tangent/oracle/efficiency、Residual-GID和k90数学逻辑不变；Dense日志不再带block-relaxation字段。

当前诊断命令预算480256，dense_rank_min=1/max=4；不自动加载任何旧adapter。
旧block实测480256；使用旧初始GID离线预算预演得到Dense478976（20层rank1、64层rank2），
余1280不足增加一个rank。预演不是新模型实跑结果，新probe后的准确值见initial_probe/budget.json。
该预算约束用于初始分配；后续明确触发的residual rank expansion可能增加参数，summary会记录前后参数量。

```bash
# 真实MiniCPM5-2B GPU smoke，4步，默认不做direction correction
bash finetune/run_minicpm5_oracle_diagnostic.sh smoke
# 连续warm-up，2/20/100/500只诊断，不改rank/不校正
bash finetune/run_minicpm5_oracle_diagnostic.sh trajectory
```

脚本名称沿用旧接口，输出已改为 outputs/minicpm5_dense_smoke 和 outputs/minicpm5_dense_trajectory。
已存在目录会拒绝覆盖。详细改动和验证见 [Dense切换报告](DENSE_LORA_REPORT.md)。

## 代码对应关系

| 文件 | 职责 |
|---|---|
| official.py | 接入本地官方类、manager、原始第一次 probe/分配/初始化 |
| gradient_probe.py | 第二次 probe；逐层暂开 base weight 梯度；不 step；恢复 RNG/mode/flags/grad |
| subspace_matching.py | 秩感知 QR+小 R 的 SVD；C_A/C_B 的 sigma² 加权覆盖 |
| residual_gradient.py | 无 identity 的双边残余；另记真实 block tangent 残余 |
| residual_gid.py | FP32 economy SVD + 与官方相同的奇异值 L1 熵有效秩 |
| tangent_metrics.py | 联合 tangent coverage、top-2r oracle、direction efficiency、residual energy rank |
| decision.py | keep/direction_mismatch/expand；联合 tangent/oracle 利用率与预算限制 |
| direction_correction.py | 互补目标子空间；默认 none；Procrustes blend 仅作 ablation；未来几何接口 |
| rank_expansion.py | 保留原官方层/scaling；默认 residual-V / zero-B；保留 symmetric ablation |
| adaptation.py | 同一 warm-up 状态逐层分析，延后统一修改，控制峰值内存 |
| diagnostics.py | modules.jsonl、summary.json、rank_pattern.json |
| checkpoint.py | adapter 保存/恢复、合并为普通 HF Linear，不改 benchmark |
| ../train_adaptive_lora.py | 共用官方初始化、两阶段训练及全新 optimizer/scheduler |
| tests/test_adaptive.py | 数学、probe 不变性、参数保留、训练、恢复与合并测试 |

与官方 SVD 的差别只在第二次分析：使用 full_matrices=False，保留 U/S/Vh；
奇异值熵定义不变，零残余定义为 GID=0。第一次完全调用原始官方函数。
不是 randomized SVD，也不把数值加速作为创新。B=0/秩亏不允许 QR 补出的假方向影响 coverage。

扩展后实际 delta 为 `old_scaling*B_old*A_old + B_add*A_add`。
旧层/分块/A/B/scaling 保留；增加分支的 scaling=1，默认新增实际更新为零；仅 symmetric ablation 为 `gamma*R_k`。
rank 统计采用 `mini_rank*n_split + added_rank`；真实参数量单独保存。

## 运行

从 MiniCPM 根目录运行。`ralora` 环境已验证能导入本地官方实现。
真实 MiniCPM-2B 权重路径必须由用户提供；不会悄悄换成 MiniCPM5。
训练与 calibration 都只取 `--train_data_path`，不接入 benchmark test/答案。

用户随后指定了 `/home/hongchang/whf/models/MiniCPM5-2B`。运行该模型必须显式加
`--model_family minicpm5`，结果记录为 MiniCPM5-2B，不再称为原始 MiniCPM-2B。
`model_loading.py` 使用 safetensors + Transformers 公开 state_dict 加载接口兼容
旧版 Transformers，严格检查 missing/unexpected keys，不改权重文件。
`training_data.py` 对该模型使用完整官方 chat template 渲染单轮训练样本，
通过 token offsets 屏蔽 prompt/padding 标签；不再逐消息重复添加 BOS。
若旧版 tokenizer 未自动加载 chat_template.jinja，显式读取该官方文件。

真实 MiniCPM5-2B GPU smoke 已完成，见
`outputs/minicpm5_2b_smoke_v2/verification.json`：42 层、84 个 q/v 目标，
4 个 training updates（2+2），84 个层均实际执行 blend 校正，最终 adapter 所有值有限。
默认诊断阈值下 keep=0、direction_mismatch=84、expand=0，参数量 480256 未变。
这是小 rank、单 calibration batch、9 条演示数据的工程验证，不是任务性能实验。
原始 MiniCPM-2B 与 MiniCPM5-2B 不混称，后者通过不代表前者已验证。

```bash
# MiniCPM5-2B GPU 少量 batch 验证；是工程 smoke，不是正式超参数配置
bash finetune/run_minicpm2b_adaptive_lora.sh \
  --model_name_or_path /home/hongchang/whf/models/MiniCPM5-2B \
  --model_family minicpm5 --device cuda --dtype bfloat16 --analysis_device cuda \
  --output_dir finetune/adaptive_lora/outputs/minicpm5_2b_smoke_new \
  --model_max_length 256 --batch_size 1 --max_steps 4 --adaptive_probe_step 2 \
  --lora_rank 2 --lora_alpha 4 --ralora_max_rank 4 --ralora_erank_max_power 2 \
  --adaptive_rank_max 32 --adaptive_rank_step_cap 2 \
  --gradient_est_n_steps 1 --adaptive_probe_batches 1 --direction_correction blend
```

```bash
# Baseline 1：原始 PEFT 入口，使用已安装 PEFT 的解释器
PEFT_PYTHON=/path/to/peft/environment/bin/python \
bash finetune/run_minicpm2b_vanilla_lora.sh \
  --model_name_or_path /path/to/MiniCPM-2B-sft-bf16 \
  --train_data_path finetune/data/AdvertiseGenChatML/train.json \
  --eval_data_path finetune/data/AdvertiseGenChatML/dev.json \
  --output_dir finetune/adaptive_lora/outputs/vanilla \
  --bf16 true --max_steps 200 --per_device_train_batch_size 1

# Baseline 2：本地官方 RaLoRA 初始化 + 单阶段训练
bash finetune/run_minicpm2b_official_ralora.sh \
  --model_name_or_path /path/to/MiniCPM-2B-sft-bf16 \
  --output_dir finetune/adaptive_lora/outputs/ralora

# Method：同一初始 RaLoRA + 第二次 probe + 两阶段训练
bash finetune/run_minicpm2b_adaptive_lora.sh \
  --model_name_or_path /path/to/MiniCPM-2B-sft-bf16 \
  --output_dir finetune/adaptive_lora/outputs/adaptive \
  --max_steps 200 --adaptive_probe_step 100 --adaptive_probe_batches 2 \
  --adaptive_direction_efficiency_threshold 0.8 --adaptive_oracle_coverage_threshold 0.9 --adaptive_residual_energy_threshold 0.1 \
  --adaptive_rank_max 256 --adaptive_rank_step_cap 4 \
  --residual_init_scale 0.001 --direction_correction blend --direction_blend 0.05 \
  --export_merged true
```

这是可执行接口示例，不是训练效果推荐参数；9 条演示数据不是论文训练集。
Vanilla 原入口默认 Gaussian A，官方 RaLoRA 桥接默认 Kaiming A，不能忽略
初始化、trainer、scheduler、数据顺序等差异直接声称严格公平的消融。
RaLoRA 与 Adaptive 共用入口和初始配置；adaptive_lora=false 不触发额外 probe、
不改变随机数状态、不修改 adapter、不重置 optimizer。

数学和小模型验证（不需要下载模型）：

```bash
TRITON_CACHE_DIR=/tmp/minicpm_adaptive_triton PYTHONPATH=finetune \
  /home/hongchang/miniconda3/envs/ralora/bin/python -B -m unittest discover \
  -s finetune/adaptive_lora/tests -v

bash finetune/run_minicpm2b_adaptive_lora.sh \
  --tiny_model --device cpu --dtype float32 --analysis_device cpu \
  --output_dir finetune/adaptive_lora/outputs/tiny_smoke \
  --max_steps 4 --adaptive_probe_step 2 \
  --gradient_est_n_steps 1 --adaptive_probe_batches 1 \
  --lora_rank 2 --lora_alpha 4 --ralora_max_rank 4 --ralora_erank_max_power 2 \
  --adaptive_rank_max 16 --direction_correction blend --export_merged true
```

MiniCPM-2B 少量 batch 验证：同上去掉 --tiny_model，设置真实模型路径、
--device cuda --dtype bfloat16。尚未运行这一步，因为目前未找到原始 2B 权重。

## 保存和恢复

输出包含 run_config.json、initial_probe/{rank,importance,n_splits,svd_profile}.json、
train.jsonl、phase1_adapter.pt、phase2_initial_adapter.pt、adapter.pt、adaptive_stats/*。
phase1 checkpoint 在第二次 probe 前保存。旧 optimizer/scheduler 在结构变化前销毁，
phase2 创建全新 AdamW 和 linear scheduler；不迁移 momentum/variance。
当前不是可中断恢复的训练管理器，未实现加载中途 optimizer checkpoint。

`adaptive_lora.checkpoint.load_adapter(base_model, path)` 可以恢复自定义 adapter；
`merge_for_export(model, target_names)` 得到普通 HF 模型，或使用 --export_merged true。
合并后保存在 output_dir/merged，供现有 benchmark 的模型路径参数使用。
不能把该自定义 adapter.pt 冒充 PEFT adapter_model.safetensors。

## 范围

目前支持单设备、非量化原始 Linear、一次 warm-up 后的结构变更。
不支持 ZeRO/FSDP 中途重构、QLoRA full-gradient probe、持续多次自适应、训练恢复、
几何优化。初次 probe 的所有目标梯度按原官方流程放在 CPU；第二次仅处理一层 G/R/SVD，
跨层只保留标量和新增小因子。所有第二次测量基于同一个 warm-up 状态，避免逐层修改
导致后面层测到不同模型。正式 MiniCPM-2B few-batch/GPU 验证和任务效果仍待完成。


## Oracle-relative decision（本次升级）

`tangent_metrics.py` 计算 `C_T=||G-R||²/(||G||²+eps)`，并独立检查 `1-rho`。
使用 float64 能量归约、eps=1e-30、容差2e-5；零梯度约定 C_T/oracle/efficiency=1、rho=0、keep。
同 rank oracle 使用 top-min(2r,valid) 的奇异值平方能量；side oracle 使用 top-r。
`direction_efficiency=C_T/oracle` 是主方向判据，CA/CB 的相对 oracle 利用率只用于定位 A/B/both。
默认 efficiency 阈值 .8、oracle 阈值 .9、residual 阈值 .1。它们是可配置诊断策略，不是理论常数。
当 oracle 和 efficiency 同时低，保守地先诊断 direction mismatch，并记录 mixed reason；
当 oracle/efficiency 已高但 residual 仍超阈值，记录 deferred keep，不宣称已满足 residual 目标。

Residual GID 只作为复杂度统计；新增 rank 由 `residual_energy_rank` 的累计 sigma² 最小 k 决定，
`--adaptive_residual_energy_target .90` 可修改。`residual_rank_k90` 始终记录真正90%结果，
另记配置目标对应的 `residual_rank_k_star`，两者不混用。
默认 `--adaptive_rank_init zero_b_residual_v`：A_add=Vh_R[:k]，B_add=0。
原有 A/B/scaling 完全保留；确定性 forward（eval/dropout关闭）扩展前后不变。
训练模式额外 dropout 分支会消耗随机数，不能据此要求整网后续随机 mask 完全一致。
`--adaptive_rank_init symmetric_residual` 只作非零函数扰动消融，仍使用 residual_init_scale。

`compute_B_target_given_A` / `compute_A_target_given_B` 从互补梯度计算目标；
JSON 记录 target_side、target_subspace 文件路径和 shape，实际小矩阵保存在 targets/*.pt。
默认不执行校正；blend 不是 basis-invariant 几何算法。尚未实现创新点3。

**分块限制**：n_split>1 时，以上 dense BA tangent/oracle 是 ambient relaxation；
它不是原分块参数的精确可达空间。继续记录 `restricted_block_residual_energy_ratio`，
不能把 dense efficiency 当成已经证明分块模型容量利用最优。

```bash
# 4步真实模型 smoke；输出目录已存在时拒绝覆盖，可另传 --output_dir
bash finetune/run_minicpm5_oracle_diagnostic.sh smoke
# 单条连续500步 warm-up轨迹，仅在2/20/100/500探测，不扩rank、不校正、不重置optimizer
bash finetune/run_minicpm5_oracle_diagnostic.sh trajectory
```

轨迹只使用现有9条训练演示样本，固定 calibration batch。保存每个 checkpoint 的逐层诊断，
以及 `warmup_trajectory.jsonl` 均值；probes 恢复 RNG、mode、requires_grad、grad，保持 optimizer state。
500步轨迹使用500步线性学习率计划，4步 smoke 使用4步计划，故二者step2不能当严格同 checkpoint。
此实验仅验证训练轨迹和工程行为，不构成任务效果或论文充分训练结论。


## Fixed calibration-8 trajectory（当前默认）

诊断脚本与CLI默认 adaptive_probe_batches=8。训练集按顺序选择前8个batch，
不重复采样、不循环补足；不足8批明确报错。output_dir/calibration_indices.json保存
训练来源、dataset长度、每批sample indices，每个checkpoint重读并校验SHA-256。
不使用validation/test。第一次rank初始化仍由gradient_est_n_steps控制（诊断脚本保持1），
仅第二次/trajectory probes改为8批，避免改变初始rank配置。

逐层probe仅backward、不step；adapter frozen；base weight逐层断言不变，恢复flags/grads/mode/RNG。
CPU梯度累积使用FP32，按批平均G=sum(G_k)/K，然后才做SVD/GID和tangent/residual分析。
loss沿用模型训练loss定义，probe采用eval模式关闭dropout，保存8个batch loss及其算术平均。
summary引用第一个模块probe的loss，同时记录各模块probe loss的最大差异（同样数据/模型重复forward）。

新增每层gradient/tangent/residual_fro_norm，summary含norm均值和gradient norm中位数、
mean_gid_full、mean_probe_loss、num_calibration_batches、manifest路径及hash。
保存max_residual_norm_consistency_error、max_energy_decomposition_error（绝对能量误差）及
max_relative_energy_decomposition_error（归一化误差，实际校验容差2e-5）。
零梯度时norm均为0、rho为0。全局energy是目标模块能量之和，global ratio是残余能量和除以梯度能量和；
其范围只覆盖所选q/v模块，不是全模型所有参数。不同层shape不同，不能仅凭跨层mean norm下论文结论。

```bash
nohup bash finetune/run_minicpm5_oracle_diagnostic.sh trajectory \
  --max_steps 500 --diagnostic_steps 2 20 100 500 \
  --adaptive_probe_batches 8 \
  --output_dir finetune/adaptive_lora/outputs/dense_trajectory_500_calib8 \
  > finetune/adaptive_lora/outputs/dense_trajectory_500_calib8.log 2>&1 &
```

trajectory设置adaptive_lora=false/direction_correction=none：只记录建议判定，不执行rank扩展或校正。
新默认输出是dense_trajectory_500_calib8，smoke是dense_smoke_calib8，不覆盖之前目录。


## 独立只读诊断 CLI

`finetune/diagnose_dense_lora.py` / `finetune/run_dense_diagnostic_suite.sh` 提供 initial、repeat、calibration、
checkpoint_k、oracle、random、drift、gaps实验入口。只做probe/离线数学计算，不包含optimizer或训练调用。
目前只通过静态语法检查，未运行这些新实验。目录存在时拒绝覆盖。

重要：已有dense_trajectory_500_calib8只保存最终adapter.pt；step2/20/100只有统计JSON，无法恢复权重。
独立dense_trajectory_100使用不同LR计划，不能冒充同一500步运行的step100。
后续trajectory默认save_diagnostic_checkpoints=true，将在checkpoints/step_N.pt保存每个诊断点的adapter。
这不会补回旧checkpoint。STEPS=500可先跑checkpoint_k/random；完整drift必须有同一次运行全部四个adapter。

Initial GID直接probe同一原始base model，不训练、不安装不同adapter；预算固定480256并继承source run的rank上下限。
各实验保存独立calibration_indices.json和diagnostic_config.json/source_run_config.json。
K敏感性使用同一manifest的前K批；calibration敏感性显式选择训练indices，当前9条样本的8样本组合高度重叠。
Repeat在同一加载checkpoint上每次重设相同seed，eval禁用dropout，逐批梯度CPU FP32累积后除K。
Oracle在临时矩阵上构造互补top-r左右子空间，不回写模型；超容差记录失败并非零退出。
Random每层100个独立Haar子空间样本，记录mean/std/empirical upper-tail；该p值未做多重比较校正，不能直接声称84层显著。
Gradient caches包含平均G、当前QA/QB、top-8数值有效U/V方向，四个时刻全84层约需数GB磁盘。
Drift输出逐层cosine、principal angles（弧度）、overlap和cross-time matrix；全局cosine按拼接目标梯度计算，
global tangent coverage按列梯度能量加权。未使用benchmark或test标签。
Gap使用每层DirectionGap=oracle-actual、CapacityGap=1-oracle，较大项用于dominant标签，近等时balanced；
全局按目标层梯度能量加权。该标签不宣称另一种gap不存在。


## Initial8 + diagnostic8（当前正式默认）

初始GID和periodic probe统一使用同一份calibration_indices.json：8个固定训练batch。
Dense入口校验gradient_est_n_steps == adaptive_probe_batches；默认均为8。
两个阶段均调用checkpoint_batches()重读相同索引并校验SHA-256。
G先按batch做FP32累加/算术平均，之后才做SVD/GID和原有Dense预算分配；分配算法未改。
480256预算不变，初始rank上限仍为4。LR、optimizer、seed、数据、benchmark和模型文件不变。

新增initial_probe/modules.jsonl、summary.json，连同budget.json、rank.json记录gid_full、
allocated_rank、rank分布、实际参数数、calibration indices、hash和batch数。
--initial_probe_only只执行初始probe/初始化和adapter保存，随后在创建optimizer之前退出。
历史章节中的initial K=1仅描述旧实验，不是当前正式默认。

```bash
# 只做initial sanity：不训练，不校正，不扩rank，使用独立新目录
bash finetune/run_minicpm5_oracle_diagnostic.sh trajectory \
  --initial_probe_only --output_dir finetune/adaptive_lora/outputs/dense_initial8_sanity_new
# 正式500步仍由用户手动启动
bash finetune/run_minicpm5_oracle_diagnostic.sh trajectory \
  --gradient_est_n_steps 8 --adaptive_probe_batches 8 \
  --max_steps 500 --diagnostic_steps 2 20 100 500 \
  --save_diagnostic_checkpoints true \
  --output_dir finetune/adaptive_lora/outputs/dense_trajectory_500_initial8_diag8
```
