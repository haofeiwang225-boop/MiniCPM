# Gradient Subspace Matching + Residual-GID Adaptive LoRA

只实现创新点1；没有 Stiefel、Oblique、SPD 或 manifold optimization。
先阅读 [审计结论](AUDIT.md)，特别是 MELoRA 分块 tangent 的限制。

## 代码对应关系

| 文件 | 职责 |
|---|---|
| official.py | 接入本地官方类、manager、原始第一次 probe/分配/初始化 |
| gradient_probe.py | 第二次 probe；逐层暂开 base weight 梯度；不 step；恢复 RNG/mode/flags/grad |
| subspace_matching.py | 秩感知 QR+小 R 的 SVD；C_A/C_B 的 sigma² 加权覆盖 |
| residual_gradient.py | 无 identity 的双边残余；另记真实 block tangent 残余 |
| residual_gid.py | FP32 economy SVD + 与官方相同的奇异值 L1 熵有效秩 |
| decision.py | keep/direction_mismatch/expand；独立 ceil(GID_R) 与预算限制 |
| direction_correction.py | 可关闭的温和 Procrustes 对齐 blend；未来几何方法接口 |
| rank_expansion.py | 保留原官方层；追加 sqrt(gamma)*U_R*sqrt(S_R) 等因子 |
| adaptation.py | 同一 warm-up 状态逐层分析，延后统一修改，控制峰值内存 |
| diagnostics.py | modules.jsonl、summary.json、rank_pattern.json |
| checkpoint.py | adapter 保存/恢复、合并为普通 HF Linear，不改 benchmark |
| ../train_adaptive_lora.py | 共用官方初始化、两阶段训练及全新 optimizer/scheduler |
| tests/test_adaptive.py | 数学、probe 不变性、参数保留、训练、恢复与合并测试 |

与官方 SVD 的差别只在第二次分析：使用 full_matrices=False，保留 U/S/Vh；
奇异值熵定义不变，零残余定义为 GID=0。第一次完全调用原始官方函数。
不是 randomized SVD，也不把数值加速作为创新。B=0/秩亏不允许 QR 补出的假方向影响 coverage。

扩展后实际 delta 为 `old_scaling*B_old*A_old + B_add*A_add`。
旧层/分块/A/B/scaling 保留；增加分支的 scaling=1，使新增实际更新为 `gamma*R_k`。
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
  --adaptive_alignment_threshold 0.6 --adaptive_residual_energy_threshold 0.1 \
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
