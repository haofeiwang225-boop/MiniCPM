# 审计：MiniCPM / 本地官方 RaLoRA

审计日期：2026-09-27。官方副本 commit：
`55d870d89e85632f1df369857dbcf34ac48a9ec4`。
没有联网替换副本；以下差异是本地 working tree 对该 commit 的差异，
不是对当前远程 HEAD 的重新认证。本任务未修改官方副本。

## 现有训练路径

- `finetune/finetune.py:141`：HF 加载，`trust_remote_code=True`。
  `:186` 的 PEFT `LoraConfig` 对原始 MiniCPM-2B 使用 q_proj、v_proj，
  r=64，alpha=32，dropout=0.1，Gaussian A、零 B。
  MiniCPM3 的分支是 q_a_proj、kv_a_proj_with_mqa、q_b_proj、kv_b_proj。
- PEFT 参数名是 `lora_A.default.weight` / `lora_B.default.weight`，
  shape 分别是 `[r,d_in]` / `[d_out,r]`；scaling=alpha/r=0.5，
  forward=base(x)+scaling*B(A(dropout(x)))。当前 ralora/minicpm5 环境未安装
  PEFT；另一个已有 fine-tuning venv 的 `peft/tuners/lora/layer.py` 可核对该实现。
  本机尚未找到原始 MiniCPM-2B 权重，故不能将其真实层维度当作已实测结果。
- `finetune/lora_finetune.sh` 使用 DeepSpeed localhost:1、ZeRO3 offload，
  训练 3000 steps；SFT、OCNLI、DPO、MLX 和 LLaMA-Factory 另有入口。
  这些入口和 configs 下四份 ZeRO 配置均保留；Factory 的部分 YAML 指向 MiniCPM4，
  不能视作原始 MiniCPM-2B baseline。
- `finetune/data/AdvertiseGenChatML/train.json` 只有 9 条示例。
  此数据只适合工程验证，不应视作正式研究训练集。

## 官方路径和本地改动

`train/u_train.py`：load_model → setup_lora → prepare_lora →
set_up_trainable_param → optimizer/DeepSpeed → train。
`common/lora_modules/__init__.py:133` 的 prepare_lora 调用
`ralora.py:415` 的 ralora_reinit；动态结构初始化发生在 optimizer 之前。
主 RaLoRA 不是训练中持续调 rank，DRaLoRA 不混入本实现。

已有本地差异：

- grad_utils.py 新增 post-accumulate hook，ralora.py 改为调用它。
  原始 leaf weight 梯度积累到 CPU，记录 iters，清除 .grad。
- common/utils/utils.py 修正 DataCollator 的 attention_masks 读取。
- train/load_model.py 的 dtype 参数改为 torch_dtype。
- lora_modules/__init__.py 移除 rasa_moe 的导入。

已存在的 SVD profiling 代码属于该本地 commit 的内容，不是本任务新增。
默认 compute_effective_rank 仍执行 torch.linalg.svd，然后只返回有效秩；
get_allocated_rank 使用 mean(abs(W*G)) 分配预算；compute_n_split_allocations
使用有效秩与分配 rank 决定 2 的幂次 split。
这些公式、分配及第一次 probe 均直接复用。

官方基础层 `lora.py`：weight_a=[r,d_in]、weight_b=[d_out,r]，
scaling=alpha/r。RaLoRA 使用 `scaling_alpha / avg_rank`（默认），
开启 dynamic_scaling 才改用该层 mini rank；rank_stablize 再开平方。
这不是无条件 alpha/(mini_rank*n_split)。

`ralora.py:dynamic_init` 的 rank 是 **每块 rank**；
`lora_rank=mini_rank*n_split` 才是拼接后的代数宽度。
n_split>1 时 `for/concat` 使用 ParameterList：
weight_a.j=[mini_rank,d_in/n_split]，weight_b.j=[d_out/n_split,mini_rank]；
einsum 使用同样分块的三维 Parameter。实际更新是 block diagonal。
forward 复用官方层，支持原三种 forward_method；默认 concat 与官方 parser 相同。

## 接入决策与局限

选择 A：保留官方 layer / initialization，不把它改写成 PEFT。
新 `official.py` 调用 LoRAManager.create_lora_layer 和原始 ralora_reinit，
后续数学和阶段切换放在独立目录。模型 HF Linear 接口兼容，
但完整原始 MiniCPM-2B 仍须权重路径及少量 batch 实测才能确认。
官方 paths.json 当前仅配置 Llama-3.1-8B，不能直接拿现有脚本当 MiniCPM-2B 命令。

原始 finetune.py 不变；新增 train_adaptive_lora.py 作为单设备训练桥接。
这复用了官方 RaLoRA **算法**，不声称与官方 DeepSpeed trainer 每个数值结果相同。
脚本中的 rank 上下限、max split power、学习率是显式实验配置，不是论文默认值声明。

分块限制：对 block-diagonal A/B 计算用户指定的双边投影残余，
得到的是 unrestricted dense BA tangent 的残余，它可能低估真实 block 结构残余。
本实现不掩盖这个问题：记录 tangent_scope 和 restricted_block_residual_energy_ratio，
后者按各块真实 tangent 计算并保留 off-block 梯度。决策仍使用用户指定的 residual。
KEEP 保留结构；blend 限定在已有块内；EXPAND 保留整个原层并附加 dense residual 分支。
尚未引入基于真实 block residual 的另一套决策算法。

新分支 unit scaling 保证增加的有效更新等于 gamma*R_k，原分支 scaling 不变。
不能为了扩 rank 而重算旧 alpha/r，否则会改变全部已学更新。
增加参数的真实数量（而不只是代数 rank）会写入 summary。

数学解释边界：绝对 coverage 也受容量影响，所以低 coverage 不能严格证明
“只有方向错”；这是本次指定的启发式策略。rho <= min(1-C_A,1-C_B)
（理想数值下），高 coverage 与很高 rho 的阈值组合可能根本不可达。
默认阈值 .6/.1 只是可配置起点。Residual-GID 是谱复杂度，不是新增 rank 的定理下界。
正 gamma 使用用户给定的 +gradient 初始化，可能短时增加 loss；没有暗中翻转符号。

## 日志路径

官方 `common/utils/utils.py:520` 支持 LOG_FOLDER，默认 ~/MyTransformers_log。
新入口默认设为本次 output_dir/official_logs，使用官方已有配置接口，
不需要改官方代码或请求写项目外的日志目录。
