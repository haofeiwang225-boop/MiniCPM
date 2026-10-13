# GID + Dense LoRA 切换报告

## 修改清单

| 文件 | 用途 |
|---|---|
| dense_init.py（新增） | 官方普通LinearWithLoRA接入；第一次逐层gradient probe；GID加权、实际参数预算分配 |
| official.py | dense_factors识别普通Dense层，直接返回二维A/B；旧block函数保留 |
| adaptation.py | Dense分支不走block结构诊断；现有tangent/residual/zero-B扩展保留 |
| diagnostics.py | 删除num_block_relaxation，Dense rank语义和日志 |
| checkpoint.py | Dense adapter保存、恢复和原版block格式兼容；Dense不保存split/mini_rank |
| ../train_adaptive_lora.py | 默认dense；结构选择、预算参数和Dense第一次probe路由 |
| ../run_minicpm2b_adaptive_lora.sh | adaptive默认dense |
| ../run_minicpm2b_official_ralora.sh | 显式block，保持原官方baseline入口 |
| ../run_minicpm5_oracle_diagnostic.sh | Dense smoke/trajectory命令、480256预算、新输出目录 |
| tests/test_dense_init.py（新增） | 实际预算、GID分配、禁用block_diag、probe不变性、二维A/B、训练及保存恢复合并 |
| README.md | 当前默认设置、历史结果边界、运行命令 |
| dense_budget_preview.json（新增） | 基于历史初始GID的离线参数预算预演 |

## 数学与预算

GID使用原有svd_with_gid：奇异值L1归一化熵的指数。
普通A=(r,in)、B=(out,r)，每层参数数r*(in+out)。
以GID决定连续rank比例，解满足预算的scale，再受上下限约束、取整、按相对连续目标的缺额分配可用预算。
零梯度层保留最小rank。所有GID来自training/calibration，未使用benchmark或测试标签。
不再调用官方RaLoRA的n_split计算；这属于GID Dense变体，不声称等同于原版RaLoRA分配算法。
初始scaling为alpha/r；第二阶段保持已有scaling。Dense下tangent投影和top-2r oracle有对应的稠密参数空间，
不再需要之前block-relaxation解释。

旧block smoke实测：480256参数。
Dense命令初始预算：480256参数。
历史初始GID离线预演：478976参数（20层rank1、64层rank2，余1280）。
本次没有运行真实模型新probe，不能把预演当成新实测；新结果将写入initial_probe/budget.json。
初始预算不约束显式的第二阶段自适应扩rank；该操作仍由Residual energy k、step cap和max rank控制。

## 验证

29项unittest全部通过；包含原有27项数学/probe/扩展/保存测试和2项Dense集成测试。
新Dense集成测试将torch.block_diag替换为抛异常，完整Dense probe/诊断/恢复合并仍通过。
CPU tiny Llama 4步集成运行完成，包含两次probe、阶段边界保存、第二阶段训练和最终adapter。
未启动MiniCPM5-2B GPU smoke或500步trajectory，未修改官方RaLoRA副本、模型文件、benchmark或数据。
CPU集成结果：outputs/dense_cpu_integration。

## 运行

```bash
bash finetune/run_minicpm5_oracle_diagnostic.sh smoke
bash finetune/run_minicpm5_oracle_diagnostic.sh trajectory
```

分别输出outputs/minicpm5_dense_smoke、outputs/minicpm5_dense_trajectory。
运行时仍可通过末尾--output_dir指定其他新目录。原始block代码和旧结果保留，不覆盖。
