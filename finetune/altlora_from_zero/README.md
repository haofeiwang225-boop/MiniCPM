# 三组从零实验运行接口（审核通过后才执行）

本次没有执行下面的命令。必须先完成 CODE_REVIEW.md 中的输入/来源/许可证审核。

## 输入准备

根据 inputs.example.json 在仓库外/新文件中填写完整审核配置，不修改历史文件。
必须提供训练外独立probe；所有source/hash/revision都须真实核验。
不要把模板中的占位符当成有效配置。

## 顺序运行三组

```bash
cd /home/hongchang/whf/MiniCPM
CONFIG=/absolute/path/reviewed_inputs.json \
  bash finetune/altlora_trajectory/RUN_FROM_ZERO_2000.sh formal
```

后台启动（每次使用新的外部日志文件名）：

```bash
CONFIG=/absolute/path/reviewed_inputs.json nohup \
  bash finetune/altlora_trajectory/RUN_FROM_ZERO_2000.sh formal \
  > /tmp/altlora_from_zero_reviewed_run.log 2>&1 &
```

默认输出：finetune/altlora_trajectory/outputs/modes_from_zero_seed42_2000steps_20261108。
输出已存在时formal拒绝覆盖。共享初始化只生成一次。每组独立进程释放GPU。

```bash
ARM=altlora_false CHECKPOINT=/absolute/path/altlora_false/attempt_ID/checkpoint_200.pt \
  bash finetune/altlora_trajectory/RUN_FROM_ZERO_2000.sh resume
# 恢复的组完成后，尚未开始的下一组显式启动：
ARM=altlora_true bash finetune/altlora_trajectory/RUN_FROM_ZERO_2000.sh train-arm
bash finetune/altlora_trajectory/RUN_FROM_ZERO_2000.sh summarize
```

## 最终导出与评测

仅在三组正式训练完成后执行。需要按现有benchmarks/math500/serve_endpoint.sh部署导出权重，
不可将final_adapter.pt当PEFT加载。部署必须记录文件hash、tokenizer及实际采样限制。

```bash
export PYTHONPATH="$PWD/finetune:$PWD"
PY=/home/hongchang/miniconda3/envs/minicpm5/bin/python
OUT=finetune/altlora_trajectory/outputs/modes_from_zero_seed42_2000steps_20261108
$PY -B -m altlora_from_zero.export --root "$OUT" --arm vanilla
# 对altlora_false和altlora_true分别导出，Base直接使用原模型。
$PY -B -m altlora_from_zero.benchmark --root "$OUT" \
  --model-key vanilla --dataset gsm8k --endpoint http://127.0.0.1:8000/v1 \
  --served-model reviewed-model --server-manifest /absolute/path/reviewed_server.json
# 对base/vanilla/altlora_false/altlora_true和gsm8k/math500分别执行。
$PY -B -m altlora_from_zero.summarize --root "$OUT" --output "$OUT/summary_final"
```

server manifest格式：

```json
{
  "model_key": "vanilla",
  "served_model": "reviewed-model",
  "files": {"config.json": "SHA256", "...": "ALL exported file hashes from export_manifest"},
  "sampling": {"temperature": 1.0, "top_p": 0.95, "n": 1, "actual_max_model_len": "RECORD", "actual_max_tokens_default": "RECORD"}
}
```

Base的files必须等于config中的base_hashes。四个服务的sampling必须一致。
不限制/追加作者协议未规定的生成token上限；服务端实际上下文限制仍须记录。

## CPU验证

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=finetune:. \
 /home/hongchang/miniconda3/envs/minicpm5/bin/python -B -m unittest discover \
 -s finetune/altlora_from_zero/tests -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=finetune:. \
 /home/hongchang/miniconda3/envs/minicpm5/bin/python -B -m unittest discover \
 -s finetune/adaptive_lora/tests -p test_gid_validation.py -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=finetune:. \
 /home/hongchang/miniconda3/envs/minicpm5/bin/python -B -m unittest discover \
 -s finetune/altlora/tests -p test_historical_forward.py -v
```
