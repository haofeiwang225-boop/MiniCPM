# 从零三模式2000步：第一阶段协议审查

状态：仅完成协议代码及测试编写；测试未执行；没有模型/梯度/训练/评测运行。
不是完整实验实现，不能启动6000步实验。启动脚本对 formal/summarize 均退出2。

## 基线与修改范围

基于 ee2e2f04bc771707f7742f3a1a456218535b189a。
独立分支 experiment/altlora-from-zero-2000-review。
旧 altlora、altlora_trajectory 及官方第三方代码在当前仓库均是未跟踪内容，
本提交不擅自包含它们、历史产物或已有 grader/global84 修改。
因此GitHub上的本阶段是可独立阅读的标准库协议模块，不是完整运行依赖包。

## 静态历史依据

- train_adaptive_lora.py:197-201：torch.optim.AdamW，仅显式传入lr和weight_decay；
  LambdaLR(max(0,1-step/steps))，无warmup。betas/eps未显式设置，依赖所用PyTorch默认值。
  后续运行前需绑定安装版本并核对实际defaults，不在此冒称已核验真实环境。
- 官方 logtrainer.py:86-92：counter=0、alter_freq=10、update_A=True。
  231-242在forward前递增并于10倍数切换；本协议累计步数=1。
- 已有 OfficialSchedule.prepare 实现同样状态机。新增 expected_side 仅用于审计预期，
  不是另一套替代官方调度器，正式运行仍必须调用现有桥接。
- 旧trajectory/run.py强绑定1600 PRE和400步。不能将它改参数后声称从零。
- 旧update_audit.py硬编码1601..2000和33600条记录，不适用于本任务。

## 新增接口

protocol.read_rank_inventory：只读JSON，不打开其中的adapter路径。
调用者必须提供plan预期SHA256；校验84个完整名称、逐层形状、rank、scaling、
成本、q/v直方图、实际480000及480256预算声明。没有重新GID计算或分秩。

training_order：固定真实train SHA256和2000记录数；本地Python Random(42)一次
无放回置换，保存全2000步索引及逐条canonical内容哈希；不消耗全局RNG。
这是新的共同顺序，不声称恢复历史数据顺序。执行器尚未实现，当前不生成文件。

learning_rate：返回已完成k次更新后下一次更新应使用的LR。
第1次使用5e-5，第2000次使用5e-5*(1-1999/2000)，之后scheduler归零。

validate_resume_identity：跨arm、初始adapter、rank、数据顺序、基座、代码或协议
不一致均报错；这只是恢复身份前置契约，不是完整optimizer/RNG恢复实现。

require_fresh_output：仅检查目录冲突，不创建目录。

## 测试（未执行）

tests/test_protocol.py：人工inventory及错误哈希/形状/rank/scaling拒绝；
2000步审计状态机逐步对照；LR边界；跨模式恢复拒绝；输出冲突拒绝。
这些测试不加载模型，但本轮遵循“不要运行，先审核代码”，没有执行。
将来审查授权后可用：

```
PYTHONPATH=finetune python -B -m unittest discover -s finetune/altlora_from_zero/tests -v
```

## 下一阶段必须完成的内容

1. 只生成一次新的Kaiming A/零B，逐元素恢复到三组；Base/初始adapter输出对照。
2. 训练、OfficialSchedule及官方optimizer完整连接；冻结参数/动量审计。
3. 新探针与整个训练集/GSM8K/MATH500原始问题隔离，缺来源时fail closed。
4. 固定G0和动态双梯度、B=0有效谱界；同一步梯度/更新配对，不能沿用旧stale参照标单步。
5. 每200步完整原子检查点、RNG/数据游标/日志截断恢复及错误报告。
6. 正式基准数据来源/评分器锁定、三组及Base汇总、配对统计。
7. 对所有复用但未跟踪依赖整理单独审查提交，防止GitHub缺依赖。

没有测试日志、模型初始化证据、完整恢复测试结果或有效正式启动命令。
目标输出仅为约定，未创建：
finetune/altlora_trajectory/outputs/modes_from_zero_seed42_2000steps_20261108/
