# 从零三模式2000步：第二阶段代码审查

基线：204ddc81a05d9d0655067df300e936d799fe3167。
分支：experiment/altlora-from-zero-2000-review。
本轮仅CPU合成测试/静态检查/JSON身份读取；没有加载MiniCPM或任何历史Adapter，没有CUDA运算、真实探针或正式评测。

## 实际调用链

RUN_FROM_ZERO_2000.sh formal
→ run.prepare（唯一一次新Adapter和G0）
→ 三个独立Python进程 run.train_arm(vanilla/altlora_false/altlora_true)。
每个进程重新加载同一Base，逐元素恢复 shared/initial_adapter.pt，新建优化器。
旧trajectory/run.py完全不调用；没有历史PRE加载入口。

prepare先做inputs.audit：固定PLAN_SHA256，另用固定HISTORY_SHA256核验旧run_config的真实480256预算字段；
读取plan的84条rank/shape/scaling记录，要求实际480000。不会读plan里记录的历史Adapter路径。
train.json实际记录数已经只读核验为2000；代码允许一般记录数，确定性逐epoch置换，截到恰好2000步。
所有arm读取同一已落盘顺序。明确不是历史PyTorch DataLoader顺序。

初始化复用 altlora.adapters.DenseAdapter：FP32 Kaiming A、零B、alpha/r、dropout=.1，cast→scale→add。
真实模型形状和84名称逐项核对。prepare保存Base logits；各arm在训练前验证同一初始参数和同一Base logits。
目前真实CUDA验证尚未执行，只通过CPU合成前向/梯度测试。

## 优化器与LR

Vanilla：torch.optim.AdamW(lr=5e-5,weight_decay=.01)。实际CPU环境torch2.5.1+cu121，
betas=(.9,.999)、eps=1e-8；runtime显式验证并记录defaults。
AltLoRA：现有 OfficialSchedule → OfficialAltLoRA →按rank分组的官方optimizer_new.py::altlora。
桥接参数沿用已有默认betas(.9,.999)、eps1e-4、reg1e-5、correct_bias=True；weight_decay明确覆写为.01。
官方核心未改。官方并非标准AdamW二阶矩公式；eps/beta2被接受不代表算法一定使用它们。

alt=False两侧始终活动。alt=True前9步both，10..19 A，20..29 B，以此重复。
整个2000步：每模块A活动1009步/冻结991步；B活动1000步/冻结1000步。
实际调用OfficialSchedule，并逐步检查其返回状态。冻结侧grad必须None、参数bitwise不变、state对象和Tensor version不变。
解冻不pop/reset动量。每步记录实际梯度范数、参数变化、优化器step；报告按rank汇总活动/冻结模块步数。

历史入口197..201行确认LambdaLR(max(0,1-step/steps))，无warmup。
第1次更新LR5e-5；第2000次更新用完成1999步的LR，完成后LR为0。global_step仅在opt.step成功后推进。
没有梯度裁剪/累积，batch1、masked mean、dynamic_length2048，非重入梯度检查点。

## RNG和恢复

每个训练步前通过 seed42+step 的SHA派生seed设置Python/NumPy/CPU/CUDA RNG；保存实际RNG状态文件及指纹。
这是新的共同逐步随机协议，不声称历史连续RNG重放。参数冻结不改变前向结构。
模型、探针数据、forward调用顺序也必须相同才能将相同RNG解释成相同dropout掩码；CUDA尚未验证。
测量用preserved事务恢复所有模式、requires_grad、grad对象及RNG；训练forward前再次设定该step共同RNG。

每200步及0步保存adapter、optimizer（含rank delegates）、LR scheduler、OfficialSchedule、step、order/cursor、RNG、identity。
临时文件fsync后原子无覆盖发布。每次resume使用新的attempt目录，不删除未提交日志。
检查点extra保存已提交日志谱系和上次动态梯度文件/哈希/时刻；恢复时复用它，不在off-grid重新定义参照。
只有report引用的成功谱系用于汇总。跨arm、rank、代码、配置、顺序、初始化身份均拒绝恢复。
结束前失败可从自己的2000步checkpoint重新完成最终验证；若final_adapter已发布但report失败，逐元素验证与恢复终点相同后复用，不覆盖。

## 梯度和指标

复用 global84_rank_direction.collect_weight_gradients_both 的CPU FP32双累积，每样本一次forward/backward、同时84模块。
所有probe为训练外固定原始问题，G0在prepare仅一次生成，各arm共用；动态梯度在0/200/400/800/1200/1600/2000测量。
复用gid_validation.coverage_diagnostics/actual_update。CT、CA、CB、残余能量、名义Top2r和活动rank(A)+rank(B)谱界分别保存。
B=0时E用活动谱界，零梯度明确undefined。固定G0和当前Gt，question/token，construction/audit都有独立字段。

实际Delta=s(Bnew Anew−Bold Aold)，核对一阶项+交叉项；alpha*=max(0,−<Delta,G>/||Delta||²)。
方向矩阵诊断只在规定阶段的最后一次更新记录，参照最近Construction探针。
**不是同一步训练批次梯度**：保存gradient_step、update_start/end、gradient_is_stale和定义字符串。
不把跨度200步的梯度参照伪称同一步梯度；Delta本身仍是一次更新，不是累计更新。
每步均保留A/B变化和梯度活动。NLL汇总使用独立训练侧probe，明确不是Dev400/正式测试成绩。

## 数据与输入配置

inputs.example.json是有意不完整的模板，不能直接正式运行。
需要人工指定并审核：独立probe聊天数据、原始MetaMath源、官方GSM8K/MATH500固定revision/hash、完整Base文件SHA清单。
不能复用历史probe，因为它来自当前完整训练集内部。
source_metadata对query/response精确匹配；检查NFKC/空白规范化original_question，训练/probe/test精确重复则停止。
不能证明语义改写的近重复完全不存在。没有自动改数据、重新抽样或下载。
官方GSM8K强制1319条、MATH500强制500条，版本/hash必须显式提供。
40GiB磁盘/28GiB空闲GPU是可配置预检下限，不是已验证峰值；CPU完整SVD可能成本较高。

## 评测调用链与限制

export.py：最终adapter→重新加载Base→合并Dense BA→同一训练题logits前后检查→HF导出。
BF16合并有舍入差异；固定atol=.05/rtol=.01，失败停止，不伪称bitwise一致。这一容差尚未真实验证，需人工审查。
benchmark.py：固定数据hash/revision→已有math500.build_request/score_response→逐题原始响应、解析、评分、tokens、停止原因。
GSM8K用官方answer的唯一####后文本作为参考答案；沿用现有boxed预测解析/数学比较，不称为官方原版评分器。
作者MATH设置thinking=true,temp1,top_p.95,n1，不新增max_tokens；记录服务端实际限制并在四模型之间验证一致。
seed42请求不等同于确定性生成；不自动重试。须三组2000步均PASS后才允许正式测试。
服务端manifest绑定导出文件/采样设置，但客户端无法远程证明服务端实际加载的权重，需要人工核查部署。
准确率Wilson CI；逐题paired bootstrap与exact McNemar；仅seed42，不代表跨训练seed证据。
正式评测必须单独启动已审核endpoint，不由训练脚本自动拉起vLLM。

## 依赖与来源

本次提交必要的既有altlora桥接、gid_validation、双梯度collector变更，均明确是复用之前未跟踪/未提交源码。
不提交现有grader修改、record_server修改、模型、数据、结果或日志。
官方optimizer源码SHA与上游commit94b219882abcaa6004d2029fd8a09cd190a8a9aa原始文件一致；ZIP注释同commit。
根LICENSE缺失，上游该commit的LICENSE返回404。保留原PEFT LICENSE但不冒称它覆盖AltLoRA核心。
这是尚未解决的许可证边界，不能声称上游核心具有已核实的开源授权条款。
详见third_party/altlora_official/PROVENANCE.json。

## 待人工审核、尚未验证

真实MiniCPM加载/显存/梯度/输出/训练、正式数据隔离审计、BF16合并容差、部署采样默认值均未验证。
没有正式NLL或Accuracy，没有方法收益结论。输入模板未绑定的新probe和正式测试文件必须先审核，当前不得直接启动正式训练。
CPU测试见TEST_RESULTS.md；不能把其PASS解释成CUDA或科学收益PASS。
