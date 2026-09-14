"""
Evaluate local MMLU-Pro Parquet data using MiniCPM5 Transformers inference.

整体评测流程：

MMLU-Pro 数据集
        |
        v
读取 parquet 测试样本
        |
        v
构造 few-shot + 当前问题 prompt
        |
        v
MiniCPM5-2B 推理生成答案
        |
        v
从模型输出中提取 A-J 选项
        |
        v
与真实答案比较
        |
        v
计算 Accuracy

目标：
复现 MiniCPM5-2B 在 MMLU-Pro 上的 benchmark 结果。
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)


# ===============================
# 导入自己实现的 prompt 和评测工具
# ===============================

try:
    # 作为 package 运行
    from .prompt import build_messages
    from .utils import (
        extract_answer,
        gold_answer,
        summarize,
    )

except ImportError:
    # 直接 python evaluate.py 运行
    from prompt import build_messages
    from utils import (
        extract_answer,
        gold_answer,
        summarize,
    )


def main():

    # ==================================================
    # 1. 解析命令行参数
    #
    # 示例：
    #
    # python evaluate.py \
    #   --model /home/.../MiniCPM5-2B \
    #   --max_samples 100
    #
    # ==================================================

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # yaml配置文件
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).with_name('config.yaml')
    )

    # 模型路径
    parser.add_argument(
        '--model'
    )  #命令行参数  就是执行命令的时候可以指定模型路径

    # MMLU-Pro 数据路径
    parser.add_argument(
        '--data_path',  #命令行参数  执行命令的时候指定数据路径
        type=Path,
        help='Local data directory or test Parquet file'
    )

    # debug模式，例如只跑100条
    parser.add_argument(
        '--max_samples',
        type=int
    )

    # 输出目录
    parser.add_argument(
        '--output_dir',
        type=Path
    )


    args = parser.parse_args()


    # ==================================================
    # 2. 加载 yaml 配置
    #
    # config.yaml 保存：
    #
    # model
    # data_path
    # few-shot设置
    # generation参数
    #
    # ==================================================

    config = yaml.safe_load(
        args.config.read_text()
    )


    # 命令行参数优先级高于 yaml
    # 如果用户命令行指定，则覆盖配置文件

    for key in (
        'model',
        'data_path',
        'max_samples',
        'output_dir'
    ):

        value = getattr(args, key)

        if value is not None:

            # Path 转字符串方便保存 yaml
            if isinstance(value, Path):
                value = str(value)

            config[key] = value



    # ==================================================
    # 3. 检查 max_samples 参数
    #
    # 用于debug：
    #
    # max_samples=100
    #
    # 只测试100条
    #
    # ==================================================

    limit = config['max_samples']

    if limit is not None:

        if (
            not isinstance(limit, int)
            or limit <= 0
        ):
            parser.error(
                'max_samples must be a positive integer or null in YAML'
            )


    # ==================================================
    # 4. 定位 MMLU-Pro 数据文件
    #
    # 支持：
    #
    # data_path/
    #       test.parquet
    #
    # 或：
    #
    # data_path/test.parquet
    #
    # ==================================================

    data_path = (
        Path(config['data_path'])
        .expanduser()
        .resolve()
    )


    # 如果传入的是具体文件
    # 直接使用

    # 如果传入目录
    # 拼接 test_file

    test_path = (
        data_path
        if data_path.is_file()
        else data_path / config['test_file']
    )


    # few-shot 示例文件

    examples_path = (
        test_path.parent
        /
        config['few_shot_file']
    ).resolve()



    # ==================================================
    # 5. 防止数据泄漏
    #
    # 禁止：
    #
    # test作为few-shot
    #
    # 否则模型提前看到答案
    #
    # ==================================================

    if (
        config['num_shots']
        and examples_path == test_path.resolve()
    ):

        raise ValueError(
            'few_shot_file must be a separate training/validation file'
        )


    # ==================================================
    # 6. 读取 MMLU-Pro parquet
    #
    # 每个sample类似：
    #
    # {
    #   question,
    #   options,
    #   answer,
    #   category
    # }
    #
    # ==================================================

    samples = (
        pq.read_table(test_path)
        .to_pylist()
    )


    # debug限制数量

    if limit is not None:

        samples = samples[:limit]


    if not samples:

        raise ValueError(
            'Test data is empty'
        )


    # ==================================================
    # 7. 读取 few-shot 示例
    #
    # 例如：
    #
    # validation.parquet
    #
    # ==================================================

    examples = []

    if config['num_shots']:

        examples = (
            pq.read_table(examples_path)
            .to_pylist()
        )



    # ==================================================
    # 8. 在加载模型之前检查数据
    #
    # 避免模型跑几个小时后发现：
    #
    # answer格式错误
    # prompt构造失败
    #
    # ==================================================

    for sample in samples:

        # 检查真实答案是否有效
        gold_answer(sample)

        # 检查prompt是否能够生成
        build_messages(
            sample,
            examples,
            config['num_shots']
        )



    # ==================================================
    # 9. 创建输出目录
    #
    # 保存：
    #
    # results.jsonl
    # summary.json
    # run_config.yaml
    #
    # ==================================================

    output_dir = Path(
        config['output_dir']
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    output_path = (
        output_dir
        /
        'results.jsonl'
    )


    # 防止覆盖历史实验

    with output_path.open(
        'x',
        encoding='utf-8'
    ) as output_file:


        # 保存本次实验配置

        (
            output_dir / 'run_config.yaml'
        ).write_text(
            yaml.safe_dump(config),
            encoding='utf-8'
        )


        # ==================================================
        # 10. 固定随机种子
        #
        # 保证实验可重复
        #
        # ==================================================

        set_seed(
            config['seed']
        )



        # ==================================================
        # 11. 加载 MiniCPM5-2B
        #
        # 官方 Transformers 加载方式
        #
        # ==================================================

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                config['model']
            )
        )


        model = (
            AutoModelForCausalLM
            .from_pretrained(
                config['model'],
                torch_dtype='auto',
                device_map='auto'
            )
        )

        #把模型切换到“评估模式（evaluation mode）”，关闭训练时的一些随机行为。
        model.eval()



        results = []

        total_generated_tokens = 0



        # ==================================================
        # 12. 遍历所有测试问题
        #
        # 每一个sample:
        #
        # prompt
        #   |
        # MiniCPM5
        #   |
        # prediction
        #
        # ==================================================

        for index, sample in enumerate(samples):


            # 构造 MMLU-Pro prompt

            messages = build_messages(
                sample,
                examples,
                config['num_shots']
            )


            # ==================================================
            # 使用 MiniCPM5 官方 chat template
            #
            # 得到模型真实输入token
            #
            # enable_thinking=True:
            # 开启<think>推理过程
            #
            # ==================================================

            inputs = (
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=True,
                    return_dict=True,
                    return_tensors='pt',
                )
                .to(model.device)
            )


            print(
                f'[{index + 1}/{len(samples)}] '
                f'prompt_tokens={inputs["input_ids"].shape[-1]}',
                flush=True
            )


            # ==================================================
            # MiniCPM5生成答案
            #
            # generation参数来自config.yaml
            #
            # ==================================================

            with torch.inference_mode():

                outputs = model.generate(
                    **inputs,
                    **config['generation']
                )



            # 去掉输入prompt，只保留生成内容

            generated = (
                outputs[0]
                [
                    inputs['input_ids'].shape[-1]:
                ]
            )


            total_generated_tokens += (
                generated.shape[-1]
            )


            # token -> 文本

            model_output = tokenizer.decode(
                generated,
                skip_special_tokens=False
            )



            # ==================================================
            # 从模型输出中提取答案
            #
            # 例如：
            #
            # The answer is (D)
            #
            # 提取：
            #
            # D
            #
            # ==================================================

            prediction = extract_answer(
                model_output
            )


            # 获取真实答案

            gold = gold_answer(
                sample
            )


            # 判断正确

            result = {

                'question_id':
                    sample.get(
                        'question_id',
                        index
                    ),

                'category':
                    sample['category'],

                'gold_answer':
                    gold,

                'prediction':
                    prediction,

                'correct':
                    prediction == gold,

                'model_output':
                    model_output,
            }


            # 保存单条结果

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                + '\n'
            )

            output_file.flush()


            results.append(
                result
            )


            print(
                f'[{index + 1}/{len(samples)}] '
                f'question_id={result["question_id"]} '
                f'prediction={prediction}',
                flush=True
            )



    # ==================================================
    # 13. 汇总实验结果
    #
    # Accuracy:
    #
    # correct / total
    #
    # ==================================================

    summary = summarize(
        results
    )


    summary.update({

        'model':
            config['model'],

        'num_samples':
            len(results),

        'num_shots':
            config['num_shots'],

        'generation':
            config['generation'],

        'enable_thinking':
            True,
    })



    (
        output_dir / 'summary.json'
    ).write_text(
        json.dumps(
            summary,
            indent=2
        )
        + '\n',
        encoding='utf-8'
    )



    print(
        f'Total samples: {summary["total"]}\n'
        f'Correct: {summary["correct"]}\n'
        f'Accuracy: {summary["accuracy"]:.6f}'
    )


    print(
        f'Average generated tokens: '
        f'{total_generated_tokens / len(results):.2f}'
    )


    print(
        f'Results: {output_path.resolve()}'
    )



if __name__ == '__main__':

    main()