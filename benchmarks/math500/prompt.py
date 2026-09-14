"""MATH-500 模型输入只包含原始 problem，不添加其他提示词。"""


def build_messages(problem: str):
    return [{'role': 'user', 'content': problem}]
