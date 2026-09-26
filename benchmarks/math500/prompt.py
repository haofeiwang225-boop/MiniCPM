"""Author-specified zero-shot, single-turn MATH-500 prompt."""


def build_messages(problem: str):
    return [{'role': 'user', 'content': problem + '\n' +
             r'Please reason step by step, and put your final answer within \boxed{}.'}]
