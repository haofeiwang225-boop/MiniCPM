"""Local data validation, answer extraction, and micro accuracy."""

import re


def gold_answer(row):
    answer = str(row['answer']).strip().upper()
    index = row['answer_index']
    if not 0 <= index < len(row['options']) <= 10:
        raise ValueError(f'Invalid answer_index/options: {row.get("question_id")}')
    if answer != chr(ord('A') + index):
        raise ValueError(f'answer disagrees with answer_index: {row.get("question_id")}')
    return answer


def extract_answer(text):
    """Take the last standalone uppercase A-J (never a letter inside a word).

    This includes `The answer is (A)`, `The answer is A`, and `Answer: A`.
    When thinking has closed, only the final response is eligible.
    """
    text = text.rsplit('</think>', 1)[-1] #如果模型有思考过程，只取最后答案部分。
    text = re.sub(r'<\|[^>]*\|>', '', text)#删除特殊 token。
    matches = re.findall(r'(?<![\w])([A-J])(?![\w])', text)#找到单独出现的 A-J 字母。
    return matches[-1] if matches else None


def summarize(results):
    total = len(results)
    correct = sum(bool(row['correct']) for row in results)
    return {'total': total, 'correct': correct,
            'accuracy': correct / total if total else 0.0}
