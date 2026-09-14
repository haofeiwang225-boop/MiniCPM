"""Build the requested MMLU-Pro prompt without model special tokens."""

import re

try:
    from .utils import gold_answer
except ImportError:
    from utils import gold_answer


SYSTEM_PROMPT = 'You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`.'


def select_few_shots(sample, examples, num_shots):
    selected = []
    seen = set()
    for row in examples:
        if len(selected) == num_shots:
            break
        if row['category'] != sample['category']:
            continue
        # Exclude duplicated test questions even when IDs differ or are absent.
        question = ' '.join(row['question'].split()).casefold()
        if question == ' '.join(sample['question'].split()).casefold():
            continue
        if (sample.get('question_id') is not None
                and row.get('question_id') == sample['question_id']):
            continue
        if question in seen:
            continue
        seen.add(question)
        selected.append(row)
    if len(selected) != num_shots:
        raise ValueError(f'{sample["category"]}: need {num_shots} distinct non-leaking examples, found {len(selected)}')
    return selected


def question_block(row):
    options = '\n'.join(f'({chr(65 + i)}): {option}'
                        for i, option in enumerate(row['options']))
    return f'Q:\n{row["question"]}\n\nOptions are:\n{options}\n\nA:\nLet\'s think step by step.'


def example_block(row):
    # Dataset CoT already contains the A: / reasoning prefix and final answer.
    # Remove only duplicate wrappers; retain its reasoning text verbatim.
    cot = (row.get('cot_content') or '').strip()
    cot = re.sub(r'^A:\s*', '', cot)
    cot = re.sub(r"^Let's think step by step\.\s*", '', cot)
    cot = re.sub(r'\s*The answer is\s*\(?[A-J]\)?\.?\s*$', '', cot)
    parts = [question_block(row)]
    if cot:
        parts.append(cot)
    parts.append(f'The answer is ({gold_answer(row)}).')
    return '\n'.join(parts)


def build_messages(sample, examples, num_shots=5):
    if num_shots < 0:
        raise ValueError('num_shots must be nonnegative')
    shots = select_few_shots(sample, examples, num_shots)
    user = '\n\n'.join([*(example_block(row) for row in shots), question_block(sample)])
    return [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user}]
