"""Extract final answers without using unfinished thinking as an answer."""

import re


def _boxed_at(text, start):
    match = re.match(r'\\boxed\s*\{', text[start:])
    if match is None:
        return None
    left = start + match.end()
    depth = 1
    for index in range(left, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
        if depth == 0:
            return text[left:index]
    return None


def extract_boxed_answer(text: str):
    matches = list(re.finditer(r'\\boxed\b', text))
    return _boxed_at(text, matches[-1].start()) if matches else None


def clean_answer(answer):
    if answer is None:
        return None
    answer = re.sub(r'<\|[^>]*\|>', '', answer).strip()
    for left, right in ((r'\[', r'\]'), (r'\(', r'\)'), ('$$', '$$'), ('$', '$')):
        if answer.startswith(left) and answer.endswith(right):
            answer = answer[len(left):-len(right)].strip()
    return answer or None


def _explicit_choice(text, problem):
    """Use only choices explicitly enumerated in the question, never the gold."""
    for group in re.findall(r'\(([^()]*)\)', problem):
        if not re.fullmatch(r'[A-Za-z ,\-]+', group) or not re.search(r'\bor\b', group):
            continue
        choices = [part.strip() for part in re.split(r',\s*|\s+or\s+', group) if part.strip()]
        if len(choices) < 2:
            continue
        sentence = re.split(r'[.!?\n]', text.strip(), maxsplit=1)[0]
        # Abstain on negation, uncertain claims, or sentences naming several choices.
        if re.search(r"\b(?:not|no|never|neither|nor|maybe|perhaps|might|could|either|if)\b|n't", sentence, re.I):
            continue
        mentioned = [choice for choice in choices
                     if re.search(r'(?<!\w)' + re.escape(choice) + r'(?!\w)', sentence, re.I)]
        if len(mentioned) == 1:
            plain = sentence.replace('**', '').strip()
            if (plain.casefold() == mentioned[0].casefold()
                    or re.search(r'\b(?:is|are|be|faces|facing|answer)\b', plain, re.I)):
                return mentioned[0]
    return None


def _explicit_person(text, problem):
    if not re.search(r'\bwho\b', problem, re.I):
        return None
    sentence = re.split(r'[.!?\n]', text.replace('**', '').strip(), maxsplit=1)[0]
    if re.search(r"\b(?:not|no|never|neither|nor|maybe|perhaps|might|could|either|if)\b|n't", sentence, re.I):
        return None
    if re.fullmatch(r"[A-Z][a-z]+(?:[ '-][A-Z][a-z]+)*", sentence) and re.search(
            r'(?<!\w)' + re.escape(sentence) + r'(?!\w)', problem):
        return sentence
    match = re.match(r"^([A-Z][a-z]+(?:[ '-][A-Z][a-z]+)*)\s+(?:would|will|is|has|does)\b", sentence)
    if match and re.search(r'(?<!\w)' + re.escape(match[1]) + r'(?!\w)', problem):
        return match[1]
    return None


def parse_answer(model_output: str, *, thinking_expected=False, problem=''):
    """Return answer and reason; the caller specifies whether thinking is enabled."""
    if '</think>' in model_output:
        text = model_output.rsplit('</think>', 1)[-1]
    elif thinking_expected or '<think>' in model_output:
        return {'answer': None, 'reason': 'unfinished_thinking'}
    else:
        text = model_output
    text = re.sub(r'<\|[^>]*\|>', '', text).strip()
    if not text:
        return {'answer': None, 'reason': 'empty_final_response'}
    markers = list(re.finditer(
        r'\\boxed\b|(?:final\s+answer\s*[:：]|(?:the\s+)?answer\s+is\s*[:：]?|answer\s*[:：])',
        text, re.IGNORECASE))
    if markers:
        marker = markers[-1]
        if marker.group().startswith('\\boxed'):
            answer = _boxed_at(text, marker.start())
            reason = 'boxed' if answer is not None else 'incomplete_box'
        else:
            tail = text[marker.end():].strip()
            answer = tail.splitlines()[0] if tail else None
            reason = 'answer_declaration'
        cleaned = clean_answer(answer)
        choice = _explicit_choice(cleaned, problem) if cleaned else None
        return {'answer': choice or cleaned, 'reason': 'explicit_question_choice' if choice else reason}
    choice = _explicit_choice(text, problem)
    if choice:
        return {'answer': choice, 'reason': 'explicit_question_choice'}
    person = _explicit_person(text, problem)
    if person:
        return {'answer': person, 'reason': 'explicit_person_answer'}
    # A single leading display followed by explanation is a common final-answer format.
    displays = list(re.finditer(r'\\\[(.*?)\\\]|\$\$(.*?)\$\$', text, re.DOTALL))
    if len(displays) == 1 and not text[:displays[0].start()].strip():
        tail = text[displays[0].end():].strip()
        if (not tail or re.match(r'^(?:Since|Because|This|Here|Indeed)\b', tail)) and not re.search(
                r'\b(?:instead|correction|actually|however)\b', tail, re.I):
            return {'answer': clean_answer(displays[0][1] or displays[0][2]), 'reason': 'final_display_math'}
    answer = text if text.startswith((r'\[', '$$')) else text.splitlines()[-1]
    return {'answer': clean_answer(answer), 'reason': 'final_text_fallback'}


def extract_answer(model_output: str, *, thinking_expected=False, problem=''):
    return parse_answer(model_output, thinking_expected=thinking_expected, problem=problem)['answer']
