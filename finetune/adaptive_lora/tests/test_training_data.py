import json
import tempfile
import unittest
from pathlib import Path
from adaptive_lora.training_data import ChatTrainingDataset


class CharacterTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = 'BOS' + ''.join(m['role'] + ':' + m['content'] + ';' for m in messages)
        return text + ('assistant:' if add_generation_prompt else '')

    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(c) for c in text],
                    offset_mapping=[(i, i+1) for i in range(len(text))])


class TrainingDataTests(unittest.TestCase):
    def test_mask_prompt_and_padding(self):
        messages = [{'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': 'A'}]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'train.json'
            p.write_text(json.dumps([{'messages': messages}]))
            dataset = ChatTrainingDataset(p, CharacterTokenizer(), 40)
            example = dataset[0]
            labels = example['labels']
            self.assertEqual(labels[labels != -100].tolist(), [ord('A'), ord(';')])
            self.assertTrue((labels[example['attention_mask'] == 0] == -100).all())

    def test_reject_fully_truncated_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'train.json'
            p.write_text(json.dumps([{'messages': [{'role': 'user', 'content': 'Q'},
                                                  {'role': 'assistant', 'content': 'A'}]}]))
            with self.assertRaises(ValueError):
                ChatTrainingDataset(p, CharacterTokenizer(), 4)


if __name__ == '__main__':
    unittest.main()
