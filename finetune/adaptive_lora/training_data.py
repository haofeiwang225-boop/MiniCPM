"""Official complete-chat rendering for MiniCPM5 single-turn training examples."""
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset


class ChatTrainingDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.examples = []
        for index, row in enumerate(json.loads(Path(path).read_text())):
            messages = row['messages']
            if messages[-1]['role'] != 'assistant' or any(m['role'] == 'assistant' for m in messages[:-1]):
                raise ValueError(f'Expected one final assistant turn at training row {index}')
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            if not full.startswith(prefix):
                raise ValueError(f'Chat template prefix differs at row {index}')
            encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
            ids = encoded['input_ids'][:max_length]
            offsets = encoded['offset_mapping'][:max_length]
            labels = [token if start >= len(prefix) else -100 for token, (start, end) in zip(ids, offsets)]
            if not any(label != -100 for label in labels[1:]):
                raise ValueError(f'No supervised answer tokens after truncation at row {index}')
            padding = max_length - len(ids)
            self.examples.append({
                'input_ids': torch.tensor(ids + [tokenizer.pad_token_id]*padding),
                'labels': torch.tensor(labels + [-100]*padding),
                'attention_mask': torch.tensor([1]*len(ids)+[0]*padding),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
