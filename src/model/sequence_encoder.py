import torch
import torch.nn as nn
from typing import List
from transformers import EsmTokenizer, EsmModel


class SequenceEncoder(nn.Module):
    def __init__(self, model_path: str, model_max_length: int = 2048):
        super(SequenceEncoder, self).__init__()
        self.model_path = model_path
        self.model_max_length = model_max_length
        self.tokenizer = EsmTokenizer.from_pretrained(self.model_path, model_max_length=self.model_max_length)
        self.model = EsmModel.from_pretrained(self.model_path)

    @staticmethod
    def remove_start_end_tokens(representation: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, length, dim = representation.shape
        mask = torch.ones_like(attention_mask, dtype=torch.bool)

        for i in range(batch_size):
            first_one_idx = (attention_mask[i] == 1).nonzero(as_tuple=True)[0][0].item()
            last_one_idx = (attention_mask[i] == 1).nonzero(as_tuple=True)[0][-1].item()

            mask[i, first_one_idx] = False
            mask[i, last_one_idx] = False

        filtered_representation = representation[mask].view(batch_size, length - 2, dim)
        filtered_attention_mask = attention_mask[mask].view(batch_size, length - 2)

        return filtered_representation, filtered_attention_mask

    def forward(self, sequences: List[str]):
        inputs = self.tokenizer(sequences, return_tensors='pt', padding=True, truncation=True).to(self.model.device)
        outputs = self.model(**inputs)
        representation = outputs['last_hidden_state'].detach().cpu()
        attention_mask = inputs['attention_mask'].detach().cpu()
        new_representation, new_attention_mask = self.remove_start_end_tokens(representation, attention_mask)
        return {
            'representation': new_representation,
            'attention_mask': new_attention_mask
        }
