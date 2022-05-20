import os

import torch
import torch.nn.functional as F

from transformers import MarianConfig

from transformers.generation_utils import GenerationMixin
from transformers.generation_logits_process import LogitsProcessor

from core.utils import create_model_for_provider


class CustomLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0.
    Disable PAD token.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`int`):
            The id of the *end-of-sequence* token.
        pad_token_id (`int`):
            The id of the *padding* token.
    """

    def __init__(self, min_length: int, eos_token_id: int, pad_token_id: int):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a positive integer, but is {min_length}")

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        if not isinstance(pad_token_id, int) or pad_token_id < 0:
            raise ValueError(f"`pad_token_id` has to be a positive integer, but is {pad_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            scores[:, self.eos_token_id] = -float("inf")

        scores[:, self.pad_token_id] = -float("inf")
        return scores


class MarianOnnx(GenerationMixin):
    def __init__(self, path: str, device: str = 'cpu'):
        self.device = torch.device(device)

        self.encoder_path = os.path.join(path, "encoder.onnx")
        self.decoder_path = os.path.join(path, "decoder.onnx")

        self.config = MarianConfig.from_pretrained(path)
        self.config.force_bos_token_to_be_generated = False

        self.final_logits_weight = torch.load(os.path.join(path, 'lm_weight.bin')).to(self.device)
        self.final_logits_bias = torch.load(os.path.join(path, 'lm_bias.bin')).to(self.device)

        self.logits_processor = CustomLogitsProcessor(
            2, self.config.eos_token_id, self.config.pad_token_id
        )

        provider = "CPUExecutionProvider" if device == 'cpu' else "CUDAExecutionProvider"

        self.encoder_session = create_model_for_provider(self.encoder_path, provider)
        self.decoder_session = create_model_for_provider(self.decoder_path, provider)

    def _init_sequence_length_for_generation(self, input_ids, max_length: int):
        unfinished_sequences = \
            torch.zeros(input_ids.shape[0], dtype=torch.int8, device=input_ids.device) + 1

        sequence_lengths = \
            torch.zeros(input_ids.shape[0], dtype=torch.int8, device=input_ids.device) + max_length

        cur_len = input_ids.shape[-1]
        return sequence_lengths, unfinished_sequences, cur_len

    def _encoder_forward(self, input_ids, attention_mask):
        inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
            "attention_mask": attention_mask.cpu().detach().numpy(),
        }
        last_hidden_state = self.encoder_session.run(None, inputs)[0]
        last_hidden_state = torch.from_numpy(last_hidden_state).to(input_ids.device)
        return last_hidden_state

    def _decoder_forward(self, input_ids, encoder_output, attention_mask):
        inputs = {
            "input_ids": input_ids.cpu().detach().numpy(),
            "encoder_hidden_states": encoder_output.cpu().detach().numpy(),
            "attention_mask": attention_mask.cpu().detach().numpy(),
        }
        decoder_output = self.decoder_session.run(None, inputs)[0]
        decoder_output = torch.from_numpy(decoder_output).to(input_ids.device)

        lm_logits = F.linear(decoder_output, self.final_logits_weight, bias=self.final_logits_bias)

        return lm_logits

    def greedy_search(self, input_ids, encoder_output, attention_mask):
        max_length = self.config.max_length
        pad_token_id = self.config.pad_token_id
        eos_token_id = self.config.eos_token_id

        sequence_lengths, unfinished_sequences, cur_len = \
            self._init_sequence_length_for_generation(input_ids, max_length)

        while cur_len < max_length:
            logits = self._decoder_forward(input_ids, encoder_output, attention_mask)
            next_token_logits = logits[:, -1, :]

            scores = self.logits_processor(input_ids, next_token_logits)

            next_tokens = torch.argmax(scores, dim=-1)
            next_tokens = \
                next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            input_ids[input_ids[:, -2] == eos_token_id, -1] = eos_token_id
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).char())

            if unfinished_sequences.max() == 0:
                break

            cur_len = cur_len + 1
        return input_ids

    def _prepare_decoder_input_ids_for_generation(self, input_ids, decoder_start_token_id : int):
        decoder_input_ids = (
            torch.ones((input_ids.shape[0], 1), dtype=torch.int64, device=input_ids.device)
            * decoder_start_token_id
        )
        return decoder_input_ids

    def generate(self, input_ids, attention_mask):
        decoder_start_token_id = self.config.decoder_start_token_id

        encoder_output = self._encoder_forward(input_ids, attention_mask)

        input_ids = self._prepare_decoder_input_ids_for_generation(
            input_ids,
            decoder_start_token_id=decoder_start_token_id,
        )

        return self.greedy_search(input_ids, encoder_output, attention_mask)
