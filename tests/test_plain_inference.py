"""CLI inputs must remain plain text, with exactly one added BOS."""
import torch
from infer import build_prompt, encode_prompt


class TokenizerStub:
    bos_token_id = 1

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        self.text = text
        return [3, 4, 5]

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError('Plain-text inference must not apply chat formatting')


def test_plain_cli_input_and_mask():
    tokenizer = TokenizerStub()
    inputs = encode_prompt(tokenizer, '안녕하세요.')
    assert tokenizer.text == '질문: 안녕하세요.\n답변:'
    assert inputs['input_ids'].tolist() == [[1, 3, 4, 5]]
    assert torch.equal(inputs['attention_mask'], torch.ones_like(inputs['input_ids']))


def test_instruction_is_plain_text_and_bos_can_be_absent():
    tokenizer = TokenizerStub()
    tokenizer.bos_token_id = None
    inputs = encode_prompt(tokenizer, '안녕하세요.', '짧게 답하세요.')
    assert tokenizer.text == '짧게 답하세요.\n\n질문: 안녕하세요.\n답변:'
    assert inputs['input_ids'].tolist() == [[3, 4, 5]]
    assert build_prompt('hello') == '질문: hello\n답변:'
