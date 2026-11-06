"""Use the exact checkpoint tokenizer with BFCL's Qwen tool-call parser."""
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from overrides import override


class Qwen35Handler(QwenFCHandler):
    @override
    def _format_prompt(self, messages, function):
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
            tools=[{'type': 'function', 'function': f} for f in function] or None,
            add_generation_prompt=True, enable_thinking=False)
