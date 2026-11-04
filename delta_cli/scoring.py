"""Public scoring rules shared by local validation and the trusted evaluator."""
import re


def score_answer(example, text):
    # Thinking traces do not count as the final answer.
    answer = text.split("</think>")[-1].strip()
    scorer = example.get("scorer", "exact")
    if scorer == "exact":
        def normalize(s):
            return re.sub(r"\s+", " ", str(s).strip()).casefold()
        return float(normalize(answer) == normalize(example["answer"]))
    if scorer == "choice":
        boxed = re.findall(r"\\boxed\{([A-E1-5])\}", answer)
        matches = boxed or re.findall(r"(?:^|\n)\s*([A-E1-5])[.)]?\s*$", answer)
        return float(bool(matches) and matches[-1] == str(example["answer"]))
    if scorer == "math":
        from math_verify import parse, verify
        from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
        gold = parse(str(example["answer"]), extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        pred = parse(answer, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
        return float(bool(gold) and bool(pred) and verify(gold, pred))
    if scorer == "ifeval":
        from lm_eval.tasks.ifeval.instructions_registry import INSTRUCTION_DICT
        passed = []
        for instruction_id, kwargs in zip(example["instruction_id_list"], example["kwargs"], strict=True):
            instruction = INSTRUCTION_DICT[instruction_id](instruction_id)
            instruction.build_description(**{k: v for k, v in kwargs.items() if v is not None})
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=example["prompt"])
            passed.append(bool(answer) and instruction.check_following(answer))
        return float(bool(passed) and all(passed))
    raise ValueError("unknown_scorer")

