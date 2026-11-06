"""Protocol names refer to lm-eval 0.4.13 / BFCL 2025.12.17, not custom approximations."""
SUITE_ID = 'qwen-small-public-v1'
SEED = 20261104
TEXT = {
    'gsm8k': ('gsm8k', 64),
    'math500': ('minerva_math500', 64),
    'arc_easy': ('arc_easy', 64),
    'arc_challenge': ('arc_challenge', 64),
    'hellaswag': ('hellaswag', 64),
    'piqa': ('piqa', 64),
    'winogrande': ('winogrande', 64),
    'boolq': ('boolq', 64),
    'openbookqa': ('openbookqa', 64),
    'commonsense_qa': ('commonsense_qa', 64),
    # Cover every subject/task instead of sampling only the first few subjects.
    'mmlu': ('mmlu', 114),
    'mmlu_pro': ('mmlu_pro', 70),
    'bbh': ('bbh', 54),
    'ifeval': ('ifeval', 64),
}
CODE = {'humaneval': 32, 'mbpp': 32}
BFCL = {
    'simple_python': 12, 'multiple': 12, 'parallel': 12, 'parallel_multiple': 12,
    'irrelevance': 12, 'live_relevance': 12,
    'multi_turn_base': 8, 'multi_turn_miss_func': 8,
    'multi_turn_miss_param': 8, 'multi_turn_long_context': 8,
}
VERSIONS = {'lm-eval': '0.4.13', 'bfcl-eval': '2025.12.17'}
