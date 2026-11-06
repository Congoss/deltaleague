"""BFCL's AST and multi-turn state/response checkers over frozen public samples."""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from .common import digest

ALLOWED_SIMULATORS = {'gorilla_file_system', 'math_api', 'message_api', 'posting_api',
                      'ticket_api', 'trading_bot', 'travel_booking', 'vehicle_control'}


def safe_simulator_call(expression, namespace):
    """Dispatch a literal call to a known simulator; never eval model-generated Python."""
    if len(expression) > 65536:
        raise ValueError('Tool call too large')
    node = ast.parse(expression, mode='eval').body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        raise ValueError('Expected a simulator method call')
    if not isinstance(node.func.value, ast.Name) or node.func.attr.startswith('_'):
        raise ValueError('Private or indirect tool access is forbidden')
    instance = namespace.get(node.func.value.id)
    module = type(instance).__module__
    prefix = 'bfcl_eval.eval_checker.multi_turn_eval.func_source_code.'
    if not module.startswith(prefix) or module[len(prefix):] not in ALLOWED_SIMULATORS:
        raise ValueError('Unknown simulator')
    if any(k.arg is None for k in node.keywords):
        raise ValueError('Expanded tool arguments are forbidden')
    args = [ast.literal_eval(a) for a in node.args]
    kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    return getattr(instance, node.func.attr)(*args, **kwargs)


def install_safe_dispatch():
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
    # The upstream executor resolves public methods to simulator instances, then
    # calls eval. Replace only that evaluator; preserve all official state checks.
    multi_turn_utils.eval = lambda expression: safe_simulator_call(expression, vars(multi_turn_utils))


def verify_data(manifest):
    from bfcl_eval.constants.eval_config import PROMPT_PATH
    actual = {str(p.relative_to(PROMPT_PATH)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(PROMPT_PATH.rglob('*.json'))}
    if actual != manifest['bfcl_data_sha256']:
        raise ValueError('BFCL bundled data changed')


def handler_for(model, tokenizer=None, endpoint=None):
    from .bfcl_handler import Qwen35Handler
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from openai import OpenAI
    alias = 'DeltaQwen35Validation'
    MODEL_CONFIG_MAPPING[alias] = ModelConfig(model_name=model, display_name=model,
        url='', org='Delta League', license='apache-2.0', model_handler=Qwen35Handler,
        is_fc_model=True, underscore_to_dot=False)
    handler = Qwen35Handler(model, 0, alias, True)
    if tokenizer is not None:
        from transformers import AutoTokenizer
        handler.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=False)
    if endpoint is not None:
        handler.client = OpenAI(base_url=endpoint, api_key='EMPTY', timeout=600, max_retries=2)
    handler.model_path_or_id = model
    handler.max_context_length = 34816
    return handler, alias


def run(manifest, endpoint, model, tokenizer, output, *, concurrency=32):
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry
    from bfcl_eval.eval_checker.eval_runner import (
        ast_file_runner, multi_turn_runner, relevance_file_runner)
    verify_data(manifest)
    install_safe_dispatch()
    handler, alias = handler_for(model, tokenizer, endpoint)
    directory = output.with_suffix('')
    directory.mkdir(parents=True, exist_ok=True)
    scores = {}
    for category, spec in manifest['bfcl'].items():
        rows = load_dataset_entry(category)
        by_id = {r['id']: r for r in rows}
        selected = [by_id[item_id] for item_id in spec['ids']]
        if [digest(r) for r in selected] != spec['doc_sha256']:
            raise ValueError(f'BFCL {category}: selected entries changed')

        def infer(row):
            # Infrastructure errors fail the run; they must not become model zeros.
            result, metadata = handler.inference(row, include_input_log=True, exclude_state_log=False)
            return {'id': row['id'], 'result': result, **metadata}

        with ThreadPoolExecutor(max_workers=min(concurrency, len(selected))) as pool:
            generated = list(pool.map(infer, selected))
        (directory / f'{category}.responses.json').write_text(json.dumps(generated, default=str, indent=2))
        # Reload because the official generator modifies entries in place.
        prompts = load_dataset_entry(category, include_prereq=False, include_language_specific_hint=False)
        positions = {r['id']: i for i, r in enumerate(prompts)}
        selected = [prompts[positions[i]] for i in spec['ids']]
        if category in ('irrelevance', 'live_relevance'):
            accuracy, count = relevance_file_runner(handler, generated, selected, alias, category, directory)
        else:
            answers = load_ground_truth_entry(category)
            truth = [answers[positions[i]] for i in spec['ids']]
            if category.startswith('multi_turn_'):
                accuracy, count = multi_turn_runner(handler, generated, selected, truth, alias, category, directory)
            else:
                accuracy, count = ast_file_runner(handler, generated, selected, truth, category, alias, directory)
        if count != spec['count']:
            raise ValueError(f'Incomplete BFCL result for {category}')
        scores[category] = {'accuracy': accuracy, 'count': count}
    report = {'suite_sha256': manifest['sha256'], 'model': model, 'scores': scores,
              'protocol': 'BFCL native AST and multi-turn state/response, Qwen tokenizer template, greedy non-thinking',
              'scope': 'sampled tool-calling and multi-turn simulation; excludes web-search and memory tracks'}
    output.write_text(json.dumps(report, indent=2))
    return report
