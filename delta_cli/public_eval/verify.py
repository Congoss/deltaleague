"""CPU-only checks of frozen task inventories and scoring contracts; no model results."""
import argparse
import copy
import json
from pathlib import Path
from .common import check_versions, read_manifest
from .text import load_tasks, verify_tasks


def verify_text(manifest):
    summary = {}
    for group, spec in manifest['text'].items():
        print(f'Checking native scoring: {group}', flush=True)
        tasks = load_tasks(spec['task'], manifest['dataset_revisions'])
        verify_tasks(tasks, spec['selection'])
        count = 0
        for name, task in tasks.items():
            for i in spec['selection'][name]['indices']:
                doc = task.eval_docs[i]
                if task.OUTPUT_TYPE == 'multiple_choice':
                    choices = task.doc_to_choice(doc)
                    target = task.doc_to_text(doc) if task.multiple_input else task.doc_to_target(doc)
                    if isinstance(target, str):
                        target = choices.index(target)
                    result = task.process_results(doc, [(0.0 if k == target else -100.0, k == target)
                                                        for k in range(len(choices))])
                    if result.get('acc', 1) != 1 or result.get('acc_norm', 1) != 1:
                        raise ValueError(f'{name}: correct-choice fixture failed: {result}')
                elif group == 'ifeval':
                    from lm_eval.tasks.ifeval.instructions_registry import INSTRUCTION_DICT
                    for instruction_id, kwargs in zip(doc['instruction_id_list'], doc['kwargs'], strict=True):
                        checker = INSTRUCTION_DICT[instruction_id](instruction_id)
                        checker.build_description(**{k: v for k, v in kwargs.items() if v is not None})
                else:
                    target = task.doc_to_target(doc)
                    if group == 'math500':
                        target = 'Final Answer: The final answer is $' + target + '$. I hope it is correct.'
                    result = task.process_results(doc, [target])
                    if not result or any(isinstance(v, (int, float)) and v != 1 for v in result.values()):
                        raise ValueError(f'{name}: reference-answer fixture failed: {result}')
                count += 1
        summary[group] = {'selected_documents': count, 'status': 'scorer-contract-checked'}
    return summary


def verify_tools(manifest, output):
    import ast
    from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry
    from bfcl_eval.eval_checker.eval_runner import multi_turn_runner, ast_file_runner, relevance_file_runner
    from .tools import handler_for, install_safe_dispatch, verify_data
    verify_data(manifest)
    install_safe_dispatch()
    handler, alias = handler_for('reference-fixtures')
    summary = {}
    for category, spec in manifest['bfcl'].items():
        print(f'Checking native simulator/state scoring: {category}', flush=True)
        prompts = load_dataset_entry(category, include_prereq=False, include_language_specific_hint=False)
        truth = [] if category in ('irrelevance', 'live_relevance') else load_ground_truth_entry(category)
        positions = {r['id']: i for i, r in enumerate(prompts)}
        selected = [copy.deepcopy(prompts[positions[i]]) for i in spec['ids']]
        answers = [truth[positions[i]] for i in spec['ids']] if truth else []
        if not category.startswith('multi_turn_'):
            responses = []
            for index, item_id in enumerate(spec['ids']):
                calls = []
                if category == 'live_relevance':
                    calls = [{'name': selected[index]['function'][0]['name'], 'arguments': {}}]
                elif category != 'irrelevance':
                    for call in answers[index]['ground_truth']:
                        for name, parameters in call.items():
                            calls.append({'name': name, 'arguments': {
                                key: values[0] for key, values in parameters.items() if values[0] != ''}})
                text = '\n'.join('<tool_call>\n' + json.dumps(c) + '\n</tool_call>' for c in calls)
                responses.append({'id': item_id, 'result': text})
            if category in ('irrelevance', 'live_relevance'):
                accuracy, count = relevance_file_runner(handler, responses, selected, alias, category, output)
            else:
                accuracy, count = ast_file_runner(handler, responses, selected, answers, category, alias, output)
            if accuracy != 1 or count != spec['count']:
                raise ValueError(f'{category}: reference-call fixture failed: {accuracy}, {count}')
            summary[category] = {'count': count, 'status': 'reference-call-checked'}
            continue
        responses = []
        for item_id, answer, prompt in zip(spec['ids'], answers, selected, strict=True):
            turns = []
            for calls in answer['ground_truth']:
                step = []
                for call in calls:
                    node = ast.parse(call, mode='eval').body
                    name = ast.unparse(node.func)
                    args = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
                    if node.args:
                        import importlib
                        import inspect
                        from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING
                        method_name = name.rsplit('.', 1)[-1]
                        methods = []
                        for class_name in prompt['involved_classes']:
                            cls = getattr(importlib.import_module(CLASS_FILE_PATH_MAPPING[class_name]), class_name)
                            if hasattr(cls, method_name):
                                methods.append(getattr(cls(), method_name))
                        if len(methods) != 1:
                            raise ValueError('Ambiguous reference method')
                        args = dict(inspect.signature(methods[0]).bind(
                            *[ast.literal_eval(x) for x in node.args], **args).arguments)
                    step.append('<tool_call>\n' + json.dumps({'name': name, 'arguments': args}) + '\n</tool_call>')
                turns.append(['\n'.join(step)])
            responses.append({'id': item_id, 'result': turns})
        accuracy, count = multi_turn_runner(handler, responses, selected, answers, alias, category, output)
        if accuracy != 1 or count != spec['count']:
            raise ValueError(f'{category}: reference-state fixture failed: {accuracy}, {count}')
        summary[category] = {'count': count, 'status': 'reference-state-checked'}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).parent / 'data/qwen-small-public-v1.json')
    parser.add_argument('--output', type=Path, default=Path('var/validation-suite/protocol-checks'))
    parser.add_argument('--tools-only', action='store_true')
    args = parser.parse_args()
    check_versions()
    manifest = read_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'suite_sha256': manifest['sha256'], 'reference_check': True,
              'text': {} if args.tools_only else verify_text(manifest),
              'tools': verify_tools(manifest, args.output)}
    (args.output / 'verification.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
