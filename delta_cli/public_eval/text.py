"""Native lm-evaluation-harness tasks, with frozen dataset revisions and samples."""
from contextlib import contextmanager
from unittest.mock import patch
from .common import digest


@contextmanager
def pinned_datasets(pins, *, discover=False):
    import datasets
    from huggingface_hub import HfApi
    original = datasets.load_dataset

    def load(path, *args, **kwargs):
        if path not in pins:
            if not discover:
                raise ValueError(f'Unpinned dataset: {path}')
            pins[path] = HfApi().dataset_info(path).sha
        kwargs['revision'] = pins[path]
        return original(path, *args, **kwargs)
    with patch.object(datasets, 'load_dataset', load):
        yield


def leaves(tree):
    result = {}
    for name, item in tree.items():
        if isinstance(item, dict):
            result.update(leaves(item))
        else:
            result[name] = item
    return result


def load_tasks(name, pins, *, discover=False):
    from lm_eval.tasks import TaskManager, get_task_dict
    with pinned_datasets(pins, discover=discover):
        return leaves(get_task_dict([name], TaskManager()))


def verify_tasks(tasks, selection):
    if set(tasks) != set(selection):
        raise ValueError('Task inventory changed')
    for name, task in tasks.items():
        entry = selection[name]
        if task.OUTPUT_TYPE != entry['output_type'] or task.get_config('num_fewshot') != entry['num_fewshot']:
            raise ValueError(f'{name}: native evaluation settings changed')
        docs = task.eval_docs
        if len(docs) != entry['population']:
            raise ValueError(f'{name}: dataset size changed')
        if [digest(docs[i]) for i in entry['indices']] != entry['doc_sha256']:
            raise ValueError(f'{name}: selected documents changed')


def run(manifest, group, endpoint, model, tokenizer, output, *, concurrency=32, generation=None):
    from lm_eval import evaluator
    from langdetect import DetectorFactory
    DetectorFactory.seed = manifest['seed']
    from lm_eval.models.openai_completions import LocalCompletionsAPI
    from .generation import sampling, final_answer
    generation = generation or {}
    thinking = generation.get('enable_thinking', False)
    spec = manifest['text'][group]
    tasks = load_tasks(spec['task'], manifest['dataset_revisions'])
    verify_tasks(tasks, spec['selection'])
    # Native likelihood tasks use their original completion prompts. Generative
    # tasks use the checkpoint's chat template and the versioned thinking setting.
    generative = all(t.OUTPUT_TYPE == 'generate_until' for t in tasks.values())
    if not generative and any(t.OUTPUT_TYPE == 'generate_until' for t in tasks.values()):
        raise ValueError('Mixed likelihood/generation group requires separate runs')

    class PublicValidationAPI(LocalCompletionsAPI):
        def apply_chat_template(self, chat_history, add_generation_prompt=True):
            return self.tokenizer.apply_chat_template(chat_history, tokenize=False,
                add_generation_prompt=add_generation_prompt, enable_thinking=thinking)

        def parse_generations(self, outputs, **kwargs):
            answers = super().parse_generations(outputs, **kwargs)
            return [final_answer(answer) for answer in answers] if thinking else answers

    backend = PublicValidationAPI(model=model, base_url=endpoint.rstrip('/') + '/completions',
        tokenizer=tokenizer, tokenizer_backend='huggingface', num_concurrent=concurrency,
        max_length=34816, max_gen_toks=generation.get('max_tokens', 4096), max_retries=3, timeout=600,
        tokenized_requests=False, seed=manifest['seed'])
    options = {'gen_kwargs': {**sampling(generation), 'max_gen_toks': generation['max_tokens'],
                              'do_sample': True, 'until': []}} if generation and generative else {}
    result = evaluator.simple_evaluate(model=backend, tasks=list(tasks.values()),
        samples={name: e['indices'] for name, e in spec['selection'].items()},
        apply_chat_template=generative, bootstrap_iters=1000, log_samples=True,
        random_seed=manifest['seed'], numpy_random_seed=manifest['seed'],
        torch_random_seed=manifest['seed'], fewshot_random_seed=manifest['seed'],
        **options, metadata={'suite_sha256': manifest['sha256'], 'benchmark': group,
                  'thinking': thinking, 'protocol': 'native-task-defaults-chat-for-generation'})
    if result is None:
        raise RuntimeError('No evaluation results')
    for name, selection in spec['selection'].items():
        seen = {r['doc_id'] for r in result.get('samples', {}).get(name, [])}
        if seen != set(selection['indices']):
            raise ValueError(f'{name}: incomplete or unexpected result coverage')
    output.write_bytes(__import__('json').dumps(result, default=str, indent=2).encode())
    return result
