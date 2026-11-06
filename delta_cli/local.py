"""Offline text inference and public validation; no platform or private-data imports."""
from contextlib import contextmanager
from importlib.metadata import version
import json
import os
from pathlib import Path
import statistics
import tempfile
import time

from .artifacts import file_hash, reconstruct, validate_manifest, verify_artifact_bytes
from .scoring import score_answer


def read_workspace(workspace):
    workspace = Path(workspace).resolve()
    identity = json.loads((workspace / 'delta-workspace.json').read_text())
    config = json.loads((workspace / 'evaluation.json').read_text())
    metadata = json.loads((workspace / 'model/delta-model.json').read_text())
    if (identity['config_id'] != config['id'] or identity['model_id'] != config['model_id']
            or identity['parent_version_id'] != metadata['id'] or identity['model_id'] != metadata['model_id']):
        raise ValueError('Workspace model, parent, or evaluation configuration mismatch. Initialize a fresh workspace.')
    return workspace, identity, config, metadata


@contextmanager
def model_directory(workspace, metadata, weights=None, patch=None, parent=None):
    """Overlay a candidate without mutating the downloaded parent or tokenizer."""
    if weights and patch:
        raise ValueError('Choose either --weights or --patch.')
    parent = parent or workspace / 'model'
    digests = verify_artifact_bytes(parent, metadata['files'], skip_names=('model.safetensors',) if weights else ())
    if not weights and not patch:
        yield parent, {'kind': 'parent', 'weights_sha256': digests['model.safetensors']}
        return
    with tempfile.TemporaryDirectory(prefix='delta-local-') as temp:
        target = Path(temp)
        for artifact in metadata['files']:
            if artifact['name'] != 'model.safetensors':
                (target / artifact['name']).symlink_to((parent / artifact['name']).resolve())
        if patch:
            from safetensors.torch import save_file
            patch = Path(patch).resolve()
            manifest = json.loads((patch / 'manifest.json').read_text())
            validate_manifest(manifest)
            if (manifest['parent_version_id'] != metadata['id'] or manifest['parent_content_hash'] != metadata['content_hash']
                    or manifest['model_config_id'] != metadata['model_id']):
                raise ValueError('Patch does not reference this workspace parent.')
            for artifact in manifest['artifacts']:
                path = patch / artifact['name']
                if path.stat().st_size != artifact['size_bytes'] or file_hash(path) != artifact['sha256']:
                    raise ValueError('Patch artifact integrity failure.')
            tensors = reconstruct(parent / 'model.safetensors', patch / manifest['artifacts'][0]['name'], manifest)
            save_file(tensors, str(target / 'model.safetensors'))
            del tensors
        else:
            weights = Path(weights).resolve()
            if not weights.is_file() or weights.suffix != '.safetensors':
                raise ValueError('--weights must be a full .safetensors checkpoint.')
            (target / 'model.safetensors').symlink_to(weights)
        yield target, {'kind': 'patch' if patch else 'weights', 'weights_sha256': file_hash(target / 'model.safetensors')}


def select_device(torch, requested):
    if requested not in {'auto', 'cpu', 'cuda', 'mps'}:
        raise ValueError('--device must be auto, cpu, cuda, or mps.')
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable. Use a supported Apple Silicon Mac or --device cpu.')
    return requested


class Runner:
    def __init__(self, model_path, device='auto'):
        import torch
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText
        self.torch = torch
        self.device = select_device(torch, device)
        # Float32 is the portable CPU/MPS path; CUDA uses BF16 when supported.
        dtype = torch.bfloat16 if self.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        auto = AutoModelForImageTextToText if type(config) in AutoModelForImageTextToText._model_mapping else AutoModelForCausalLM
        self.model, info = auto.from_pretrained(model_path, config=config, local_files_only=True, trust_remote_code=False,
                                                use_safetensors=True, dtype=dtype, output_loading_info=True)
        if info.get('missing_keys') or info.get('unexpected_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise ValueError('Checkpoint tensors do not match the model configuration; refusing partially initialized weights.')
        self.model.to(self.device).eval()
        self.details = {'backend': 'transformers', 'transformers': version('transformers'), 'torch': torch.__version__,
                        'device': self.device, 'dtype': str(dtype),
                        'mps_cpu_fallback': self.device == 'mps' and os.getenv('PYTORCH_ENABLE_MPS_FALLBACK') == '1'}

    def generate(self, prompt, generation, seed):
        from transformers import LogitsProcessor, LogitsProcessorList, set_seed
        torch = self.torch
        rendered = self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=generation.get('enable_thinking', True))
        inputs = self.tokenizer(rendered, return_tensors='pt', add_special_tokens=False).to(self.device)
        prompt_length = inputs['input_ids'].shape[1]
        max_tokens = generation.get('max_tokens', 32768)
        if prompt_length + max_tokens > generation.get('max_model_len', 34816):
            raise ValueError('Prompt plus output budget exceeds max_model_len; reduce --max-new-tokens.')
        penalty = generation.get('presence_penalty', 1.5)

        class PresencePenalty(LogitsProcessor):
            def __call__(self, input_ids, scores):
                # Match vLLM: presence penalty applies to generated tokens, not the prompt.
                for row in range(input_ids.shape[0]):
                    ids = input_ids[row, prompt_length:].unique()
                    scores[row, ids] -= penalty
                return scores

        temperature = generation.get('temperature', .6)
        kwargs = {'max_new_tokens': max_tokens, 'do_sample': temperature > 0,
                  'logits_processor': LogitsProcessorList([PresencePenalty()])}
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=generation.get('top_p', .95), top_k=generation.get('top_k', 20))
        set_seed(seed)
        start = time.monotonic()
        with torch.inference_mode():
            tokens = self.model.generate(**inputs, **kwargs)[0, prompt_length:]
        count = tokens.numel()
        # Preserve </think> so the same scorer can remove reasoning traces as on the server.
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        for token in [self.tokenizer.eos_token, self.tokenizer.pad_token]:
            if token:
                text = text.replace(token, '')
        return {'text': text, 'tokens': count, 'truncated': count == max_tokens,
                'generation_seconds': time.monotonic() - start}


def public_examples(workspace, config, benchmark=None, limit=None):
    path = workspace / 'validation.json'
    if file_hash(path) != config['validation_sha256']:
        raise ValueError('Public validation data changed; initialize a fresh workspace.')
    data = json.loads(path.read_text())
    names = config['validation_names']
    if set(data) != set(names) or not names:
        raise ValueError('Public validation benchmark inventory mismatch.')
    if benchmark and benchmark not in names:
        raise ValueError('Unknown benchmark; choose from: ' + ', '.join(names))
    selected = [benchmark] if benchmark else names
    rows = []
    for name in selected:
        if not data[name]:
            raise ValueError(f'Empty benchmark: {name}')
        for index, example in enumerate(data[name][:limit] if limit else data[name]):
            scorer = example.get('scorer', 'exact')
            if scorer not in {'exact', 'choice', 'math', 'ifeval'}:
                raise ValueError(f'Unsupported scorer: {scorer}')
            # Fail on missing scoring dependencies before spending time on inference.
            if scorer == 'math':
                __import__('math_verify')
            elif scorer == 'ifeval':
                __import__('lm_eval.tasks.ifeval.instructions_registry')
            samples = example.get('samples', 1)
            if type(samples) is not int or not 1 <= samples <= 16:
                raise ValueError('Invalid sample count.')
            for sample in range(samples):
                rows.append({'benchmark': name, 'example_index': index, 'sample': sample,
                             'seed': 1000 + sample, 'example': example})
    return rows, selected, not benchmark and all(limit is None or limit >= len(v) for v in data.values())


def generations_path(report):
    """Path of the generations file written beside an evaluation report."""
    return Path(report).with_suffix('.generations.jsonl')


def example_means(rows):
    """Per benchmark: mean sample score by example, truncation rate and mean output tokens."""
    grouped = {}
    for row in rows:
        group = grouped.setdefault(row['benchmark'], {'scores': {}, 'truncated': [], 'tokens': []})
        group['scores'].setdefault(row['example_index'], []).append(row['score'])
        group['truncated'].append(bool(row['truncated']))
        group['tokens'].append(row['tokens'])
    return {name: {'examples': {k: statistics.mean(v) for k, v in g['scores'].items()},
                   'truncation_rate': statistics.mean(g['truncated']), 'mean_tokens': statistics.mean(g['tokens'])}
            for name, g in grouped.items()}


def evaluate(runner, tasks, generation, stream, progress):
    rows, tokens, seconds, truncated = [], 0, 0, 0
    for index, task in enumerate(tasks, 1):
        result = runner.generate(task['example']['prompt'], generation, task['seed'])
        score = score_answer(task['example'], result['text'])
        row = {k: v for k, v in task.items() if k != 'example'}
        row.update(example_id=task['example']['id'], score=score, **result)
        rows.append({k: row[k] for k in ('benchmark', 'example_index', 'example_id', 'score', 'truncated', 'tokens')})
        stream.write(json.dumps(row) + '\n')
        stream.flush()
        tokens += result['tokens']
        seconds += result['generation_seconds']
        truncated += result['truncated']
        progress(index, len(tasks), task['benchmark'])
    components = {name: statistics.mean(v['examples'].values()) * 100 for name, v in example_means(rows).items()}
    return {'validation_score': statistics.mean(components.values()), 'validation_benchmarks': components,
            'output_tokens': tokens, 'generation_seconds': seconds, 'truncation_rate': truncated / len(tasks)}
