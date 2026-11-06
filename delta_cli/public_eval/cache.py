"""Bake pinned public task datasets and scorer resources into the worker image."""
import os
from .suite import manifest
from .common import check_versions
from .text import load_tasks, verify_tasks
from .tools import verify_data


def main():
    check_versions()
    suite = manifest()
    for name, spec in suite['text'].items():
        print('Caching public task:', name, flush=True)
        verify_tasks(load_tasks(spec['task'], suite['dataset_revisions']), spec['selection'])
    verify_data(suite)
    import nltk
    for name in ('punkt', 'punkt_tab'):
        if not nltk.download(name, download_dir=os.environ.get('NLTK_DATA'), quiet=True):
            raise RuntimeError('NLTK resource download failed: ' + name)


if __name__ == '__main__':
    main()
