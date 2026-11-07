"""Explicit public generation settings; absent settings preserve legacy evaluations."""
SAMPLING_KEYS = ('temperature', 'top_p', 'top_k', 'presence_penalty', 'repetition_penalty', 'min_p')


def public_settings(generation):
    return generation.get('public_validation', {})


def sampling(settings):
    return {key: settings[key] for key in SAMPLING_KEYS if key in settings}


def final_answer(text):
    # Completion prompts already open <think>; never score an unfinished thought.
    return text.rsplit('</think>', 1)[1].strip() if '</think>' in text else ''
