"""Persistent memory — corrections survive server restarts.

Uses JSON file storage (zero dependencies). Each correction is saved
and reloaded on boot, so interactive teaching persists across reboots.
"""

import json, time
from pathlib import Path

MEMORY_FILE = Path('memory/corrections.json')


def load_memory() -> dict:
    """Load all saved corrections."""
    if not MEMORY_FILE.exists():
        return {'corrections': [], 'total': 0}
    try:
        with open(MEMORY_FILE) as f:
            return json.load(f)
    except:
        return {'corrections': [], 'total': 0}


def save_correction(expr: str, correct_answer: str, skill: str = 'general'):
    """Save a correction to persistent memory."""
    mem = load_memory()
    entry = {
        'expr': expr,
        'answer': correct_answer,
        'skill': skill,
        'timestamp': time.time(),
    }
    mem['corrections'].append(entry)
    mem['total'] = len(mem['corrections'])
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_FILE, 'w') as f:
        json.dump(mem, f, indent=2)
    return mem['total']


def get_all_corrections(skill: str = None) -> list:
    """Get corrections, optionally filtered by skill."""
    mem = load_memory()
    if skill:
        return [c for c in mem['corrections'] if c['skill'] == skill]
    return mem['corrections']


def replay_corrections(model_handler, skill: str = None):
    """Re-train a model on all saved corrections."""
    corrections = get_all_corrections(skill)
    if not corrections:
        return 0
    pairs = [(c['expr'], c['answer']) for c in corrections]
    # Deduplicate
    seen = set()
    unique_pairs = []
    for e, a in pairs:
        key = f'{e}={a}'
        if key not in seen:
            seen.add(key)
            unique_pairs.append((e, a))
    if unique_pairs:
        result = model_handler.train_step(unique_pairs, epochs=3)
        return len(unique_pairs)
    return 0


if __name__ == '__main__':
    # Test
    save_correction('2+3', '5', 'math/add')
    save_correction('10-4', '6', 'math/sub')
    print(f'Saved {load_memory()["total"]} corrections')
    print(json.dumps(get_all_corrections(), indent=2))
