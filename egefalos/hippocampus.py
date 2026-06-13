"""Hippocampus — Fast, short-term memory during awake state.

When the model encounters surprising data (high prediction error),
it's instantly saved here. No heavy training — just storage.

This mirrors the human hippocampus: fast encoding of experiences
for later consolidation into permanent memory.
"""

import sqlite3, json, time, math
from pathlib import Path
from typing import Optional
import torch

DB_PATH = Path('memory/hippocampus.db')


def get_db() -> sqlite3.Connection:
    """Get or create the hippocampus database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            input_text TEXT NOT NULL,
            output_text TEXT,
            prediction_error REAL NOT NULL,
            skill TEXT DEFAULT 'general',
            is_correction INTEGER DEFAULT 0,
            consolidated INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_consolidated
        ON experiences(consolidated)
    """)
    conn.commit()
    return conn


def store_experience(input_text: str, prediction_error: float,
                     output_text: str = None, skill: str = 'general',
                     is_correction: bool = False):
    """Store a surprising experience in the hippocampus."""
    conn = get_db()
    conn.execute("""
        INSERT INTO experiences (timestamp, input_text, output_text,
                                 prediction_error, skill, is_correction)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (time.time(), input_text, output_text,
          prediction_error, skill, 1 if is_correction else 0))
    conn.commit()
    conn.close()


def get_unconsolidated(limit: int = 100) -> list[dict]:
    """Get experiences that haven't been consolidated yet."""
    conn = get_db()
    rows = conn.execute("""
        SELECT id, input_text, output_text, prediction_error, skill, is_correction
        FROM experiences WHERE consolidated = 0
        ORDER BY prediction_error DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    return [
        {
            'id': r[0],
            'input': r[1],
            'output': r[2],
            'error': r[3],
            'skill': r[4],
            'is_correction': bool(r[5]),
        }
        for r in rows
    ]


def get_old_memories(limit: int = 50) -> list[dict]:
    """Get random old consolidated memories for replay."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM experiences WHERE consolidated = 1").fetchone()[0]
    if total == 0:
        conn.close()
        return []

    rows = conn.execute("""
        SELECT id, input_text, output_text, prediction_error, skill, is_correction
        FROM experiences WHERE consolidated = 1
        ORDER BY RANDOM() LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    return [
        {
            'id': r[0],
            'input': r[1],
            'output': r[2],
            'error': r[3],
            'skill': r[4],
            'is_correction': bool(r[5]),
        }
        for r in rows
    ]


def mark_consolidated(ids: list[int]):
    """Mark experiences as consolidated after training."""
    if not ids:
        return
    conn = get_db()
    placeholders = ','.join('?' * len(ids))
    conn.execute(f"""
        UPDATE experiences SET consolidated = 1
        WHERE id IN ({placeholders})
    """, ids)
    conn.commit()
    conn.close()


def get_stats() -> dict:
    """Get hippocampus statistics."""
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
    unconsolidated = conn.execute("SELECT COUNT(*) FROM experiences WHERE consolidated = 0").fetchone()[0]
    avg_error = conn.execute("SELECT AVG(prediction_error) FROM experiences").fetchone()[0]
    conn.close()
    return {
        'total_experiences': total,
        'unconsolidated': unconsolidated,
        'avg_error': round(avg_error, 3) if avg_error else 0,
    }


def clear():
    """Clear the hippocampus (for testing)."""
    conn = get_db()
    conn.execute("DELETE FROM experiences")
    conn.commit()
    conn.close()


# ─── Surprise Detector ──────────────────────────────────────────────

@torch.no_grad()
def compute_surprise(model, tokenizer, text: str) -> float:
    """Compute prediction error (surprise) for a text."""
    import torch
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 3:
        return 0.0

    device = next(model.parameters()).device
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)

    model.eval()
    _, loss, _ = model(x, y)
    return loss.item()


def auto_store(model, tokenizer, text: str, threshold: float = 1.0,
               output: str = None, skill: str = 'general'):
    """Automatically compute surprise and store if above threshold."""
    error = compute_surprise(model, tokenizer, text)
    if error >= threshold:
        store_experience(text, error, output, skill)
        return True
    return False


if __name__ == '__main__':
    # Test
    store_experience('What is a mitochondrion?', 2.5, 'The powerhouse of the cell', 'biology')
    store_experience('E=mc^2', 1.8, 'Energy equals mass times speed of light squared', 'physics')
    print(f'Stats: {get_stats()}')
    print(f'Unconsolidated: {len(get_unconsolidated())}')
    print(f'Old memories: {len(get_old_memories())}')
