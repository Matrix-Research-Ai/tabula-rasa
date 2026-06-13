"""Sleep Cycle — background self-improvement when the system is idle.

Like humans consolidating memories during sleep, this daemon continues
training specialists when nobody is actively using the system.

Two-stage brain architecture:
  Hippocampus (Awake):   Fast storage of surprising experiences
  Neocortex  (Sleep):    Slow consolidation with EWC protection

Usage:
    python3 sleep_cycle.py                        # Run once
    python3 sleep_cycle.py --daemon               # Run continuously
    python3 sleep_cycle.py --daemon --interval 30 # Run every 30 min
"""

import sys, time, json, random, math
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW

from config import Config
from tokenizer import MathTokenizer
from model import MathTransformer, count_parameters


# ─── Problem Generation ────────────────────────────────────────────

OPS = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}

def generate_problem(op_name: str, min_digits=1, max_digits=4):
    """Generate a problem for a specific operation."""
    op = OPS[op_name]
    a_digits = random.randint(min_digits, max_digits)
    b_digits = random.randint(min_digits, max_digits)
    a = random.randint(10**(a_digits-1), 10**a_digits - 1)
    b = random.randint(10**(b_digits-1), 10**b_digits - 1)

    if op_name == 'add':
        ans = a + b
    elif op_name == 'sub':
        if a < b: a, b = b, a
        ans = a - b
    elif op_name == 'mul':
        ans = a * b
    elif op_name == 'div':
        ans = random.randint(1, max(1, a // 2))
        b = random.randint(1, max(1, a // 2))
        if b == 0: b = 1
        a = b * ans
        if a == 0: a = b * max(1, ans)
        ans = a // b if b != 0 else 1

    return f'{a}{op}{b}', str(ans)


# ─── Specialist Loader ─────────────────────────────────────────────

def load_specialist(skill_dir: str):
    """Load a specialist model from its directory."""
    d = Path(skill_dir)
    ckpt = d / 'best.pt'
    if not ckpt.exists():
        ckpt = d / 'final.pt'
    if not ckpt.exists():
        return None, None

    tok = MathTokenizer.load(str(d / 'tokenizer.json'))
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    tok.max_seq_len = cfg.max_seq_len

    model = MathTransformer(cfg)
    state = torch.load(ckpt, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model, tok


def find_specialists() -> list:
    """Find all specialist directories."""
    base = Path('specialists')
    results = []
    for domain in base.iterdir():
        if domain.is_dir():
            for skill in domain.iterdir():
                if skill.is_dir() and (skill / 'best.pt').exists():
                    op_name = skill.name
                    if op_name in OPS:
                        results.append((op_name, str(skill)))
    return results


# ─── Self-Improvement Cycle ────────────────────────────────────────

def self_improve_cycle(skill_name: str, skill_dir: str,
                       num_problems=200, train_epochs=3):
    """Run one self-improvement cycle for a specialist.

    1. Generate N problems
    2. Have the model solve them
    3. Keep only correct solutions
    4. Fine-tune on correct solutions
    """
    print(f'  [{skill_name}] Starting sleep cycle...')

    model, tok = load_specialist(skill_dir)
    if model is None:
        print(f'  [{skill_name}] No checkpoint found')
        return

    device = torch.device('cpu')
    model = model.to(device)

    # Phase 1: Generate and verify
    correct_pairs = []
    total = 0
    while len(correct_pairs) < num_problems:
        expr, expected = generate_problem(skill_name)
        prompt = f'{expr}='
        out = model.generate(tok, prompt, max_new_tokens=10,
                             temperature=0.4, top_k=5)
        pred = ''.join(c for c in (out.split('=')[-1] if '=' in out else '') if c.isdigit() or c == '-')
        total += 1
        if pred == expected:
            correct_pairs.append((expr, expected))

    acc = len(correct_pairs) / total * 100
    print(f'  [{skill_name}] Generated {len(correct_pairs)} correct / {total} attempts ({acc:.1f}%)')

    if len(correct_pairs) < 10:
        print(f'  [{skill_name}] Too few correct answers, skipping training')
        return

    # Phase 2: Train on correct solutions
    model.train()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    for epoch in range(train_epochs):
        random.shuffle(correct_pairs)
        total_loss = 0.0
        batch_count = 0
        for i in range(0, len(correct_pairs), 32):
            batch = correct_pairs[i:i+32]
            input_ids_list = []
            target_ids_list = []
            for expr, ans in batch:
                text = f'{expr}={ans}'
                ids = tok.encode(text, add_special_tokens=True)
                if len(ids) > 64:
                    ids = ids[:64]
                padded = ids + [tok.pad_id] * (64 - len(ids))
                x = torch.tensor(padded[:-1], dtype=torch.long).unsqueeze(0)
                y = torch.tensor(padded[1:], dtype=torch.long).unsqueeze(0)
                input_ids_list.append(x)
                target_ids_list.append(y)
            if not input_ids_list:
                continue
            x_batch = torch.cat(input_ids_list, dim=0)
            y_batch = torch.cat(target_ids_list, dim=0)

            optimizer.zero_grad()
            _, loss = model(x_batch, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / max(1, batch_count) if total_loss > 0 else 0
        print(f'  [{skill_name}]  Epoch {epoch+1}: loss={avg_loss:.4f}')

    # Save checkpoint
    model.eval()
    save_dir = Path(skill_dir)
    torch.save({
        'model_state_dict': model.state_dict(),
        'acc': acc,
        'sleep_cycle': True,
    }, save_dir / 'sleep_cycle.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'acc': acc,
        'sleep_cycle': True,
    }, save_dir / 'best.pt')

    # Also run Socratic consolidation if possible
    try:
        from socratic_stage3 import DialecticalEngine, ValueJudge, consolidate_winners
        engine = DialecticalEngine(model, tok)
        judge = ValueJudge(model, tok)
        topics = [
            'If all A are B, and all B are C, then all A are C.',
            'A statement cannot be both true and false at the same time.',
            'If premise A implies B, and B is false, then A must be false.',
        ]
        for topic in topics:
            debate = engine.debate(topic, 'logician', 'philosopher', turns=2)
            result = judge.judge_debate(debate)
            if result['winning_arguments']:
                consolidate_winners(model, tok, [result], epochs=1)
    except Exception as e:
        print(f'  [{skill_name}] Socratic consolidation: {e}')

    print(f'  [{skill_name}] Sleep cycle complete. Saved best.pt')


# ─── Sleep Scheduler ───────────────────────────────────────────────

def run_all_cycles(num_problems=200):
    """Run a sleep cycle with full Hippocampus → Pythagorean Review → Neocortex consolidation."""
    from egefalos.hippocampus import get_stats, get_unconsolidated, get_old_memories, mark_consolidated
    from egefalos.neocortex import full_sleep_cycle
    from egefalos.pythagoras import pythagorean_review, get_tomorrow_focus

    stats = get_stats()
    print(f'[*] Hippocampus: {stats["total_experiences"]} total, '
          f'{stats["unconsolidated"]} unconsolidated')

    # ─── Step 1: Retrieve today's experiences ───
    new_exp = get_unconsolidated(limit=500)
    if not new_exp:
        print('[*] No new experiences to consolidate today.')
        print('[*] Resting weights peacefully.')
        # Still show tomorrow's curriculum if any
        focus = get_tomorrow_focus()
        if focus:
            print(f'[*] {len(focus)} topics in tomorrow\'s curriculum.')
        return

    # ─── Step 2: The Pythagorean Review (Audit the day) ───
    print()
    print('  ╔══════════════════════════════════════════╗')
    print('  ║     PYTHAGOREAN EXAMINATION OF          ║')
    print('  ║     CONSCIENCE — Nightly Audit          ║')
    print('  ╚══════════════════════════════════════════╝')
    print()

    review = pythagorean_review(new_exp)

    # ─── Step 3: Load model for consolidation ───
    # (Existing specialist loading code)
    specialists = find_specialists()
    if not specialists:
        print('[*] No specialists found')
        return

    purified = review['purified']
    if not purified:
        print()
        print('  [Pythagoras] No logically sound memories to consolidate.')
        print('  [Pythagoras] Resting weights peacefully.')
        # Show tomorrow's curriculum
        focus = get_tomorrow_focus()
        if focus:
            print(f'[*] Tomorrow\'s curriculum: {len(focus)} topics')
            for item in focus[:5]:
                print(f'    - {item["topic"][:60]} ({item["reason"]})')
        return

    # ─── Step 4: Neocortex Consolidation (EWC) ───
    # Only the purified, golden examples reach the weights
    print()
    print(f'[*] Consolidating {len(purified)} verified memories into Neocortex...')

    for skill_name, skill_dir in specialists:
        model, tok = load_specialist(skill_dir)
        if model is None:
            continue

        print(f'\n[*] Consolidating {skill_name}...')
        try:
            result = full_sleep_cycle(model, tok, skill_dir=skill_dir)
        except Exception as e:
            print(f'  [!] Error: {e}')
            import traceback
            traceback.print_exc()

    # ─── Step 5: Show tomorrow's curriculum ───
    focus = get_tomorrow_focus()
    if focus:
        print()
        print(f'[*] Tomorrow\'s curriculum ({len(focus)} neglected topics):')
        for item in focus[:8]:
            print(f'    - {item["topic"][:60]}')
            print(f'      Reason: {item["reason"]}')
        if len(focus) > 8:
            print(f'    ... and {len(focus)-8} more')
    else:
        print()
        print('[*] No neglected duties. Curious mind at rest.')
    
    # ─── Optional Step 6: Language AlphaZero Gymnasium ───
    # Only runs if use_language_az = True in config
    try:
        cfg = Config()
        if getattr(cfg, 'use_language_az', False):
            print()
            print('  ╔══════════════════════════════════════════╗')
            print('  ║  ALPHAZERO LANGUAGE GYMNASIUM          ║')
            print('  ║  Nightly self-play for language        ║')
            print('  ╚══════════════════════════════════════════╝')
            print()
            
            from egefalos.semantic_game import alphazero_gymnasium_session
            from egefalos.grammar_engine import create_grammar_rules
            from egefalos.mcts import create_mcts_fn
            
            # Load or create language model
            lang_model = None
            lang_tok = None
            for skill_name, skill_dir in specialists:
                if 'language' in skill_name or 'general' in skill_name:
                    lang_model, lang_tok = load_specialist(skill_dir)
                    if lang_model:
                        break
            
            if lang_model is None:
                # Create a fresh LanguageAlphaZero model
                print('  [*] Creating fresh LanguageAlphaZero model...')
                from model import MathTokenizer
                lang_tok = MathTokenizer()
                cfg.vocab_size = lang_tok.vocab_size
                cfg.max_seq_len = getattr(cfg, 'language_max_seq_len', 128)
                cfg.use_value_head = True
                from egefalos.language_az import LanguageAlphaZero
                lang_model = LanguageAlphaZero(cfg)
            
            # Setup MCTS and grammar
            grammar_rules = create_grammar_rules()
            mcts_fn = create_mcts_fn(
                lang_model, lang_tok,
                simulations=getattr(cfg, 'language_mcts_simulations', 16),
                top_k=5,
            )
            
            num_games = getattr(cfg, 'language_games_per_session', 200)
            difficulty = getattr(cfg, 'language_difficulty', 'medium')
            
            print(f'  [*] Running {num_games} reconstruction games ({difficulty})...')
            
            gym_results = alphazero_gymnasium_session(
                lang_model, lang_tok,
                num_games=num_games,
                difficulty=difficulty,
                mcts_fn=mcts_fn,
                grammar_rules=grammar_rules,
                device='cpu',
            )
            
            print(f'  [*] Gymnasium complete:')
            print(f'      Games: {gym_results["games_played"]}')
            print(f'      Avg reward: {gym_results["avg_reward"]:.3f}')
            print(f'      Perfect rate: {gym_results["perfect_rate"]:.1f}%')
            print(f'      Training loss: {gym_results["training_loss"]:.4f}')
            
            # Save the language model
            save_path = Path(f'specialists/language/general')
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': lang_model.state_dict(),
                'gym_results': gym_results,
                'timestamp': time.time(),
            }, save_path / 'best.pt')
            print(f'  [*] Language model saved to {save_path / "best.pt"}')
            
            # Add results to the output
            review['language_gymnasium'] = gym_results
    except Exception as e:
        print(f'  [!] Language Gymnasium skipped: {e}')
        import traceback
        traceback.print_exc()
    
    # ─── Optional Step 7: Code AlphaZero Gymnasium ───
    try:
        cfg = Config()
        if getattr(cfg, 'use_code_az', False):
            print()
            print('  ╔══════════════════════════════════════════╗')
            print('  ║  CODE ALPHAZERO GYMNASIUM              ║')
            print('  ║  Self-play programming via sandbox     ║')
            print('  ╚══════════════════════════════════════════╝')
            print()
            
            from egefalos.code_curriculum import full_code_training_session
            
            # Load or create code model
            code_model = None
            code_tok = None
            for skill_name, skill_dir in specialists:
                if 'code' in skill_name or 'general' in skill_name:
                    code_model, code_tok = load_specialist(skill_dir)
                    if code_model:
                        break
            
            if code_model is None:
                print('  [*] Creating fresh CodeAlphaZero model...')
                from model import MathTokenizer
                code_tok = MathTokenizer()
                cfg.vocab_size = code_tok.vocab_size
                cfg.max_seq_len = getattr(cfg, 'code_max_seq_len', 256)
                cfg.use_value_head = True
                from egefalos.code_specialist import CodeAlphaZero
                code_model = CodeAlphaZero(cfg)
            
            num_syntax = getattr(cfg, 'code_syntax_games', 100)
            num_fuzzing = getattr(cfg, 'code_fuzzing_games', 50)
            num_algo = getattr(cfg, 'code_algorithm_games', 30)
            
            print(f'  [*] Syntax={num_syntax} Fuzzing={num_fuzzing} Algorithm={num_algo}')
            
            code_results = full_code_training_session(
                code_model, code_tok,
                num_syntax=num_syntax,
                num_fuzzing=num_fuzzing,
                num_algorithm=num_algo,
                learning_rate=getattr(cfg, 'code_learning_rate', 1e-4),
                device='cpu',
            )
            
            print(f'  [*] Code Gymnasium complete:')
            for stage_key, sr in code_results.items():
                if isinstance(sr, dict):
                    print(f'      {stage_key}: acc={sr.get("syntax_accuracy", sr.get("fuzz_pass_rate", sr.get("solve_rate", "?")))}')
            
            # Save the code model
            save_path = Path(f'specialists/code/general')
            save_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': code_model.state_dict(),
                'results': code_results,
                'timestamp': time.time(),
            }, save_path / 'best.pt')
            print(f'  [*] Code model saved to {save_path / "best.pt"}')
            
            review['code_gymnasium'] = code_results
    except Exception as e:
        print(f'  [!] Code Gymnasium skipped: {e}')
        import traceback
        traceback.print_exc()


def daemon_mode(interval_minutes=60, problems_per_cycle=200):
    """Run sleep cycles continuously."""
    print(f'[*] Sleep Cycle Daemon started')
    print(f'    Interval: {interval_minutes} min')
    print(f'    Problems per cycle: {problems_per_cycle}')
    print(f'    Ctrl+C to stop')
    print()

    cycle = 0
    try:
        while True:
            cycle += 1
            print(f'\n{"="*60}')
            print(f'  Sleep Cycle #{cycle}')
            print(f'  {time.strftime("%Y-%m-%d %H:%M:%S")}')
            print(f'{"="*60}')
            run_all_cycles(problems_per_cycle)
            print(f'\n[*] Next cycle in {interval_minutes} min...')
            time.sleep(interval_minutes * 60)
    except KeyboardInterrupt:
        print('\n[*] Sleep Cycle Daemon stopped')


# ─── Main ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    if '--daemon' in sys.argv:
        interval = int(sys.argv[sys.argv.index('--interval') + 1]) if '--interval' in sys.argv else 60
        daemon_mode(interval)
    else:
        problems = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200
        run_all_cycles(problems)
