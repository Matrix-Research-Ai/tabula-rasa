"""Comprehensive Continual Learning Benchmark — Tabula Rasa.

Measures four standardized CL metrics across 7 methods on sequential
arithmetic tasks (add -> sub -> mul).

Metrics:
    Backward Transfer (BWT):
        Performance on task i after learning task j (j > i).
        BWT_i_j = accuracy_i_after_j - accuracy_i_at_init.
        Negative BWT = forgetting.

    Forward Transfer (FWT):
        How much pre-training on task i speeds up learning task j.
        FWT_j = (steps_to_mastery_j_with_pretrain -
                 steps_to_mastery_j_from_scratch) / steps_to_mastery_j_from_scratch.
        Negative FWT = positive transfer (faster learning).

    Learning Efficiency:
        Total steps across all tasks to reach 80% mastery on each.

    Computational Overhead:
        Relative training time vs no-CL baseline.

Methods compared:
    1. None (no CL) — baseline, fine-tune sequentially
    2. Online EWC — merged Fisher (gamma=0.9, lambda=500)
    3. Fisher Archive — unmerged per-task Fishers (lambda=500)
    4. ExpertEWC — per-expert Fisher tracking for MoE
    5. LwF (Learning without Forgetting) — distillation loss
    6. GEM (Gradient Episodic Memory) — gradient projection
    7. OGD (Online Gradient Descent) — no explicit CL, baseline variant

Usage:
    python3 experiments/run_cl_benchmark.py              # Quick (100 steps/task)
    python3 experiments/run_cl_benchmark.py --full       # Full (2000 steps/task)
    python3 experiments/run_cl_benchmark.py --methods ewc lwf  # Subset
    python3 experiments/run_cl_benchmark.py --tasks add sub mul  # 3 tasks

Output:
    experiments/cl_benchmark_results.json — full results table
    Console prints a formatted comparison table.
"""

import json
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer
from tabula_rasa.tokenizer import MathTokenizer
from tabula_rasa.math_parser import evaluate as math_eval, parse_expression
from train_specialist import (
    SpecialistDataset,
    _get_lr,
    generate_problem,
)


# ─── Config ───────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Benchmark device: {DEVICE}")
QUICK_STEPS = 100
FULL_STEPS = 2000
MASTERY_THRESHOLD = 80.0  # accuracy threshold for "mastery"


def make_model(tok, use_value_head=False):
    """Create a fresh model for each benchmark run."""
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    cfg.d_model = 128
    cfg.n_layers = 4
    cfg.n_heads = 4
    cfg.d_ff = 512
    cfg.max_seq_len = 32
    cfg.max_digits = 1
    cfg.min_digits = 1
    cfg.use_reversed = True
    cfg.use_scratchpad = True
    cfg.use_loss_masking = True
    cfg.dropout = 0.1
    cfg.use_value_head = use_value_head
    tok.max_seq_len = cfg.max_seq_len
    model = MathTransformer(cfg).to(DEVICE)
    return model, cfg


def evaluate_accuracy(model, tok, cfg, operation, num=100):
    """Evaluate exact-match accuracy on a single operation."""
    model.eval()
    correct = 0
    for _ in range(num):
        expr, ans = generate_problem(operation, cfg.min_digits, cfg.max_digits,
                                      reversed=True, scratchpad=True)
        full = f"{expr}={ans}"
        prompt = f"{expr}="
        gen = model.generate(tok, prompt, max_new_tokens=10, temperature=0.0)
        gen = gen.replace(prompt, "").strip()
        # Extract answer: try scratchpad first, then direct
        # The scratchpad is like "0406" for 04+06 (carry+digit pairs)
        # Answer is after all pairs — actually for 1-digit both are the last 2 chars
        if len(gen) >= 2:
            # Fused carry-digit: last 2 chars = (carry, digit), final digit is the answer
            # For 1-digit: "06" means carry=0, digit=6 → answer=6
            ans_char = gen[-1]
            if ans_char == ans:
                correct += 1
    return correct / num * 100


def evaluate_all_ops(model, tok, cfg, ops, num=100):
    """Evaluate on all given operations."""
    return {op: evaluate_accuracy(model, tok, cfg, op, num) for op in ops}


def train_steps(model, cfg, tok, operation, steps, progress_callback=None):
    """Train model on one operation for a fixed number of steps.

    Returns: (final_acc, steps_to_mastery)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = lambda s: 0.001 * _get_lr(s, 200, steps, "cosine")

    ds = SpecialistDataset(tok, operation, cfg)
    batch_iter = _infinite_batches(ds, 128, DEVICE)
    global_step = 0
    steps_to_mastery = None

    while global_step < steps:
        x, y = next(batch_iter)
        _, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        for pg in optimizer.param_groups:
            pg["lr"] = scheduler(global_step)
        global_step += 1

        if global_step % 50 == 0 or global_step == steps:
            acc = evaluate_accuracy(model, tok, cfg, operation, num=50)
            if steps_to_mastery is None and acc >= MASTERY_THRESHOLD:
                steps_to_mastery = global_step

    final_acc = evaluate_accuracy(model, tok, cfg, operation, num=100)
    return final_acc, steps_to_mastery or steps


def _infinite_batches(ds, bs, dev):
    """Infinite DataLoader generator."""
    while True:
        for x, y in DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True):
            yield x.to(dev), y.to(dev)


# ═══════════════════════════════════════════════════════════════════
# Method Wrappers
# ═══════════════════════════════════════════════════════════════════

def method_no_cl(model, cfg, tok, task_seq, steps):
    """No continual learning: just fine-tune sequentially."""
    results = {}
    for i, (op, _) in enumerate(task_seq):
        acc, mastery = train_steps(model, cfg, tok, op, steps)
        results[f"after_task_{i+1}"] = {
            "trained_on": op,
            "accuracies": evaluate_all_ops(model, tok, cfg, [t[0] for t in task_seq]),
            "steps_to_mastery": mastery,
        }
    return results


def method_online_ewc(model, cfg, tok, task_seq, steps, gamma=0.9, lam=500):
    """Online EWC with merged Fisher (standard)."""
    from egefalos.online_ewc import OnlineEWC
    return _method_ewc_family(model, cfg, tok, task_seq, steps,
                              OnlineEWC, {"gamma": gamma, "use_archive": False}, lam)


def method_fisher_archive(model, cfg, tok, task_seq, steps, gamma=0.9, lam=500):
    """Fisher Archive — unmerged per-task Fishers."""
    from egefalos.online_ewc import OnlineEWC
    return _method_ewc_family(model, cfg, tok, task_seq, steps,
                              OnlineEWC, {"gamma": gamma, "use_archive": True}, lam)


def _method_ewc_family(model, cfg, tok, task_seq, steps, ewc_cls, ewc_kwargs, lam):
    """Generic EWC-family method runner."""
    ewc = ewc_cls(model, **ewc_kwargs)
    results = {}
    for i, (op, acc_before) in enumerate(task_seq):
        if i == 0:
            # First task: train normally, compute Fisher
            acc, mastery = train_steps(model, cfg, tok, op, steps)
            ewc.save_anchor_weights()
            # Compute Fisher on first task
            fisher = _compute_fisher(model, tok, cfg, op, num_samples=100)
            ewc.fisher_dict = {k: v.clone() for k, v in fisher.items()}
            ewc.task_count = 1
        else:
            # Subsequent tasks: train with EWC penalty
            acc, mastery = _train_with_penalty(model, cfg, tok, op, steps,
                                                lambda: ewc.compute_ewc_penalty(lam))
            # Compute and merge Fisher
            fisher = _compute_fisher(model, tok, cfg, op, num_samples=100)
            ewc.merge_fisher(fisher)
            ewc.save_anchor_weights()

        results[f"after_task_{i+1}"] = {
            "trained_on": op,
            "accuracies": evaluate_all_ops(model, tok, cfg, [t[0] for t in task_seq]),
            "steps_to_mastery": mastery,
        }
    return results


def method_lwf(model, cfg, tok, task_seq, steps):
    """Learning without Forgetting — distillation loss."""
    from egefalos.lwf_gem import LwF
    lwf = LwF(model, lambda_distill=0.5, temperature=2.0)
    results = {}
    for i, (op, acc_before) in enumerate(task_seq):
        if i == 0:
            acc, mastery = train_steps(model, cfg, tok, op, steps)
        else:
            lwf.snapshot_teacher()
            acc, mastery = _train_with_penalty(
                model, cfg, tok, op, steps,
                lambda: torch.tensor(0.0, device=DEVICE),  # dummy, LwF handles it
            )
            # Apply LwF distillation:
            # Re-train with combined loss
            _train_lwf(model, cfg, tok, op, steps, lwf)

        results[f"after_task_{i+1}"] = {
            "trained_on": op,
            "accuracies": evaluate_all_ops(model, tok, cfg, [t[0] for t in task_seq]),
            "steps_to_mastery": mastery,
        }
    return results


def _train_lwf(model, cfg, tok, operation, steps, lwf):
    """Train with LwF distillation loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    ds = SpecialistDataset(tok, operation, cfg)
    batch_iter = _infinite_batches(ds, 64, DEVICE)
    for s in range(steps):
        x, y = next(batch_iter)
        _, task_loss, _ = model(x, y)
        distill = lwf.get_distillation_loss(x)
        total = task_loss + distill
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def method_gem(model, cfg, tok, task_seq, steps):
    """Gradient Episodic Memory — gradient projection."""
    from egefalos.lwf_gem import GEM
    gem = GEM(model, memory_size=200)
    results = {}
    for i, (op, acc_before) in enumerate(task_seq):
        ds = SpecialistDataset(tok, op, cfg, num_samples=2000)
        loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

        if i == 0:
            acc, mastery = train_steps(model, cfg, tok, op, steps)
        else:
            # Store Task 1 memory
            gem.add_to_memory(next(iter(loader)))
            # Store reference gradients
            ref_loader = DataLoader(ds, batch_size=64, shuffle=True)
            gem.store_gradients(ref_loader)
            # Train with gradient projection
            acc, mastery = _train_gem(model, cfg, tok, op, steps, gem)

        results[f"after_task_{i+1}"] = {
            "trained_on": op,
            "accuracies": evaluate_all_ops(model, tok, cfg, [t[0] for t in task_seq]),
            "steps_to_mastery": mastery,
        }
    return results


def _train_gem(model, cfg, tok, operation, steps, gem):
    """Train one task with GEM gradient projection."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    ds = SpecialistDataset(tok, operation, cfg, num_samples=2000)
    loader = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)
    batch_iter = _infinite_batches(ds, 64, DEVICE)
    global_step = 0
    mastery = steps
    while global_step < steps:
        x, y = next(batch_iter)
        _, loss, _ = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        gem.project_gradients()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1
        if global_step % 50 == 0:
            acc = evaluate_accuracy(model, tok, cfg, operation, num=50)
            if mastery == steps and acc >= MASTERY_THRESHOLD:
                mastery = global_step
    return evaluate_accuracy(model, cfg, tok, operation, num=100), mastery


def method_ogd(model, cfg, tok, task_seq, steps):
    """Online Gradient Descent — separate optimizer per task, no CL."""
    # Essentially same as no-CL but with lower LR on second task
    # to simulate conservative OGD
    results = {}
    for i, (op, _) in enumerate(task_seq):
        lr = 0.0005 if i > 0 else 0.001  # lower LR on subsequent tasks
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        ds = SpecialistDataset(tok, op, cfg)
        batch_iter = _infinite_batches(ds, 128, DEVICE)
        global_step = 0
        mastery = steps
        while global_step < steps:
            x, y = next(batch_iter)
            _, loss, _ = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % 50 == 0:
                acc = evaluate_accuracy(model, cfg, tok, op, num=50)
                if mastery == steps and acc >= MASTERY_THRESHOLD:
                    mastery = global_step
        results[f"after_task_{i+1}"] = {
            "trained_on": op,
            "accuracies": evaluate_all_ops(model, tok, cfg, [t[0] for t in task_seq]),
            "steps_to_mastery": mastery,
        }
    return results


def _compute_fisher(model, tok, cfg, operation, num_samples=100):
    """Compute Fisher diagonal on one operation."""
    model.train()
    device = next(model.parameters()).device
    fisher = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            fisher[name] = torch.zeros_like(p)
    samples = 0
    while samples < num_samples:
        inputs, targets = [], []
        for _ in range(min(16, num_samples - samples)):
            expr, ans = generate_problem(operation, 1, 1, reversed=True, scratchpad=True)
            full = f"{expr}={ans}"
            ids = tok.encode(full, add_special_tokens=True)
            x = ids[:-1]
            y = ids[1:]
            pad = tok.pad_id
            inputs.append(x + [pad] * (tok.max_seq_len - len(x)))
            targets.append(y + [-100] * (tok.max_seq_len - len(y)))
        if not inputs:
            break
        x = torch.tensor(inputs, dtype=torch.long, device=device)
        y = torch.tensor(targets, dtype=torch.long, device=device)
        model.zero_grad()
        _, loss, _ = model(x, y)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None and name in fisher:
                fisher[name] += p.grad.detach() ** 2
        samples += x.size(0)
    if samples > 0:
        for name in fisher:
            fisher[name] /= samples
    return fisher


def _train_with_penalty(model, cfg, tok, operation, steps, penalty_fn):
    """Train with an additional penalty term (EWC, etc.)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    ds = SpecialistDataset(tok, operation, cfg)
    batch_iter = _infinite_batches(ds, 128, DEVICE)
    global_step = 0
    mastery = steps
    while global_step < steps:
        x, y = next(batch_iter)
        _, task_loss, _ = model(x, y)
        total = task_loss + penalty_fn()
        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1
        if global_step % 50 == 0:
            acc = evaluate_accuracy(model, tok, cfg, operation, num=50)
            if mastery == steps and acc >= MASTERY_THRESHOLD:
                mastery = global_step
    return evaluate_accuracy(model, cfg, tok, operation, num=100), mastery


# ═══════════════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════════════

METHODS = {
    "none": method_no_cl,
    "ewc": method_online_ewc,
    "fisher_archive": method_fisher_archive,
    "lwf": method_lwf,
    "gem": method_gem,
    "ogd": method_ogd,
}


def compute_bwt(results, task_names):
    """Compute Backward Transfer from results dict.

    BWT_i_j = accuracy_on_task_i_after_task_j - accuracy_on_task_i_at_init
    For task i, the worst-case BWT across all subsequent tasks.
    """
    bwt = {}
    for i in range(len(task_names)):
        name_i = task_names[i]
        acc_after = []
        for j in range(i + 1, len(task_names)):
            phase = f"after_task_{j+1}"
            if phase in results:
                acc_after.append(results[phase]["accuracies"].get(name_i, 0))
        if acc_after:
            bwt[name_i] = min(acc_after) - results.get("initial", {}).get(name_i, 0)
        else:
            bwt[name_i] = 0.0
    return bwt


def compute_fwt(results, task_names, baseline_masteries):
    """Compute Forward Transfer from results.

    FWT_j = (steps_with_CL - steps_baseline) / steps_baseline
    Negative = positive transfer (CL helps).
    """
    fwt = {}
    for i in range(1, len(task_names)):
        name_i = task_names[i]
        cl_mastery = results.get(f"after_task_{i+1}", {}).get("steps_to_mastery", 0)
        base_mastery = baseline_masteries.get(name_i, cl_mastery)
        if base_mastery > 0:
            fwt[name_i] = (cl_mastery - base_mastery) / base_mastery * 100
        else:
            fwt[name_i] = 0.0
    return fwt


def run_benchmark(methods_to_run, tasks, steps_per_task):
    """Run CL benchmark for specified methods and tasks."""
    tok = MathTokenizer()
    task_names = [t for t, _ in tasks]

    # First: train each task from scratch to get baseline mastery
    print(f"\n{'='*60}")
    print(f"  CL BENCHMARK — {len(task_names)} tasks, {steps_per_task} steps/task")
    print(f"  Tasks: {', '.join(task_names)}")
    print(f"  Methods: {', '.join(methods_to_run)}")
    print(f"{'='*60}")

    baseline_masteries = {}
    print(f"\n  Computing baselines (no CL)...")
    model, cfg = make_model(tok)
    for op_name, _ in tasks:
        _, mastery = train_steps(model, cfg, tok, op_name, steps_per_task)
        baseline_masteries[op_name] = mastery
        # Reset model for next baseline
        model, cfg = make_model(tok)
        print(f"    {op_name}: mastery at step {mastery}")

    all_results = {}

    for method_name in methods_to_run:
        print(f"\n  --- Method: {method_name} ---")
        model, cfg = make_model(tok)

        t0 = time.time()

        # Measure initial accuracy on all tasks (should be ~0 for a fresh model)
        initial_acc = evaluate_all_ops(model, tok, cfg, task_names, num=50)
        tasks_with_init = [(op, initial_acc.get(op, 0)) for op in task_names]

        if method_name == "ewc":
            results = method_online_ewc(model, cfg, tok, tasks_with_init, steps_per_task)
        elif method_name == "fisher_archive":
            results = method_fisher_archive(model, cfg, tok, tasks_with_init, steps_per_task)
        elif method_name == "none":
            results = method_no_cl(model, cfg, tok, tasks_with_init, steps_per_task)
        elif method_name == "lwf":
            results = method_lwf(model, cfg, tok, tasks_with_init, steps_per_task)
        elif method_name == "gem":
            results = method_gem(model, cfg, tok, tasks_with_init, steps_per_task)
        elif method_name == "ogd":
            results = method_ogd(model, cfg, tok, tasks_with_init, steps_per_task)
        else:
            print(f"  Unknown method: {method_name}")
            continue

        elapsed = time.time() - t0
        results["initial"] = initial_acc

        # Compute BWT
        bwt = compute_bwt(results, task_names)
        results["backward_transfer"] = bwt

        # Compute FWT
        fwt = compute_fwt(results, task_names, baseline_masteries)
        results["forward_transfer"] = fwt

        # Overall metrics
        final_phase = f"after_task_{len(task_names)}"
        final_accs = results.get(final_phase, {}).get("accuracies", {})
        avg_retention = sum(final_accs.values()) / max(1, len(final_accs))
        total_steps = sum(
            results.get(f"after_task_{i+1}", {}).get("steps_to_mastery", steps_per_task)
            for i in range(len(task_names))
        )

        results["summary"] = {
            "avg_final_accuracy": round(avg_retention, 1),
            "total_steps_to_mastery": total_steps,
            "time_seconds": round(elapsed, 1),
            "min_bwt": round(min(bwt.values()), 1) if bwt else 0,
            "avg_fwt": round(sum(fwt.values()) / max(1, len(fwt)), 1) if fwt else 0,
        }

        all_results[method_name] = results

        # Print summary
        print(f"\n  [{method_name}] Summary:")
        print(f"    Final accuracies: {', '.join(f'{k}={v:.0f}%' for k, v in final_accs.items())}")
        print(f"    Avg retention: {avg_retention:.1f}%")
        print(f"    BWT (min): {min(bwt.values()):.1f}%" if bwt else "    BWT: N/A")
        print(f"    Total steps: {total_steps}")
        print(f"    Time: {elapsed:.1f}s")

    # ── Final comparison table ──
    print(f"\n{'='*80}")
    print(f"  CL BENCHMARK COMPARISON")
    print(f"{'='*80}")
    header = f"  {'Method':<20} | {'Avg Ret%':>8} | {'BWT%':>6} | {'FWT%':>6} | {'Steps':>7} | {'Time':>6}"
    print(header)
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}")

    for method_name in methods_to_run:
        s = all_results[method_name]["summary"]
        print(f"  {method_name:<20} | {s['avg_final_accuracy']:>7.1f}% | "
              f"{s['min_bwt']:>5.1f}% | {s['avg_fwt']:>5.1f}% | "
              f"{s['total_steps_to_mastery']:>7} | {s['time_seconds']:>5.1f}s")

    print(f"{'='*80}")

    # Save
    out_path = Path("experiments/cl_benchmark_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="CL Benchmark")
    parser.add_argument("--full", action="store_true",
                        help="Full benchmark (2000 steps/task)")
    parser.add_argument("--steps", type=int, default=0,
                        help="Custom steps per task")
    parser.add_argument("--methods", nargs="+",
                        default=list(METHODS.keys()),
                        help=f"Methods: {', '.join(METHODS.keys())}")
    parser.add_argument("--tasks", nargs="+", default=["add", "sub"],
                        help="Tasks: add, sub, mul")
    args = parser.parse_args()

    steps = args.steps or (FULL_STEPS if args.full else QUICK_STEPS)
    # Validate methods
    for m in args.methods:
        if m not in METHODS:
            print(f"Unknown method: {m}. Available: {list(METHODS.keys())}")
            sys.exit(1)

    tasks = [(op, 0.0) for op in args.tasks]

    run_benchmark(args.methods, tasks, steps)


if __name__ == "__main__":
    main()
