"""Quick test: train subtraction with borrow scratchpad fix."""
import sys
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

import torch
from tabula_rasa.config import Config
from tabula_rasa.model import MathTransformer, count_parameters
from tabula_rasa.tokenizer import MathTokenizer
import train_specialist as ts
from torch.utils.data import DataLoader, Dataset

log = open("test_borrow_log.txt", "w", encoding="utf-8")
def p(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()

# Config
cfg = Config()
cfg.d_model = 64
cfg.n_layers = 3
cfg.n_heads = 4
cfg.d_ff = 256
cfg.min_digits = 1
cfg.max_digits = 2
cfg.use_reversed = True
cfg.use_scratchpad = True
cfg.use_loss_masking = True
cfg.batch_size = 32
cfg.max_steps = 300
cfg.eval_every = 100
cfg.eval_samples = 100
cfg.eval_max_tokens = 12
cfg.vocab_size = 50

tok = MathTokenizer()
model = MathTransformer(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=0.001)

p(f"Model: {count_parameters(model):,} params")
p("Scratchpad validation:")
all_ok = True
for _ in range(10):
    expr, ans = ts.generate_problem("sub", min_digits=1, max_digits=2,
                                     reversed=True, scratchpad=True, cot=False)
    parsed = ts.parse_scratchpad_answer(ans)
    rev = parsed[::-1].lstrip("0") or "0"
    # Verify by computing actual answer
    op = expr.replace("=", "")
    if "-" in op:
        parts = op.split("-")
        a_val = int(parts[0][::-1])  # reversed input
        b_val = int(parts[1][::-1])
        expected = str(max(a_val,b_val) - min(a_val,b_val))
        ok = rev == expected
        if not ok: all_ok = False
        p(f"  {a_val}-{b_val}={expected} | prompt={expr!r} | sp={ans!r} -> {rev} {'✓' if ok else '✗'}")

p(f"All scratchpad correct: {all_ok}")

class SubDataset(Dataset):
    def __init__(self, cfg, tok):
        self.data = []
        max_seq = cfg.max_seq_len
        pad_id = tok.pad_id
        for _ in range(2000):
            expr, ans = ts.generate_problem("sub", cfg.min_digits, cfg.max_digits,
                                              reversed=True, scratchpad=True, cot=False)
            text = f"{expr}={ans}"
            ids = tok.encode(text, add_special_tokens=True)
            if len(ids) > max_seq:
                ids = ids[:max_seq]
            ids = ids + [pad_id] * (max_seq - len(ids))
            self.data.append((torch.tensor(ids[:-1]), torch.tensor(ids[1:])))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

ds = SubDataset(cfg, tok)
loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

import time
t_start = time.time()
p(f"\nTraining {cfg.max_steps} steps...")
for step in range(cfg.max_steps):
    for bx, by in loader:
        opt.zero_grad()
        _, loss, _ = model(bx, by)
        loss.backward()
        opt.step()
    if (step + 1) % cfg.eval_every == 0 or step == 0:
        acc = ts.evaluate(model, tok, cfg, "sub", num=cfg.eval_samples)
        elapsed = time.time() - t_start
        p(f"  Step {step+1:>3}/{cfg.max_steps} | loss={loss.item():.4f} | acc={acc:.1f}% | {elapsed:.0f}s")
        model.train()

elapsed = time.time() - t_start
p(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
acc = ts.evaluate(model, tok, cfg, "sub", num=500)
p(f"Final accuracy (1-2 digit, scratchpad): {acc:.1f}%")
log.close()
