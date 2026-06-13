"""Batch test a partial checkpoint with sample problems."""
import torch, sys
sys.path.insert(0, '.')
from config import Config
from tokenizer import MathTokenizer
from model import MathTransformer

ckpt = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/best.pt'

cfg = Config()
tok = MathTokenizer.load('checkpoints/tokenizer.json')
cfg.vocab_size = tok.vocab_size
tok.max_seq_len = cfg.max_seq_len

model = MathTransformer(cfg)
state = torch.load(ckpt, map_location='cpu', weights_only=True)
model.load_state_dict(state['model_state_dict'])
model.eval()

problems = [
    '1+1=', '2+3=', '5+5=', '10-4=', '3*3=',
    '12+34=', '56-18=', '7*8=', '99+1=', '100-1=',
    '0+0=', '9+10=', '20-5=', '4*25=', '100/2=',
]

print(f'{"Problem":<12} {"Expected":<10} {"Got":<15} {"Match"}')
print('-' * 49)
for expr in problems:
    expected = str(eval(expr[:-1]))
    out = model.generate(tok, expr, max_new_tokens=10, temperature=0.3, top_k=3)
    pred = out.split('=')[-1].strip() if '=' in out else '?'
    pred = ''.join(c for c in pred if c.isdigit() or c == '-')
    match = 'YES' if pred == expected else 'no'
    print(f'{expr:<12} {expected:<10} {pred:<15} {match}')
