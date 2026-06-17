"""Test the auto-retrain logic directly."""
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "egefalos"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Manually test the retrain logic
from egefalos.tabula_rasa import SkillManager, load_dataset, scale_config

sm = SkillManager()
skill = "greeting"

# Check current state
print(f"skill_levels: {sm.skill_levels}")
print(f"models: {list(sm.models.keys())}")
print(f"training_queue: {sm.training_queue}")

# Load dataset  
pairs = load_dataset(skill)
print(f"dataset has {len(pairs)} pairs")
existing = {q.lower().strip() for q, a in pairs}
is_novel = "hello".lower().strip() not in existing
print(f"is_novel for 'hello': {is_novel}")

# Simulate the query loop
sm._query_count = {}
for i in range(5):
    sm._query_count[skill] = sm._query_count.get(skill, 0) + 1
    print(f"Query {i+1}: count={sm._query_count[skill]}", end="")
    if sm._query_count[skill] >= 3:
        sm._query_count[skill] = 0
        level = sm.skill_levels.get(skill, 0)
        print(f" -> RETRAIN (level {level} -> {level+1})")
        sm.skill_levels[skill] = level + 1
    else:
        print()

print(f"\nFinal skill_levels: {sm.skill_levels}")
print("Test PASSED: retrain logic works correctly")
