"""고정 스텝 상한 K별 '잃는 성공 비율'과 '절약 스텝 비율' 계산.

스텝 = ai 메시지 1개 (+ 뒤따르는 user 관측). 상한 K는 K스텝을 넘는 궤적을 K에서 끊는 규칙.
- 잃는 성공 = 스텝 수 > K인 성공 궤적 / 전체 성공 궤적
- 절약 스텝 = sum(max(0, 스텝 수 - K)) / 전체 스텝 수

설치: pip install datasets pandas
실행: python code/fixed_cap_k.py
"""
import numpy as np
from datasets import load_dataset

ds = load_dataset("nebius/SWE-agent-trajectories", split="train",
                  columns=["trajectory", "target"])

steps, succ = [], []
for row in ds:
    steps.append(sum(1 for m in row["trajectory"] if m.get("role") == "ai"))
    succ.append(bool(row["target"]))
steps, succ = np.array(steps), np.array(succ)

print(f"궤적 {len(steps):,}개, 성공률 {succ.mean():.1%}, 전체 스텝 {steps.sum():,}")
print("검증용 기존 값: K=30 → 7.0% / 29.4%, K=40 → 3.7% / 20.8%, K=50 → 2.2% / 15.8%\n")
print(" K  잃는성공  절약스텝")
best = None
for K in list(range(30, 41)) + [50]:
    lost = (steps[succ] > K).mean()
    saved = np.clip(steps - K, 0, None).sum() / steps.sum()
    print(f"{K:>2}  {lost:7.1%}  {saved:7.1%}")
    if lost <= 0.05 and best is None:
        best = (K, lost, saved)
print(f"\n성공 손실 5% 이하인 최소 K = {best[0]} (잃는 성공 {best[1]:.1%}, 절약 스텝 {best[2]:.1%})")
