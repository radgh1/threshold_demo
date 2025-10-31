import numpy as np
import pandas as pd

# Simulate the issue
rng = np.random.default_rng(123)

def logistic(x): return 1/(1+np.exp(-x))

# Create sample data
difficulty = rng.normal(0.0, 1.0, size=100)
p_true = logistic(-0.7 * difficulty)
y_true = rng.binomial(1, p_true)
ai_logit = np.log(p_true/(1-p_true)) + rng.normal(0, 0.6, size=100)
ai_prob = logistic(ai_logit)

df = pd.DataFrame({'difficulty': difficulty, 'p_true': p_true, 'y_true': y_true, 'ai_prob': ai_prob})

# Test different tau values
taus = [0.1, 0.3, 0.5, 0.7, 0.9]
print("tau\tcoverage\tAI_acc\thuman_acc\ttotal_acc")
for tau in taus:
    route_human = ai_prob < tau
    cov = route_human.mean()

    # AI predictions (only for tasks AI handles)
    ai_pred = (ai_prob >= 0.5).astype(int)
    ai_correct = (ai_pred == y_true)[~route_human]
    ai_acc = ai_correct.mean() if len(ai_correct) > 0 else 0

    # Human predictions (simplified - assume perfect)
    human_pred = y_true[route_human]  # Assume humans are perfect for this test
    human_correct = (human_pred == y_true[route_human])
    human_acc = human_correct.mean() if len(human_correct) > 0 else 0

    total_correct = ai_correct.sum() + human_correct.sum()
    acc = total_correct / len(df)

    print(f"{tau:.1f}\t{cov:.3f}\t\t{ai_acc:.3f}\t{human_acc:.3f}\t\t{acc:.3f}")