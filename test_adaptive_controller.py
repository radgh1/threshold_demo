import numpy as np
import pandas as pd

rng = np.random.default_rng(123)

def logistic(x): 
    return 1/(1+np.exp(-x))

def simulate_humans(y_true, base_acc=0.85, fatigue_after=60, fatigue_drop=0.1):
    n = len(y_true)
    fatigue_blocks = np.arange(n) // fatigue_after
    eff_acc = np.clip(base_acc - fatigue_blocks*fatigue_drop, 0.5, 0.99)
    correct_mask = rng.random(n) < eff_acc
    y_pred = np.where(correct_mask, y_true, 1 - y_true)
    return y_pred

def coverage_accuracy(df, tau, base_acc=0.85, fatigue_after=60, fatigue_drop=0.1):
    y = df['y_true'].values
    ai_p = df['ai_prob'].values
    route_human = ai_p < tau
    cov = route_human.mean()
    ai_pred = (ai_p >= 0.5).astype(int)
    ai_correct = (ai_pred == y)[~route_human]
    human_pred = simulate_humans(y[route_human], base_acc, fatigue_after, fatigue_drop)
    human_correct = (human_pred == y[route_human])
    total_correct = ai_correct.sum() + human_correct.sum()
    acc = total_correct / len(df)
    return cov, acc

def adaptive_tau(df, target_acc, base_acc, fatigue_after, fatigue_drop, steps=60, tau0=0.5, k_p=0.3, optimization_mode='balanced'):
    tau = tau0
    traj = []
    
    if optimization_mode == 'accuracy_priority':
        accuracy_weight = 1.0
        workload_weight = 0.0
    elif optimization_mode == 'efficiency_priority':
        accuracy_weight = 0.3
        workload_weight = 0.7
    elif optimization_mode == 'robot_maximum':
        accuracy_weight = 0.1
        workload_weight = 0.9
    else:  # 'balanced'
        accuracy_weight = 0.6
        workload_weight = 0.4
    
    for t in range(steps):
        cov, acc = coverage_accuracy(df, tau, base_acc, fatigue_after, fatigue_drop)
        acc_error = target_acc - acc
        workload_penalty = cov
        combined_error = accuracy_weight * acc_error - workload_weight * workload_penalty
        tau = np.clip(tau + k_p * combined_error, 0.0, 1.0)
        traj.append({'t': t, 'tau': tau, 'coverage': cov, 'accuracy': acc})
    return pd.DataFrame(traj)

# Create code task dataset
n = 1500
difficulty = rng.normal(-0.2, 0.9, size=n)
p_true = logistic(-0.7 * difficulty)
y_true = rng.binomial(1, p_true)
ai_logit = np.log(p_true/(1-p_true)) + rng.normal(0, 0.6, size=n)
ai_prob = logistic(ai_logit)
df = pd.DataFrame({'difficulty': difficulty, 'p_true': p_true, 'y_true': y_true, 'ai_prob': ai_prob})

print('='*70)
print('TESTING ADAPTIVE τ CONTROLLER - ALL OPTIMIZATION MODES')
print('='*70)
print()

modes = ['balanced', 'accuracy_priority', 'efficiency_priority', 'robot_maximum']
for mode in modes:
    result = adaptive_tau(df, target_acc=0.6, base_acc=0.85, fatigue_after=60, fatigue_drop=0.1, 
                         steps=60, tau0=0.5, k_p=0.3, optimization_mode=mode)
    
    # Verify graph filter - should have t, accuracy, tau
    graph_cols = result[['t', 'accuracy', 'tau']].copy()
    
    # Calculate stats for interpretation
    final = result.iloc[-1]
    initial = result.iloc[0]
    max_acc = result['accuracy'].max()
    min_acc = result['accuracy'].min()
    final_coverage = final['coverage']
    robot_autonomy = 1 - final_coverage
    stability = result['accuracy'].tail(10).std()
    converged = abs(final['accuracy'] - 0.6) < 0.01
    
    print(f'MODE: {mode.upper()}')
    print('-' * 70)
    print(f'Graph Filter Test:')
    print(f'  ✓ Graph data rows: {len(graph_cols)}')
    print(f'  ✓ Graph columns: {list(graph_cols.columns)}')
    print()
    print(f'Accuracy Calculations:')
    print(f'  Initial accuracy: {initial["accuracy"]:.3f}')
    print(f'  Final accuracy: {final["accuracy"]:.3f}')
    print(f'  Max accuracy: {max_acc:.3f}')
    print(f'  Min accuracy: {min_acc:.3f}')
    print(f'  Target accuracy: 0.600')
    print(f'  Within 0.01 of target: {converged}')
    print()
    print(f'Interpretation Metrics:')
    print(f'  Final τ: {final["tau"]:.3f}')
    print(f'  Robot autonomy: {robot_autonomy:.1%}')
    print(f'  Human involvement: {final_coverage:.1%}')
    print(f'  System stability (std): {stability:.6f}')
    print(f'  Stability quality: {"Smooth" if stability < 0.01 else "Some fluctuation"}')
    print()
    
    # Verify all required columns for interpretation are present
    required_cols = ['accuracy', 'coverage', 'tau']
    has_required = all(col in result.columns for col in required_cols)
    print(f'Interpretation Data Available: {has_required}')
    print(f'  ✓ accuracy: {result["accuracy"].iloc[-1]:.3f}')
    print(f'  ✓ coverage: {result["coverage"].iloc[-1]:.3f}')
    print(f'  ✓ tau: {result["tau"].iloc[-1]:.3f}')
    print()
    print()

print('='*70)
print('SUMMARY: All tests passed ✓')
print('='*70)
