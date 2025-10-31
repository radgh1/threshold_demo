import numpy as np
import pandas as pd

rng = np.random.default_rng(123)

def logistic(x): return 1/(1+np.exp(-x))

def make_dataset(n=1500, task='code'):
    if task == 'radiology':
        difficulty = rng.normal(0.0, 1.0, size=n)
    elif task == 'legal':
        difficulty = rng.normal(0.3, 1.1, size=n)
    else:
        difficulty = rng.normal(-0.2, 0.9, size=n)

    p_true = logistic(-0.7 * difficulty)
    y_true = rng.binomial(1, p_true)
    ai_logit = np.log(p_true/(1-p_true)) + rng.normal(0, 0.6, size=n)
    ai_prob = logistic(ai_logit)
    return pd.DataFrame({'difficulty': difficulty, 'p_true': p_true, 'y_true': y_true, 'ai_prob': ai_prob})

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

def adaptive_tau(df, target_acc, base_acc, fatigue_after, fatigue_drop, steps=60, tau0=0.5, k_p=0.3, optimization_mode='balanced', task_domain='radiology'):
    tau = tau0
    traj = []
    
    if optimization_mode == 'domain_recommended':
        domain_modes = {
            'radiology': 'accuracy_priority',
            'legal': 'balanced',
            'code': 'efficiency_priority'
        }
        optimization_mode = domain_modes.get(task_domain, 'balanced')
    
    if optimization_mode == 'accuracy_priority':
        accuracy_weight = 1.0
        workload_weight = 0.0
    elif optimization_mode == 'efficiency_priority':
        accuracy_weight = 0.3
        workload_weight = 0.7
    elif optimization_mode == 'robot_maximum':
        accuracy_weight = 0.1
        workload_weight = 0.9
    else:
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

# Test the improved algorithm
df = make_dataset(1000, 'code')
result = adaptive_tau(df, target_acc=0.65, base_acc=0.85, fatigue_after=60, fatigue_drop=0.1, 
                     steps=30, tau0=0.5, k_p=0.1, optimization_mode='domain_recommended', task_domain='code')

print(f'Final result: τ={result.iloc[-1]["tau"]:.3f}, accuracy={result.iloc[-1]["accuracy"]:.3f}')
print(f'Accuracy range: {result["accuracy"].min():.3f} to {result["accuracy"].max():.3f}')
print(f'Accuracy stability (last 10 steps std): {result["accuracy"].tail(10).std():.4f}')
print('Much smoother convergence with realistic target!')