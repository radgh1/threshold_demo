import gradio as gr
import numpy as np
import pandas as pd

rng = np.random.default_rng(123)

def logistic(x): return 1/(1+np.exp(-x))

def make_dataset(n=1500, task="radiology"):
    if task == "radiology":
        difficulty = rng.normal(0.0, 1.0, size=n)
    elif task == "legal":
        difficulty = rng.normal(0.3, 1.1, size=n)
    else:
        difficulty = rng.normal(-0.2, 0.9, size=n)

    p_true = logistic(-0.7 * difficulty)
    y_true = rng.binomial(1, p_true)
    ai_logit = np.log(p_true/(1-p_true)) + rng.normal(0, 0.6, size=n)
    ai_prob = logistic(ai_logit)
    return pd.DataFrame({"difficulty": difficulty, "p_true": p_true, "y_true": y_true, "ai_prob": ai_prob})

def simulate_humans(y_true, base_acc=0.85, fatigue_after=60, fatigue_drop=0.1):
    n = len(y_true)
    # approximate average fatigue over stream
    fatigue_blocks = np.arange(n) // fatigue_after
    eff_acc = np.clip(base_acc - fatigue_blocks*fatigue_drop, 0.5, 0.99)
    correct_mask = rng.random(n) < eff_acc
    y_pred = np.where(correct_mask, y_true, 1 - y_true)
    return y_pred

def coverage_accuracy(df, tau, base_acc=0.85, fatigue_after=60, fatigue_drop=0.1):
    y = df["y_true"].values
    ai_p = df["ai_prob"].values
    route_human = ai_p < tau
    cov = route_human.mean()
    # AI preds
    ai_pred = (ai_p >= 0.5).astype(int)
    ai_correct = (ai_pred == y)[~route_human]
    # Humans
    human_pred = simulate_humans(y[route_human], base_acc=base_acc,
                                 fatigue_after=fatigue_after, fatigue_drop=fatigue_drop)
    human_correct = (human_pred == y[route_human])
    total_correct = ai_correct.sum() + human_correct.sum()
    acc = total_correct / len(df)
    return cov, acc

def sweep_curve(df, base_acc, fatigue_after, fatigue_drop, tmin=0.0, tmax=1.0, step=0.02):
    taus = np.arange(tmin, tmax+1e-9, step)
    rows = []
    for tau in taus:
        cov, acc = coverage_accuracy(df, tau, base_acc, fatigue_after, fatigue_drop)
        rows.append({"tau": tau, "coverage": cov, "accuracy": acc})
    return pd.DataFrame(rows)

def adaptive_tau(df, target_acc, base_acc, fatigue_after, fatigue_drop, steps=60, tau0=0.5, k_p=0.3):
    tau = tau0
    traj = []
    for t in range(steps):
        cov, acc = coverage_accuracy(df, tau, base_acc, fatigue_after, fatigue_drop)
        err = target_acc - acc
        # proportional controller: if acc < target -> increase human routing (raise tau)
        tau = np.clip(tau + k_p*err, 0.0, 1.0)
        traj.append({"t": t, "tau": tau, "coverage": cov, "accuracy": acc})
    return pd.DataFrame(traj)

with gr.Blocks(title="Plan B — Dynamic Thresholding") as demo:
    gr.Markdown("# Plan B — Dynamic Confidence Thresholding")
    gr.Markdown("Fixed τ sweep vs. simple adaptive controller nudging τ toward a target accuracy.")

    with gr.Accordion("About This App", open=False):
        gr.Markdown("""
Plan B
To address the research scenario, we design a flexible workload distribution mechanism 
that enables dynamic control of the trade-off between human expert coverage and AI 
classifier efficiency, while ensuring reliable performance evaluation through the coverage-
accuracy curve. The core objective is to identify a principled, adaptive strategy for 
workload allocation that maximizes system reliability without overburdening human 
experts.
We begin by defining the system's operational constraints: (1) human experts are costly 
and limited in capacity, (2) AI classifiers have high throughput but may exhibit 
unpredictable errors, and (3) the coverage-accuracy curve serves as the primary evaluation 
metric, with coverage defined as the fraction of inputs processed by humans and accuracy 
as the fraction of correct predictions among all outputs. To assess the trade-off, we must 
control the allocation rule between humans and AI across varying input distributions.
The mechanism we propose is a dynamic confidence thresholding system with adjustable 
human-in-the-loop (HITL) trigger conditions. Specifically, for each input, the AI classifier 
produces a prediction along with a confidence score. We define a tunable threshold τ such 
that if the AI's confidence exceeds τ, the system uses the AI's prediction; otherwise, the 
input is routed to a human expert. This threshold τ becomes the primary control variable 
for workload distribution. The system maintains a coverage ratio C(τ) = proportion of 
inputs routed to humans, and an accuracy A(τ) = overall correctness rate under threshold 
τ.
To implement this, we first collect a representative dataset D of real-world inputs, stratified 
by task complexity and domain diversity (e.g., medical diagnosis, legal document review). 
We then train a robust AI classifier on a labeled subset of D and evaluate its performance 
on a held-out test set. For each input in the test set, we record the AI's prediction and 
confidence score. This forms the basis for simulating the coverage-accuracy curve under 
varying τ.
Next, we conduct controlled simulations across a grid of τ values (e.g., 0.1 to 0.9 in steps 
of 0.05). For each τ, we compute C(τ) and A(τ) by applying the threshold rule to the test 
set. The resulting curve reveals the trade-off: as τ increases (more confident AI decisions), 
coverage decreases but accuracy may improve or degrade depending on the AI's 
calibration. We also compute the expected human workload as a function of τ, defined as 
the number of inputs routed to experts, to quantify efficiency gains.
To account for human variability, we introduce a stochastic human expert model. We 
simulate human performance using a probabilistic response function: for each input, the 
expert has a base accuracy p_h and a workload cost c_h (e.g., time per decision). We 
further model expert fatigue by allowing accuracy to degrade over time when processing 
consecutive inputs. This allows us to evaluate system performance under realistic 
operational conditions, not just static accuracy.
We then extend the framework to adaptive thresholding. Instead of a fixed τ, we allow τ to 
vary based on real-time feedback: for example, if a human expert's error rate exceeds a threshold, the system lowers τ to increase human involvement. We implement a feedback 
loop that monitors human performance and adjusts τ accordingly. This adaptive 
mechanism enables the system to self-regulate based on changing task demands or 
expert availability.
To validate the mechanism, we conduct a controlled user study with expert participants 
(e.g., domain experts in healthcare or law) who label a subset of D. We compare the 
performance of the fixed-threshold and adaptive systems under identical input streams. 
We measure coverage, accuracy, and human workload across conditions. We also assess 
system responsiveness and stability during adaptation.
Finally, we analyze the results through statistical modeling. We fit a piecewise regression 
model to the coverage-accuracy curve to identify optimal τ values for different operational 
goals (e.g., maximum accuracy under minimum human coverage). We also examine the 
sensitivity of performance to changes in AI confidence calibration and human error rates.
This plan enables systematic exploration of the human-AI workload trade-off. By using a 
tunable threshold as the control variable and grounding it in empirical data and human-in-
the-loop feedback, the mechanism supports flexible, real-time adaptation. The method is 
transparent, scalable, and directly interpretable via the coverage-accuracy curve, making it 
suitable for deployment in real-world L2D systems.
""")

    with gr.Row():
        task = gr.Dropdown(["radiology", "legal", "code"], value="radiology", label="Task domain")
        dataset_size = gr.Slider(300, 5000, value=1500, step=100, label="Dataset size")
        base_acc = gr.Slider(0.6, 0.99, value=0.85, step=0.01, label="Human base accuracy")
        fatigue_after = gr.Slider(20, 200, value=60, step=5, label="Fatigue after N tasks")
        fatigue_drop = gr.Slider(0.0, 0.3, value=0.10, step=0.01, label="Fatigue accuracy drop")

    with gr.Tab("Fixed τ sweep"):
        tmin = gr.Slider(0.0, 1.0, value=0.0, step=0.02, label="τ min")
        tmax = gr.Slider(0.0, 1.0, value=1.0, step=0.02, label="τ max")
        tstep = gr.Slider(0.01, 0.2, value=0.02, step=0.01, label="τ step")
        sweep_btn = gr.Button("Compute Curve")
        curve_plot = gr.LinePlot(x="coverage", y="accuracy", color="tau", label="Coverage–Accuracy Curve")
        curve_table = gr.Dataframe(interactive=False)

        def on_sweep(task, dataset_size, base_acc, fatigue_after, fatigue_drop, tmin, tmax, tstep):
            df = make_dataset(int(dataset_size), task)
            curve = sweep_curve(df, float(base_acc), int(fatigue_after), float(fatigue_drop),
                                float(tmin), float(tmax), float(tstep))
            return curve[["coverage","accuracy","tau"]], curve
        sweep_btn.click(
            fn=on_sweep,
            inputs=[task, dataset_size, base_acc, fatigue_after, fatigue_drop, tmin, tmax, tstep],
            outputs=[curve_plot, curve_table]
        )

    with gr.Tab("Adaptive τ controller"):
        target_acc = gr.Slider(0.6, 0.99, value=0.9, step=0.01, label="Target accuracy")
        steps = gr.Slider(5, 200, value=60, step=5, label="Steps")
        tau0 = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Initial τ")
        kp = gr.Slider(0.05, 1.0, value=0.3, step=0.05, label="Proportional gain k_p")
        adapt_btn = gr.Button("Run Adaptive")
        traj_plot = gr.LinePlot(x="coverage", y="accuracy", color="tau", label="Adaptive trajectory")
        traj_table = gr.Dataframe(interactive=False)

        def on_adapt(task, dataset_size, base_acc, fatigue_after, fatigue_drop, target_acc, steps, tau0, kp):
            df = make_dataset(int(dataset_size), task)
            traj = adaptive_tau(df, float(target_acc), float(base_acc), int(fatigue_after),
                                float(fatigue_drop), int(steps), float(tau0), float(kp))
            return traj[["coverage","accuracy","tau"]], traj
        adapt_btn.click(
            fn=on_adapt,
            inputs=[task, dataset_size, base_acc, fatigue_after, fatigue_drop, target_acc, steps, tau0, kp],
            outputs=[traj_plot, traj_table]
        )

    demo.load(lambda: None, None, None)

demo.launch()
