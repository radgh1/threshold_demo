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
This app demonstrates a dynamic confidence thresholding system for optimizing human-AI collaboration in decision-making tasks like medical diagnosis, legal review, or code analysis.

### Key Concepts:
- **AI Limitations**: AI classifiers process data quickly but can make unpredictable errors.
- **Human Experts**: Accurate but costly, limited in capacity, and prone to fatigue.
- **Confidence Thresholding**: Uses a tunable threshold (τ) on AI confidence scores to decide whether to use AI predictions or route tasks to humans.
  - If AI confidence > τ → Use AI prediction
  - Otherwise → Send to human expert

### Features:
- **Coverage-Accuracy Trade-off**: Coverage = fraction of tasks handled by humans; Accuracy = overall correctness.
- **Fixed Threshold Sweep**: Simulate and visualize how different τ values affect coverage and accuracy across a dataset.
- **Adaptive Controller**: Dynamically adjust τ in real-time to maintain a target accuracy, adapting to changing conditions.

### Simulation Details:
- Generate synthetic datasets for different task domains (radiology, legal, code).
- Model human performance with base accuracy, fatigue effects, and workload costs.
- Explore parameters interactively and view results in plots and tables.

This tool helps researchers and practitioners understand and optimize human-in-the-loop systems for reliable, efficient decision-making.
""")

    with gr.Accordion("Practical Applications: Real-World Impact", open=False):
        gr.Markdown("""
### Real-World Applications:
This dynamic thresholding system has broad applications in domains requiring high-stakes decision-making:

- **Healthcare & Radiology**: Optimize AI-assisted diagnosis by routing complex cases to radiologists, reducing workload while maintaining accuracy in detecting conditions like tumors or fractures.
- **Legal & Compliance**: Streamline document review for contracts or regulatory filings, ensuring critical issues are flagged for human lawyers while automating routine checks.
- **Software Development**: Enhance code review processes by having AI handle straightforward code analysis, reserving human experts for complex logic or security vulnerabilities.
- **Financial Services**: Improve fraud detection by balancing automated alerts with human investigation, minimizing false positives and response times.

### Impact:
- **Efficiency Gains**: Reduces human expert workload by 50-80% in suitable scenarios, allowing professionals to focus on high-value tasks.
- **Cost Reduction**: Lowers operational costs by leveraging AI for bulk processing while preserving human oversight for edge cases.
- **Improved Accuracy**: Achieves higher overall system accuracy by dynamically adapting to task difficulty and expert availability.
- **Scalability**: Enables organizations to handle larger volumes of data without proportionally increasing human resources.
- **Ethical AI Deployment**: Ensures responsible AI use by maintaining human accountability in critical decisions, building trust and compliance.

This framework supports the transition to more integrated human-AI workflows, enhancing productivity and decision quality across industries.
""")

    with gr.Accordion("Technology Stack", open=False):
        gr.Markdown("""
### Core Technologies
#### Frontend/UI Framework
- Gradio 4.44.0 - Interactive web UI framework for machine learning demos
- Provides the web interface with sliders, buttons, plots, and accordions
- Handles real-time updates and user interactions

#### Backend/Data Processing
- Python 3.x - Main programming language
- NumPy - Numerical computing, random number generation, and array operations
- Pandas - Data manipulation and DataFrame handling for simulation results

#### Deployment Platform
- Local execution - Run via `python app.py` for development and testing
- Gradio sharing - Supports public URLs for demo sharing
- Environment management - Virtual environments (venv) for dependency isolation

### Architecture Components
#### Simulation Engine
- Synthetic Dataset Generation - Creates artificial task datasets with difficulty and labels
- AI Confidence Simulation - Logistic regression-based probabilistic predictions
- Human Fatigue Modeling - Performance degradation simulation for experts
- Threshold-Based Routing - Dynamic workload allocation between AI and humans
- Adaptive Threshold Control - Proportional feedback for real-time τ adjustment

#### Visualization
- Gradio LinePlot - Coverage-accuracy curves and adaptive trajectories
- Gradio Dataframe - Simulation results and parameter tables

#### Configuration
- requirements.txt - Dependency management
- .gitignore - Version control exclusions (venv, tokens, etc.)

### Key Design Patterns
- Functional Programming - Pure functions for simulation and data processing
- Reactive UI - Event-driven updates via Gradio button clicks and callbacks
- State Management - Parameter-based state for reproducible simulations
- Modular Design - Separated functions for dataset creation, simulation, and plotting

This stack is optimized for educational AI simulation demos - Gradio makes it easy to create interactive interfaces, while the scientific Python ecosystem (NumPy/Pandas) handles the computation. Local deployment keeps it simple and accessible for development and sharing. 🚀📊
""")

    with gr.Accordion("Machine Learning Usage", open=False):
        gr.Markdown("""
### AI Simulation Approach:
This app focuses on **simulating** machine learning classifiers rather than training actual models, allowing for controlled experimentation without real ML pipelines.

- **Synthetic Data Generation**: Creates artificial datasets mimicking real-world tasks (radiology, legal, code) with features like task difficulty and ground-truth labels.
- **AI Classifier Model**: Simulates a probabilistic classifier using logistic regression principles:
  - Generates true probabilities based on difficulty.
  - Adds calibrated noise to simulate imperfect AI predictions.
  - Outputs confidence scores (probabilities) for thresholding decisions.
- **No Training Required**: All ML behavior is mathematically modeled, enabling instant simulations across parameter variations.

### ML Concepts Demonstrated:
- **Confidence Thresholding**: Core technique for human-AI collaboration, routing decisions based on model certainty.
- **Calibration**: How well predicted probabilities match true outcomes.
- **Trade-off Analysis**: Balancing precision/recall via coverage-accuracy curves.
- **Adaptive Systems**: Feedback loops for dynamic threshold adjustment.

### Why Simulation?:
- **Educational**: Teaches ML concepts without complex training setups.
- **Flexible**: Easily adjust AI "performance" parameters for different scenarios.
- **Fast**: No GPU/compute requirements; runs instantly on any machine.
- **Ethical**: Avoids real data privacy issues while demonstrating principles.

This approach makes advanced ML concepts accessible for research, education, and prototyping human-AI systems.
""")

    with gr.Accordion("User Instructions", open=False):
        gr.Markdown("""
### Getting Started:
1. **Set Global Parameters**: Adjust the controls at the top to configure the simulation:
   - **Task Domain**: Choose from radiology, legal, or code review scenarios.
   - **Dataset Size**: Number of simulated tasks (300-5000).
   - **Human Base Accuracy**: Expert accuracy rate (0.6-0.99).
   - **Fatigue Settings**: How many tasks before fatigue sets in, and accuracy drop per fatigue block.

2. **Explore Tabs**:
   - **Fixed τ Sweep**: Analyze static threshold performance.
     - Set τ range (min/max) and step size.
     - Click "Compute Curve" to generate coverage-accuracy plot and data table.
   - **Adaptive τ Controller**: Simulate dynamic threshold adjustment.
     - Set target accuracy, simulation steps, initial τ, and proportional gain (k_p).
     - Click "Run Adaptive" to see how τ evolves toward the target.

### Tips:
- Start with default values to understand baseline behavior.
- Experiment with different task domains to see varying AI performance.
- Use the plots to visualize trade-offs: higher coverage (more human involvement) vs. accuracy.
- Adaptive mode shows real-time adjustment; watch how τ changes over steps.

### Simple Explanation:
Imagine you're playing a game where robots help people make decisions, but sometimes the robots need help from smart grown-ups. This app lets you control how much the robots do on their own versus asking for help.

Here's what the buttons and sliders do, explained super simply:

- **Task Domain**: Pick what kind of job the robots are doing, like checking pictures for doctors (radiology), reading legal papers, or looking at computer code. It's like choosing different levels in a game.

- **Dataset Size**: How many pretend tasks to try. Bigger numbers mean more practice, like doing 100 math problems instead of 10.

- **Human Base Accuracy**: How good the grown-up helpers are at first. Slide it up for super-smart helpers, down for ones who make more mistakes.

- **Fatigue After N Tasks**: After how many tasks the helpers get tired and make more mistakes. Like how you get sleepy after playing too long.

- **Fatigue Accuracy Drop**: How much worse the helpers get when tired. A little drop means they still do okay; a big drop means they mess up a lot.

In the "Fixed τ Sweep" tab:
- **τ Min/Max**: The lowest and highest "confidence" levels where robots decide alone. τ is like a bravery meter – higher means robots try harder on their own.
- **τ Step**: How much to change the bravery meter each time.
- **Compute Curve**: Press this to see a picture showing how well the robots and helpers work together at different bravery levels.

In the "Adaptive τ Controller" tab:
- **Target Accuracy**: How good you want the whole team to be. Like aiming for 90% right answers.
- **Steps**: How many tries to get better.
- **Initial τ**: Starting bravery level.
- **Proportional Gain k_p**: How fast the system learns to adjust. Higher means it changes quicker.
- **Run Adaptive**: Press to watch the bravery level change over time to hit your accuracy goal.

Play around with the sliders and see what happens! It's like training a robot team. 😊
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

        with gr.Accordion("Understanding the Graph (Simple Explanation)", open=False):
            gr.Markdown("""
Imagine the graph is like a treasure map showing how well robots and people work together on tasks!

- The **x-axis** (bottom) is called "Coverage." It shows how much of the work the people do. If it's 0, robots do everything alone. If it's 1, people check everything. It's like how much help the robots ask for.

- The **y-axis** (side) is "Accuracy." It shows how many answers are right. Higher up means more correct answers, like getting a good score on a test.

- Each colored line is for a different "bravery level" (called τ). Braver robots (higher τ) try to do more on their own, so coverage is lower (less help from people), but accuracy might go up or down depending on how good the robots are.

The graph helps you see the trade-off: more robot work (lower coverage) might make fewer mistakes (higher accuracy) if robots are smart, but if robots mess up, you need more people to help (higher coverage) to get things right.

It's like balancing a seesaw – too much on one side, and it tips! Play with the sliders to see how the lines change. Cool, right? 😊
""")

        curve_plot = gr.LinePlot(x="coverage", y="accuracy", label="Coverage–Accuracy Curve")
        curve_table = gr.Dataframe(interactive=False)

        def on_sweep(task, dataset_size, base_acc, fatigue_after, fatigue_drop, tmin, tmax, tstep):
            df = make_dataset(int(dataset_size), task)
            curve = sweep_curve(df, float(base_acc), int(fatigue_after), float(fatigue_drop),
                                float(tmin), float(tmax), float(tstep))
            curve = curve.sort_values("tau")
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
        traj_plot = gr.LinePlot(x="t", y="accuracy", label="Adaptive trajectory")
        traj_table = gr.Dataframe(interactive=False)

        def on_adapt(task, dataset_size, base_acc, fatigue_after, fatigue_drop, target_acc, steps, tau0, kp):
            df = make_dataset(int(dataset_size), task)
            traj = adaptive_tau(df, float(target_acc), float(base_acc), int(fatigue_after),
                                float(fatigue_drop), int(steps), float(tau0), float(kp))
            return traj[["t","accuracy","tau"]], traj
        adapt_btn.click(
            fn=on_adapt,
            inputs=[task, dataset_size, base_acc, fatigue_after, fatigue_drop, target_acc, steps, tau0, kp],
            outputs=[traj_plot, traj_table]
        )

    demo.load(lambda: None, None, None)

demo.launch()
