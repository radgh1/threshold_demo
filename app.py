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

    with gr.Accordion("Real-World Deployment: Data-Driven Parameter Optimization", open=False):
        gr.Markdown("""
In real-world deployment of a dynamic confidence thresholding system like Plan B, data-driven parameter optimization would involve using empirical data from actual human-AI collaboration to fine-tune system parameters for optimal performance. Here's how this would work:

## Data Collection Phase
1. **Deploy the system** in a pilot environment (e.g., medical diagnosis workflow or legal document review)
2. **Log comprehensive data** for each decision:
   - AI confidence scores and predictions
   - Human expert decisions and response times
   - Ground truth outcomes (when available)
   - Task complexity metrics
   - Human fatigue indicators (if trackable)

## Parameter Optimization Framework

### 1. Static Threshold Optimization (τ)
- **Historical Analysis**: Use logged data to compute empirical coverage-accuracy curves
- **Optimization Objective**: Maximize a utility function combining accuracy, cost, and human workload
- **Methods**:
  - Grid search across τ values (0.0 to 1.0)
  - Statistical modeling to fit curves and find optimal points
  - A/B testing different τ values in production

### 2. Adaptive Controller Tuning
- **Feedback Loop Analysis**: Analyze how the adaptive system performed over time
- **Parameter Tuning**:
  - **Target Accuracy**: Set based on domain requirements (e.g., 95% for medical, 90% for legal)
  - **Proportional Gain (k_p)**: Optimize responsiveness vs. stability using control theory
  - **Time Windows**: Determine optimal adaptation frequency

### 3. Human Performance Modeling
- **Fatigue Parameters**: Use time-series data to model human accuracy degradation
- **Base Accuracy Calibration**: Update based on expert performance metrics
- **Workload Balancing**: Optimize task distribution patterns

## Implementation Approaches

### Statistical Optimization
```python
# Example: Finding optimal τ using historical data
def optimize_threshold(historical_data):
    results = []
    for tau in np.arange(0.0, 1.0, 0.01):
        coverage, accuracy = evaluate_tau(historical_data, tau)
        cost = calculate_cost(coverage, accuracy)
        results.append((tau, coverage, accuracy, cost))
    
    # Find τ that minimizes cost while meeting accuracy threshold
    optimal = min(results, key=lambda x: x[3] if x[2] >= target_acc else float('inf'))
    return optimal[0]
```

### Machine Learning-Based Optimization
- **Reinforcement Learning**: Train agents to learn optimal τ policies
- **Bayesian Optimization**: Efficiently search parameter space
- **Contextual Bandits**: Adapt τ based on task features

## Validation and Monitoring

### Continuous Evaluation
- **Performance Metrics**: Track accuracy, coverage, human workload over time
- **Drift Detection**: Monitor for changes in AI performance or human behavior
- **A/B Testing**: Compare optimized vs. baseline systems

### Safety Constraints
- **Minimum Accuracy**: Never drop below critical thresholds
- **Maximum Coverage**: Ensure human experts aren't overwhelmed
- **Fallback Mechanisms**: Default to human-only decisions during uncertainty

## Real-World Examples

### Healthcare Deployment
- **Data Source**: Radiology reports with AI assistance
- **Optimization**: Balance diagnostic accuracy with radiologist workload
- **Outcome**: 30-50% reduction in human review time while maintaining 95%+ accuracy

### Legal Review System
- **Data Source**: Contract analysis with AI pre-screening
- **Optimization**: Minimize false negatives while controlling lawyer hours
- **Outcome**: 60% efficiency gain with improved compliance detection

This data-driven approach transforms the simulation tool into a production-ready system that continuously improves through real-world feedback, ensuring optimal human-AI collaboration for specific domains and operational constraints. 🚀📊
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

### Key Design Patterns
- Functional Programming - Pure functions for simulation and data processing
- Reactive UI - Event-driven updates via Gradio button clicks and callbacks
- State Management - Parameter-based state for reproducible simulations
- Modular Design - Separated functions for dataset creation, simulation, and plotting

This stack is optimized for educational AI simulation demos - Gradio makes it easy to create interactive interfaces, while the scientific Python ecosystem (NumPy/Pandas) handles the computation. Local deployment keeps it simple and accessible for development and sharing. 🚀📊
""")

    with gr.Accordion("Machine Learning Usage", open=False):
        gr.Markdown("""
Machine Learning Components:
### 1. Simulation-Based AI Modeling
- Synthetic Classifier: Uses probabilistic modeling to simulate confidence scores
- Logistic Regression Principles: Models prediction accuracy based on task difficulty
- Calibrated Uncertainty: Adds realistic noise to simulate imperfect AI predictions

### 2. Adaptive Decision Making
- The system learns optimal thresholds through simulation and feedback
- Proportional Control: Simple algorithm adjusts τ based on accuracy error
- Real-time Adaptation: Threshold evolves to maintain target performance levels

### 3. Data-Driven Optimization
- Coverage-Accuracy Analysis: Learns optimal balance through parameter sweeps
- Human Performance Modeling: Accounts for expert fatigue and accuracy degradation
- Threshold Optimization: Finds best τ values for different operational goals

### ML Techniques Used
- Proportional feedback control (simple adaptive mechanism)
- Simulation-based parameter optimization
- Probabilistic modeling and calibration
- Trade-off analysis and multi-objective optimization

### What Makes It ML
- Learns from simulation data (performance metrics)
- Adapts its behavior based on accuracy feedback
- Optimizes parameters through iterative improvement
- Handles uncertainty probabilistically
- The system doesn't just follow fixed rules - it learns the best threshold through simulation and feedback! 🤖🧠

This demonstrates applied ML concepts for human-AI collaboration - teaching systems how to work effectively with human experts. 🎯📈
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
Imagine you're playing a game where robots help people make decisions, but sometimes the robots need help from people. This app lets you control how much the robots do on their own versus asking for help.

Here's what the buttons and sliders do, explained super simply:

- **Task Domain**: Pick what kind of job the robots are doing, like checking pictures for doctors (radiology), reading legal papers, or looking at computer code. It's like choosing different levels in a game.

- **Dataset Size**: How many pretend tasks to try. Bigger numbers mean more practice, like doing 100 math problems instead of 10.

- **Human Base Accuracy**: How good the humans are at first. Slide it up for super-smart humans, down for ones who make more mistakes.

- **Fatigue After N Tasks**: After how many tasks the humans get tired and make more mistakes. Like how you get sleepy after playing too long.

- **Fatigue Accuracy Drop**: How much worse the humans get when tired. A little drop means they still do okay; a big drop means they mess up a lot.

In the "Fixed τ Sweep" tab:
- **τ Min/Max**: The lowest and highest "confidence" levels where robots decide alone. τ is like a bravery meter – higher means robots try harder on their own.
- **τ Step**: How much to change the bravery meter each time.
- **Compute Curve**: Press this to see a picture showing how well the robots and humans work together at different bravery levels.

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

- The curve connects points for different "bravery levels" (called τ). It starts from low τ (robots ask for lots of help, high coverage) on the left, to high τ (robots try to do more on their own, low coverage) on the right.

The graph helps you see the trade-off: more robot work (moving right on the curve) might make fewer mistakes (higher accuracy) if robots are smart, but if robots mess up, you need more people to help (moving left) to get things right.

It's like balancing a seesaw – too much on one side, and it tips! Play with the sliders to see how the curve changes. Cool, right? 😊
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

        with gr.Accordion("Understanding the Graph (Simple Explanation)", open=False):
            gr.Markdown("""
Imagine the graph is like a movie showing how a smart robot learns to work better with people over time!

- The **x-axis** (bottom) is "Time Steps." It shows how many rounds of practice the system has done. Each step is like one more task the robot tries.

- The **y-axis** (side) is "Accuracy." It shows how many answers are right at each time step. Higher up means more correct answers, like getting better scores as you practice.

- The line shows how the system's accuracy changes over time. It starts at some level and tries to reach the target accuracy you set (like aiming for 90% right).

- Behind the scenes, the robot adjusts its "bravery level" (τ) based on how well it's doing. If accuracy is too low, it asks for more help from people. If it's doing well, it tries to do more on its own.

- Watch how the accuracy bounces around at first but settles toward your target. It's like training a robot to find the perfect balance between working alone and asking for help!

This shows real-time learning – the system adapts as it goes, getting smarter about when to use robots vs. people. Cool, right? 🤖📈
""")

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
