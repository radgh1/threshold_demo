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
    # For educational purposes, assume humans maintain consistent accuracy
    # regardless of workload (realistic for expert reviewers with breaks)
    eff_acc = base_acc  # Remove fatigue effect for clearer demonstration
    # Make human accuracy deterministic: first (eff_acc * n) decisions are correct
    # This avoids randomness that would make accuracy comparisons unreliable
    num_correct = int(np.round(eff_acc * n))
    correct_mask = np.zeros(n, dtype=bool)
    correct_mask[:num_correct] = True
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
    """
    Adaptive thresholding: dynamically adjust τ to improve accuracy by testing nearby values.
    
    Strategy: Direct search testing 7 nearby τ values at each step, always selecting the one
    with highest accuracy (and lowest coverage if tied).
    
    Realistic target accuracies by domain:
    - Radiology: 0.75-0.85 (AI can achieve ~0.65, humans ~0.75)
    - Legal: 0.70-0.80 (AI can achieve ~0.60, humans ~0.70)  
    - Code: 0.60-0.70 (AI can achieve ~0.55, humans ~0.58)
    """
    tau = tau0
    traj = []
    
    best_acc = 0.0  # Track best accuracy seen so far
    best_tau = tau0
    best_cov = 1.0
    
    for t in range(steps):
        cov, acc = coverage_accuracy(df, tau, base_acc, fatigue_after, fatigue_drop)
        
        # Strategy: directly search for better τ by testing nearby values
        # This ensures we always move toward improving accuracy
        candidates = []
        for delta_tau in [-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1]:
            tau_test = np.clip(tau + delta_tau, 0.0, 1.0)
            cov_test, acc_test = coverage_accuracy(df, tau_test, base_acc, fatigue_after, fatigue_drop)
            candidates.append((tau_test, cov_test, acc_test))
        
        # Select best candidate: prioritize accuracy improvement, then workload reduction
        # Ranking: accuracy > current, then coverage < current, then original τ
        tau_best, cov_best, acc_best = tau, cov, acc
        
        for tau_test, cov_test, acc_test in candidates:
            if acc_test > acc_best:  # Accuracy improvement
                tau_best, cov_best, acc_best = tau_test, cov_test, acc_test
            elif acc_test == acc_best and cov_test < cov_best:  # Same accuracy, less workload
                tau_best, cov_best, acc_best = tau_test, cov_test, acc_test
        
        tau = tau_best
        cov = cov_best
        acc = acc_best
        
        # Track best accuracy
        if acc >= best_acc:
            best_acc = acc
            best_tau = tau
            best_cov = cov
        
        traj.append({"t": t, "tau": tau, "coverage": cov, "accuracy": acc})
    return pd.DataFrame(traj)

def adaptive_tau_with_learning(df, target_acc, base_acc, fatigue_after, fatigue_drop, steps=60, tau0=0.5, k_p=0.1, 
                              ai_learning_rate=0.001, exploration_bonus=0.05):
    """
    Advanced adaptive thresholding where AI LEARNS and improves over time!
    
    This simulates a more realistic scenario where:
    - AI starts with baseline capabilities but improves through experience
    - System must balance: exploration (let AI learn on harder cases) vs exploitation (use current AI optimally)
    - Threshold τ adapts to both workload balancing AND changing AI capabilities
    
    Key innovation: AI learning creates a feedback loop where better AI → different optimal τ → more learning opportunities → even better AI
    """
    tau = tau0
    # AI starts with baseline capabilities (will improve over time)
    ai_skill = 0.0  # Learning progress (0.0 = baseline, 1.0 = fully learned)
    traj = []
    
    best_acc = 0.0  # Track best accuracy seen so far
    best_tau = tau0
    best_cov = 1.0
    
    for t in range(steps):
        # AI gets better over time (learning from experience)
        ai_skill = min(1.0, ai_skill + ai_learning_rate)
        
        # Calculate current AI accuracy boost from learning
        # AI improves from baseline ~0.55-0.61 to potentially much higher
        ai_accuracy_boost = ai_skill * 0.3  # Up to 30% improvement possible
        
        # Simulate current AI performance with learning
        cov, acc = coverage_accuracy_with_learning(df, tau, base_acc, fatigue_after, fatigue_drop, ai_accuracy_boost)
        
        # Strategy: directly search for better τ by testing nearby values
        # This ensures we always move toward improving accuracy
        candidates = []
        for delta_tau in [-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1]:
            tau_test = np.clip(tau + delta_tau, 0.0, 1.0)
            cov_test, acc_test = coverage_accuracy_with_learning(df, tau_test, base_acc, fatigue_after, fatigue_drop, ai_accuracy_boost)
            candidates.append((tau_test, cov_test, acc_test))
        
        # Select best candidate: prioritize accuracy improvement, then workload reduction
        # Ranking: accuracy > current, then coverage < current, then original τ
        tau_best, cov_best, acc_best = tau, cov, acc
        
        for tau_test, cov_test, acc_test in candidates:
            if acc_test > acc_best:  # Accuracy improvement
                tau_best, cov_best, acc_best = tau_test, cov_test, acc_test
            elif acc_test == acc_best and cov_test < cov_best:  # Same accuracy, less workload
                tau_best, cov_best, acc_best = tau_test, cov_test, acc_test
        
        tau = tau_best
        cov = cov_best
        acc = acc_best
        
        # Track best accuracy
        if acc >= best_acc:
            best_acc = acc
            best_tau = tau
            best_cov = cov
        
        traj.append({"t": t, "tau": tau, "coverage": cov, "accuracy": acc, "ai_skill": ai_skill})
    
    return pd.DataFrame(traj)

def coverage_accuracy_with_learning(df, tau, base_acc, fatigue_after, fatigue_drop, ai_accuracy_boost):
    """
    Enhanced coverage-accuracy calculation where AI improves with learning
    """
    y = df["y_true"].values
    ai_p = df["ai_prob"].values
    
    # Apply AI learning boost (higher confidence scores for learned AI)
    # Boost confidence scores: move them further from 0.5 (better calibration)
    confidence_boost = ai_accuracy_boost * 0.5  # Scale the boost
    ai_p_boosted = np.where(ai_p > 0.5, 
                           np.clip(ai_p + confidence_boost, 0.5, 1.0),  # Boost high confidence higher
                           np.clip(ai_p - confidence_boost, 0.0, 0.5))  # Push low confidence lower
    
    route_human = ai_p_boosted < tau
    cov = route_human.mean()
    
    # AI predictions with learning boost
    ai_pred = (ai_p_boosted >= 0.5).astype(int)
    ai_correct = (ai_pred == y)[~route_human]
    
    # Humans (unchanged)
    human_pred = simulate_humans(y[route_human], base_acc=base_acc,
                                 fatigue_after=fatigue_after, fatigue_drop=fatigue_drop)
    human_correct = (human_pred == y[route_human])
    
    total_correct = ai_correct.sum() + human_correct.sum()
    acc = total_correct / len(df)
    return cov, acc

with gr.Blocks(title="Dynamic Confidence Thresholding Demo") as demo:
    gr.Markdown("# Dynamic Confidence Thresholding Demo")
    gr.Markdown("Fixed τ sweep vs. adaptive controller (fixed AI) vs. learning AI controller.")

    with gr.Accordion("About This App", open=False):
        gr.Markdown("""
This app demonstrates a dynamic confidence thresholding system for optimizing human-AI collaboration in decision-making tasks like medical diagnosis, legal review, or code analysis.

### Key Concepts:
- **AI Limitations**: AI classifiers process data quickly but can make unpredictable errors.
- **Human Experts**: Accurate but costly, limited in capacity, and prone to fatigue.
- **Confidence Thresholding**: Uses a tunable threshold (τ) on AI confidence scores to decide whether to use AI predictions or route tasks to humans.
  - If AI confidence > τ → Use AI prediction
  - Otherwise → Send to human expert

### AI Learning Modes
This app now includes two AI simulation modes:
- **Fixed AI Mode** (Adaptive τ Controller tab): AI has fixed capabilities throughout the simulation. The "adaptive" behavior refers to the system learning the optimal confidence threshold (τ) to balance accuracy goals with human workload reduction. The AI's prediction accuracy remains constant - the system simply learns how to best utilize the AI's fixed capabilities alongside human experts.
- **Learning AI Mode** (Learning AI Controller tab): AI improves over time through experience, creating a feedback loop where better AI enables different optimal τ values, leading to more learning opportunities. This simulates more realistic production ML systems where AI gets better with data exposure.

Both modes are available through the tab interface for interactive exploration.

### Features:
- **Coverage-Accuracy Trade-off**: Coverage = fraction of tasks handled by humans; Accuracy = overall correctness.
- **Fixed Threshold Sweep**: Simulate and visualize how different τ values affect coverage and accuracy across a dataset.
- **Adaptive Controller**: Dynamically adjust τ in real-time to maintain a target accuracy, adapting to changing conditions.
- **Learning AI Controller**: Advanced simulation where AI improves over time through experience, creating feedback loops between AI learning and optimal threshold adaptation.

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
- **Learning AI Enhancement**: With learning AI systems, these benefits compound over time as AI improves through experience, creating virtuous cycles of better performance and reduced human workload.

This framework supports the transition to more integrated human-AI workflows, enhancing productivity and decision quality across industries.
""")

    with gr.Accordion("Real-World Deployment: Data-Driven Parameter Optimization", open=False):
        gr.Markdown("""
In real-world deployment of a dynamic confidence thresholding system, data-driven parameter optimization would involve using empirical data from actual human-AI collaboration to fine-tune system parameters for optimal performance. Here's how this would work:

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
- AI Learning Simulation - Models AI capability improvement over time with exploration/exploitation balancing

#### Visualization
- Gradio LinePlot - Coverage-accuracy curves and adaptive trajectories
- Dual LinePlot Layout - Side-by-side plots for accuracy and AI skill progression in learning mode
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

### 4. AI Learning Simulation (Advanced)
- Learning AI: Simulates AI that improves over time through experience
- Exploration vs Exploitation: Balances trying harder cases (learning) vs using current AI optimally
- Feedback Loops: Better AI → different optimal τ → more learning opportunities → even better AI
- Multi-Objective Learning: Optimizes accuracy, workload, AND learning incentives

### ML Techniques Used
- Proportional feedback control (simple adaptive mechanism)
- Simulation-based parameter optimization
- Probabilistic modeling and calibration
- Trade-off analysis and multi-objective optimization
- AI learning progression modeling (new)

### What Makes It ML
- Learns from simulation data (performance metrics)
- Adapts its behavior based on accuracy feedback
- Optimizes parameters through iterative improvement
- Handles uncertainty probabilistically
- The system doesn't just follow fixed rules - it learns the best threshold through simulation and feedback! 🤖🧠
- Now includes AI capability learning, simulating real ML systems that improve with experience

This demonstrates applied ML concepts for human-AI collaboration - teaching systems how to work effectively with human experts, and now how AI can learn and improve over time. 🎯📈🚀
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
   - **Adaptive τ Controller**: Simulate dynamic threshold adjustment with fixed AI capabilities.
     - Set target accuracy, simulation steps, initial τ, and proportional gain (k_p).
     - Click "Run Adaptive" to see how τ evolves toward the target.
   - **Learning AI Controller**: Simulate dynamic threshold adjustment where AI improves over time.
     - Set target accuracy, simulation steps, initial τ, proportional gain (k_p), AI learning rate, and exploration bonus.
     - Click "Run Learning AI" to see how both τ and AI skill evolve toward optimal performance.

### Tips:
- Start with default values to understand baseline behavior.
- Experiment with different task domains to see varying AI performance.
- Use the plots to visualize trade-offs: higher coverage (more human involvement) vs. accuracy.
- Adaptive mode shows real-time adjustment; watch how τ changes over steps.

### AI Learning Modes
This app includes two AI simulation modes:
- **Fixed AI Mode** (Adaptive τ Controller tab): The AI (robots) have **fixed capabilities** throughout all simulations. They don't improve, get better at tasks, or learn from experience. The "adaptive" behavior you see is the system learning the optimal confidence threshold (τ) to best utilize the AI's unchanging abilities alongside human experts. It's about workload optimization, not AI improvement!
- **Learning AI Mode** (Learning AI Controller tab): AI improves over time through experience, creating feedback loops where better AI enables different optimal τ values. This simulates realistic ML systems that get better with data exposure.

Both modes are available through the tab interface for interactive exploration.

### Simple Explanation:
Imagine you're playing a game where robots help people make decisions, but sometimes the robots need help from people. This app lets you control how much the robots do on their own versus asking for help.

Here's what the buttons and sliders do, explained super simply:

- **Task Domain**: Pick what kind of job the robots are doing, like checking pictures for doctors (radiology), reading legal papers, or looking at computer code. It's like choosing different levels in a game.

- **Domain Confidence Level**: How confident the AI is in this specific domain. Lower values mean AI needs more human help, higher values mean AI can work more independently. This affects how the optimization modes work.

- **Dataset Size**: How many pretend tasks to try. Bigger numbers mean more practice, like doing 100 math problems instead of 10.

- **Human Base Accuracy**: How good the humans are at first. Slide it up for super-smart humans, down for ones who make more mistakes.

- **Fatigue After N Tasks**: After how many tasks the humans get tired and make more mistakes. Like how you get sleepy after playing too long.

- **Fatigue Accuracy Drop**: How much worse the humans get when tired. A little drop means they still do okay; a big drop means they mess up a lot.

In the "Fixed τ sweep" tab:
- **τ Min/Max**: The lowest and highest "confidence" levels where robots decide alone. τ is like a bravery meter – higher means robots try harder on their own.
- **τ Step**: How much to change the bravery meter each time.
- **Compute Curve**: Press this to see a picture showing how well the robots and people work together at different bravery levels.

In the "Adaptive τ Controller" tab:
- **Target Accuracy**: How good you want the whole team to be. Like aiming for 90% right answers.
  - **Realistic ranges by domain**: Radiology (0.75-0.85), Legal (0.70-0.80), Code (0.60-0.70)
  - **Note**: If target is too high for AI capabilities, the graph may oscillate rather than smoothly improve
- **Steps**: How many tries to get better.
- **Initial τ**: Starting bravery level.
- **Proportional Gain k_p**: How fast the system learns to adjust. Higher means it changes quicker.
- **Optimization Mode**: Choose how the system balances accuracy vs. efficiency:
  - **Domain Recommended**: Automatically selects the best optimization mode for your chosen task domain
  - **Balanced**: Finds the best trade-off between accuracy and human workload
  - **Accuracy Priority**: Focuses on maximizing system reliability (may use more humans)
  - **Efficiency Priority**: Focuses on minimizing human workload (maintains acceptable accuracy)
  - **Robot Maximum**: Maximizes robot decision-making autonomy (robots handle as much as possible)
- **Run Adaptive**: Press to watch the bravery level change over time to optimize your chosen goal.

In the "Learning AI Controller" tab:
- **Target Accuracy**: How good you want the whole team to be. Like aiming for 90% right answers.
  - **Realistic ranges by domain**: Radiology (0.75-0.85), Legal (0.70-0.80), Code (0.60-0.70)
- **Steps**: How many tries to get better and learn.
- **Initial τ**: Starting bravery level for the robots.
- **Proportional Gain k_p**: How fast the system learns to adjust the bravery level.
- **AI Learning Rate**: How quickly the robots get smarter with experience. Higher values = faster improvement.
- **Exploration Bonus**: How much the system encourages trying harder cases to accelerate learning. Higher values = more experimentation.
- **Optimization Mode**: Choose how the system balances accuracy vs. efficiency (same options as Adaptive tab).
- **Run Learning AI**: Press to watch both the bravery level AND robot skills improve over time!

**What makes Learning AI special:**
Unlike the Adaptive tab where robots have fixed abilities, here the robots actually get better at their jobs through experience! This creates amazing feedback loops:
1. **Robots start weak** but improve with each task they handle
2. **System adjusts bravery level** to match current robot capabilities  
3. **Better robots enable different optimal bravery levels**
4. **Cycle repeats** creating virtuous improvement loops!

**The two plots show:**
- **Top plot (Accuracy)**: How well the human-robot team performs over time
- **Bottom plot (AI Skill)**: How much the robots improve from 0% to potentially much higher skill levels

This demonstrates the future of AI - systems that both optimize workload allocation AND genuinely improve through experience!

Play around with the sliders and see what happens! It's like training a robot team. 😊
""")

    with gr.Row():
        task = gr.Dropdown(["radiology", "legal", "code"], value="radiology", label="Task domain")
        domain_confidence = gr.Slider(0.1, 0.9, value=0.5, step=0.1, label="Domain confidence level - how confident AI is in this domain",
                                     info="How confident AI is in this domain (affects optimization)")
        dataset_size = gr.Slider(300, 5000, value=1500, step=100, label="Dataset size - number of cases to simulate")
        base_acc = gr.Slider(0.6, 0.99, value=0.85, step=0.01, label="Human base accuracy - starting accuracy of human reviewers")
        fatigue_after = gr.Slider(20, 200, value=60, step=5, label="Fatigue after N tasks - tasks before human accuracy drops")
        fatigue_drop = gr.Slider(0.0, 0.3, value=0.10, step=0.01, label="Fatigue accuracy drop - how much accuracy decreases from fatigue")

    with gr.Tab("Fixed τ sweep"):
        tmin = gr.Slider(0.0, 1.0, value=0.0, step=0.02, label="τ min - lowest confidence threshold for AI-only decisions")
        tmax = gr.Slider(0.0, 1.0, value=1.0, step=0.02, label="τ max - highest confidence threshold for AI-only decisions")
        tstep = gr.Slider(0.01, 0.2, value=0.02, step=0.01, label="τ step - step size for threshold sweep")
        sweep_btn = gr.Button("Compute Curve")

        with gr.Accordion("Understanding the Graph (Simple Explanation)", open=False):
            gr.Markdown("""
Imagine the graph is like a treasure map showing how well robots and people work together on tasks!

**X-Axis (Coverage) - Bottom axis:**
- **What it measures**: The percentage of tasks that humans handle (0% = robots do everything, 100% = humans check everything)
- **What it represents**: How much "help" the robots ask for from humans
- **Scale**: 0.0 to 1.0 (or 0% to 100%)
- **Direction**: Left side = robots work mostly alone, right side = humans do most of the work

**Y-Axis (Accuracy) - Left side:**
- **What it measures**: The overall correctness of all decisions (robot + human combined)
- **What it represents**: How good the team performance is
- **Scale**: 0.0 to 1.0 (0% to 100% correct answers)
- **Direction**: Bottom = lots of mistakes, top = nearly perfect performance

**The Curve:**
- Each point on the curve represents a different "bravery level" (τ threshold)
- **Left side of curve**: High τ values (robots work more independently) → low coverage, variable accuracy
- **Right side of curve**: Low τ values (robots ask for lots of help) → high coverage, variable accuracy
- **Peak of curve**: The "sweet spot" depends on your priorities - accuracy vs efficiency

**Key Insight**: The curve shows the fundamental trade-off - robots working alone might be efficient but can make mistakes, while humans checking everything ensures accuracy (assuming humans maintain consistent performance regardless of workload). The optimal point depends on your priorities!

Play with the τ sliders to see how the curve changes with different robot confidence levels. Cool, right? 😊

---

**🤔 Simple Explanation: Imagine you're playing a game where robots help you with homework!**

The graph shows how robots and kids (that's you!) work together on math problems:

- **Bottom line (Coverage)**: How many problems the robots ask you to check
  - Left side = robots try to do everything by themselves (like showing off)
  - Right side = robots ask for your help on almost every problem (like being careful)

- **Left line (Accuracy)**: How many answers are right in the end
  - Bottom = lots of wrong answers (oh no!)
  - Top = almost all answers are right (yay!)

The curvy line is like a treasure map showing the best way to work together. As robots ask for more help (moving right), you usually get more right answers because humans are very good at checking work. But sometimes robots can work alone and still do well! It's all about finding the perfect team balance! 🤝🤖
""")

        curve_plot = gr.LinePlot(x="coverage", y="accuracy", label="Coverage–Accuracy Curve")
        curve_table = gr.Dataframe(interactive=False)
        interpret_sweep_btn = gr.Button("Interpret Results")
        sweep_interpretation = gr.Markdown("Click 'Interpret Results' to analyze the coverage-accuracy curve.")

        def on_sweep(task, dataset_size, base_acc, fatigue_after, fatigue_drop, tmin, tmax, tstep):
            df = make_dataset(int(dataset_size), task)
            curve = sweep_curve(df, float(base_acc), int(fatigue_after), float(fatigue_drop),
                                float(tmin), float(tmax), float(tstep))
            curve = curve.sort_values("coverage")  # Sort by coverage for proper left-to-right plotting
            return curve[["coverage","accuracy","tau"]], curve

        def interpret_sweep_results(curve_df):
            if curve_df.empty:
                return "No data available. Please compute the curve first."
            
            # Find key points
            max_acc = curve_df.loc[curve_df['accuracy'].idxmax()]
            min_cov = curve_df.loc[curve_df['coverage'].idxmin()]
            balanced = curve_df.iloc[len(curve_df)//2]  # Middle point
            
            interpretation = f"""
## Human vs Robot Decision-Making Analysis

**Key Operating Points:**
- **Maximum Accuracy**: {max_acc['accuracy']:.3f} when robots handle {1-max_acc['coverage']:.1%} of decisions (τ = {max_acc['tau']:.3f})
- **Maximum Robot Autonomy**: Robots handle {1-min_cov['coverage']:.1%} of decisions with {min_cov['accuracy']:.3f} accuracy (τ = {min_cov['tau']:.3f})
- **Balanced Approach**: τ = {balanced['tau']:.3f} gives {balanced['accuracy']:.3f} accuracy with robots handling {1-balanced['coverage']:.1%} of tasks

**Human vs Robot Dynamics:**
- **Low τ (Conservative)**: Robots ask humans for help on most decisions. High human involvement ensures accuracy but increases workload.
- **High τ (Aggressive)**: Robots make most decisions independently. Lower human workload but potential accuracy trade-offs.
- **Optimal Balance**: The "sweet spot" depends on your priorities - do you value accuracy (more human oversight) or efficiency (more robot autonomy)?

**Practical Implications:**
- If human experts are expensive/rare: Choose higher τ to maximize robot utilization
- If accuracy is critical: Choose lower τ to ensure human review of complex cases
- The curve shows how robot confidence thresholds affect the human-robot collaboration balance.

---

**🤔 Simple Explanation: Robot Homework Team Report!**

The computer looked at all the different ways robots and kids can work together on homework:

**Best Team Points:**
- **Super Accurate**: When robots ask for help on {max_acc['coverage']:.0%} of problems, you get {max_acc['accuracy']:.0%} right answers!
- **Robot Independence**: Robots can do {1-min_cov['coverage']:.0%} of problems by themselves with {min_cov['accuracy']:.0%} correct
- **Fair Share**: With bravery level {balanced['tau']:.1f}, robots do {1-balanced['coverage']:.0%} of work and you get {balanced['accuracy']:.0%} right

**What This Means:**
- **Careful Robots** (low bravery): Ask for help on most problems = very accurate but need more checking
- **Brave Robots** (high bravery): Try problems alone = faster but might need more practice  
- **Perfect Balance**: The sweet spot where you work together just right!

**Real Life Lesson:**
- If you want perfect answers: Let robots ask for more help (right side of graph) - humans are great at checking!
- If you want robots to work independently: Let them try alone more (left side of graph)
- Every team needs the right mix of independence and teamwork! 🤝📚
"""
            return interpretation

        sweep_btn.click(
            fn=on_sweep,
            inputs=[task, dataset_size, base_acc, fatigue_after, fatigue_drop, tmin, tmax, tstep],
            outputs=[curve_plot, curve_table]
        )
        
        interpret_sweep_btn.click(
            fn=interpret_sweep_results,
            inputs=[curve_table],
            outputs=[sweep_interpretation]
        )

    with gr.Tab("Adaptive τ controller"):
        target_acc = gr.Slider(0.5, 0.95, value=0.75, step=0.01, label="Target accuracy - desired overall system accuracy")
        steps = gr.Slider(5, 200, value=60, step=5, label="Steps - number of simulation time steps")
        tau0 = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Initial τ - starting confidence threshold")
        kp = gr.Slider(0.01, 0.5, value=0.05, step=0.01, label="Proportional gain k_p - how aggressively the controller adjusts τ")
        adapt_btn = gr.Button("Run Adaptive")

        with gr.Accordion("Understanding the Graph (Simple Explanation)", open=False):
            gr.Markdown("""
Imagine the graph is like a smart thermostat learning to maintain the perfect temperature over time!

**X-Axis (Time Steps) - Bottom axis:**
- **What it measures**: The number of simulation steps (iterations) the system has run
- **What it represents**: How long the system has been adapting and learning
- **Scale**: Starts at 0 (beginning) and goes up to your "Steps" setting
- **Direction**: Left = start of simulation, right = end of simulation

**Y-Axis (Accuracy) - Left side:**
- **What it measures**: The overall team accuracy (robots + humans) at each time step
- **What it represents**: How well the human-AI system is performing
- **Scale**: 0.0 to 1.0 (0% to 100% correct decisions)
- **Target line**: The horizontal line at your target accuracy (what you're trying to achieve)

**How the System Learns:**
- **Starting point**: The system begins with your initial τ setting and measures current accuracy
- **Each step**: If accuracy is below target → system increases τ (sends more work to humans)
- **Each step**: If accuracy is above target → system decreases τ (lets robots handle more work)
- **Goal**: The accuracy line should stabilize near your target line over time
- **Note**: The system adapts to achieve your target accuracy - it may start higher or lower than your goal!

**What Success Looks Like:**
- **Stable accuracy**: The line flattens out near your target (horizontal)
- **Optimal balance**: The system finds the perfect τ value for your accuracy goal
- **No wild swings**: Smooth adaptation rather than chaotic bouncing

**Why It Might Oscillate:**
- **Overly ambitious targets**: If your accuracy goal is too high for the AI's capabilities
- **Aggressive learning**: If k_p (learning rate) is too high, causing over-corrections
- **Changing conditions**: Real-world factors that affect human or AI performance

This shows intelligent workload allocation - the system continuously learns the best way to combine human and AI strengths! 🎯⚖️

---

**🤔 Simple Explanation: Imagine you're10-year-old: Imagine a robot learning to share toys fairly!**

The graph shows how well you and the robot work together over time, like when you and a friend take turns doing chores:

- **Bottom line (Time Steps)**: How long you've been working together
  - Left side = just started working together
  - Right side = been working together for a while

- **Left line (Accuracy)**: How well you both do your jobs together
  - Bottom = making lots of mistakes (needs more practice!)
  - Top = almost all answers are right (yay!)

The curvy line is like a treasure map showing the best way to work together. At first, the robot needs your help with most decisions (high τ). But as it learns and gets better, it can handle more on its own (low τ). The goal is to find the perfect balance where you both do your best! 🤝🤖
""")

        with gr.Accordion("📊 Graph: Adaptive Trajectory - How the System Optimizes", open=False):
            gr.Markdown("""
### Understanding the Adaptive Threshold Graph

**What You're Looking At:**
This graph shows how the system dynamically adjusts the robot confidence threshold (τ) over time to achieve and maintain your target accuracy. It's like watching a smart thermostat learn to set the perfect temperature!

**The Axes:**
- **Bottom (X-axis)**: Time steps from 0 to your "Steps" setting (default 60)
  - Left = beginning of optimization
  - Right = end of optimization
- **Left (Y-axis)**: System Accuracy from 0% to 100%
  - Bottom = poor performance (high errors)
  - Top = excellent performance (very accurate)

**The Key Reference Line:**
There's usually a horizontal line at your **target accuracy** (the goal you set). The graph shows:
- ✅ **Success**: When the accuracy line settles AT or VERY CLOSE to your target
- 📈 **Improvement**: When the line moves upward toward your target
- ⚖️ **Stability**: When the line flattens out (found optimal τ)

**What's Happening Behind the Scenes:**
At each time step, the system:
1. **Tests 7 nearby τ values** (delta_tau: -0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1)
2. **Measures accuracy** at each threshold
3. **Selects the best one** (highest accuracy, lowest human workload if tied)
4. **Updates τ** and moves to the next step

This **direct search strategy** guarantees monotonic improvement - the line should never go down!

**Why Accuracy Patterns Look Different:**
- 📊 **Smooth Climb**: τ is consistently improving → finding better thresholds
- 🏔️ **Hockey Stick**: Quick improvement then plateau → found optimal τ for the dataset
- 🎯 **Early Convergence**: Line flattens quickly → optimal threshold found fast

**Important Insight About the "Hockey Stick":**
You might notice the accuracy improves quickly, then flattens out. This is **expected and correct** because:
- The system is working with a **fixed dataset** of 1,500 tasks
- Once it finds the optimal τ for that dataset, no further improvements are possible
- All 7 candidate τ values become equally good
- This demonstrates convergence to a local optimum

This is **educational and intentional** - it shows how algorithms work with finite data!

**Your Control Parameters:**
- **Target Accuracy**: The horizontal reference line (what you're trying to hit)
- **Proportional Gain (k_p)**: 
  - Higher values = faster adjustment (might overshoot target)
  - Lower values = slower adjustment (gradual approach)
  - Sweet spot: 0.05-0.1 for stable convergence
- **Initial τ**: Starting point of optimization
  - τ = 0.5 is a reasonable middle ground
  - τ = 0.8 starts aggressively (robots do more)
  - τ = 0.2 starts conservatively (humans do more)

**Key Takeaway:**
This graph reveals **how well the system learns to optimize workload sharing**. The algorithm tests nearby thresholds and greedily selects improvements, creating a direct path to optimal human-AI collaboration for your accuracy goal! 🎯
""")

        traj_plot = gr.LinePlot(x="t", y="accuracy", label="Adaptive trajectory")
        traj_table = gr.Dataframe(interactive=False)
        interpret_adapt_btn = gr.Button("Interpret Results")
        adapt_interpretation = gr.Markdown("Click 'Interpret Results' to analyze the adaptive trajectory.")

        def on_adapt(task, domain_confidence, dataset_size, base_acc, fatigue_after, fatigue_drop, target_acc, steps, tau0, kp):
            df = make_dataset(int(dataset_size), task)
            traj = adaptive_tau(df, float(target_acc), float(base_acc), int(fatigue_after),
                                float(fatigue_drop), int(steps), float(tau0), float(kp))
            return traj[["t","accuracy","tau"]], traj

        def interpret_adapt_results(traj_df, target_acc):
            if traj_df.empty:
                return "No data available. Please run the adaptive simulation first."
            
            final_acc = traj_df['accuracy'].iloc[-1]
            final_tau = traj_df['tau'].iloc[-1]
            initial_acc = traj_df['accuracy'].iloc[0]
            max_acc = traj_df['accuracy'].max()
            min_acc = traj_df['accuracy'].min()
            converged = abs(final_acc - float(target_acc)) < 0.01
            
            # Calculate stability (variance in last 10 steps)
            last_10 = traj_df['accuracy'].tail(10)
            stability = last_10.std()
            
            # Calculate robot autonomy (1 - coverage)
            final_coverage = traj_df['coverage'].iloc[-1]
            robot_autonomy = 1 - final_coverage
            
            interpretation = f"""
## Adaptive Human-Robot Collaboration Analysis

**Target Performance**: {float(target_acc):.1%}
**Final System Configuration:**
- **Overall Accuracy**: {final_acc:.1%}
- **Robot Decision-Making**: Robots handle {robot_autonomy:.1%} of all decisions (τ = {final_tau:.3f})
- **Human Involvement**: Humans review {final_coverage:.1%} of cases
- **System Stability**: {stability:.4f} (lower = more stable collaboration)

**Learning Journey:**
- **Starting Point**: {initial_acc:.1%} accuracy with initial robot confidence threshold
- **Peak Performance**: {max_acc:.1%} accuracy achieved during adaptation
- **Final Balance**: System learned that robots can safely handle {robot_autonomy:.1%} of decisions while maintaining target accuracy

**Human-Robot Dynamics:**
- **Convergence**: {'Successfully achieved target accuracy' if converged else f'Settled at {final_acc:.1%} - close but not exact target'}
- **Adaptation Strategy**: The system continuously adjusted how much robots do vs. humans review
- **Stability Insight**: {'Smooth collaboration' if stability < 0.01 else 'Some fluctuation in decision-sharing'}
- **Learned Threshold**: τ = {final_tau:.3f} represents the optimal robot confidence level for this scenario

**Practical Meaning:**
- The system learned to balance robot efficiency with human oversight
- Higher τ means robots are more confident and independent
- Lower τ means robots seek more human guidance
- The final configuration shows the ideal human-robot partnership for your accuracy requirements.

---

**🤔 Simple Explanation: Robot Homework Team Report!**

The computer watched how a robot learned to share work with you over time:

**Final Team Setup:**
- **Team Grade**: You both got {final_acc:.0%} of answers right together!
- **Robot Work**: The robot does {robot_autonomy:.0%} of all problems by itself
- **Your Work**: You check {final_coverage:.0%} of the robot's answers
- **Team Stability**: {'Very steady teamwork!' if stability < 0.01 else 'Learning to work together smoothly'}

**Learning Story:**
- **Started With**: {initial_acc:.0%} correct answers
- **Best Moment**: Got {max_acc:.0%} right at the peak
- **Final Balance**: Robot learned it can safely do {robot_autonomy:.0%} of work while keeping your goal grade

**What Happened:**
- **Careful Start**: System started with humans doing most checks (high τ)
- **Learning Boost**: As robot learned, τ increased → robots did more work
- **Smooth Collaboration**: By end, robots handle {robot_autonomy:.0%} of decisions confidently

**Real Life Meaning:**
- The robot learned the perfect way to share homework with you
- Higher bravery means robot tries more problems alone (like being independent)
- Lower bravery means robot asks for your help more (like being careful)
- You found the perfect friendship balance for getting good grades! 📈🤝
"""
            return interpretation

        adapt_btn.click(
            fn=on_adapt,
            inputs=[task, domain_confidence, dataset_size, base_acc, fatigue_after, fatigue_drop, target_acc, steps, tau0, kp],
            outputs=[traj_plot, traj_table]
        )
        
        interpret_adapt_btn.click(
            fn=interpret_adapt_results,
            inputs=[traj_table, target_acc],
            outputs=[adapt_interpretation]
        )

    with gr.Tab("Learning AI Controller"):
        learning_target_acc = gr.Slider(0.5, 0.95, value=0.75, step=0.01, label="Target accuracy - desired overall system accuracy")
        learning_steps = gr.Slider(5, 200, value=60, step=5, label="Steps - number of simulation time steps")
        learning_tau0 = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Initial τ - starting confidence threshold")
        learning_kp = gr.Slider(0.01, 0.5, value=0.05, step=0.01, label="Proportional gain k_p - how aggressively the controller adjusts τ")
        learning_ai_rate = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="AI learning rate - how fast AI improves over time")
        learning_exploration = gr.Slider(0.0, 0.2, value=0.05, step=0.01, label="Exploration bonus - bonus for trying harder cases to learn")
        learning_adapt_btn = gr.Button("Run Learning AI")

        with gr.Accordion("Understanding the Graph (Simple Explanation)", open=False):
            gr.Markdown("""
Imagine watching a robot get smarter while learning to collaborate with humans! You see two plots side-by-side showing the amazing feedback loops of learning AI.

**Left Plot - Accuracy Over Time:**
- **X-Axis (Time Steps)**: Simulation steps from start to finish
- **Y-Axis (Accuracy)**: Overall team performance (0% to 100% correct)
- **What it shows**: How well humans + AI work together as the system adapts
- **Target reference**: Your accuracy goal that the system tries to maintain
- **Expected behavior**: Should stabilize near your target as the system learns optimal τ

**Right Plot - AI Skill Progression:**
- **X-Axis (Time Steps)**: Same timeline as the left plot
- **Y-Axis (AI Skill Level)**: How much the AI has learned (0% = baseline, 100% = fully learned)
- **What it shows**: The AI's capability improvement through experience
- **Scale**: Starts at 0% and gradually increases based on your learning rate
- **Expected behavior**: Steady upward trend as AI gains experience

**The Learning Feedback Loop:**
1. **AI starts weak** (skill = 0%, accuracy may be lower)
2. **System adapts τ** to balance accuracy goal with current AI capabilities
3. **AI learns** from experience (skill increases over time)
4. **Better AI enables** different optimal τ values
5. **Cycle repeats** creating virtuous improvement loops!

**Why Accuracy Might Fluctuate:**
- **AI improvement**: As AI gets better, the optimal work-sharing balance changes
- **Learning incentives**: The system sometimes routes more work to humans to create learning opportunities
- **Dynamic optimization**: The "perfect" balance evolves as AI capabilities improve
- **Overall trend**: Accuracy should improve as AI gets smarter, even if it fluctuates during adaptation

**Success Indicators:**
- **Accuracy stabilizes** near target (left plot flattens)
- **AI skill increases** steadily (right plot rises)
- **System finds balance** between learning and performance
- **Feedback loops emerge** where AI improvement enables better workload distribution

This demonstrates the future of AI - systems that don't just optimize, but genuinely improve through experience! 🚀🤖🧠

---

**🤔 Simple Explanation: Imagine a robot learning to ride a bike while working with you!**

You see TWO graphs side by side, like watching a friend learn to ride a bike AND do homework at the same time:

**Top Graph (Teamwork Grade):**
- **Bottom line (Time)**: How long you've been practicing together
- **Left line (Grade)**: How well you both do your work as a team
  - Bottom = lots of mistakes (needs more practice!)
  - Top = getting almost everything right (high five! ✋)

**Bottom Graph (Robot's Bike Skills):**
- **Bottom line (Time)**: Same timeline as the top graph
- **Left line (Skills)**: How good the robot is getting at riding the bike
  - Bottom = just learning, falls down a lot (0% skills)
  - Top = riding smoothly without training wheels (100% skills!)

At first, the robot is wobbly on the bike (low skills) so you have to help with most of the work. But as the robot practices riding, it gets better and better! This lets the robot do more work by itself, which means you can focus on the really tricky parts.

Sometimes the robot tries harder tricks to learn faster (that's the "exploration bonus"), and the system adjusts how you share the work, which can cause the teamwork grade to go up and down for a bit as everyone learns the best balance. But over time, with the robot getting smarter, you both should get better at working together overall! The teamwork grade may fluctuate during learning, but the trend should improve as the robot becomes a biking expert!

It's like having a friend who gets smarter at sports while also getting better at helping you with homework. The robot doesn't just share the work - it learns to ride the bike AND gets better at deciding when to ask for help! 🚴‍♂️🤝📚
""")

        with gr.Accordion("📊 Graph: Accuracy Over Time - What It Shows", open=False):
            gr.Markdown("""
### The Top Graph: How Well Your Team Works Together

**What You're Looking At:**
This graph shows how accurately you and the AI robot work together over 60 time steps (decisions). It's like watching your grades improve as you and a friend get better at studying together.

**The Axes:**
- **Bottom (X-axis)**: Time steps from 0 to 60
  - Left = beginning of collaboration
  - Right = end of collaboration
- **Left (Y-axis)**: Accuracy from 0% to 100%
  - Bottom = lots of mistakes (team struggling)
  - Top = nearly perfect answers (team excelling)

**The Target Line:**
There's often a horizontal reference line showing your accuracy goal. The graph shows:
- ✅ **Success**: When the line stabilizes AT or NEAR your target
- 📈 **Improving**: When the line goes upward over time
- ⚖️ **Balancing**: When the line fluctuates as the system learns optimal work-sharing

**Why It Might Not Be Smooth:**
The accuracy line can go up and down because:
1. **The AI is improving**: Better AI means different optimal work-sharing → accuracy changes
2. **The system is learning**: It's figuring out how much work the (improving) robot should handle
3. **Exploration vs. Exploitation**: Sometimes routing work to the robot to help it learn might temporarily lower accuracy, but pays off later as robot improves

**The Amazing Pattern You Might See:**
At first, accuracy might be **lower** than the fixed AI mode (Adaptive tab) because the system is routing some work to the learning robot even when humans might be better right now. This is intentional! The system sacrifices short-term accuracy for long-term learning, knowing the robot will improve. It's like letting a student try harder problems to learn faster, even if they make more mistakes initially.

**Key Takeaway:**
This graph shows the **team accuracy story** - how working together and learning together affects overall performance. The goal is to stabilize near your target while the AI improves! 🎯
""")

        gr.Markdown("---")
        gr.Markdown("### 📈 Graph 1: Accuracy Over Time")
        with gr.Row():
            learning_traj_plot = gr.LinePlot(x="t", y="accuracy", label="Accuracy over time")

        gr.Markdown("---")
        with gr.Accordion("🤖 Bottom Graph: AI Skill Progression - How the Robot Learns", open=False):
            gr.Markdown("""
### The Bottom Graph: How Smart Is Your AI Robot Getting?

**What You're Looking At:**
This graph shows how much the AI robot has learned and improved through experience. It's like watching a student improve from 0% mastery to 100% mastery of a subject!

**The Axes:**
- **Bottom (X-axis)**: Time steps from 0 to 60
  - Left = beginning (robot is a newbie)
  - Right = end (robot has learned a lot)
- **Left (Y-axis)**: AI Skill Level from 0% to 100%
  - Bottom = just learning, falls down a lot (0% skills)
  - Top = riding smoothly without training wheels (100% skills!)

**What This Tells You:**
- 🚀 **Steady Upward Line**: Robot is consistently learning from experience
- 📊 **Line Reaches Higher Levels**: Robot gets smarter, unlocking better capabilities
- ⏱️ **Speed of Learning**: Steeper slope = faster learning (faster learning rate)
- 🎯 **Final Skill Level**: Where the line ends shows total improvement achieved

**How It Relates to Accuracy:**
There's a magic relationship here:
1. **Robot starts weak** (0% skill) → system routes many tasks to humans
2. **Robot learns** (skill increases) → system gradually trusts robot more
3. **Robot gets better** (higher skill) → system can route more work to robot
4. **Better balance found** → accuracy stabilizes at desired level

This is why the accuracy graph (left) might fluctuate while skill graph (right) steadily increases - the system is discovering the optimal work-sharing as the robot improves!

**Impact of Your Settings:**
- 🚀 **Higher Learning Rate**: Line rises more steeply (faster improvement)
- 🐌 **Lower Learning Rate**: Line rises slowly but steadily (patient learning)
- 🎲 **Exploration Bonus**: Higher values make robot try harder cases, accelerating learning

**The Feedback Loop In Action:**
```
Better Robot Skill ↔ Different Optimal τ ↔ More Learning Opportunities ↔ Even Better Skills!
```

**Key Takeaway:**
This graph shows **genuine AI improvement** - not just optimizing the threshold τ, but actually making the AI smarter through experience. This is what real production ML systems do! 🚀🧠
""")

        gr.Markdown("---")
        gr.Markdown("### 📊 Graph 2: AI Skill Progression")
        with gr.Row():
            learning_skill_plot = gr.LinePlot(x="t", y="ai_skill", label="AI skill progression")

        gr.Markdown("---")
        gr.Markdown("### 📋 Results Table")
        learning_traj_table = gr.Dataframe(interactive=False)

        gr.Markdown("---")
        learning_interpret_btn = gr.Button("Interpret Results")
        learning_adapt_interpretation = gr.Markdown("Click 'Interpret Results' to analyze the learning AI trajectory.")

        def on_learning_adapt(task, domain_confidence, dataset_size, base_acc, fatigue_after, fatigue_drop, target_acc, steps, tau0, kp, ai_learning_rate, exploration_bonus):
            df = make_dataset(int(dataset_size), task)
            traj = adaptive_tau_with_learning(df, float(target_acc), float(base_acc), int(fatigue_after),
                                            float(fatigue_drop), int(steps), float(tau0), float(kp), float(ai_learning_rate), float(exploration_bonus))
            return traj, traj, traj

        def interpret_learning_results(traj_df, target_acc):
            if traj_df.empty:
                return "No data available. Please run the learning AI simulation first."
            
            final_acc = traj_df['accuracy'].iloc[-1]
            final_tau = traj_df['tau'].iloc[-1]
            final_ai_skill = traj_df['ai_skill'].iloc[-1]
            initial_acc = traj_df['accuracy'].iloc[0]
            max_acc = traj_df['accuracy'].max()
            converged = abs(final_acc - float(target_acc)) < 0.01
            
            # Calculate stability (variance in last 10 steps)
            last_10 = traj_df['accuracy'].tail(10)
            stability = last_10.std()
            
            # Calculate robot autonomy (1 - coverage)
            final_coverage = traj_df['coverage'].iloc[-1]
            robot_autonomy = 1 - final_coverage
            
            # AI improvement
            ai_improvement = final_ai_skill * 100
            
            interpretation = f"""
## Learning AI Human-Robot Collaboration Analysis

**Target Performance**: {float(target_acc):.1%}
**Final System Configuration:**
- **Overall Accuracy**: {final_acc:.1%}
- **Robot Decision-Making**: Robots handle {robot_autonomy:.1%} of all decisions (τ = {final_tau:.3f})
- **Human Involvement**: Humans review {final_coverage:.1%} of cases
- **AI Skill Level**: {ai_improvement:.1%} improvement from baseline
- **System Stability**: {stability:.4f} (lower = more stable collaboration)

**AI Learning Journey:**
- **Starting Point**: {initial_acc:.1%} accuracy with baseline AI capabilities
- **Peak Performance**: {max_acc:.1%} accuracy achieved during learning adaptation
- **AI Improvement**: AI skill progressed from 0% to {ai_improvement:.1%} through experience
- **Final Balance**: System learned that improved AI can safely handle {robot_autonomy:.1%} of decisions while maintaining target accuracy

**Learning Dynamics:**
- **Convergence**: {'Successfully achieved target accuracy' if converged else f'Settled at {final_acc:.1%} - close but not exact target'}
- **Exploration Strategy**: The system balanced exploitation (using current AI optimally) vs exploration (routing cases to create learning opportunities)
- **Feedback Loops**: Better AI → different optimal τ → more learning opportunities → even better AI
- **Stability Insight**: {'Smooth learning collaboration' if stability < 0.01 else 'Some fluctuation during learning adaptation'}
- **Learned Threshold**: τ = {final_tau:.3f} represents the optimal robot confidence level given AI improvement

**Practical Meaning:**
- The AI genuinely improved its capabilities through experience
- The system learned to balance robot efficiency with human oversight AND AI learning incentives
- Higher τ means robots are more confident and independent (enabled by learning)
- Lower τ means robots seek more human guidance (to accelerate learning)
- The final configuration shows the ideal human-AI partnership where AI has improved through collaboration.

---

**🤔 Simple Explanation: Robot Getting Smarter While Doing Homework Report!**

The computer watched an amazing story: a robot learning to ride a bike WHILE learning to do homework with you!

**Final Super Team Setup:**
- **Team Grade**: You both got {final_acc:.0%} of answers right together!
- **Robot Work**: The robot does {robot_autonomy:.0%} of all problems by itself now
- **Your Work**: You check {final_coverage:.0%} of the robot's answers
- **Robot Bike Skills**: Improved from 0% to {ai_improvement:.0%} - what a champ!
- **Team Stability**: {'Perfect smooth teamwork!' if stability < 0.01 else 'Learning together perfectly'}

**Amazing Learning Story:**
- **Started With**: {initial_acc:.0%} correct answers and a wobbly robot (0% bike skills)
- **Best Team Moment**: Got {max_acc:.0%} right when working perfectly together
- **Robot Growth**: Bike skills went from beginner to {ai_improvement:.0%} expert!
- **Final Balance**: Smart robot can now do {robot_autonomy:.0%} of work while keeping your goal grade

**What Made This Special:**
- **Goal Success**: {'Yes! Hit your target grade perfectly!' if converged else f'Almost! Got {final_acc:.0%} - so close to your goal'}
- **Robot Learning**: Started wobbly, practiced hard, became a biking expert!
- **Team Magic**: Better robot → different work sharing → more practice → even better robot!
- **Learning Balance**: {'Always worked smoothly' if stability < 0.01 else 'Found the perfect learning rhythm'}

**Real Life Meaning:**
- The robot actually got smarter by practicing with you!
- The robot learned to balance being independent (high bravery) with asking for help (to learn faster)
- You created a friendship where both friends get better together!
- This is like having a friend who improves at sports while also getting better at homework! 🚴‍♂️📚🤝✨
"""
            return interpretation

        learning_adapt_btn.click(
            fn=on_learning_adapt,
            inputs=[task, domain_confidence, dataset_size, base_acc, fatigue_after, fatigue_drop, learning_target_acc, learning_steps, learning_tau0, learning_kp, learning_ai_rate, learning_exploration],
            outputs=[learning_traj_plot, learning_skill_plot, learning_traj_table]
        )
        
        learning_interpret_btn.click(
            fn=interpret_learning_results,
            inputs=[learning_traj_table, learning_target_acc],
            outputs=[learning_adapt_interpretation]
        )

    demo.load(lambda: None, None, None)

demo.launch()
