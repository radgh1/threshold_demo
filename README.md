---
title: Dynamic Confidence Thresholding Demo
emoji: "📊"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
short_description: Interactive simulation of AI confidence thresholds for human-AI collaboration
---

# 📊 Dynamic Confidence Thresholding Demo

**Human-AI Collaboration** - An interactive simulation that shows how AI systems can work perfectly with humans by learning when to ask for help!

[![Live Demo](https://img.shields.io/badge/🚀-Live_Demo-00ADD8)](https://huggingface.co/spaces/raddev1/threshold_demo)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717)](https://github.com/radgh1/threshold_demo)

## 🎯 What This App Does

Imagine you have a smart AI assistant that helps with important decisions. Sometimes the AI is very confident and can make decisions quickly. But other times, the AI gets uncertain and might make mistakes.

**This app teaches the AI when to handle decisions alone and when to ask human experts for help.** Through three different simulation modes, you can see how adjustable confidence thresholds create the perfect balance between AI efficiency and human expertise.

### 🌟 The Big Idea

Instead of asking **"Can AI replace humans?"**, we explore **"How can AI and humans collaborate most effectively?"**

This app demonstrates how to build systems where:
- 🤖 **AI handles** routine decisions with high confidence
- 👥 **Humans focus** on complex cases requiring judgment
- 🧠 **The system adapts** thresholds to optimize collaboration

## 🎮 What You Can Do

### **Run Three Types of Simulations**

#### **1. Fixed Threshold Sweep**
- Test different confidence levels across a range
- See how coverage (human involvement) affects accuracy
- Visualize the coverage-accuracy trade-off curve
- Compare performance across task domains

#### **2. Adaptive Controller**
- Watch thresholds adjust in real-time to maintain target accuracy
- See how the system responds to changing conditions
- Learn about dynamic workload optimization
- Understand adaptive human-AI collaboration

#### **3. Learning AI Controller**
- Observe how AI improves its confidence calibration over time
- See the system learn from human feedback
- Understand how AI can become a better collaborative partner
- Explore the future of adaptive AI systems

### **See Amazing Visualizations**
- **Coverage-Accuracy Curves**: Optimal operating points for different thresholds
- **Adaptive Trajectories**: Real-time threshold adjustments
- **Performance Metrics**: Accuracy, coverage, and efficiency tracking
- **Workload Distribution**: How tasks are shared between AI and humans

### **Get Smart Analysis**
- **Result Interpretation**: AI-powered insights into simulation outcomes
- **Educational Explanations**: Understand complex concepts through analogies
- **Parameter Effects**: See how different settings impact collaboration

## 🧠 How It Works (Simple Version)

### **The AI's Job**
The AI makes predictions with confidence scores for tasks like:
- 🏥 **Medical diagnosis** (analyzing patient data)
- ⚖️ **Legal review** (checking documents)
- 💻 **Code analysis** (reviewing programming)

For each task, AI provides: **"My answer is X, and I'm Y% confident"**

### **The Human Experts**
- Professional experts who can also evaluate tasks
- Performance affected by fatigue (just like real experts!)
- Provide ground truth for training and validation

### **The Smart Threshold System**
- **Confidence Routing**: Tasks below threshold go to humans
- **Adaptive Learning**: System learns optimal thresholds
- **Quality Balance**: Maintains target accuracy levels
- **Workload Optimization**: Minimizes human involvement while ensuring quality

## 🔬 Technical Details (For Experts)

### **Core Algorithm: Confidence-Based Routing**
- **Threshold Decision**: Route to humans when `confidence < τ`
- **Coverage Calculation**: `Coverage = fraction of tasks routed to humans`
- **Accuracy Balance**: `Accuracy = weighted average of AI and human performance`

### **Adaptive Controller**
- **PID-like Control**: Maintains target accuracy through threshold adjustment
- **Real-time Adaptation**: Responds to changing task difficulty and human fatigue
- **Feedback Loop**: Uses performance metrics to optimize τ values

### **Learning AI Controller**
- **Reinforcement Learning**: AI learns optimal confidence calibration
- **Feedback Integration**: Incorporates human corrections into learning
- **Confidence Calibration**: Improves alignment between confidence and actual performance

### **Fatigue Modeling**
- Expert accuracy degrades over time: `current_acc = base_acc × (1 - fatigue_factor × tasks_completed)`
- Simulates realistic human performance limitations

## 📊 What You'll Learn

### **Key Insights**
1. **Optimal Thresholds Exist**: Different tasks need different confidence levels
2. **Adaptation Matters**: Fixed thresholds can't handle changing conditions
3. **Learning Improves Collaboration**: AI can become better partners over time
4. **Balance is Key**: Too much or too little human involvement reduces effectiveness

### **Real-World Applications**
- 🏥 **Healthcare**: AI pre-screens cases, doctors review uncertain diagnoses
- ⚖️ **Legal**: AI flags issues, lawyers review complex cases
- 💻 **Software**: AI checks code, developers focus on architecture
- 🏦 **Finance**: AI detects patterns, experts investigate anomalies

## 🚀 Getting Started

### **Online Demo**
Visit the [live demo](https://huggingface.co/spaces/raddev1/threshold_demo) to try it right now!

### **Local Installation**
```bash
# Clone the repository
git clone https://github.com/radgh1/threshold_demo.git
cd threshold_demo

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

### **Quick Start Guide**
1. **Choose a Tab**: Start with "Fixed τ sweep" to understand basics
2. **Pick a Domain**: Try "radiology" for medical decision simulation
3. **Set Parameters**: Adjust dataset size and human settings
4. **Run Simulation**: Click buttons and watch the visualizations!
5. **Experiment**: Change thresholds and see how they affect results
6. **Interpret**: Use "Interpret Results" for AI-powered analysis

## 🎛️ Controls Guide

| Control | What It Does | Recommended Setting |
|---------|-------------|-------------------|
| **Task Domain** | Type of decisions (affects difficulty) | radiology |
| **Dataset Size** | Number of simulated decisions | 1000 |
| **Threshold Range** | Min/Max confidence levels to test | 0.1 - 0.9 |
| **Threshold Steps** | Number of different thresholds to try | 20 |
| **Target Accuracy** | Desired system accuracy level | 0.85 - 0.95 |
| **Human Base Accuracy** | Expert performance without fatigue | 0.8 - 0.9 |
| **Fatigue Factor** | How quickly experts tire | 0.01 - 0.05 |
| **Learning Steps** | Rounds of AI learning | 50 - 100 |

## 📈 Understanding the Results

### **Coverage-Accuracy Curves**
- **X-axis**: Coverage (fraction of tasks going to humans)
- **Y-axis**: Accuracy (fraction of correct decisions)
- **Curve Shape**: Shows trade-off between efficiency and quality
- **Optimal Points**: Where you get best accuracy for given coverage

### **Adaptive Trajectories**
- **Real-time Updates**: Watch threshold adjustments live
- **Target Tracking**: See how system maintains desired accuracy
- **Stability Indicators**: Observe when system finds optimal balance

### **Performance Tables**
- Shows detailed metrics for each simulation run
- Helps understand the impact of different parameters

## 🧪 Testing & Validation

The app includes tests validating:
- ✅ Mathematical accuracy of threshold calculations
- ✅ Proper coverage-accuracy curve generation
- ✅ Correct adaptive threshold adjustments
- ✅ Valid fatigue modeling
- ✅ Realistic simulation outcomes

Run tests with: `python -m pytest` (if pytest is available)

## 🤝 Contributing

Found a bug or have an idea? Open an issue or submit a pull request!

## 📄 License

This project demonstrates educational concepts for human-AI collaboration research.

## 🙏 Acknowledgments

Built with:
- **Gradio** for the interactive interface
- **NumPy & Pandas** for numerical computing
- **Matplotlib** for visualizations
- **Adaptive algorithms** for threshold optimization

---

**Ready to see how AI and humans can work together perfectly?** 🚀🤖👥

[Try the Live Demo Now!](https://huggingface.co/spaces/raddev1/threshold_demo)