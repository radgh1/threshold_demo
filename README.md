# Plan B — Dynamic Confidence Thresholding Demo

An interactive web application demonstrating dynamic confidence thresholding systems for optimizing human-AI workload distribution in decision-making tasks.

## Description

This app simulates "Plan B", a flexible mechanism for balancing AI efficiency and human expert coverage through adjustable confidence thresholds. It explores the trade-off between coverage (human involvement) and accuracy in scenarios like medical diagnosis, legal review, and code analysis.

### Key Features
- **Fixed Threshold Sweep**: Analyze coverage-accuracy trade-offs across different confidence levels
- **Adaptive Controller**: Real-time threshold adjustment to maintain target accuracy
- **Interactive Simulations**: Adjust parameters for task domains, human performance, and fatigue effects
- **Visual Analytics**: Coverage-accuracy curves and adaptive trajectories
- **Educational Content**: Built-in explanations and simple analogies for concepts

## Installation

1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # macOS/Linux
   ```

## Usage

Run the application:
```bash
python app.py
```

Open the provided local URL in your browser to access the interactive interface.

### Controls Overview
- **Task Domain**: Select scenario (radiology, legal, code)
- **Dataset Size**: Number of simulated tasks
- **Human Settings**: Base accuracy and fatigue parameters
- **Threshold Controls**: Range and step for fixed sweeps, or adaptive parameters

## Tech Stack

- **Frontend/UI**: Gradio 4.44.0 - Interactive web interface
- **Backend**: Python 3.x with NumPy and Pandas
- **Simulation**: Custom algorithms for AI confidence modeling and human fatigue
- **Visualization**: Gradio's plotting components

## Architecture

The app uses mathematical simulations rather than trained ML models, making it fast and educational. Key components include:
- Synthetic dataset generation
- Probabilistic AI predictions
- Human performance modeling with fatigue
- Threshold-based routing algorithms
- Real-time adaptive feedback loops

## Files

- `app.py` - Main Gradio application
- `designspecs.txt` - Detailed design specifications
- `requirements.txt` - Python dependencies
- `huggingfacetoken.txt` - API token (ignored in version control)
- `.gitignore` - Git exclusions

## Contributing

This is an educational demo. For modifications, ensure compatibility with the simulation framework and update documentation accordingly.

## License

Educational use permitted. See design specifications for research context.