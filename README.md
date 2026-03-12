SkySentinel 🚀

Radar + AI + Threat Simulation 

SkySentinel holo ekta next-gen radar simulator ja moving targets track kore, AI diye trajectory predict kore, ar interactive dashboard e sob visualize kore. Defense research or hobby sim, dono vibe korte pare.

 Why It’s Cool
Track multiple targets at once (missiles, drones, aircraft)
AI predicts where targets will go (RNN/LSTM magic)
Add realistic radar noise & jamming
Real-time 2D/3D plots, heatmaps & trajectories
Predefined scenarios for instant testing
Measure everything: accuracy, prediction error, detection
Dashboard = your mission control room
Easy to tweak & extend – code = fully modular
Run fast, see instant results, repeat
ML models ready, but retrainable if you want

Structure
SkySentinel/
 ├ radar_simulation/
 │   ├ generate_signals.py
 │   ├ noise_model.py
 │   └ filters.py
 ├ ml_models/
 │   ├ trajectory_predictor.py
 │   ├ threat_classifier.py
 │   └ train_models.py
 ├ multi_target_tracking/
 │   └ kalman_tracker.py
 ├ visualization/
 │   ├ dashboard.py
 │   └ plots.py
 ├ scenarios/
 │   └ sample_scenarios.json
 ├ utils/
 │   ├ data_loader.py
 │   └ metrics.py
 ├ main.py
 └ requirements.txt
 


Tech Stack

Python 3.14
Plotly Dash – interactive dashboards
NumPy / SciPy – signal processing
PyTorch / scikit-learn – ML models
Kalman & Particle filters – multi-target tracking

Next-Level Ideas

Real radar hardware integration 
Cloud-based multi-user dashboard 
Advanced evasive AI prediction 
Mobile-friendly interface 
