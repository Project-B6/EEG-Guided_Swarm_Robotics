“EchoSwarm” EEG-Guided Swarm Robotics: Optimal Control and Distributed Coordination via Neural Signal Processing
Data Pipeline

Key Preprocessing Pipeline

Raw EEG → Bandpass Filter → Segment → Extract Features → 
Spatial Mapping → Normalize → Train/Test Split
This pipeline converts raw brain signals into structured spatial-spectral representations suitable for ConvLSTM/CNN models while preserving critical emotional response patterns. The smooth variant adds advanced nonlinear features for enhanced emotion discrimination
