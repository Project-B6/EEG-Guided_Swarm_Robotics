# "EchoSwarm" EEG-Guided Swarm Robotics: Optimal Control and Distributed Coordination via Neural Signal Processing

## Data Pipeline
**Key Preprocessing Pipeline**  

Raw EEG → Bandpass Filter → Segment → Extract Features → 
Spatial Mapping → Normalize → Train/Test Split
This pipeline converts raw brain signals into structured spatial-spectral representations suitable for ConvLSTM/CNN models while preserving critical emotional response patterns.

---

## Butterworth Bandpass Filter Implementation

### 1. Filter Design Mathematics
**Transfer Function**:  
$$
H(s) = \frac{1}{\sqrt{1 + \left(\frac{s}{j\omega_c}\right)^{2n}}}
$$  
- $$ n = 4 $$ (filter order)  
- $$ \omega_c $$ = cutoff frequency (radians/sample)  

**Pole-Zero Placement**:  
- Digital frequencies: $$ \omega_1 = \frac{2\pi \cdot 1}{f_s} $$, $$ \omega_2 = \frac{2\pi \cdot 75}{f_s} $$  
- Poles placed on unit circle between $$ \omega_1 $$ and $$ \omega_2 $$

### 2. Bilinear Transform
$$
s = \frac{2}{T_s}\frac{z - 1}{z + 1} \quad \text{with prewarping:} \quad \omega_{digital} = \frac{2}{T_s} \tan\left(\frac{\omega_{analog}T_s}{2}\right)
$$

### 3. Linear Algebra Implementation
Digital filter representation:  
$$
\begin{bmatrix}
y[n] \\
\vdots \\
y[n-4]
\end{bmatrix}
= 
\mathbf{A}^{-1} \mathbf{B} 
\begin{bmatrix}
x[n] \\
\vdots \\
x[n-4]
\end{bmatrix}
$$

### 4. Numerical Optimization (filtfilt)
**Forward-backward filtering process**
y_forward = lfilter(b, a, x)
y_backward = lfilter(b, a, y_forward[::-1])
y_final = y_backward[::-1]

---

## Feature Extraction Mathematics

### 1. Power Spectral Density (PSD)
**Welch's Method**:  
$$
X(f) = \int_{-\infty}^\infty x(t)e^{-j2\pi ft}dt \quad \text{(Fourier Transform)}
$$  
**Matrix Representation**:  
$$
\mathbf{S} = \begin{bmatrix}
s_1 & s_1[1] & \cdots \\
s_2 & s_2[1] & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}
$$

### 2. Differential Entropy (DE)
$$
DE = \frac{1}{2}\log(2\pi e\sigma^2) \quad \text{(Gaussian assumption)}
$$

### 3. Fractal Dimensions
**Higuchi Method**:  
$$
L(k) = \frac{N-1}{k^2}\sum_{m=1}^k \sum_{i=1}^{\lfloor(N-m)/k\rfloor} |x[m+ik] - x[m+(i-1)k]|
$$

**Katz Method**:  
$$
D = \frac{\log(L/\Delta)}{\log(d/\Delta)}
$$

### 4. Spatial Feature Mapping
**8x9 Electrode Grid**:  
$$
\mathbf{F}_{spatial} = \begin{bmatrix}
F_{AF3} & F_{FP1} & \cdots \\
\vdots & \ddots & \\
F_{O2} & \cdots & F_{CB2}
\end{bmatrix}
$$

---

## Key Mathematical Relationships

| Feature Type         | Math Domain          | Key Operations                  |
|----------------------|----------------------|---------------------------------|
| PSD/DE               | Fourier Analysis     | FFT, Variance Calculation       |
| Fractal Dimensions   | Nonlinear Dynamics   | Curve Length Analysis           |
| Wavelets             | Linear Algebra       | Basis Decomposition             |
| Spatial Mapping      | Matrix Algebra       | Grid-based Feature Arrangement  |
| Deep Learning        | Optimization         | Gradient Descent Backpropagation|
