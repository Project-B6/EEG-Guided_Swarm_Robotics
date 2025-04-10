“EchoSwarm” EEG-Guided Swarm Robotics: Optimal Control and Distributed Coordination via Neural Signal Processing
Data Pipeline

Key Preprocessing Pipeline

Raw EEG → Bandpass Filter → Segment → Extract Features → 
Spatial Mapping → Normalize → Train/Test Split
This pipeline converts raw brain signals into structured spatial-spectral representations suitable for ConvLSTM/CNN models while preserving critical emotional response patterns. The smooth variant adds advanced nonlinear features for enhanced emotion discrimination

The Butterworth bandpass filter implementation involves several mathematical concepts across signal processing, linear algebra, and numerical methods. Here's a detailed technical explanation:

## 1. **Filter Design Mathematics**

A filter is like a tool that lets certain parts of a signal through while blocking others. The transfer function is the formula that describes how the filter behaves. For the Butterworth filter:

It has a cutoff frequency, which determines what part of the signal gets through and what gets blocked. For example, if you want to filter out low-frequency noise, the filter will let frequencies higher than a certain threshold pass through.

The filter is designed so that it smoothly reduces the signal after the cutoff point without any sharp edges, providing a smooth transition from allowed to blocked frequencies.

### Transfer Function Formulation
The 4th-order Butterworth filter is defined by its transfer function:

$$
H(s) = \frac{1}{\sqrt{1 + \left(\frac{s}{j\omega_c}\right)^{2n}}}
$$

Where:
- $$ n = 4 $$ (filter order)
- $$ \omega_c $$ = cutoff frequency (radians/sample)
- $$ s = \sigma + j\omega $$ (complex frequency)

### Pole-Zero Placement

When designing a filter, we place "poles" (special points in the complex frequency space) at specific locations to control the filter’s behavior. For the Butterworth filter:

Poles are placed in a way that the filter’s frequency response has a smooth, gradual decline, without any sudden jumps.

For a bandpass filter (which lets only a specific range of frequencies through), the poles are placed in the right spot on the frequency spectrum to allow the desired frequencies (like brainwave bands in EEG signals) to pass while blocking everything else.

For bandpass (1-75 Hz) with sampling frequency $$ f_s $$:
- Digital frequencies: $$ \omega_1 = \frac{2\pi \cdot 1}{f_s} $$, $$ \omega_2 = \frac{2\pi \cdot 75}{f_s} $$
- Poles are placed on the unit circle in the z-plane at angles between $$ \omega_1 $$ and $$ \omega_2 $$

## 2. **Bilinear Transform**

This is a mathematical trick used to convert analog filters (which work in continuous time) into digital filters (which work in discrete time, like in computers).

The problem is that analog filters (like the Butterworth filter) are continuous, but computers deal with signals in steps, not continuously. So, we convert them using the bilinear transform.

The prewarping part helps make sure the filter doesn’t distort important frequencies (like 1 Hz and 75 Hz) when converting from analog to digital.

$$
s = \frac{2}{T_s}\frac{z - 1}{z + 1}
$$

Where $$ T_s = 1/f_s $$ is the sampling period. This preserves stability but warps frequencies, compensated via prewarping:

$$
\omega_{digital} = \frac{2}{T_s} \tan\left(\frac{\omega_{analog}T_s}{2}\right)
$$

## 3. **Linear Algebra Implementation**
The final digital filter is represented as:

$$
\sum_{k=0}^{4} a_k y[n-k] = \sum_{k=0}^{4} b_k x[n-k]
$$

Which can be expressed in matrix form for batch processing:

$$
\begin{bmatrix}
y[n] \\
y[n-1] \\
\vdots \\
y[n-4]
\end{bmatrix}
= 
\mathbf{A}^{-1} \mathbf{B} 
\begin{bmatrix}
x[n] \\
x[n-1] \\
\vdots \\
x[n-4]
\end{bmatrix}
$$

Where matrices $$ \mathbf{A} $$ and $$ \mathbf{B} $$ contain the filter coefficients $$ a_k $$ and $$ b_k $$.

## 4. **Numerical Optimization**

The filtfilt function is a smart way of filtering that reduces unwanted effects like phase distortion (which shifts the timing of the signal).

First, the signal is filtered normally, then the time direction is flipped, and it is filtered again. Flipping it back ensures that no phase shifts occur.

This is important for EEG signals because we want to preserve the exact timing of the brain activity without distortion.

The **filtfilt** function uses forward-backward filtering to achieve zero phase delay:

1. Solve $$ \mathbf{A}\mathbf{y} = \mathbf{B}\mathbf{x} $$ (forward filter)
2. Reverse time axis: $$ \mathbf{x'} = \mathbf{x}[::-1] $$
3. Solve $$ \mathbf{A}\mathbf{y'} = \mathbf{B}\mathbf{x'} $$ (backward filter)
4. Reverse result: $$ y_{final} = \mathbf{y'}[::-1] $$

This effectively squares the filter's magnitude response while eliminating phase distortion.

## 5. **Frequency Domain Analysis**
Using Discrete Fourier Transform (DFT):

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}
$$

The ideal bandpass filter can be represented as:

$$
H_{ideal}[k] = \begin{cases}
1 & \omega_1 \leq |\omega_k| \leq \omega_2 \\
0 & \text{otherwise}
\end{cases}
$$

Where $$ \omega_k = 2\pi k/N $$. The Butterworth approximation provides smooth transition bands instead of abrupt transitions.

## 6. **Computational Complexity**
The 4th-order filter requires:
- 8 multiplications/sample (4 zeros + 4 poles)
- 8 additions/sample
- $$ O(N) $$ operations for N samples

The filtfilt operation doubles this to $$ O(2N) $$ but preserves phase characteristics critical for EEG analysis.

## 7. **Stability Considerations**
The filter coefficients must satisfy:

$$
\sum_{k=0}^{4} |a_k| < \infty \quad \text{(BIBO stability)}
$$

Guaranteed by Butterworth design's pole placement within unit circle.

This mathematical foundation enables effective noise removal while preserving crucial EEG waveforms like alpha (8-12 Hz) and gamma (30-50 Hz) rhythms critical for emotion recognition.

The feature extraction process in your EEG emotion detection system combines signal processing, information theory, and nonlinear dynamics. Here's a simplified mathematical breakdown:

## 1. **Power Spectral Density (PSD) with Welch's Method**

What you're doing: Breaking down brain signals into different frequencies like Delta, Theta, Alpha, Beta, and Gamma to see which ones are more powerful.

How: You use Welch's Method — think of it as slicing the EEG into overlapping chunks, turning them into frequency form using Fourier Transform, and then averaging the results.

Why: Emotions affect different brain rhythms. PSD tells us which rhythms are active.

**Mathematical Foundation**:  
- **Fourier Transform**:  
  $$
  X(f) = \int_{-\infty}^\infty x(t)e^{-j2\pi ft}dt
  $$
  - Convert time-domain EEG to frequency-domain  
  - Welch's method averages PSD over overlapping segments  
  ```python
  freqs, psd = welch(signal, fs=200, nperseg=800)  # 4s window
  ```

**Linear Algebra Connection**:  
- Segmented data forms a matrix:  
  $$
  \mathbf{S} = \begin{bmatrix}
  s_1 & s_1[1] & \cdots \\
  s_2 & s_2[1] & \cdots \\
  \vdots & \vdots & \ddots
  \end{bmatrix}
  $$
  - Each row is a 4-second EEG segment  
  - PSD computed as mean of FFT magnitude squared across rows

## 2. **Differential Entropy (DE)**

What you're doing: Calculating how unpredictable or complex the brainwaves are within each frequency band.

How: If the brain signal behaves like a bell-shaped curve (Gaussian), there's a shortcut to calculate entropy using just the variance (how much the signal spreads).

Why: Emotional states like anxiety or calmness affect how "chaotic" the EEG becomes.

**Information Theory Foundation**:  
- For Gaussian-distributed signals:  
  $$
  DE = \frac{1}{2}\log(2\pi e\sigma^2)
  $$
  - Measures signal complexity in frequency bands  
  - Implemented via bandpass filtering + variance calculation

**Calculus Connection**:  
- Original entropy definition requires integration:  
  $$
  H(X) = -\int p(x)\log p(x)dx
  $$
  - Gaussian assumption avoids numerical integration

## 3. **Fractal Dimensions**
### Higuchi Fractal Dimension  
- Calculates curve length at different scales:  
  $$
  L(k) = \frac{N-1}{k^2}\sum_{m=1}^k \sum_{i=1}^{\lfloor(N-m)/k\rfloor} |x[m+ik] - x[m+(i-1)k]|
  $$
  - Slope of $$\log L(k)$$ vs $$\log(1/k)$$ gives dimension

### Katz Dimension  
- Geometric approach:  
  $$
  D = \frac{\log(L/\Delta)}{\log(d/\Delta)}
  $$
  Where $$L$$=total curve length, $$d$$=max displacement

## 4. **Wavelet Decomposition**
**Linear Algebra Implementation**:  
- Discrete Wavelet Transform (DWT) using filter banks:  
  $$
  \mathbf{W} = \mathbf{\Psi}\mathbf{x}
  $$
  - `pywt.wavedec` implements this as:  
  ```python
  coeffs = [cA, cD4, cD3, cD2, cD1]  # 5-level decomposition
  ```

## 5. **Spatial Feature Mapping**
**Matrix Operations**:  
- EEG channels mapped to 8x9 grid:  
  $$
  \mathbf{F}_{spatial} = \begin{bmatrix}
  F_{AF3} & F_{FP1} & \cdots \\
  \vdots & \ddots & \\
  F_{O2} & \cdots & F_{CB2}
  \end{bmatrix}
  $$
  - Each position contains spectral/entropy features

## 6. **Deep Learning Integration**
**ConvLSTM Architecture**:  
- Combines CNN spatial processing with LSTM temporal modeling:  
  $$
  \mathbf{h}_t = \sigma(\mathbf{W}_h \ast [\mathbf{X}_t, \mathbf{h}_{t-1}] + \mathbf{b}_h)
  $$
  - **Key Components**:  
  - 2D convolutions for spatial patterns  
  - LSTM gates for temporal dynamics  
  - Final dense layers for classification

## Optimization Foundations
1. **Filter Design**:  
   Butterworth filter coefficients optimized via bilinear transform:  
   $$
   s = \frac{2}{T_s}\frac{z-1}{z+1}
   $$

2. **Model Training**:  
   Adam optimizer minimizes cross-entropy loss:  
   $$
   \mathcal{L} = -\sum y_i\log(\hat{y}_i)
   $$

## Key Mathematical Relationships
| Feature Type         | Math Domain          | Key Operations                  |
|----------------------|----------------------|---------------------------------|
| PSD/DE               | Fourier Analysis     | FFT, Variance Calculation       |
| Fractal Dimensions   | Nonlinear Dynamics   | Curve Length Analysis           |
| Wavelets             | Linear Algebra       | Basis Decomposition             |
| Spatial Mapping      | Matrix Algebra       | Grid-based Feature Arrangement  |
| Deep Learning        | Optimization         | Gradient Descent Backpropagation|

This multi-domain approach transforms raw EEG signals into features that capture both rhythmic patterns (PSD/DE) and complex nonlinear dynamics (fractal dimensions), while spatial mapping and deep learning handle electrode geometry and temporal evolution.

