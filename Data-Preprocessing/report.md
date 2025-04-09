# Swarm Robotics + EEG emotion detection

## **EEG Signal Processing & Mathematical Foundations for Swarm Robotics**

### **1. Multivariate Time-Series Representation**
EEG data is structured as a matrix:  
$$
\mathbf{S} \in \mathbb{R}^{T \times C} \quad \text{(Time samples } \times \text{ Channels)}
$$  
- **Example**: 64-channel EEG sampled at 1,000 Hz for 5 seconds → $T = 5,000$, $C = 64$.  
- **Linear Algebra**: Channels are mapped to spatial coordinates via a **sensor geometry matrix** $\mathbf{G} \in \mathbb{R}^{C \times 3}$, encoding 3D electrode positions[1].

---

### **2. Signal Filtering & Transformations**
#### **Bandpass Filtering**
- **Convolution**: Implemented via Toeplitz matrix multiplication:  
  $$
  \mathbf{Y} = \mathbf{F}_{\text{Toeplitz}} \mathbf{S}
  $$  
  where $\mathbf{F}_{\text{Toeplitz}}$ encodes filter coefficients (e.g., 8-30 Hz bandpass)[1].

#### **Fourier Transform**
- **Integral Calculus**: Converts time-domain to frequency-domain:  
  $$
  \mathcal{F}(s(t)) = \int_{-\infty}^\infty s(t)e^{-j\omega t} dt
  $$  
- **Discrete Implementation**: DFT as matrix multiplication:  
  $$
  \mathbf{F} = \mathbf{W} \mathbf{S}, \quad \mathbf{W} \in \mathbb{C}^{F \times T} \text{ (Fourier basis)}
  $$  
  Power Spectral Density (PSD):  
  $$
  P(f) = \frac{1}{T}|\mathcal{F}(s(t))|^2 \quad \text{(Squared magnitude)}
  $$  

#### **Wavelet Transform**
- **Continuous Form**:  
  $$
  WT(a,b) = \frac{1}{\sqrt{a}} \int_{-\infty}^\infty s(t)\psi\left(\frac{t-b}{a}\right)dt
  $$  
- **Discrete Implementation**: Filter bank convolutions across scales[1].

---

### **3. Feature Extraction via Calculus**
#### **Power Spectral Density (PSD)**
- **Integration**: Bandpower computed via:  
  $$
  \text{Bandpower} = \int_{f_1}^{f_2} P(f) df
  $$  

#### **Approximate Entropy**
- **Probability & Limits**:  
  $$
  \text{ApEn} = \lim_{m \to \infty} \left[ \Phi^m(r) - \Phi^{m+1}(r) \right]
  $$  
  where $\Phi^m(r)$ measures the log-probability of similar signal patterns[1].

#### **Fractal Dimension (Higuchi’s Method)**
- **Derivatives & Limits**: Computes signal complexity via:  
  $$
  FD = \lim_{k \to \infty} \frac{\log(L(k))}{\log(1/k)}
  $$  
  where $L(k)$ is the curve length at scale $k$.

#### **Statistical Moments**
- **Variance (2nd moment)**:  
  $$
  \sigma^2 = \frac{1}{T}\sum_{t=1}^T (s_t - \mu)^2
  $$  
- **Kurtosis (4th moment)**:  
  $$
  \kappa = \frac{1}{T\sigma^4}\sum_{t=1}^T (s_t - \mu)^4
  $$  

---

### **4. Emotion Classification & Optimization**
#### **Feature Matrix**  
Processed features form:  
$$
\mathbf{X} \in \mathbb{R}^{n \times d} \quad \text{(Samples } \times \text{ Features)}
$$  
- **Example**: 64 channels × 5 frequency bands × 3 features = 960 features/row[1].

#### **Cross-Entropy Loss & Gradients**
For a neural network with weights $\theta$:  
$$
J(\theta) = -\frac{1}{n} \sum_{i=1}^n \sum_{c=1}^C y_{ic} \log(p_{ic})
$$  
- **Gradient Descent Update**:  
  $$
  \theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta)
  $$  
  where $\nabla_\theta J(\theta)$ is computed via backpropagation (chain rule)[1].

#### **Backpropagation Mechanics**
- **Partial Derivatives**: For a weight $w_{jk}^l$ in layer $l$:  
  $$
  \frac{\partial J}{\partial w_{jk}^l} = \delta_k^l a_j^{l-1}
  $$  
  where $\delta_k^l$ is the error gradient at neuron $k$[1].

---

### **5. Mathematical Significance for Swarm Robotics**
#### **Spatial Mapping Pipeline**
1. **EEG → Features**: Raw signals → PSD/entropy via calculus.  
2. **Features → Emotion**: Classification via optimization.  
3. **Emotion → Formation**:  
   - **Affine Transformations**: Robot positions $\mathbf{q}_i$ from shape templates:  
     $$
     \mathbf{q}_i = \mathbf{R}(\theta)\mathbf{p}_i + \mathbf{t}
     $$  
   - **Consensus Optimization**: Minimize formation error:  
     $$
     \min_{\mathbf{q}_1, ..., \mathbf{q}_k} \sum_{i=1}^k \|\mathbf{q}_i - \mathbf{p}_i\|^2 \quad \text{s.t. collision avoidance}
     $$  

#### **Key Mathematical Bridges**
- **Fourier/Wavelet**: Functional analysis → Linear algebra (matrix multiplications).  
- **Classification**: Non-convex optimization (gradient descent in high-dim spaces).  
- **Swarm Control**: Differential geometry (Lie groups for transformations) + convex optimization.  

---

## **Symbolic Summary**
$$
\boxed{
\begin{aligned}
\text{EEG Signal} &\xrightarrow[\text{Calculus}]{\text{FT/Wavelet}} \text{Spectral Features} \xrightarrow[\text{Optimization}]{\text{ML}} \text{Emotion} \\
&\xrightarrow[\text{Linear Algebra}]{\text{Affine Transform}} \text{Robot Coordinates} \xrightarrow[\text{Convex Opt.}]{\text{Consensus}} \text{Formation}
\end{aligned}
}
$$  
This mathematical pipeline enables precise translation of brain activity into robotic spatial configurations, leveraging layered abstractions from calculus, linear algebra, and optimization[1].

## **Mathematical Pipeline from EEG Emotions to Swarm Formations**

### **1. Shape Representation & Affine Transformations**
#### **Shape Matrices**
Each emotion maps to a predefined 2D formation represented as a matrix:  
$$
\mathbf{P}_{\text{emotion}} \in \mathbb{R}^{2 \times k} \quad \text{(2D coordinates for } k \text{ robots)}
$$  
**Example**: A "happy" emotion might use:  
$$
\mathbf{P}_{\text{happy}} = \begin{bmatrix}
0 & 1 & -1 & 0.5 & -0.5 \\
1 & 1 & 1 & 0.5 & 0.5
\end{bmatrix} \quad \text{(smiley face template)}
$$

#### **Affine Transformations**
To adapt the template to the environment, apply:  
$$
\mathbf{q}_i = \mathbf{R}(\theta)\mathbf{S}(s)\mathbf{p}_i + \mathbf{t}
$$  
where:  
- $\mathbf{R}(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ (rotation)  
- $\mathbf{S}(s) = \begin{bmatrix} s & 0 \\ 0 & s \end{bmatrix}$ (scaling)  
- $\mathbf{t} = [t_x, t_y]^\top$ (translation)  

---

### **2. Formation Interpolation Using Calculus**
For smooth transitions between emotions, use **Bézier curves**:  
$$
\mathbf{P}(t) = \sum_{i=0}^n \mathbf{P}_i B_i^n(t), \quad t \in[1]
$$  
where $B_i^n(t)$ are Bernstein polynomials.  
**Velocity/Acceleration Constraints**:  
$$
\frac{d\mathbf{P}}{dt} \leq v_{\text{max}}, \quad \frac{d^2\mathbf{P}}{dt^2} \leq a_{\text{max}}
$$  
Solved via quadratic programming to ensure kinematic feasibility.

---

### **3. Robot-to-Goal Assignment: Hungarian Algorithm**
Formulate as a bipartite matching problem:  
- **Cost Matrix**: $C_{ij} = \|\mathbf{x}_i - \mathbf{p}_j\|^2$ (Euclidean distance)  
- **Optimization**:  
$$
\min \sum_{i,j} C_{ij} X_{ij} \quad \text{s.t. } X_{ij} \in \{0,1\}, \sum_i X_{ij} = 1, \sum_j X_{ij} = 1
$$  
Solved in $O(k^3)$ time via the Hungarian Algorithm[1].

---

### **4. Multi-Agent Path Planning**
#### **Convex Optimization Formulation**
For collision-free trajectories:  
$$
\min_{\mathbf{u}_i} \sum_{i=1}^k \int_0^T \|\mathbf{u}_i(t)\|^2 dt \quad \text{s.t.}
$$  
$$
\|\mathbf{x}_i(t) - \mathbf{x}_j(t)\| \geq d_{\text{safe}}, \quad \forall t, i \neq j
$$  
where $\mathbf{u}_i(t) = \ddot{\mathbf{x}}_i(t)$ (acceleration control). Discretized via direct transcription into a quadratic program.

#### **Potential Fields**
Navigation using gradient descent on a potential function:  
$$
U(\mathbf{x}) = U_{\text{goal}}(\mathbf{x}) + \sum_{j \neq i} U_{\text{rep}}(\mathbf{x}, \mathbf{x}_j)
$$  
$$
\mathbf{u}_i = -\nabla U(\mathbf{x}_i)
$$  

---

### **5. Consensus Protocols via Linear Algebra**
Swarm cohesion is maintained using graph Laplacians:  
$$
\dot{\mathbf{x}}_i = -\sum_{j \in \mathcal{N}_i} (\mathbf{x}_i - \mathbf{x}_j)
$$  
In matrix form:  
$$
\dot{\mathbf{X}} = -\mathbf{L} \mathbf{X}
$$  
where $\mathbf{L}$ is the Laplacian matrix. Convergence guaranteed if $\lambda_2(\mathbf{L}) > 0$ (algebraic connectivity).

---

## **Key Mathematical Bridges**
| Stage                  | Mathematical Tools                          | Purpose                                  |
|------------------------|---------------------------------------------|------------------------------------------|
| Shape Definition       | Matrix Algebra (ℝ²ˣᵏ)                      | Template representation                  |
| Affine Transforms      | Lie Groups (SO(2), Translation)             | Scaling/Rotation/Translation             |
| Assignment             | Combinatorial Optimization (Hungarian)      | Optimal robot-goal pairing               |
| Path Planning          | Convex Optimization (QP)                    | Collision-free trajectories              |
| Swarm Dynamics         | Spectral Graph Theory (Laplacian eigenvalues)| Stability analysis                       |

---

## **Geometric Interpretation**
The pipeline transforms abstract emotion labels into spatial configurations through:  
1. **Manifold Embedding**: Emotions → Lie group elements (SE(2) for transformations)  
2. **Configuration Space**: Robot positions live on $\mathbb{R}^{2k}$, constrained by collision avoidance.  
3. **Optimization**: Maps SE(2) × Assignment × Trajectories → Feasible swarm motion.  

---

## **Integration with EEG Classification**
The neural network's emotion output (e.g., "happy" → class 3) indexes into a library of $\mathbf{P}_{\text{emotion}}$ matrices. Real-time updates to $\mathbf{P}(t)$ are computed at 10 Hz (matching the EEG system's 100 ms latency), ensuring the swarm reacts dynamically to brain signals.


## **Swarm Control in Emotion-Driven Robotics: A Dynamical Systems Approach**

### **1. Individual Robot Dynamics via Second-Order ODEs**
Each robot's motion is governed by Newtonian mechanics:  
$$
m_i \frac{d^2 \mathbf{x}_i}{dt^2} = \mathbf{u}_i(t) \quad \text{where } \mathbf{u}_i = \underbrace{-\nabla U_{\text{goal}}}_{\text{Attraction}} + \underbrace{\sum_{j \neq i} \nabla U_{\text{rep}}}_{\text{Repulsion}} + \underbrace{\alpha (\mathbf{\bar{x}}_i - \mathbf{x}_i)}_{\text{Cohesion}}
$$  
- **Attraction**: Proportional to gradient of goal potential $$ U_{\text{goal}} = \frac{1}{2}\|\mathbf{x}_i - \mathbf{g}\|^2 $$  
- **Repulsion**: Lennard-Jones potential $$ U_{\text{rep}}(r) = \frac{A}{r^{12}} - \frac{B}{r^6} $$ (prevents collisions)  
- **Cohesion**: Average neighbor position $$ \mathbf{\bar{x}}_i = \frac{1}{|\mathcal{N}_i|} \sum_{j \in \mathcal{N}_i} \mathbf{x}_j $$  

**State-Space Representation**:  
$$
\frac{d}{dt} \begin{bmatrix} \mathbf{x}_i \\ \mathbf{v}_i \end{bmatrix} = \begin{bmatrix} \mathbf{v}_i \\ \frac{1}{m_i} \mathbf{u}_i \end{bmatrix}
$$  
This formulation enables real-time trajectory optimization[1].

---

### **2. Swarm as a Dynamical System**
The $$ N $$-robot system evolves as:  
$$
\frac{d\mathbf{X}}{dt} = F(\mathbf{X}), \quad \mathbf{X} = [\mathbf{x}_1, \mathbf{v}_1, ..., \mathbf{x}_N, \mathbf{v}_N]^\top
$$  
**Key Properties**:  
- **Connectivity**: Governed by interaction radius $$ R $$ in $$ \mathcal{N}_i = \{ j : \|\mathbf{x}_i - \mathbf{x}_j\| \leq R \} $$  
- **Symmetry**: $$ F $$ is permutation-invariant for homogeneous swarms  

---

### **3. Stability via Lyapunov Theory**
**Lyapunov Function**:  
$$
V(\mathbf{X}) = \underbrace{\sum_i \|\mathbf{x}_i - \mathbf{g}\|^2}_{\text{Goal convergence}} + \underbrace{\sum_{i,j} U_{\text{rep}}(\|\mathbf{x}_i - \mathbf{x}_j\|)}_{\text{Collision avoidance}}
$$  
**Stability Proof**:  
1. $$ V(\mathbf{X}) \geq 0 $$  
2. $$ \frac{dV}{dt} = \nabla V \cdot F(\mathbf{X}) \leq 0 $$ (semi-definite negative)  
Guarantees swarm converges to goal while avoiding collisions[1].

---

### **4. Large-Scale Density Modeling with PDEs**
For $$ N \to \infty $$, define swarm density $$ \rho(\mathbf{x},t) $$:  
**Continuity Equation**:  
$$
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0
$$  
**Momentum Equation**:  
$$
\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla)\mathbf{v} = -\frac{1}{\rho} \nabla P + \nu \nabla^2 \mathbf{v} + \mathbf{f}_{\text{ext}}
$$  
Where $$ P $$ = internal pressure, $$ \nu $$ = viscosity, $$ \mathbf{f}_{\text{ext}} $$ = emotion-modulated forces.

---

### **5. Emotion-to-Control Mapping**
Emotion labels modulate control parameters:  

| Emotion   | Parameters Affected               | Geometric Effect               |
|-----------|-----------------------------------|---------------------------------|
| Happy     | $$ \alpha \uparrow $$, $$ A \downarrow $$ | Tighter formation, fluid motion|
| Sad       | $$ \alpha \downarrow $$, $$ B \uparrow $$ | Loose cluster, sluggishness    |
| Fear      | $$ R \downarrow $$, $$ \nu \uparrow $$    | Rapid dispersion, jerky motion |

**Example**: For "happy" emotion:  
$$
\alpha \leftarrow 2\alpha_0, \quad A \leftarrow 0.5A_0 \quad \Rightarrow \text{Smiley face formation}
$$  

---

### **6. Significance in Emotion-Driven Systems**
1. **Abstraction Hierarchy**:  
   EEG Emotions → ODE Parameters → PDE Density Patterns → Geometric Shapes  
2. **Real-Time Stability**: Lyapunov guarantees prevent chaotic behavior during emotion transitions.  
3. **Scalability**: PDE models enable efficient simulation of 10³-10⁶ robots.  

---

## **Mathematical Pipeline**
$$
\boxed{
\begin{aligned}
\text{Emotion Class} &\xrightarrow{\text{Parameter Mapping}} \text{ODE Controls} \\
&\xrightarrow{\text{Lyapunov Analysis}} \text{Stable Trajectories} \\
&\xrightarrow{\text{Continuum Limit}} \text{Density PDEs} \\
&\xrightarrow{\text{IC/Boundary Conditions}} \text{Shape Patterns}
\end{aligned}
}
$$  
This framework ensures emotion-driven swarms behave as programmable matter, blending control theory, dynamical systems, and continuum mechanics[1].

## **Mathematical Pipeline from EEG to Swarm Robotics**

### **1. EEG Signal Acquisition & Preprocessing**
#### **Time-Series Representation**
Raw EEG data is captured as a matrix:  
$$
\mathbf{S} \in \mathbb{R}^{T \times C} \quad \text{(Time samples } \times \text{ Channels)}
$$  
- **Sampling**: At 1,000 Hz, a 5-second recording yields $$ T = 5,000 $$.  
- **Noise Removal**: Bandpass filtering (e.g., 8-30 Hz) via **convolution** with Toeplitz matrices.  

#### **Spectral Transformations**
- **Fourier Transform (FFT)**:  
  $$
  \mathcal{F}(s(t)) = \int_{-\infty}^\infty s(t)e^{-j\omega t} dt \quad \text{(Continuous)}  
  $$  
  Discrete implementation:  
  $$
  \mathbf{F} = \mathbf{W} \mathbf{S} \quad \text{where } \mathbf{W} \in \mathbb{C}^{F \times T} \text{ is the Fourier basis matrix.}
  $$  
- **Wavelet Transform**:  
  $$
  WT(a,b) = \frac{1}{\sqrt{a}} \int_{-\infty}^\infty s(t)\psi\left(\frac{t-b}{a}\right)dt \quad \text{(Multi-scale analysis)}
  $$  

---

### **2. Feature Extraction via Calculus & Linear Algebra**
#### **Entropy & Fractal Dimension**
- **Approximate Entropy**:  
  $$
  \text{ApEn} = \lim_{m \to \infty} \left[ \Phi^m(r) - \Phi^{m+1}(r) \right] \quad \text{(Measures signal unpredictability)}
  $$  
- **Higuchi’s Fractal Dimension**:  
  $$
  FD = \lim_{k \to \infty} \frac{\log(L(k))}{\log(1/k)} \quad \text{(Quantifies signal complexity)}
  $$  

#### **Statistical Moments**
- **Variance**:  
  $$
  \sigma^2 = \frac{1}{T}\sum_{t=1}^T (s_t - \mu)^2  
  $$  
- **Kurtosis**:  
  $$
  \kappa = \frac{1}{T\sigma^4}\sum_{t=1}^T (s_t - \mu)^4  
  $$  

---

### **3. Emotion Classification with SVM/XGBoost**
#### **Support Vector Machines (SVM)**
- **Hyperplane Optimization**:  
  $$
  \min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t. } y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1  
  $$  
  Solved via **Lagrange multipliers** and kernel tricks (e.g., RBF).  

#### **XGBoost**
- **Gradient Boosting**: Minimizes loss $$ L(y, \hat{y}) $$ using additive trees:  
  $$
  \hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)  
  $$  
  With **regularization**:  
  $$
  \Omega(f_t) = \gamma T + \frac{1}{2}\lambda \|\mathbf{w}\|^2  
  $$  

#### **Performance**  
- Accuracy: ~72-74% in leave-one-subject-out validation (see code output)[1].  

---

### **4. Emotion-to-Formation Geometric Mapping**
#### **Shape Definition**  
Each emotion maps to a 2D template:  
$$
\mathbf{P}_{\text{emotion}} \in \mathbb{R}^{2 \times k} \quad \text{(Coordinates for } k \text{ robots)}
$$  
**Example**: A "happy" formation might resemble a smiley face.  

#### **Affine Transformations**  
Robot positions $$ \mathbf{q}_i $$ are derived via:  
$$
\mathbf{q}_i = \mathbf{R}(\theta)\mathbf{S}(s)\mathbf{p}_i + \mathbf{t}  
$$  
- $$ \mathbf{R}(\theta) $$: Rotation matrix  
- $$ \mathbf{S}(s) $$: Scaling matrix  
- $$ \mathbf{t} $$: Translation vector  

#### **Robot Assignment**  
**Hungarian Algorithm** solves:  
$$
\min \sum_{i,j} \|\mathbf{x}_i - \mathbf{p}_j\|^2 X_{ij} \quad \text{s.t. } X_{ij} \in \{0,1\}, \sum_i X_{ij} = 1  
$$  

---

### **5. Swarm Control via Differential Equations**
#### **Dynamical Model**  
Each robot follows:  
$$
m_i \frac{d^2 \mathbf{x}_i}{dt^2} = \underbrace{-\nabla U_{\text{goal}}}_{\text{Attraction}} + \underbrace{\sum_{j \neq i} \nabla U_{\text{rep}}}_{\text{Collision Avoidance}}  
$$  
- **Attraction Potential**: $$ U_{\text{goal}} = \frac{1}{2}\|\mathbf{x}_i - \mathbf{g}\|^2 $$  
- **Repulsion Potential**: $$ U_{\text{rep}}(r) = \frac{A}{r^{12}} - \frac{B}{r^6} $$  

#### **Stability Analysis**  
**Lyapunov Function**:  
$$
V(\mathbf{X}) = \sum_i \|\mathbf{x}_i - \mathbf{g}\|^2 + \sum_{i,j} U_{\text{rep}}(\|\mathbf{x}_i - \mathbf{x}_j\|)  
$$  
Guarantees convergence to goal while avoiding collisions.  

---

## **Integrated Pipeline & Mathematical Significance**

### **Closed-Loop Workflow**
1. **EEG → Features**: Spectral transforms (FFT/wavelets) and calculus-driven features (entropy, fractal dimension).  
2. **Features → Emotion**: SVM/XGBoost classify using hyperplanes or gradient boosting.  
3. **Emotion → Formation**: Affine transformations (linear algebra) and optimal assignment (combinatorial optimization).  
4. **Formation → Motion**: Stabilized via Lyapunov-based control theory.  

### **Key Mathematical Bridges**
| Stage                  | Tools                              | Purpose                                  |
|------------------------|-----------------------------------|------------------------------------------|
| Signal Processing      | Fourier/Wavelet Transforms        | Noise reduction, spectral feature extraction |
| Feature Extraction     | Calculus (entropy, fractals)      | Emotion-relevant pattern isolation       |
| Classification         | SVM (Lagrange multipliers)        | High-dimensional decision boundaries     |
| Geometry               | Lie Groups (SE(2))                | Shape scaling/rotation                   |
| Control                | Lyapunov Stability, ODEs          | Collision-free trajectory convergence    |

---

## **Symbolic Summary**
$$
\boxed{
\begin{aligned}
\text{EEG} &\xrightarrow[\text{Calculus}]{\text{FFT/Wavelet}} \text{Features} \xrightarrow[\text{Optimization}]{\text{SVM/XGBoost}} \text{Emotion} \\
&\xrightarrow[\text{Linear Algebra}]{\text{Affine Transform}} \text{Formation} \xrightarrow[\text{Control Theory}]{\text{Lyapunov}} \text{Swarm Motion}
\end{aligned}
}
$$  
This pipeline exemplifies how interdisciplinary mathematics transforms biological signals into coordinated robotic behavior.
