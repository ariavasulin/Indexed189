---
course: CS 189
semester: Fall 2025
type: lecture
title: Linear Regression (3)
source_type: slides
source_file: Lecture 08 -- Linear Regression (3).pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Lecture 8

## Linear Regression (3)

## Probabilistic view of linear regression

## EECS 189/289, Fall 2025 @ UC Berkeley

Joseph E. Gonzalez and Narges Norouzi

# III Join at slido.com \#4234276 

## Roadmap

- Recap of Regularization ${ }^{4234276}$
- When Normal Equation Gets Tricky
- Least-Squares $\cong$ Maximum Likelihood
- Uncertainty $\approx$ Confidence Intervals
- Decision Theory


## Recap of Regularization

- Recap of Regularization 4234276
- When Normal Equation Gets Tricky
- Least-Squares $\cong$ Maximum Likelihood
- Uncertainty $\approx$ Confidence Intervals
- Decision Theory

L2 Regularization (Ridge) $\quad L_{2}[w]=\sqrt{\sum_{d} w_{d}^{2}}$ $\mathrm{w}^{*}=\arg \min \mathrm{E}_{\mathrm{D}}[w]+\lambda \mathrm{E}_{\mathrm{w}}[w]$
$\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}$

$$
w_{2} \quad\|w\|_{2}=c
$$
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-05.jpg?height=107&width=405&top_left_y=595&top_left_x=918)

**Image Description:** The image displays a diagram resembling a bifurcation point or a flow chart split, characterized by two curved lines that extend outward and then converge at a central point. The top and bottom ends of the lines are open, indicating a transition between two states or conditions. This could represent processes in systems theory, decision-making frameworks, or fluid dynamics, emphasizing divergence and convergence in a system's evolution. The lines suggest smooth transitions, likely illustrating a qualitative change in parameters or variables.

$$
\underbrace{}_{\frac{1}{2} \boldsymbol{L}_{2}[w]^{2}=\frac{1}{2} \sum_{d} w_{d}^{2}}
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-05.jpg?height=882&width=1102&top_left_y=586&top_left_x=2228)

**Image Description:** The image is a contour plot representing a two-dimensional function, likely a loss function in a machine learning context. The x-axis is labeled \( w_1 \) and the y-axis is labeled \( w_2 \), indicating two parameters or weights. The plot illustrates a gradient descent path toward a minimum, with concentric contour lines depicting function values. A circular path is shown, indicating the optimization trajectory. Additionally, a highlighted point signifies the optimal weight \( E_D[w] \). The color gradient transitions from dark blue to lighter shades, illustrating value changes in the function.

$\frac{1}{2}\left(\left\|t_{n}-y\left(\mathrm{x}_{n}, w\right)\right\|_{2}\right)^{2}$
$\frac{1}{2}\left(\left\|e_{n}\right\|_{2}\right)^{2}$
length of the residual vector

## L2 Regularizatio n Demo

## Interactive link

$$
4234276
$$
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-06.jpg?height=950&width=950&top_left_y=467&top_left_x=2287)

**Image Description:** The image is a graph depicting a curve analysis in relation to a parameter \(\lambda\). The x-axis represents the parameter \(\lambda\), ranging from 0 to 15. The y-axis indicates the value of the functions plotted. Four curves are displayed, including:

1. A red star for \( w(\lambda=0) \) (no regularization).
2. An orange curve labeled \( E_D(w(\lambda)) \).
3. A green curve for \( \lambda \cdot E_{W}(w(\lambda)) \).
4. A blue curve denoted as \( E(w(\lambda)) \).
5. A purple curve labeled \( ||w|| = c(\lambda) \).

The graph interprets the relationship between \(\lambda\) and various cost functions.


L1 Regularization (Lasso) $\quad L_{1}[w]=\sum_{j}\left|w_{j}\right|$ $\mathrm{w}^{*}=\arg \min \mathrm{E}_{\mathrm{D}}[w]+\lambda \mathrm{E}_{\mathrm{w}}[w]$
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-07.jpg?height=128&width=426&top_left_y=591&top_left_x=910)

**Image Description:** This image consists of a simple line diagram resembling a curve or a "u" shape, typically used to denote a function or a boundary in geometric contexts. The ends of the curve are open and stretched horizontally, while the center dips downwards, creating a concave structure. There are no axes or labels present, indicating that it may represent a concept such as a potential energy curve or a theoretical boundary condition in academic discourse. The overall style is minimalist with a solid blue color.

![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-07.jpg?height=125&width=422&top_left_y=586&top_left_x=1573)

**Image Description:** The image depicts a smooth, symmetrical curve resembling a stylized "U" shape. This diagram lacks axis labels but appears to represent a concept in physics or engineering, possibly related to potential energy minima or a connection in a graphical representation of a system. The lines are colored blue and are of uniform thickness, suggesting a clean and simple design intended for clarity in an academic setting. The visual is abstract and does not contain additional elements or text.

$\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}$
$\frac{1}{2} \boldsymbol{L}_{\mathbf{1}}[w]=\frac{1}{2} \sum_{\boldsymbol{d}}\left|w_{\boldsymbol{d}}\right|$
$\frac{1}{2}\left(\left\|t_{n}-y\left(\mathrm{x}_{n}, w\right)\right\|_{2}\right)^{2}$
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-07.jpg?height=933&width=1102&top_left_y=535&top_left_x=2228)

**Image Description:** The image depicts a contour plot representing the optimization landscape for a function \( E_D[w] \) over two dimensions, \( w_1 \) and \( w_2 \). The plot features blue gradient contours indicating levels of the function, with an inner diamond shape illustrating the constraint \( \|w\|_2 = c \). The \( w_1 \) axis is horizontal, and the \( w_2 \) axis is vertical. A green point marks the optimization solution, and a labeled arrow points towards this point from the contour. This visualization effectively conveys the relationship between parameters and the optimization objective.

$\frac{1}{2}\left(\left\|e_{n}\right\|_{2}\right)^{2}$
length of the residual vector
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-08.jpg?height=728&width=814&top_left_y=510&top_left_x=1394)

**Image Description:** The image depicts a contour plot, represented in gradient blue shades, indicating levels of a scalar field. The x-axis ranges from -5 to 10, while the y-axis extends from -5 to 10. A pink diamond shape is overlaid on the contour plot, with two markers: a red star and a green cross positioned inside the diamond. These markers likely indicate specific points of interest within the plotted scalar field. The concentric contours represent areas of equal value, suggesting optimization or minimization contexts in a multidimensional space.

$\lambda=4.00$
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-08.jpg?height=767&width=924&top_left_y=484&top_left_x=2313)

**Image Description:** The image is a multi-curve plot illustrating a relationship between a parameter \( \lambda \) (x-axis) and various functions \( w(\lambda) \), \( E_D(w(\lambda)) \), \( E(w(\lambda)) \), and \( ||w||_1 = c(\lambda) \) (y-axis). Each curve is color-coded: orange for \( w(\lambda) \), purple for \( E(w(\lambda)) \), blue for \( E_D(w(\lambda)) \), and green for \( ||w||_1 \). A star indicates a point at \( \lambda = 0 \) with no regularization. A vertical dashed line marks the position of \( \lambda \) where \( E_w \) is noted as approximately 14.731.

$\lambda=14.00$

- Recap of Regularization ${ }^{4234276}$


## When Normal Equation Gets Tricky

- When Normal Equation Gets Tricky
- Least-Squares $\cong$ Maximum Likelihood
- Uncertainty $\approx$ Confidence Intervals
- Decision Theory


## Normal Equation (Reminder)

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}=\frac{1}{2}\left(\|\vec{e}\|_{2}\right)^{2} \quad \begin{aligned}
& E(w) \text { is minimum when } \vec{e} \text { is } \\
& \text { orthogonal projection of } \mathrm{t} .
\end{aligned}
$$

Vector $\vec{e}$ is orthogonal to the span( $\mathbb{X}$ ), meaning that:

$$
\begin{array}{ll}
\mathbb{X}^{T} \vec{e}=0 & \begin{array}{l}
\text { Adding the definition of } \\
\text { residual }
\end{array} \\
\mathbb{X}^{T}\left(\mathrm{t}-\mathbb{Y}\left(\mathbb{X}, w^{*}\right)\right)=0 & \text { Adding formula for } \mathbb{Y}\left(\mathbb{X}, w^{*}\right)=\mathbb{X} w^{*} \\
\mathbb{X}^{T} \mathbf{t}-\mathbb{X}^{T} \mathbb{X} w^{*}=0 & \text { Noving terms } \\
\mathbb{X}^{T} \mathbf{t}=\mathbb{X}^{T} \mathbb{X} w^{*} & w^{*}=\left(\mathbb{X}^{T} \mathbb{X}\right)^{-1} \mathbb{X}^{T} \mathbf{t} \\
\text { Equation } \quad & \begin{array}{l}
\text { If } \mathbb{X}^{T} \mathbb{X} \text { is } \\
\text { invertible }
\end{array}
\end{array}
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-10.jpg?height=451&width=792&top_left_y=659&top_left_x=2419)

**Image Description:** The image is a 3D diagram representing a coordinate system with three axes labeled as \( X_1 \), \( X_2 \), and \( t \) (time). Each axis is colored differently: the \( X_1 \) and \( X_2 \) axes are shown in black and green, respectively, while the \( t \) axis is orange. The diagram illustrates vector representations in a three-dimensional space, possibly indicating relationships between variables. The surface grid provides a spatial reference, enhancing comprehension of the vector directions and interactions. An interactive link is noted at the bottom of the slide.


## When Normal Equation Gets Tricky

Normal Equation
When is $\mathbb{X}^{T} \mathbb{X}$ invertible?

- Full column rank: $\operatorname{rank}(\mathbb{X})=D+1$
- Requires $\mathrm{N} \geq \mathrm{D}+1$ (at least as many samples as features)
- No perfect collinearity: $\operatorname{det}\left(\mathbb{X}^{T} \mathbb{X}\right) \neq 0$
- All singular values must be non-zero: $\sigma_{1}, \sigma_{2}, \ldots, \sigma_{D+1}>0$.

$$
\begin{aligned}
w^{*} & =\overbrace{\left(\mathbb{X}^{T} \mathbb{X}\right)^{-1} \mathbb{X}^{T} t}^{D+1} \\
\mathbb{X} & =\overbrace{\left[\begin{array}{llll}
1 & \mathrm{x}_{11} & \cdots & \mathrm{x}_{1 \mathrm{D}} \\
1 & \cdots & \cdots & \cdots \\
1 & \mathrm{x}_{\mathrm{N} 1} & \cdots & \mathrm{x}_{\mathrm{ND}}
\end{array}\right]}\}_{N}
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-12.jpg?height=481&width=431&top_left_y=561&top_left_x=229)

**Image Description:** The image represents a checkbox or checklist icon, featuring a checkmark and horizontal lines. It is designed in a simple, minimalist style, with a rectangular outline and rounded corners. The checkmark indicates completion or verification, while the horizontal lines suggest items or tasks within a list. The icon typically symbolizes tasks, selections, or forms, facilitating user interactions in digital interfaces.


## If $N=D+1$ and $X$ is full rank, what is the training error?

## Solution

If $N=D+1$ and $\mathbb{X}$ is full rank, what is the training error?
$\mathbb{X} \in \mathbb{R}^{N \times D+1}$
$\mathbb{X} \in \mathbb{R}^{D+1 \times D+1} \quad$ Square matrix that is also full rank

We can solve directly $\quad \mathbb{X}^{T} \vec{e}=0$
Multiply both sides by $\left(\mathbb{X}^{T}\right)^{-1} \quad \vec{e}=0$
Training error will always be 0.
$2-$
$0-$
0
In which of the following situations is X^TX likely to be singular or nearly singular?

## When Normal Equation Gets Tricky

Normal Equation
When is $\mathbb{X}^{T} \mathbb{X}$ invertible?

- Full column rank: $\operatorname{rank}(\mathbb{X})=D+1$
- Requires $\mathrm{N} \geq \mathrm{D}+1$ (at least as many samples as features)
- No perfect collinearity: $\operatorname{det}\left(\mathbb{X}^{T} \mathbb{X}\right) \neq 0$
- All singular values must be non-zero: $\sigma_{1}, \sigma_{2}, \ldots, \sigma_{D+1}>0$.

$$
\begin{aligned}
w^{*} & =\overbrace{\left(\mathbb{X}^{T} \mathbb{X}\right)^{-1} \mathbb{X}^{T} t}^{D+1} \\
\mathbb{X} & =\overbrace{\left[\begin{array}{llll}
1 & \mathrm{x}_{11} & \cdots & \mathrm{x}_{1 \mathrm{D}} \\
1 & \cdots & \cdots & \cdots \\
1 & \mathrm{x}_{\mathrm{N} 1} & \cdots & \mathrm{x}_{\mathrm{ND}}
\end{array}\right]}\}_{N}
\end{aligned}
$$

- Ill-conditioning (numerical instability): Even if $\operatorname{det}\left(\mathbb{X}^{T} \mathbb{X}\right) \neq 0$

For any matrix $\mathrm{A}, \operatorname{det}(\mathrm{A})=\prod_{i=1}^{K} \sigma_{i}$ where $\sigma_{i}$ are eigenvalues.
When $\operatorname{det}(\mathrm{A}) \approx 0$, means at least one singular value $\sigma_{i} \approx 0$.

## When Normal Equation Gets Tricky

Normal Equation
When is $\mathbb{X}^{T} \mathbb{X}$ invertible?

- Full column rank: $\operatorname{rank}(\mathbb{X})=D+1$
- Requires $\mathrm{N} \geq \mathrm{D}+1$ (at least as many samples as features)
- No perfect collinearity: $\operatorname{det}\left(\mathbb{X}^{T} \mathbb{X}\right) \neq 0$
- All singular values must be non-zero: $\sigma_{1}, \sigma_{2}, \ldots, \sigma_{D+1}>0$.

$$
\begin{aligned}
w^{*} & =\overbrace{\left(\mathbb{X}^{T} \mathbb{X}\right)^{-1} \mathbb{X}^{T} t}^{D+1} \\
\mathbb{X} & =\overbrace{\left[\begin{array}{cccc}
1 & \mathrm{x}_{11} & \cdots & \mathrm{x}_{1 \mathrm{D}} \\
1 & \cdots & \cdots & \cdots \\
1 & \mathrm{x}_{\mathrm{N} 1} & \cdots & \mathrm{x}_{\mathrm{ND}}
\end{array}\right]}\}_{N}
\end{aligned}
$$

- III-conditioning (numerical instability): Even if $\operatorname{det}\left(\mathbb{X}^{T} \mathbb{X}\right) \neq 0$
- Condition number $\kappa=\frac{\sigma_{\max }}{\sigma_{\min }} \gg 10^{8}$ small noise in the data creates huge weight swing
- Why?
- Rounding error will be amplified when calculating $\kappa \varepsilon_{\text {machine }} \cdot$


## Fix \#1: Ridge Trick

In Ridge regression, the closed form solution is

$$
w^{*}=\left(\mathbb{X}^{\mathrm{T}} \mathbb{X}+\lambda I\right)^{-1} \mathbb{X}^{\mathrm{T}} \mathrm{t}
$$

Proof in discussion 4 this week

- Ridge replaces $X^{T} \mathbb{X}$ in normal equation with $\mathbb{X}^{T} \mathbb{X}+\lambda I$.


## Fix \#1: Ridge Trick

In Ridge regression, the closed form solution is

$$
w^{*}=\left(\mathbb{X}^{\mathrm{T}} \mathbb{X}+\lambda I\right)^{-1} \mathbb{X}^{\mathrm{T}} \mathrm{t}
$$

Proof in discussion 4 this week

- Ridge replaces $\mathbb{X}^{\mathrm{T}} \mathbb{X}$ in normal equation with $\mathbb{X}^{\mathrm{T}} \mathbb{X}+\lambda I$.

What are the eigenvalues of $\mathbb{X}^{\mathrm{T}} \mathbb{X}+\lambda I$ ?
A. $\sigma_{i}$
B. $\sigma_{i}+\lambda$
C. $\sigma_{i}^{2}$
D. $\sigma_{i}^{2}+\lambda$

## Which of the following options are correct?

## Fix \#1: Ridge Trick

In Ridge regression, the closed form solution is

$$
w^{*}=\left(\mathbb{X}^{\mathrm{T}} \mathbb{X}+\lambda I\right)^{-1} \mathbb{X}^{\mathrm{T}} \mathrm{t}
$$

Proof in discussion 4 this week

- Ridge replaces $\mathbb{X}^{\mathrm{T}} \mathbb{X}$ in normal equation with $\mathbb{X}^{\mathrm{T}} \mathbb{X}+\lambda I$.
- This adds $\lambda$ to every eigenvalue of $\mathbb{X}^{T} \mathbb{X}$ :

$$
\sigma_{i}^{2} \rightarrow \sigma_{i}^{2}+\lambda
$$
- $\lambda>0$ boosts all eigenvalues.
- Condition number shrinks:
$$
\kappa_{\text {ridge }}=\frac{\sigma_{\max }^{2}+\lambda}{\sigma_{\min }^{2}+\lambda} \ll \kappa
$$

This stabilizes inversion

## Illconditionin g Example

With the condition factor that is large, due to very small eigenvalues resulting from very similar features, the weights become unstable.

Try changing the noise and you will see substantial swings in weights.

```
N, D = 500, 2 # one bias + 2 features
X = np.ones((N, D + 1)) # first column = bias
x1 = np.random.randn(N)
x2 = x1 + 1e-4 * np.random.randn(N) # almost identical feature
X[:, 1] = x1 Condition ( }\mp@subsup{\mathbb{X}}{}{T}\mathbb{X})
X[:, 2] = x2 420140223.27925146
t = 3 + 2*x1 - 1*x2 + 0.1*np.random.randn(N) # targets
XtX = X.T @ X
Xty = X.T @ t
w_ne = np.linalg.solve(XtX, Xty)
lam = 1e-2
w_ridge = np.linalg.solve(XtX + lam*np.eye(D+1), Xty)
w_neq: [ 3.00273423 12.87110771
-11.87562795]
w_ridge : [3.00275573 0.50070697
0.49481053]
```

How to resolve the issue of ill-conditioned design matrix?

## SVD Recap: Stretch + Rotate

- Every real matrix $\mathbb{X}$ can be factored as

$$
\mathbb{X}=U \Sigma V^{T}
$$
$U$ and $V$ are rotations and have orthogonal columns ( $U^{T} U=V^{T} V=I$ ).
$\Sigma$ is diagonal scaling by singular values $\sigma_{i}$.
$\Sigma=\operatorname{diag}\left(\sigma_{1}, \ldots, \sigma_{r}\right)$ with $r=\operatorname{rank}(\mathbb{X})$.
$$
A=\quad U \quad \quad \Sigma \quad V^{\top}
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-23.jpg?height=413&width=1643&top_left_y=1335&top_left_x=884)

**Image Description:** The image consists of four diagrams depicting ellipsoids in a Cartesian coordinate system. Each diagram illustrates the geometric representation of different vector transformations. 

1. The first diagram shows an ellipse with axes labeled \( \sigma_1^{2} \) and \( \sigma_2^{2} \) (red arrows), indicating variances along principal components.
2. The second diagram focuses on a transformed ellipse with labels \( \sigma_1^{2} \) and \( \sigma_2^{2} \) (purple arrows).
3. The third diagram presents a circular cross-section with vectors \( \mathbf{e}_1 \) and \( \mathbf{e}_2 \) (blue arrows).
4. The last diagram illustrates another circular shape with vectors \( \mathbf{v}_1 \) and \( \mathbf{v}_2 \).

All shapes emphasize relationships between different basis vectors and transformations.

https://eecs16b.org/notes/sp24/note15.pdf

## Fix \#2: Flipping Singular Values with Moore-Penrose Pseudo-Inverse

Ordinary inverse of $\mathbb{X}=U \Sigma V^{T}$ is $\mathbb{X}^{-1}=V \Sigma^{-1} U^{T}$

- If some $\sigma_{i}=0$, we cannot invert directly.
- Moore-Penrose Pseudo-inverse:

$$
\sigma_{i i}^{+}= \begin{cases}\frac{1}{\sigma_{i}} & \text { if } \sigma_{i}>\varepsilon \sigma_{\max } \\ 0 & \text { otherwise }\end{cases}
$$
- We safely "flip" only the non-tiny singular values.
- Pseudo-Inverse:
$$
\mathbb{X}^{+}=V \Sigma^{+} U^{T}
$$

## Fix \#2: Flipping Singular Values with Moore-Penrose Pseudo-Inverse

Pseudo-Inverse:

$$
\mathbb{X}^{+}=V \Sigma^{+} U^{T}
$$
- Properties:
- $K X^{+} \mathbb{X}=\mathbb{X}$,
- $\mathbb{X}^{+} \mathbb{X} \mathbb{X}^{+}=\mathbb{X}^{+}$,
- $\left(\mathrm{XN}^{+}\right)^{T}=\mathrm{XN}^{+}$,
- $\left(\mathbb{X}^{+} \mathbb{X}\right)^{T}=\mathbb{X}^{+} \mathbb{X}$.
- These guarantee uniqueness of $X^{+}$.

## $\mathbb{X} \mathbb{X}^{+}$Is the Orthogonal Projector

Let $P=X X^{+}$. Then:

1. Projector: $P^{2}=X X^{+} X^{+}=X X^{+}=P$
2. Symmetric: $P^{T}=\mathrm{P}$
3. Range: $\quad P y=\mathbb{X} \mathbb{X}^{+} \mathbb{X} w=\mathbb{X} w=y \left\{\begin{array}{l}P \text { acts as identity on } \operatorname{span}(\mathbb{X}) . \\ \operatorname{range}(P) \text { is } \operatorname{span}(\mathbb{X}) .\end{array}\right.$
$P=X X^{+}$is the orthogonal projector onto $\operatorname{span}(\mathbb{X})$.

## Obtaining $w^{*}$

$P=X X^{+}$is the orthogonal projector onto $\operatorname{span}(\mathbb{X})$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-27.jpg?height=175&width=90&top_left_y=650&top_left_x=680)

We will use $P$ to find the projection of $t$ onto the $\operatorname{span}(\mathbb{X})$.

$$
\begin{array}{ll}
y=P t & \text { Replacing } \mathrm{P}=\mathrm{XX}^{+} \\
y=\mathrm{XX}^{+} \mathrm{t} & \text { Knowing that } y=\mathrm{X} w \\
w^{*}=\mathbb{X}^{+} \mathrm{t} & \text { Very important result!!! }
\end{array}
$$

## Fix \#2: Moore-Penrose Pseudo-Inverse

## How can Penrose Pseudo-Inverse help us with solving normal equation?

- Moore-Penrose Pseudo-inverse:

$$
\sigma_{i i}^{+}=\left\{\begin{array}{lr}
\frac{1}{\sigma_{i}} & \text { if } \sigma_{i}>\varepsilon \sigma_{\max } \\
0 & \text { otherwise }
\end{array}\right.
$$

- We safely "flip" only the non-tiny singular values.

Closed-form solution for least squares:

$$
w^{*}=\mathbb{X}^{+} t
$$

This works no matter whether $\mathbb{X}$ is tall ( $N>D$ ), square, or wide ( $N<D$ ). Also regardless of singular values.

## Why This $w^{*}$ Amongst So Many?

$P=X X^{+}$is the orthogonal projector onto $\operatorname{span}(\mathbb{X})$.
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-29.jpg?height=171&width=90&top_left_y=654&top_left_x=680)

We will use $P$ to find the projection of $t$ onto the $\operatorname{span}(X)$.

$$
\begin{array}{ll}
y=P t & \text { Replacing } \mathrm{P}=\mathbb{X X}^{+} \\
y=\mathbb{X} \mathbb{X}^{+} t & \text { Knowing that } y=\mathbb{X} w \\
w^{*}=\mathbb{X}^{+} t &
\end{array}
$$

## Why This $w^{*}$ Amongst So Many?

If $X$ is rank-deficient, there are infinitely many $w$ with the same prediction $y$.
You can add any vector from the nullspace of $\mathbb{X}$.

The Pseudo-inverse picks the one with smallest Euclidean form:

$$
w^{*}=\underset{w: \mathbb{X} w=y}{\operatorname{argmin}}\|w\|_{2}
$$

## Fix 2: Why It Helps?

| Issue | How pseudo-inverse fixes it? |  |
| :--- | :--- | :--- |
| Rank deficiency (perfect multicollinearity) | Issue Rank deficiency (perfect | How pseudo-inverse fixes it? Sets O for $\sigma=\mathrm{O} \Rightarrow$ removes redundant directions, gives |

## Fix 2: Why It Helps?

| Issue | How pseudo-inverse fixes it? |  |
| :--- | :--- | :--- |
| Rank deficiency (perfect multicollinearity) | Issue Rank deficioncy (perfect multicollinearity) Ml-conditioning | How pseudo-inverse fixes it? Sets o for $\sigma=0 \rightarrow$ removes redundant directions, gives the minimum-norm weight vector. Truncation threshold discards numerically meaningless directions, implicitly adds ridge-like regularization. |
| III-conditioning |  | Truncation threshold discards numerically meaningless directions, implicitly adds ridgelike regularization. |

## Fix 2: Why It Helps?

| Issue | How pseudo-inverse fixes it? |  |
| :--- | :--- | :--- |
| Rank deficiency (perfect multicollinearity) | Ramk aefinioney perfect Rank define iency multiodlinearity Ill-conditioning <br> Undordotexmined |  |
| III-conditioning | Truncation threshold discards numerically meaningless directions, implicitly adds ridgelike regularization. |  |
| ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-33.jpg?height=50&width=881&top_left_y=1149&top_left_x=135) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-33.jpg?height=48&width=1032&top_left_y=1171&top_left_x=140) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-33.jpg?height=60&width=1038&top_left_y=1225&top_left_x=135) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-33.jpg?height=66&width=1038&top_left_y=1273&top_left_x=135) | reaming amonemor coerect <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-33.jpg?height=60&width=460&top_left_y=1203&top_left_x=1220) <br> प्रत्ये विश्ववर्त्या | E0. N..................................... <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-33.jpg?height=87&width=1238&top_left_y=1171&top_left_x=1922) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-33.jpg?height=80&width=1249&top_left_y=1221&top_left_x=1922) поштык. |

## Fix 2：Why It Helps？

| Issue | How pseudo－inverse fixes it？ |  |
| :--- | :--- | :--- |
| Rank deficiency （perfect multicollinearity） | ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=76&width=676&top_left_y=523&top_left_x=1220) multicolineomity <br> Ynaoxdotermined Datams（N＜D <br> morromicomation | How parando－invorea fixeas itz <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=70&width=1237&top_left_y=544&top_left_x=1922) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=55&width=1017&top_left_y=593&top_left_x=1922) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=71&width=1249&top_left_y=620&top_left_x=1922) Rotumns tho uniquite olution with tho smallest $L_{2}$－ norm amons infinitely many exact fits <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=71&width=1302&top_left_y=749&top_left_x=1922) |
| III－conditioning | Truncation threshold discards numerically meaningless directions，implicitly adds ridge－ like regularization． |  |
|  |  |  |
| ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=60&width=892&top_left_y=1149&top_left_x=140) Nummon <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=65&width=1033&top_left_y=1252&top_left_x=140) No．（Now． | ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=93&width=449&top_left_y=1160&top_left_x=1220) प्रत्या木 <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=93&width=460&top_left_y=1289&top_left_x=1220) | ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=59&width=956&top_left_y=1149&top_left_x=1922) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=103&width=1292&top_left_y=1160&top_left_x=1922)

**Image Description:** The image contains a lecture slide summarizing the concept of pseudo-inverse in relation to minimizing redundancy in weight vector solutions. It emphasizes setting σ to 0, which results in the minimum-norm weight vector. Additionally, it discusses a truncation threshold that discards directions viewed as numerically insignificant, contributing to a ridge-like effect in the data. The text is formatted with a blue header and explanation points, suggesting it’s part of a larger discussion on linear algebra or optimization techniques in machine learning.
 <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=61&width=1249&top_left_y=1235&top_left_x=1922) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=65&width=1296&top_left_y=1273&top_left_x=1922) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=71&width=1314&top_left_y=1311&top_left_x=1922) |
| Implementation convenience | ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=109&width=465&top_left_y=1381&top_left_x=1215)

**Image Description:** The image is a text-based information slide from an academic lecture. It presents two key issues in statistical analysis or regression modeling: "Rank deficiency (multicollinearity)" and "Ill-conditioning." The layout suggests a structured approach to identifying and categorizing problems related to model stability and interpretability in statistical methods. No diagrams, equations, or graphical representations are present; the content is purely textual and addresses common computational issues in data analysis.
 प्रत⿻丷木 준준…⿴囗十 | ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=62&width=1238&top_left_y=1402&top_left_x=1922) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=65&width=1249&top_left_y=1441&top_left_x=1922) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=60&width=1287&top_left_y=1495&top_left_x=1922) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-34.jpg?height=77&width=1319&top_left_y=1532&top_left_x=1922) |

## Least-Squares @ Maximum Likelihood

- Recap of Regularization ${ }^{4234276}$
- When Normal Equation Gets Tricky
- Least-Squares $\cong$ Maximum Likelihood
- Uncertainty $\approx$ Confidence Intervals
- Decision Theory


## Least Squares $\cong$ Maximum Likelihood

Recall: We fit a linear featurized function by Minimizing the sum-of-squares error.

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}
$$
this just algebraic convenience ... or is there a probabilistic justificatio

## Additive-Gaussian Noise Assumption

Assume the target $t$ is the true signal plus noise:

$$
t=y(\mathrm{x}, w)+\epsilon
$$
where $\epsilon \sim \mathcal{N}\left(0, \sigma^{2}\right)$ (zero-mean Gaussian random variable) $\rightarrow$ i.i.d. across points.
- The conditional likelihood of one observation is:
$$
p\left(t \mid \mathrm{x}, w, \sigma^{2}\right)=\mathcal{N}\left(t \mid y(\mathrm{x}, w), \sigma^{2}\right)
$$

Gaussian noise: $p\left(t \mid x, w, \sigma^{2}\right)$
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-37.jpg?height=587&width=1349&top_left_y=1263&top_left_x=990)

**Image Description:** The image is a probabilistic density plot representing a continuous distribution. The horizontal axis (t) ranges from -1 to 5, while the vertical axis indicates density, scaling from 0 to approximately 0.5. A smooth, orange curve illustrates the density function, peaking at t = 2. A dashed vertical line at t = 2 denotes the mean, labeled as \( \text{mean } y(x, w) \). The legend is positioned in the upper right corner, providing context for the mean indicator. The overall layout includes a grid for enhanced readability.


## Maximum Likelihood: Over the Datasetitation

- Consider the data set of i.i.d. inputs $\mathbb{X}=\left\{\mathrm{x}_{1}, \ldots, \mathrm{x}_{N}\right\}$ with corresponding target values $t_{1}, \ldots, t_{N}$. We group the target variables $\left\{t_{n}\right\}$ into a column vector that we denote by t .

$$
\begin{aligned}
p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right) & =\mathcal{N}\left(\mathrm{t} \mid y(\mathbb{X}, w), \sigma^{2}\right) \\
& =\prod_{n=1}^{N} \mathcal{N}\left(t_{n} \mid w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right), \sigma^{2}\right) \\
p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right) & =\prod_{n=1}^{N} \mathcal{N}\left(t_{n} \mid w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right), \sigma^{2}\right)
\end{aligned}
$$

## Least Squares $\cong$ Maximum Likelihood

$$
p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)=\prod_{n=1}^{N} \mathcal{N}\left(t_{n} \mid w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right), \sigma^{2}\right)
$$

The optimum value of $w$ in the above equation equals the optimum value of the log of the function (because log is a monotonically

$$
\begin{array}{rlr}
\text { increasing function) } & \mu & \\
\begin{aligned}
\ln \left(p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)\right) & =\sum_{n=1}^{N} \ln \left(\mathcal{N}\left(t_{n} \mid w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right), \sigma^{2}\right)\right)
\end{aligned} & & \begin{array}{l}
\text { Writing the equation for a } \\
\text { Normal distribution }
\end{array} \\
& =\sum_{n=1}^{N} \ln \left(\frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-\frac{1}{2 \sigma^{2}}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2}}\right) & \\
& =\sum_{n=1}^{N}\left(\ln \left(\frac{1}{\sqrt{2 \pi \sigma^{2}}}\right)-\frac{1}{2 \sigma^{2}}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2}\right) & \\
\ln \left(p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)\right) & =-\frac{N}{2} \ln \left(2 \pi \sigma^{2}\right)-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2} &
\end{array}
$$

## Least Squares $\cong$ Maximum Likelihood

$$
\begin{aligned}
\ln \left(p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)\right) & =-\frac{N}{2} \ln \left(2 \pi \sigma^{2}\right)-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2} \\
& =-\frac{N}{2} \ln (2 \pi)-\frac{N}{2} \ln \left(\sigma^{2}\right)-\frac{1}{\sigma^{2}} \mathrm{E}_{\mathrm{D}}(\mathrm{w}){ }^{(*)} \\
& \mathrm{E}_{\mathrm{D}}(\mathrm{w})=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2}
\end{aligned}
$$

- The first two terms in can be treated as constants when determining $w$ because they are independent of $w$.
- Therefore, maximizing the likelihood function under a Gaussian noise distribution is equivalent to minimizing the sum-of-squares error function.


## Maximum Likelihood

$\nabla_{\mathrm{w}} \ln \left(p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)\right)=-\frac{1}{\sigma^{2}} \nabla_{\mathrm{w}} \mathrm{E}_{\mathrm{D}}(\mathrm{w})$
Sum rule $\quad=\nabla_{\mathrm{w}}\left(-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2}\right)$
Chain rule $\quad=-\frac{1}{2 \sigma^{2}} \nabla_{\mathrm{w}}\left(\sum_{n=1}^{N}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2}\right)$
Simplify

$$
\begin{aligned}
& =-\frac{1}{2 \sigma^{2}} 2 \sum_{n=1}^{N}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right)\left(-\phi\left(\mathrm{x}_{\mathrm{n}}\right)\right) \\
& =\frac{1}{\sigma^{2}} \sum_{n=1}^{N}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right) \phi\left(\mathrm{x}_{\mathrm{n}}\right)
\end{aligned}
$$

Setting the gradient to 0 gives:

$$
\begin{aligned}
& \frac{1}{\sigma^{2}} \sum_{n=1}^{N}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right) \phi\left(\mathrm{x}_{\mathrm{n}}\right)=0 \\
& \sum_{n=1}^{N} t_{n} \phi\left(\mathrm{x}_{\mathrm{n}}\right)-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right) \phi\left(\mathrm{x}_{\mathrm{n}}\right)=0 \\
& \sum_{n=1}^{N} t_{n} \phi\left(\mathrm{x}_{\mathrm{n}}\right)=\sum_{n=1}^{N} w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right) \phi\left(\mathrm{x}_{\mathrm{n}}\right) \\
& \sum_{n=1}^{N} t_{n} \phi\left(\mathrm{x}_{\mathrm{n}}\right)=\sum_{n=1}^{N} \phi\left(\mathrm{x}_{\mathrm{n}}\right) \phi\left(\mathrm{x}_{\mathrm{n}}\right)^{T} w
\end{aligned}
$$

## Maximum Likelihood

$\Gamma \sum_{n=1}^{N} t_{n} \phi\left(\mathrm{x}_{\mathrm{n}}\right)=\sum_{n=1}^{N} \phi\left(\mathrm{x}_{\mathrm{n}}\right) \phi\left(\mathrm{x}_{\mathrm{n}}\right)^{T} w$
If we stack features $\phi\left(\mathrm{x}_{\mathrm{n}}\right)$ row-wise in $\Phi(\mathbb{X})=\left[\begin{array}{c}\phi\left(\mathrm{x}_{1}\right)^{T} \\ \phi\left(\mathrm{x}_{2}\right)^{T} \\ \cdots \\ \text { the design matrix } \Phi(\mathbb{X})\end{array}\right]$
$\Phi(\mathbb{X})^{T} \mathbf{t}=\Phi(\mathbb{X})^{T} \Phi(\mathbb{X}) w_{M L}$
$w_{M L}=\left(\Phi(\mathbb{X})^{T} \Phi(\mathbb{X})\right)^{-1} \Phi(\mathbb{X})^{T} \mathrm{t}$
Normal Equations for the least squared problem

## Least Squares $\cong$ Maximum Likelihood

$$
\begin{aligned}
& p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)=\prod_{n=1}^{N} \mathcal{N}\left(t_{n} \mid w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right), \sigma^{2}\right) \\
& \ln \left(p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)\right)=-\frac{N}{2} \ln \left(2 \pi \sigma^{2}\right)-\frac{1}{2 \sigma^{2}} E_{D}(w) \quad \text { with } \quad \mathrm{E}_{\mathrm{D}}(\mathrm{w})=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2} \\
& \nabla_{\mathrm{w}} \ln \left(p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)\right)=-\frac{1}{\sigma^{2}} \nabla_{\mathrm{w}} \mathrm{E}_{\mathrm{D}}(\mathrm{w}) \longrightarrow w_{M L}=\left(\Phi(\mathbb{X})^{T} \Phi(\mathbb{X})\right)^{-1} \Phi(\mathbb{X})^{T} \mathrm{t} \\
& \quad \Phi(\mathbb{X})=\left[\begin{array}{c}
\phi\left(\mathrm{x}_{1}\right)^{T} \\
\phi\left(\mathrm{x}_{2}\right)^{T} \\
\ldots \\
\phi\left(\mathrm{x}_{\mathrm{N}}\right)^{T}
\end{array}\right]
\end{aligned}
$$

Hence the least-squares solution is the MLE under Gaussian noise.

## MLE Estimate of $w_{0}$

$$
\begin{array}{l|l}
\mathrm{E}_{\mathrm{D}}(\mathrm{w})=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2} & \frac{\partial E_{D}(w)}{\partial w_{0}}=\frac{\partial\left(\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-w_{0}-\sum_{j=1}^{M-1} w_{j} \phi_{j}\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2}\right)}{\partial w_{0}} \\
& w_{0}=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-w_{0}-\sum_{j=1}^{M-1} w_{j} \phi_{j}\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2} \\
& \underbrace{\frac{1}{N} \sum_{n=1}^{N} t_{n}}_{t} \times 2 \times(-1) \times \sum_{n=1}^{N}-\underbrace{\frac{1}{N} \sum_{n=1}^{N} \sum_{j=1}^{M-1} w_{j} \phi_{j}\left(\mathrm{x}_{\mathrm{n}}\right)}_{\sum_{j=1}^{M-1} w_{j} \bar{\phi}_{j}} \text { with }{ }_{\overline{\phi_{j}}}=\frac{1}{N} \sum_{n=1}^{N} \bar{\phi}_{j}\left(\mathrm{x}_{n}\right)
\end{array}
$$

## MLE Estimate of $w_{0}$

$$
\begin{aligned}
& w_{0}=\underbrace{\frac{1}{N} \sum_{n=1}^{N} t_{n}}_{\bar{t}}-\underbrace{\frac{1}{N} \sum_{n=1}^{N} \sum_{j=1}^{M-1} w_{j} \phi_{j}\left(\mathrm{x}_{\mathrm{n}}\right)}_{\sum_{j=1}^{M-1} w_{j} \overline{\phi_{j}}} \text { with } \overline{\phi_{j}}=\frac{1}{N} \sum_{n=1}^{N} \overline{\phi_{j}}\left(\mathrm{x}_{n}\right) \\
& w_{0}=\bar{t}-\sum_{j=1}^{M-1} w_{j} \overline{\phi_{j}}
\end{aligned}
$$

The bias component is the difference between averages of target values and weighted sum of averages of basis function values.

## MLE Estimate of $\sigma_{M L}$

$$
\begin{aligned}
& w_{0}=\underbrace{\frac{1}{n} \sum_{n=1}^{N} t_{n}}_{\bar{t}}-\underbrace{\frac{1}{n} \sum_{n=1}^{N} \sum_{j=1}^{M-1} w_{j} \phi_{j}\left(\mathrm{x}_{\mathrm{n}}\right)}_{\sum_{j=1}^{M-1} w_{j} \overline{\phi_{j}}} \\
& w_{0}=\bar{t}-\sum_{j=1}^{M-1} w_{j} \overline{\phi_{j}}
\end{aligned}
$$

The bias component is the difference between averages of target values and weighted sum of averages of basis function values.

We can also prove that:

$$
\sigma_{M L}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left\{t_{n}-w_{M L}^{T} \phi\left(\mathrm{x}_{n}\right)\right\}^{2}
$$

The maximum likelihood value of the variance parameter is given by the residual variance of the target values around the regression function.

## Uncertainty $\approx$ Confidence Intervals

- Recap of Regularization ${ }^{4234276}$
-When Normal Equation Gets Tricky
- Least-Squares $\cong$ Maximum Likelihood
- Uncertainty $\approx$ Confidence Intervals
- Decision Theory


## Posterior Predictive Distribution

What we actually care about is not $w$, but predictions for new data.

- The posterior predictive integrates over parameter uncertainty:

$$
p\left(\mathrm{t} \mid \mathrm{x}_{\text {test }}, D\right)=\int p\left(\mathrm{t} \mid \mathrm{x}_{\text {test }}, w\right) p(w \mid D) d w
$$
$p\left(\mathrm{t} \mid \mathrm{x}_{\text {test }}, w\right)$ : Likelihood of t given parameters $w$.
$p(w \mid D)$ : Posterior distribution over weights after seeing training data $D$.
Averages predictions across all plausible parameter settings, weighted by how likely they are under the posterior.

## Uncertainty from Noise $\approx$ Confidence Intervals

$$
p\left(\mathrm{t} \mid \mathrm{x}_{\text {test }}, D\right)=\int p\left(\mathrm{t} \mid \mathrm{x}_{\text {test }}, w\right) p(w \mid D) d w
$$
- Predictive distribution (with fixed parameters, noise-only)
$$
p(t \underbrace{\mid \mathrm{x}_{\text {test }}, w, \sigma^{2}})=\mathcal{N}\left(t \mid y\left(\phi\left(\mathrm{x}_{\text {test }}\right), w\right), \sigma^{2}\right)
$$

Predictive given fixed $w$
$95 \% \mathrm{CI}$ (single prediction) $\in y\left(\phi\left(\mathrm{x}_{\text {test }}\right), w\right) \pm 1.96 \sigma$
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-49.jpg?height=792&width=1221&top_left_y=582&top_left_x=2092)

**Image Description:** The image contains a scatter plot with a fitted curve illustrating predictive risk. The x-axis ranges from -1.00 to 1.00, representing the predictor variable \( x \). The y-axis spans from -1.5 to 1.5, displaying the response variable \( t \). Data points are represented by blue dots. The orange line denotes the predicted function \( y(x, w) \), while the light blue area around it indicates the 95% confidence interval for a single draw, and the darker blue area represents the confidence interval for the mean of 10 predictions.


If you average $k$ draws:
s.e. $=\frac{\sigma}{\sqrt{k}} \rightarrow$ confidence interval ribbon shrinks by $\sqrt{k}$.

## Uncertainty $\approx$ Confidence Intervals

- Quantifies risk: wide band $\Rightarrow$ cautious decisions.
- Drives outlier detection
- Lets you compare models: same mean, different uncertainty $\Rightarrow$ picl tighter band.

Confidence ribbons illustrate predictive risk
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-50.jpg?height=890&width=1455&top_left_y=646&top_left_x=1837)

**Image Description:** The image is a scatter plot with a fitted curve. The x-axis represents variable \( x \), ranging from -1.0 to 1.0. The y-axis denotes variable \( t \), also spanning between approximately -1.5 and 1.5. Blue data points indicate observations, while an orange line illustrates the predicted function \( y(x, w) \). The shaded area surrounding the prediction shows a 95% confidence interval (CI), with a lighter blue area representing a single draw of the CI and the darker blue area indicating the mean of the CI.


## Least-Squares + Gaussian Prior $\Rightarrow$ Ridge Regression

Posterior (Bayes Rule):

$$
p(\mathrm{w} / \mathrm{t}) \propto p(\mathrm{t} \mid \mathrm{w}) p(\mathrm{w})
$$

Work with the $-\log : \quad-\log (p(w \mid t))=-\log (p(t \mid w))-\log (p(w))+$ const
Plugging in Gaussian:

$$
\begin{aligned}
p\left(w \mid \tau^{2}\right)=\frac{1}{\left(2 \pi \tau^{2}\right)^{\frac{M}{2}}} e^{-\frac{\|w\|^{2}}{2 \tau^{2}}} & \longrightarrow \log (p(t \mid w))=\frac{1}{2 \sigma^{2}} \sum_{n}\left(t_{n}-y(\phi(\mathrm{x}), w)\right)^{2} \\
w_{M A P}^{*} & =\underset{w}{\arg \min } \sum_{n}\left(t_{n}-y(\phi(\mathrm{x}), w)\right)^{2}+\frac{\|w\|^{2}}{2 \tau^{2}}+\text { const } \overbrace{\begin{array}{c}
\text { Combine and } \\
\text { multiply by } 2 \sigma^{2}
\end{array}}^{\tau^{2}}{ }_{\lambda \text { for ridge regression }}
\end{aligned}
$$

## Noise Model $\Longleftrightarrow$ Error Function

## Zero-mean Gaussian Noise

$p(\epsilon)=\frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{\epsilon^{2}}{2 \sigma^{2}}\right)$
$-\log p(\epsilon)=\frac{\epsilon^{2}}{2 \sigma^{2}}+$ const
$\Rightarrow M S E$ loss
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-52.jpg?height=779&width=916&top_left_y=1071&top_left_x=416)

**Image Description:** The image presents a comparison of Probability Density Functions (PDFs) for Gaussian and Laplace distributions. It features a 2D plot with the x-axis ranging from -4 to 4, representing the random variable values, and the y-axis extending from 0 to 1, illustrating the density. The Gaussian distribution is depicted in blue, showing a bell-shaped curve centered at 0, while the Laplace distribution, shown in orange, has a sharper peak at 0 and heavier tails. The graph effectively highlights the differences in shape and spread between the two distributions.


## Zero-mean Laplacian Noise

$p(\epsilon)=\frac{1}{2 b} \exp \left(-\frac{|\epsilon|}{b}\right)$
$-\log p(\epsilon)=\frac{|\epsilon|}{b}+$ const
$\Rightarrow M A E$ loss
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-52.jpg?height=770&width=885&top_left_y=1080&top_left_x=1824)

**Image Description:** The diagram is a plot showing two loss functions: Mean Squared Error (MSE) and Mean Absolute Error (MAE). The x-axis represents the error values, ranging from -4 to 4, while the y-axis indicates the loss values, extending from 0 to approximately 6. The blue curve, representing MSE, is a parabolic shape, opening upwards, with its vertex at the origin (0,0). The orange line represents MAE, forming a 'V' shape, also meeting the y-axis at 0. The legend in the top right distinguishes the two curves.


## Decision Theory

- Recap of Regularization ${ }^{4234276}$
-When Normal Equation Gets Tricky
- Least-Squares $\cong$ Maximum Likelihood
- Uncertainty $\approx$ Confidence Intervals
- Decision Theory


## Decision Theory for Regression

- Why decision theory?
- A probabilistic predictive model outputs a distribution over possible targets, not a single value.
- We need to choose one actionable prediction from that distribution.
- Decision theory tells us how to pick the "best" point estimate given a loss function.


## Inference vs. Decision Stages

## Inference stage:

- Fit model parameters to data $\rightarrow$ obtain a predictive distribution

$$
p\left(\mathrm{t} \mid \mathrm{x}, w^{*}, \sigma^{* 2}\right)=\mathcal{N}\left(\mathrm{t} \mid y\left(\mathrm{x}, w^{*}\right)\right)
$$
- Captures our uncertainty about the true target value.

## - Decision stage:

- Given $p(\mathrm{t} \mid \mathrm{x})$, choose a single value $f(\mathrm{x})$ to minimize expected penalty.


## Defining Loss Functions

- A loss $L(t, f)$ measures penalty when true value is $t$ but we predict $f$.
- Common choices:
- Squared loss: $L(t, f)=(f-t)^{2}$
- Absolute loss: $L(t, f)=|f-t|$
- General Minkowski loss: $L_{q}(t, f)=|f-t|^{q}$
- Different losses $\rightarrow$ different "best" decision rule.


## Expected Loss

Since $t$ is unknown, minimize the expected loss over the conditional $\mathrm{p}(\mathrm{t} \mid \mathrm{x})$ :

$$
E[L \mid \mathrm{x}]=\iint L(\mathrm{t}, f(\mathrm{x})) p(\mathrm{t} \mid \mathrm{x}) d \mathrm{x} d \mathrm{t}
$$

Where we are averaging over both input and target variables, weighted by their joint distribution $\mathrm{p}(\mathrm{t} \mid \mathrm{x})$.

- Goal: Choose $f(\mathrm{x})$ to minimize $E[L \mid \mathrm{x}]$.


## Optimal Predictor for Squared Loss

- For $L(\mathrm{t}, f)=(f-\mathrm{t})^{2}$, we set the derivative of $E[L \mid \mathrm{x}]$ with respect to $f$ to 0 .

$$
\begin{array}{ll}
E[L \mid \mathrm{x}]=\iint L(\mathrm{t}, f(\mathrm{x})) p(\mathrm{t} \mid \mathrm{x}) d \mathrm{t} & \text { Using squared loss } \\
E[L \mid \mathrm{x}]=\iint(f-\mathrm{t})^{2} p(\mathrm{t} \mid \mathrm{x}) d \mathrm{t} & \begin{array}{l}
\text { Calculating the } \\
\text { derivative wrt } f
\end{array} \\
\frac{\partial E[L \mid \mathrm{x}]}{\partial f(\mathrm{x})}=\int 2(f-\mathrm{t}) p(\mathrm{t} \mid \mathrm{x}) d \mathrm{t}=0 & \text { Rearranging terms } \\
f^{*}(\mathrm{x}) \int p(\mathrm{t} \mid \mathrm{x}) d \mathrm{t}=\int \mathrm{t} p(\mathrm{t} \mid \mathrm{x}) d \mathrm{t} & \text { Simplifying } \\
f^{*}(\mathrm{x})=\frac{\int \mathrm{t} p(\mathrm{t} \mid \mathrm{x}) d \mathrm{t}}{\int p(\mathrm{t} \mid \mathrm{x}) d \mathrm{t}}=\frac{\int \mathrm{t} p(\mathrm{t} \mid \mathrm{x}) d t}{1}=E_{\mathrm{t}}[\mathrm{t} \mid \mathrm{x}]
\end{array}
$$

Interpretation :
Under squared loss, the mean of the predictive distribution is optimal.
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-59.jpg?height=456&width=541&top_left_y=557&top_left_x=178)

**Image Description:** The image is a stylized representation of clouds, characterized by a simplified, rounded shape. It features a larger cloud overlapping a smaller one, both outlined in purple with light purple fill. The design lacks intricate details, focusing on a flat, graphical aesthetic typical in educational contexts. The clouds are positioned in a way that suggests they are floating in the sky, simplifying the concept of cloud formations for illustrative purposes in a lecture on meteorology or atmospheric science.


$$
\begin{aligned}
& \text { Suppose } p(t \mid x) \text { is: } \\
& t=1(0.2) \text {, } \\
& t=3(0.5), \\
& t=10(0.3) \text {. } \\
& \text { What is } f(x) \text { under squared } \\
& \text { loss? }
\end{aligned}
$$

## Beyond Squared Loss Minkowski Family

Loss: $L_{q}(t, f)=|f-t|^{q}$

## Optimal solutions:

- $\mathrm{q}=2 \Rightarrow$ mean (conditional average)
- $\mathrm{q}=1 \Rightarrow$ median ( $50^{\text {th }}$ percentile)
- $q \rightarrow 0 \Rightarrow$ mode (peak of $p(t \mid \mathrm{x})$ )

Minkowski Loss Functions
![](https://cdn.mathpix.com/cropped/2025_10_01_d7f20f4e50dde858fb01g-60.jpg?height=1269&width=1664&top_left_y=318&top_left_x=1411)

**Image Description:** The image is a 2D plot showing multiple curves representing the function \( f - t^q \) as a function of \( f - t \). The x-axis denotes \( f - t \), ranging approximately from -2 to 2, while the y-axis represents the value of \( |f - t^q| \), showing values up to 8. Curves for different values of \( q \) (0.5, 1, 2, and 3) are color-coded: blue for \( q = 0.5 \), orange for \( q = 1 \), green for \( q = 2 \), and red for \( q = 3 \). A legend indicates the corresponding colors for each \( q \).


Under the Gaussian noise mean, median, and mode are the same.

## Lecture 8

## Linear Regression (3)

Credit: Joseph E. Gonzalez and Narges Norouzi
Reference Book Chapters: Chapter sections 4.1 and 4.2

