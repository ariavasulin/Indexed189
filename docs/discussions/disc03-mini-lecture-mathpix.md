---
course: CS 189
semester: Fall 2025
type: discussion
title: Discussion 3
source_type: slides
source_file: Discussion Mini Lecture 03.pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Discussion Mini Lecture 3

# K-Means Clustering, Probability \& MLE 

K-Means Clustering, Probability Theory \& Maximum Likelihood Estimation

CS 189/289A, Fall 2025 @ UC Berkeley<br>Sara Pohland

## Concepts Covered

1. K-Means Clustering
2. Probability Theory
a) Probability Fundamentals
b) Common Distributions
c) Multiple Random Variables
3. Maximum Likelihood Estimation (MLE)
4. Convex Optimization

## K-Means Clustering

1. K-Means Clustering
2. Probability Theory
a) Probability Fundamentals
b) Common Distributions
c) Multiple Random Variables
3. Maximum Likelihood Estimation (MLE)
4. Convex Optimization

## Goal of K-Means Clustering

Goal: Partition a set of $N D$-dimensional data points, $\mathcal{D}=\left\{\boldsymbol{x}_{\mathbf{1}}, \ldots, \boldsymbol{x}_{\mathbf{N}}\right\}$ where $\boldsymbol{x}_{\boldsymbol{n}} \in \mathbb{R}^{D}$, into $K$ disjoint clusters.

Unlabeled Data
![](https://cdn.mathpix.com/cropped/2025_10_01_1900c9840f0af548fa40g-04.jpg?height=856&width=1302&top_left_y=922&top_left_x=153)

**Image Description:** The image is a scatter plot diagram depicting clusters of gray dots arranged in a heart-like shape. The dots are evenly distributed, forming three distinct clusters, suggestive of some form of data grouping or clustering analysis. No axes are visible, indicating that the plot may be conceptual rather than quantitative. The overall design emphasizes the pattern formed by the arrangement of dots.


Clustered Data
![](https://cdn.mathpix.com/cropped/2025_10_01_1900c9840f0af548fa40g-04.jpg?height=882&width=1371&top_left_y=905&top_left_x=1815)

**Image Description:** The image depicts a clustered diagram illustrating five distinct groups represented by different colored circles: green, blue, purple, orange, and yellow. Each color signifies a unique cluster, with circles arranged in a loose formation. The clusters vary in size and density, indicating potential relationships or classifications among the data points. No axes are present, suggesting that this is a qualitative representation of groupings rather than a quantitative measure. The overall layout emphasizes the concept of clustering in data analysis or similar fields.


## K-Means Clustering Problem

Objective: Assign each data point, $\boldsymbol{x}_{\boldsymbol{n}}$, to a cluster $z_{n} \in\{1, \ldots, K\}$ such that the sum of the sample variances for each cluster is minimized.
![](https://cdn.mathpix.com/cropped/2025_10_01_1900c9840f0af548fa40g-05.jpg?height=1060&width=1638&top_left_y=731&top_left_x=157)

**Image Description:** The image displays a mathematical equation formatted in LaTeX. It shows the optimization problem represented by:

$$
\arg \min_z \sum_{k=1}^K \sum_{n: z_n=k} \| x_n - \mu_k \|^2
$$

The equation indicates the objective of minimizing the sum of squared distances between data points \(x_n\) and their corresponding cluster centers \(\mu_k\) across \(K\) clusters, emphasizing the relationship to variance within each cluster.

![](https://cdn.mathpix.com/cropped/2025_10_01_1900c9840f0af548fa40g-05.jpg?height=694&width=1294&top_left_y=693&top_left_x=1900)

**Image Description:** The slide presents a formula for calculating the mean of cluster \( k \) in a clustering algorithm. The formula is displayed prominently:

$$
\mu_k = \frac{\sum_{n: z_n = k} x_n}{\sum_{n: z_n = k} 1}
$$

It depicts two arrows pointing towards the numerator and denominator, indicating the sum of points assigned to cluster \( k \) over the total points assigned to that cluster, respectively. The label "mean of cluster k" is placed below the equation for clarity.


## Lloyd's Algorithm ("naïve k-means")

Given an initial set of $K$ means, $\boldsymbol{\mu}_{\mathbf{1}}, \ldots, \boldsymbol{\mu}_{K}$, alternate between the following two update steps until convergence:

1. Update Assignments: Assign each data point to the cluster with the nearest mean: $N$

$$
z=\arg \min _{z} \sum_{n=1}\left\|x_{n}-\mu_{z_{n}}\right\|_{2}^{2}
$$
2. Update Means: Recalculate means based on data points assigned to each cluster:
\$\$\begin{aligned}

\& ster: <br>
\& \mu=\arg \min _{\mu} \sum_{n=1}^{N}\left\|x_{n}-\mu_{z_{n}}\right\|_{2}^{2}

\end{aligned}\$\$

## Random Variables \& Statistical Measures

1. K-Means Clustering
2. Probability Theory
a) Probability Fundamentals
b) Common Distributions
c) Multiple Random Variables
3. Maximum Likelihood Estimation (MLE)
4. Convex Optimization

## Probability Mass Function (PMF)

A discrete random variable is defined over a countable set, $\mathcal{X}$.
For a discrete random variable, $X$, we define the probability mass function (PMF) as $p_{X}(x)=P(\{X=x\})$, where $x$ is a realization.

A valid PMF satisfies the following properties:

1. $p_{X}(x) \geq 0 \forall x \in \mathcal{X}$
2. $\sum_{x \in \mathcal{X}} p_{X}(x)=1$
3. $\mathrm{P}(\mathrm{A})=\sum_{x \in A} p_{X}(x)$

## Probability Density Function (PDF)

A continuous random variable, $X$, is defined over an interval of values, $X$. Because it is defined over an uncountable range of values, the probability that it takes on a value of exactly $x$ is zero.

We thus define the probability density function (PDF), $f_{X}(x)^{*}$, to capture the probability that $X$ is near a realization $x$.

A valid PDF satisfies the following properties:

1. $f_{X}(x) \geq 0 \forall x \in \mathcal{X}$
2. $\int_{X} f_{X}(x) d x=1$
3. $\mathrm{P}(\{\mathrm{a}<\mathrm{X} \leq \mathrm{b}\})=\int_{a}^{b} f_{X}(x) d x$

* The text uses $p(x)$ to denote probability density, but $f_{X}(x)$ is the more common convention, so this is what I'm using so that it is also familiar to you.


## Expectation of a Random Variable

Expected Value:
Discrete: $\quad \mathbb{E}[X]=\sum_{x \in \mathcal{X}} x p_{X}(x)$

Continuous: $\mathbb{E}[X]=\int_{x} x f_{X}(x) d x \quad \mathbb{E}[g(X)]=\int_{x} g(x) f_{X}(x) d x$

Linearity of Expectation: $\mathbb{E}[a X+b]=a \mathbb{E}[X]+b$

## Law of the Unconscious Statistician:

$$
\mathbb{E}[g(X)]=\sum_{x \in X} g(x) p_{X}(x)
$$

## Variance \& Standard Deviation

Variance - measures the dispersion around the expected value

$$
\operatorname{Var}(X)=\mathbb{E}\left[(X-\mathbb{E}[X])^{2}\right]=\mathbb{E}\left[X^{2}\right]-\mathbb{E}[X]^{2}
$$

A Useful Property: $\operatorname{Var}(a X+b)=a^{2} \operatorname{Var}(X)$

Standard Deviation - another measure of dispersion/spread

$$
\sigma_{X}=\sqrt{\operatorname{Var}(X)}
$$

## Common Probability Distributions

1. K-Means Clustering
2. Probability Theory
a) Probability Fundamentals
b) Common Distributions
c) Multiple Random Variables
3. Maximum Likelihood Estimation (MLE)
4. Convex Optimization

## Bernoulli Random Variable

Bernoulli random variables are used to represent the outcome a single binary experiment.

$$
X \sim \operatorname{Bern}(\mu), p=\text { probability of success }
$$

PMF: $\quad p_{X}(x ; p)=p^{x}(1-p)^{1-x}=\left\{\begin{array}{cl}1-p & \text { if } x=0 \\ p & \text { if } x=1 \\ 0 & \text { otherwise }\end{array}\right.$

Expected Value: $\quad \mathbb{E}[X]=p$
Variance:

$$
\operatorname{Var}(X)=p(1-p)
$$

## Binomial Random Variable

Binomial random variables are used to represent the outcome of multiple binary (Bernoulli) experiments.
$X \sim \operatorname{Bin}(n, \mathrm{p}), n=$ number of trials, $p=$ probability of success
PMF: $\quad p_{X}(x ; n, p)=\left\{\begin{array}{cc}\binom{n}{x} p^{x}(1-p)^{n-x} & \text { if } x=0,1, \ldots, n \\ 0 & \text { otherwise }\end{array}\right.$

Expected Value: $\quad \mathbb{E}[X]=n p$
Variance: $\quad \operatorname{Var}(X)=n p(1-p)$

## Exponential Random Variable

Exponential random variables are used to model the amount of time that passes before an event occurs.

$$
X \sim \operatorname{Exp}(\lambda), \lambda=\text { rate/inverse scale }
$$

PDF: $\quad f_{X}(x ; \lambda)=\left\{\begin{array}{cc}\lambda e^{-\lambda x} & \text { if } x \geq 0 \\ 0 & \text { otherwise }\end{array}\right.$

Expected Value: $\quad \mathbb{E}[X]=1 / \lambda$
Variance:

$$
\operatorname{Var}(X)=1 / \lambda^{2}
$$

## Gaussian Random Variable

Gaussian random variables are used to represent many continuous real-world processes.

$$
X \sim \mathcal{N}\left(\mu, \sigma^{2}\right), \mu=\text { mean, } \sigma^{2}=\text { variance }
$$

PDF: $\quad f_{X}\left(x ; \mu, \sigma^{2}\right)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right)$

Expected Value: $\quad \mathbb{E}[X]=\mu$
Variance:

$$
\operatorname{Var}(X)=\sigma^{2}
$$

## Standard Normal Random Variable

The standard normal random variable is the Gaussian random variable with zero mean and unit variance.

$$
X \sim \mathcal{N}(0,1)
$$

PDF: $\quad f_{X}(x)=\frac{1}{\sqrt{2 \pi}} \exp \left(-\frac{1}{2} x^{2}\right)$

Expected Value: $\quad \mathbb{E}[X]=0$
Variance:

$$
\operatorname{Var}(X)=1
$$

## Functions of Multiple Random Variables

1. K-Means Clustering
2. Probability Theory
a) Probability Fundamentals
b) Common Distributions
c) Multiple Random Variables
3. Maximum Likelihood Estimation (MLE)
4. Convex Optimization

## Joint Probability Mass Function (PMF)

Consider two discrete random variables, $X$ and $Y$, which are defined over the countable sets $\mathcal{X}$ and $\mathcal{Y}$ respectively.

We define the joint probability mass function (PMF) as $p_{X, Y}(x, y)=P(\{X=x, Y=y\})$, where $x$ and $y$ are realizations.

We can find the marginal PMFs from the joint PMFs as such:

$$
p_{X}(x)=\sum_{y \in \mathcal{Y}} p_{X, Y}(x, y) \quad p_{Y}(y)=\sum_{x \in \mathcal{X}} p_{X, Y}(x, y)
$$

## Joint Probability Density Function (PDF)

Now consider two continuous random variables, $X$ and $Y$, which are defined over the intervals $\mathcal{X}$ and $\mathcal{Y}$ respectively.

We define the joint probability density function (PDF), $f_{X, Y}(x, y)$, to model the probability that $X$ is near $x$ and $Y$ is near $y$.

We can find the marginal PDFs from the joint PDFs as such:

$$
f_{X}(x)=\int_{\mathcal{Y}} f_{X, Y}(x, y) d y \quad f_{Y}(y)=\int_{x} f_{X, Y}(x, y) d x
$$

## Expected Value for Multiple RVs

Discrete: $\quad \mathbb{E}[g(X, Y)]=\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} g(x, y) p_{X, Y}(x, y)$
Continuous: $\quad \mathbb{E}[g(X, Y)]=\int_{X} \int_{\mathcal{Y}} g(x, y) f_{X, Y}(x, y) d x d y$

For any two random variables, $X$ and $Y$,

$$
\mathbb{E}[X+Y]=\mathbb{E}[X]+\mathbb{E}[Y]
$$

## Variance \& Covariance for Multiple RVs

For any two random variables, $X$ and $Y$,

$$
\operatorname{Var}(X+Y)=\operatorname{Var}(X)+\operatorname{Var}(Y)+2 \underbrace{\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]}_{\operatorname{Cov}(X, Y)}
$$

The covariance, $\operatorname{Cov}(X, Y)$, describes how the pair of random variables, $X$ and $Y$, vary together.

- If $\operatorname{Cov}(X, Y)>0$, then as X increases, Y tends to increase.
- If $\operatorname{Cov}(X, Y)<0$, then as X increases, Y tends to decrease.
- If $\operatorname{Cov}(X, Y)=0$, then X and Y are said to be uncorrelated.


## Independent Random Variables

Two random variables are independent if knowing information about one does not tell us information about the other.

Discrete random variables, $X$ and $Y$, are independent if and only if

$$
p_{X, Y}(x, y)=p_{X}(x) p_{Y}(y) \forall x \in \mathcal{X}, y \in \mathcal{Y}
$$

Continuous random variables, $X$ and $Y$, are independent if and only if

$$
f_{X, Y}(x, y)=f_{X}(x) f_{Y}(y) \forall x \in \mathcal{X}, y \in \mathcal{Y}
$$

## Independent and Identically Distributed

Random variables, $X_{1}, X_{2}, \ldots, X_{n}$, are said to be independent and identically distributed (IID) if and only if

1. the random variables are independent and
2. they have the same probability distribution.

Discrete random variables, $X_{1}, X_{2}, \ldots, X_{n}$, are IID if and only if $p_{X_{1}, X_{2}, \ldots X_{n}}\left(x_{1}, x_{2}, \ldots, x_{n}\right)=\prod_{i=1}^{n} p_{X}\left(x_{i}\right)$, where $p_{X}$ denotes the shared PMF

Continuous random variables, $X_{1}, X_{2}, \ldots, X_{n}$, are IID if and only if $f_{X_{1}, X_{2}, \ldots X_{n}}\left(x_{1}, x_{2}, \ldots, x_{n}\right)=\prod_{i=1}^{n} f_{X}\left(x_{i}\right)$, where $f_{X}$ denotes the shared PDF

## Conditional Probability Mass Function (PMF)

The conditional PMF tells us the probability of one discrete random variable given our observation of another and is defined as:

$$
p_{X \mid Y}(x \mid y)=\frac{p_{X, Y}(x, y)}{p_{Y}(y)} \quad p_{Y \mid X}(y \mid x)=\frac{p_{Y, X}(y, x)}{p_{X}(x)}
$$

If random variables, $X$ and $Y$, are independent, then

$$
p_{X \mid Y}(x \mid y)=p_{X}(x) \quad \text { and } \quad p_{Y \mid X}(y \mid x)=p_{Y}(y) .
$$

## Conditional Probability Density Function (PDF)

The conditional PDF tells us the probability of one continuous random variable given our observation of another and is defined as:

$$
f_{X \mid Y}(x \mid y)=\frac{f_{X, Y}(x, y)}{f_{Y}(y)} \quad f_{Y \mid X}(y \mid x)=\frac{f_{Y, X}(y, x)}{f_{X}(x)}
$$

If random variables, $X$ and $Y$, are independent, then

$$
f_{X \mid Y}(x \mid y)=f_{X}(x) \quad \text { and } \quad f_{Y \mid X}(y \mid x)=f_{Y}(y) .
$$

## Maximum Likelihood Estimation (MLE)

1. K-Means Clustering
2. Probability Theory
a) Probability Fundamentals
b) Common Distributions
c) Multiple Random Variables
3. Maximum Likelihood Estimation (MLE)
4. Convex Optimization

## Maximum Likelihood Estimation (MLE)

What we ultimately want to know:
Underlying probability distribution of our population: $p_{\text {true }}(x)^{*}$

What we have available to us:
Samples from that distribution: $\mathcal{D}=\left\{x_{1}, \ldots, x_{N}\right\}$

What we can realistically determine:
Parameters of assumed distribution: w s.t. $p_{\text {assumed }}(x ; w) \approx p_{\text {true }}(x)$

* We always express the probability distribution as $p(x)$, but this could be a PMF for discrete random variables or a PDF for continuous random variables.


## What is the Assumed Distribution?

## Usually, one of the common distributions listed in the probability theory section of slides:

| Distribution | Parameters |  |
| :--- | :--- | :--- |
| Bernoulli | Bernoulli Binomial | mean (K) and variance ( $\sigma^{2}$ ) |
|  | Distribution | Probability of success |
| Binomial | Binomial Exponential | probability of success ( $\boldsymbol{p}$ ) |
|  | Gaussian |  |
| Exponential | Binomial | probability of success ( $P$ ) mumbex of trials ( $m$ ) ama probability of success ( $P$ ) |
|  | Exponential Caussian |  |
| Gaussian |  | mean (K) and variance ( $\sigma^{2}$ ) |

## How do we Estimate these Parameters?

Approach: Maximize the likelihood of our data under the assumed
probability distribution.
Likelihood: $\quad p(\mathcal{D} \mid w)=\prod_{n=1} p_{\text {assumed }}\left(x_{n} ; w\right)^{*}$

Optimization problem: $\widehat{w}=\arg \max _{w} p(\mathcal{D} \mid w)$

* This assumes that our data is independent and identically distributed
(IID)- a very common through rarely true assumption.


## Introduction to Convex Optimization

1. K-Means Clustering
2. Probability Theory
a) Probability Fundamentals
b) Common Distributions
c) Multiple Random Variables
3. Maximum Likelihood Estimation (MLE)
4. Convex Optimization

## Two Types of Problems

Minimizatio n $\min _{x} f_{1}(x)$ convex function

Maximizatio n $\max _{x} f_{2}(x)$ concave function

## Convex and Concave Functions

## Convex Functions

![](https://cdn.mathpix.com/cropped/2025_10_01_1900c9840f0af548fa40g-33.jpg?height=992&width=1604&top_left_y=586&top_left_x=21)

**Image Description:** The diagram is a plot illustrating a convex function \( f(x) \). The x-axis represents the variable \( x \), while the y-axis represents the function value \( f(x) \). It features points \( x_1 \) and \( x_2 \), with corresponding function values \( f(x_1) \) and \( f(x_2) \). A point \( z \) is shown on the curve, indicating a mixture of \( x_1 \) and \( x_2 \) defined by the equation \( z = \lambda x_1 + (1 - \lambda)x_2 \), where \( \lambda \) is a weight parameter ranging from 0 to 1.

$\mathbf{2}^{\text {nd }}$ order condition: $f^{\prime \prime}(x) \geq 0$

## Concave Functions

![](https://cdn.mathpix.com/cropped/2025_10_01_1900c9840f0af548fa40g-33.jpg?height=992&width=1612&top_left_y=586&top_left_x=1705)

**Image Description:** The image is a diagram representing a piecewise function \( f(x) \) plotted on Cartesian coordinates. The x-axis denotes the variable \( x \), while the y-axis represents the function value \( f(x) \). Two points on the curve, \( f(x_1) \) and \( f(x_2) \), are marked with green dots, indicating specific function values at \( x_1 \) and \( x_2 \). The interpolated point \( z \) is shown along the curve, with a relationship expressed as \( z = \lambda x_1 + (1 - \lambda)x_2 \) supporting the concept of linear interpolation between the two points.

$2^{\text {nd }}$ order condition: $f^{\prime \prime}(x) \leq 0$

## Common Convex／Concave Functions

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Name |  | $\log _{a}(x) \log _{-x} \log _{a}(x)$ | convex arced corrorts for all a．b $\in \mathbb{E}$ convex on $\mathbb{R}$ for a $\geq$ | ![](https://cdn.mathpix.com/cropped/2025_10_01_1900c9840f0af548fa40g-34.jpg?height=27&width=316&top_left_y=698&top_left_x=1433) | ![](https://cdn.mathpix.com/cropped/2025_10_01_1900c9840f0af548fa40g-34.jpg?height=32&width=162&top_left_y=612&top_left_x=1887) | $\Rightarrow$ ＝＝ <br> $\Leftrightarrow \infty .=1$ <br> 工要 ， ＝ ＝＝ $\square$ － <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_1900c9840f0af548fa40g-34.jpg?height=24&width=41&top_left_y=648&top_left_x=3029) － C－ － ＝＝ c $\Longrightarrow$ <br> $\Rightarrow$ <br> 三 <br> $\square$ |  |  |  |  |  |  |  |  |  |
|  |  | $a x+b a x$ | ![](https://cdn.mathpix.com/cropped/2025_10_01_1900c9840f0af548fa40g-34.jpg?height=23&width=397&top_left_y=788&top_left_x=1005) | Name | Eloxim | Properts |  |  |  |  |  |  |  |  |  |
|  | Quadratic | $a x^{2}+b x+c$ |  | Affine Quadratic | $x^{2}+1-5 x+c$ |  |  |  |  |  |  |  |  |  |  |
|  | Exponential | $e^{a x}$ | ![](https://cdn.mathpix.com/cropped/2025_10_01_1900c9840f0af548fa40g-34.jpg?height=24&width=280&top_left_y=872&top_left_x=1072) | Fixponential | eax | convex on IR\＆ <br> 101 <br> $>1$ ． <br> a <br> 당 <br> convex on $18 \Psi_{+-1}$ for a $\geqslant 1 . a \leqslant 0$ |  |  |  |  |  |  |  |  |  |
|  | Logarithmic | $\log _{\mathrm{a}}(x)$ | concave on $\mathbb{R}_{++}$for all $a \in \mathbb{R}$ | Iosarithmnic | los．（x） | concave |  |  |  |  |  |  |  |  |  |
|  | Entropy | $x \log _{a}(x)$ | concave on $\mathbb{R}_{++}$for all $\mathbf{a} \in \mathbb{R}$ | Entionoy | $x$ losa（ $x$ ） |  |  |  |  |  |  |  |  |  |  |
|  | Name | $a x+b$ | Property <br> Property | IName | Foxim | O <br> $\mathbb{O}$ <br> 娪 |  |  |  |  |  |  |  |  |  |
|  | Quadratic | $a x^{2}+b x+c$ | convex and concave on $\mathbb{R}$ for all $\mathrm{a}, \mathrm{b} \in \mathbb{R}$ | Quadratic | $x^{2}$ |  |  |  |  |  |  |  |  |  |  |
|  | Power | $x^{a}$ | convex on $\mathbb{R}_{++}$for $\mathrm{a} \geq 1, \mathrm{a} \leq 0$ <br> oncave on $\mathbb{R}_{++}$for $0 \leq \mathrm{a} \leq 1$ | Fourser | $x^{c x}$ |  |  |  |  |  |  |  |  |  |  |
|  | Exponential | $e^{a x}$ | convex on $\mathbb{R}$ for all $\mathrm{a} \in \mathbb{R}$ | Fixpuncential | exx | BC |  |  |  |  |  |  |  |  |  |
|  | Logarithmic | $x \log _{a}(x)$ | concave on $\mathbb{R}_{++}$for all $a \in \mathbb{R}$ | Finatroms | x 108 （x） |  |  |  |  |  |  |  |  |  |  |
| Exponential | Affine | $a x$ | $\_\_\_\_$ | ![](https://cdn.mathpix.com/cropped/2025_10_01_1900c9840f0af548fa40g-34.jpg?height=33&width=173&top_left_y=1325&top_left_x=1502) |  | == <br> ccccuscance 11:Es |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## (Unconstrained) Convex Optimization Problems

Minimizatio n $\min _{x} f_{1}(x)$
convex function

Maximizatio
n
$\max _{x} f_{2}(x)$
concave function

## Solving Convex Optimization Problems

1. Take the derivative of the objective w.r.t. optimization variable:

$$
\frac{\partial}{\partial x} f(\mathrm{x})
$$
2. Set the derivative equal to zero and solve for the optimal solution:
$$
\frac{\partial}{\partial x} f(\mathrm{x})=0 \rightarrow \hat{x}
$$

3. (Optionally) Compute the optimal value by plugging the optimal solution into the objective function:

$$
p^{*}=f(\hat{x})
$$

## Equivalent Optimization Problems

![](https://cdn.mathpix.com/cropped/2025_10_01_1900c9840f0af548fa40g-37.jpg?height=1043&width=2892&top_left_y=412&top_left_x=221)

**Image Description:** The image presents a conceptual equivalence in optimization. It consists of two sections: on the left, "Minimization" is represented by the equation \( \min_{x} f(x) \), indicating the search for the minimum of a convex function; on the right, "Maximization" is shown as \( \max_{x} -f(x) \), indicating the maximization of the negative of a convex function, which corresponds to a concave function. The presentation utilizes clear headings, directional arrows, and formatted mathematical expressions to convey the relationship between minimization and maximization in optimization theory.


These problems have the same optimal solution! (And the optimal value of one is the negation of the other.)

## Equivalent Optimization Problems

Monotonically increasing

$$
\phi\left(x_{2}\right) \geq \phi\left(x_{1}\right) \forall x_{2} \geq x_{1}
$$
functionies: Logarithmic growth, exponential growth,
positive affine, convex quadratic (for $x \geq 0$ )

Equivalent problems via monotone objective:

$$
\begin{aligned}
\min _{x} f(\mathrm{x}) & \equiv \min _{x} \phi(f(\mathrm{x})) \\
\max _{x} f(\mathrm{x}) & \equiv \max _{x} \phi(f(\mathrm{x}))
\end{aligned}
$$

These problems have the same optimal solution!

## Equivalent MLE Problems

Because the natural log is a monotonically increasing function,

$$
\widehat{w}=\arg \max _{w} p(\mathcal{D} \mid w) \equiv \widehat{w}=\arg \max _{w} \ln p(\mathcal{D} \mid w)
$$

We very often will solve this second optimization problem!

## Discussion Mini Lecture 3

## K-Means Clustering, Probability \& MLE

Contributors: Sara Pohland

## Additional Resources

1. K-Means Clustering
2. Deep Learning Foundations and Concepts - Chapter 15.1
3. Sara's notes on Machine Learning - Section 15.2
4. Probability Theory

- Deep Learning Foundations and Concepts - Chapter 2.1, 2.2, 3.1
- Sara's notes on Probability \& Random Processes - Sections 3-7

3. Maximum Likelihood Estimation

- Deep Learning Foundations and Concepts - Chapter 2.3
- Sara's notes on Machine Learning - Section 7

4. Convex Optimization

- Sara's notes on Convex Optimization - Sections 2-3

