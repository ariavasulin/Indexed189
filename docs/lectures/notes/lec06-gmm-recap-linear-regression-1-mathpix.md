---
course: CS 189
semester: Fall 2025
type: lecture
title: GMM Recap + Linear Regression (1)
source_type: slides
source_file: Lecture 06 -- GMM Recap + Linear Regression (1).pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Lecture 6

# GMM Recap + Linear Regression (1) 

Getting started with linear regression

EECS 189/289, Fall 2025 @ UC Berkeley
Joseph E. Gonzalez and Narges Norouzi

# III Join at slido.com <br> '1َيْL \#3689302 

## \section*{\section*{Roadmap <br> <br> <br> Roadmap <br> <br> <br> Roadmap}}

- Gaussian Mixture Modeß689302
- Linear Regression Formulation
- Basis Functions
- Vectorizing Calculations
- Error Function
- Error Function Minimization
- Geometric Interpretation
- Evaluation


## Gaussian Mixture Model

- Gaussian Mixture Modeß689302
- Linear Regression Formulation
- Basis Functions
- Vectorizing Calculations
- Error Function
- Error Function Minimization
- Geometric Interpretation
- Evaluation


## Gaussian Mixture Model (GMM)

The Gaussian mixture model defines the probability of sampling a $D$-dimensional vector $\mathbf{x}$ as a weighted combination of Gaussians:

- Example: $K=3$
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-05.jpg?height=677&width=3093&top_left_y=876&top_left_x=25)

**Image Description:** The image consists of two diagrams depicting probability density functions (PDFs) of Gaussian mixtures. The left diagram shows three Gaussian distributions, each with distinct means ($\mu_1$, $\mu_2$, $\mu_3$) and variances ($\sigma_1^2$, $\sigma_2^2$, $\sigma_3^2$), indicated by colored curves: red, green, and blue. Data points are represented as dots in matching colors. The right diagram illustrates the resultant mixture distribution, with all three components combined, showing the cumulative PDF. The x-axis represents the variable \(x\), and the y-axis indicates the probability density.



## Gaussian Mixture Model (GMM)

The Gaussian mixture model defines the probability of sampling a $D$-dimensional vector $\mathbf{x}$ as a weighted combination of Gaussians:

$$
p\left(\mathbf{x} \mid \pi, \mu_{1}, \Sigma_{1}, \ldots, \mu_{K}, \Sigma_{\mathrm{K}}\right)=\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\mathbf{x} \mid \mu_{k}, \Sigma_{k}\right)
$$
where $\sum_{k=1}^{K} \pi_{k}=1$ is a probability distribution over clusters.
- Example: $K=3$
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-06.jpg?height=592&width=1561&top_left_y=1203&top_left_x=1097)

**Image Description:** The image is a colored diagram depicting three overlapping probability density functions (PDFs) in a Gaussian mixture model. The x-axis represents the value of the variable, while the y-axis shows the probability density. Each curve is color-coded: blue for the first PDF, orange for the second, and green for the third. Each PDF is labeled with its mixing coefficient ($\pi$), mean ($\mu$), and variance ($\sigma^2$). Small colored points along the x-axis represent data points associated with each distribution.


## Demo

Gaussian Mixture Model
from sklearn.mixture import GaussianMixture
\# Create a Gaussian Mixture Model with 4 components
gmm = GaussianMixture(n_components=4, random_state=42, )
\# Fit the model to the data
gmm.fit(bikes[['Speed', 'Length']])
\# Get the cluster labels
bikes['scikit gmm'] = gmm.predict(bikes[['Speed', 'Length']]).astype(str)

3689302
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-07.jpg?height=1319&width=1532&top_left_y=489&top_left_x=1445)

**Image Description:** The diagram is a two-dimensional contour plot displaying four distinct clusters represented by ellipsoidal shapes. The x-axis ranges from approximately 5 to 20, while the y-axis ranges from 0 to 30. Each cluster is filled with color gradients indicating density, with colors ranging from green to red. A color bar on the right illustrates the density values, with lower densities in light hues and higher densities in darker hues. The plot visually represents data distribution and clustering in a feature space, useful for analyzing unsupervised learning outcomes.


## The GMM is a Latent Variable Model

'Ne introduce a latent (unobserved) assignment variable $z$ for each $\mathbf{x}$.

$$
p(z)=\pi_{z}
$$

We then define the likelihood of $\mathbf{x}$ given $z$ as:

$$
p(\mathbf{x} \mid z)=\mathcal{N}\left(\mathbf{x} \mid \mu_{z}, \Sigma_{z}\right)
$$

The joint probability of the $\mathbf{x}$ and $z$ is then:

$$
p(\mathbf{x}, z)=p(\mathbf{x} \mid z) p(z)=\mathcal{N}\left(\mathbf{x} \mid \mu_{z}, \Sigma_{z}\right) \pi_{z}
$$

If we marginalize over $\boldsymbol{z}$ we obtain the GMM:

$$
p(\mathbf{x})=\sum_{k=1}^{K} p(\mathbf{x}, z=k)=\sum_{k=1}^{K} \mathcal{N}\left(\mathbf{x} \mid \mu_{k}, \Sigma_{k}\right) \pi_{k}
$$

## The GMM is a Generative Model

A generative model is a model that can generate the data.

- The GMM is a generative model.

Recall, the joint distribution has the following factorization:

$$
p(\mathbf{x}, z)=p(z) p(\mathbf{x} \mid z)
$$

We can draw a graph where the parents of a variable are the variables in its conditional distribution.

- This is a graphical model and is used to represent the
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-09.jpg?height=456&width=175&top_left_y=786&top_left_x=2900)

**Image Description:** The image depicts a simple diagram featuring two circles. The upper circle is labeled "z," and the lower circle is labeled "x." A blue arrow points downward from the "z" circle to the "x" circle, indicating a directional relationship or flow from "z" to "x." The circles are colored yellow with a thick dark outline, enhancing visibility. This diagram likely represents a conceptual or causal relationship in a theoretical framework.

conditional structure of a probability model.
We can sample data from a graphical model using ancestor sampling.
- Sample all the variables that have no parents: $p(z)$
- Continue sampling variables after all their parents have been sampled: $p(\mathbf{x} \mid z)$

```
# Ancestor Sampling to create a synthetic dataset
np.random.seed(42)
N = 100
mu = np.array([-1, 2, 5])
pi = np.array([0.2, 0.5, 0.3])
Sigma = np.array([0.2, 0.5, .1])
0.0s
```


## Demo

Sampling from a GMM

```
z = np.random.choice(len(mu), size=N, p=pi)
x = np.random.normal(mu[z], np.sqrt(Sigma[z]))
0.0s
```

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-10.jpg?height=260&width=235&top_left_y=833&top_left_x=918)

**Image Description:** The image appears to be a screenshot from a programming or editing environment, showing a log output section. It features a vertical line indicating a response time or duration measurement, labeled "0.0s," likely representing a timing metric for a function or operation. The presence of a check mark suggests successful execution. There are no visible axes or diagrams; instead, it is a demonstration of logging output relevant to code performance evaluation.


Synthetic Dataset from GMM (Log Likelihood: 181.82)
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-10.jpg?height=745&width=1779&top_left_y=1012&top_left_x=1224)

**Image Description:** The image is a distribution plot consisting of three distinct kernel density estimates (KDEs) represented by colored curves: blue, red, and green. The x-axis is labeled "x" and ranges approximately from -4 to 8, while the y-axis is labeled "y" and indicates the density values. Each color corresponds to a set of data points (blue dots for one distribution, red dots for another, and green dots for the third) scattered along the x-axis, illustrating the underlying probability distributions of the data sets. The curves visually summarize the density of values in each dataset.


## Latent Variable Posteriors

'Ne are interested in modeling the distribution (the uncertainty) over the cluster assignment $z$.

- Prior: $p(z)=\pi_{z}$
- Likelihood: $p(\mathbf{x} \mid z)=\mathcal{N}\left(\mathbf{x} \mid \mu_{z}, \Sigma_{z}\right)$

We can use Bayes rule to compute the posterior:

$$
p\left(z \mid \mathbf{x}, \pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}\right)=\frac{p(\mathbf{x} \mid z) p(z)}{\sum_{k=1}^{K} p(\mathbf{x} \mid z=k) p(z=k)}=\frac{\mathcal{N}\left(\mathbf{x} \mid \mu_{z}, \Sigma_{z}\right) \pi_{z}}{\sum_{k=1}^{K} \mathcal{N}\left(\mathbf{x} \mid \mu_{k}, \Sigma_{k}\right) \pi_{k}}
$$

If we could estimate $\pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}$ from the data we could compute this distribution over cluster assignments.

## Estimating the GMM Parameters

We could try to use the method of maximum likelihood:

$$
\ln \left(p\left(\left\{\mathbf{x}_{n}\right\}_{n=1}^{N} \mid \pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}\right)\right)=\sum_{n=1}^{N} \ln \left(\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{k}, \Sigma_{k}\right)\right)
$$
- however, the summation inside of the log couples the parameters.

But, if we extend the model to include the latent variable $z$
$\ln \left(p\left(\left\{\mathbf{x}_{n}, z_{n}\right\}_{n=1}^{N} \mid \pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}\right)\right)=\sum_{n=1}^{N}\left(\ln \left(\pi_{z_{n}}\right)+\ln \left(\mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{z_{n}}, \Sigma_{z_{n}}\right)\right)\right)$

- We can optimize the parameters directly (sum of log terms).
- But we don't know the values of $\left\{z_{n}\right\}_{n=1}^{N}$


## Quick Recap

If we knew the model parameters, we could easily compute the cluster assignments.
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-13.jpg?height=656&width=2714&top_left_y=625&top_left_x=344)

**Image Description:** The image is a diagram illustrating a process loop involving model parameters and cluster assignments, commonly used in clustering algorithms like K-means. It features two labeled sections: "Model Parameters" on the left and "Cluster Assignments" on the right. The loop is represented by arrows pointing in a circular direction, indicating an iterative feedback mechanism. Below "Model Parameters," there is a mathematical representation labeled with parameters $\{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$, and below "Cluster Assignments," it shows $\{z_n\}_{n=1}^N$. This structure highlights the relationship between parameters and their assignments in clustering.


If we knew the cluster assignments, we could easily estimate the model parameters.

## How can we solve this cyclic dependency?

## The EM Algorithm

The Expectation-Maximization (EM) algorithm is a standard technique for fitting latent variable models like the Gaussian Mixture Model.

1. Initialize the model parameters $\theta$.
2. Iterate until convergence.

- E-Step: Compute expected log-likelihood.

$$
Q\left(\theta^{\prime}\right)=\sum_{Z} \ln \left(p\left(X, Z \mid \theta^{\prime}\right)\right) p(Z \mid X, \theta)
$$
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-14.jpg?height=243&width=647&top_left_y=748&top_left_x=1960)

**Image Description:** The image is a speech bubble graphic containing the text "Easy to optimize joint probability." The background is a dark blue color, and the text is prominently displayed in white, emphasizing clarity and focus on the concept of joint probability in statistical or probabilistic contexts. The design is simplistic and aimed at engaging viewers in a lecture setting on optimization or probability theory.


Current distribution
over the latent Z

- M-Step: Maximize the expected log-likelihood.

$$
\theta=\arg \max _{\theta^{\prime}} Q\left(\theta^{\prime}\right)
$$

Updates distribution over Z.

## The EM Algorithm: E-step

E-Step: We don't know the value of $\left\{z_{n}\right\}_{n=1}^{N}$ but we can compute

$$
\gamma_{n k}=p\left(z_{n}=k \mid \mathbf{x}_{n}, \pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}\right)=\frac{p\left(\mathbf{x}_{n} \mid z_{n}\right) p\left(z_{n}\right)}{\sum_{k^{\prime}=1}^{K} p\left(\mathbf{x}_{n} \mid k^{\prime}\right) p\left(k^{\prime}\right)}=\frac{\mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{z_{n}}, \Sigma_{z_{n}}\right) \pi_{z_{n}}}{\sum_{k^{\prime}=1}^{K} \mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{k^{\prime}}, \Sigma_{k^{\prime}}\right) \pi_{k^{\prime}}}
$$
- $\gamma_{n k}$ are "soft" assignments to each class.

Using the $\gamma_{n k}$ we can compute the expected log likelihood:
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-15.jpg?height=563&width=431&top_left_y=824&top_left_x=2840)

**Image Description:** The image depicts a colored rectangle, labeled with the letters "N" and "K" on the left and top edges, respectively. Inside the rectangle, a stylized, bold Greek letter "γ" (gamma) is centered in a contrasting color (yellow or gold). The rectangle's blue background may represent a specific set or concept, while the letters likely indicate variables or parameters relevant to the topic being discussed. The positioning of the labels suggests potential orientations or relations within a scientific or mathematical context.


$$
\begin{aligned}
Q\left(\pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}\right) & =\sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k} \ln \left(p\left(\mathbf{x}_{n}, z_{n}=k \mid \pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}\right)\right) \\
& =\sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k}\left(\ln \left(\pi_{k}\right)+\ln \left(\mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{k}, \Sigma_{k}\right)\right)\right)
\end{aligned}
$$

## The EM Algorithm: M-Step ( $\pi_{k}$ )

M-Step: Compute the parameters that maximize $Q$ :

$$
Q\left(\pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}\right)=\sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k}\left(\ln \left(\pi_{k}\right)+\ln \left(\mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{k}, \Sigma_{k}\right)\right)\right)
$$

Lagrangian for the
Optimizing with respect to $\pi_{k}$ :

$$
\begin{array}{ccc}
\frac{\partial}{\partial \pi_{k}} \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k}\left(\ln \left(\pi_{k}\right)+\ln \left(\mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{k}, \Sigma_{k}\right)\right)\right)+\lambda\left(\sum_{k=1}^{K} \pi_{k}-1\right)=\sum_{n=1}^{N} \frac{\gamma_{n k}}{\pi_{k}}+\lambda=0 & \\
1=\sum_{k=1}^{K} \pi_{k}=-\frac{1}{\lambda} \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k}=-\frac{1}{\lambda} \sum_{n=1}^{N} 1 & \pi_{k}=-\frac{1}{\lambda} \sum_{n=1}^{N} \gamma_{n k} & N_{k}=\sum_{n=1}^{N} \gamma_{n k} \\
=-\frac{N}{\lambda} \Rightarrow \lambda=-N & =\frac{1}{N} \sum_{n=1}^{N} \gamma_{n k}=\frac{N_{k}}{N} &
\end{array}
$$

## The EM Algorithm: M-Step ( $\mu_{k}$ )

M-Step: Compute the parameters that maximize $Q$ :

$$
Q\left(\pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}\right)=\sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k}\left(\ln \left(\pi_{k}\right)+\ln \left(\mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{k}, \Sigma_{k}\right)\right)\right)
$$

Optimizing with respect to $\mu_{k}$ :

$$
\begin{gathered}
\frac{\partial}{\partial \mu_{k}} \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k}\left(\ln \left(\pi_{k}\right)+\ln \left(\mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{k}, \Sigma_{k}\right)\right)\right)=\sum_{n=1}^{N} \gamma_{n k} \Sigma_{k}^{-1}\left(\mathbf{x}_{n}-\mu_{k}\right)=0 \\
\mu_{k}=\frac{1}{\sum_{n=1}^{N} \gamma_{n k}} \sum_{n=1}^{N} \gamma_{n k} \mathbf{x}_{n}=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{n k} \mathbf{x}_{n}
\end{gathered}
$$

## The EM Algorithm: M-Step $\left(\Sigma_{k}\right)$

M-Step: Compute the parameters that maximize $Q$ :

$$
Q\left(\pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}\right)=\sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k}\left(\ln \left(\pi_{k}\right)+\ln \left(\mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{k}, \Sigma_{k}\right)\right)\right)
$$

Optimizing with respect to $\Sigma_{k}$ requires matrix calculus (out of scope):

$$
\Sigma_{k}=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{n k}\left(\mathbf{x}_{n}-\mu_{k}\right)\left(\mathbf{x}_{n}-\mu_{k}\right)^{T}
$$

Summarizing the other parameters:

$$
N_{k}=\sum_{n=1}^{N} \gamma_{n k}, \quad \pi_{k}=\frac{N_{k}}{N}, \quad \mu_{k}=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{n k} \mathbf{x}_{n}
$$

## The EM Algorithm for GMMs

2. Initialize the model parameters $\pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}$ (often using k -means)
3. Iterate until convergence

- E-Step: compute expected log-likelihood

$$
\gamma_{n k}=\frac{\mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{z_{n}}, \Sigma_{z_{n}}\right) \pi_{z_{n}}}{\sum_{k^{\prime}=1}^{K} \mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{k^{\prime}}, \Sigma_{k^{\prime}}\right) \pi_{k^{\prime}}}
$$

- M-Step: maximize the expected log-likelihood

$$
\begin{gathered}
N_{k}=\sum_{n=1}^{N} \gamma_{n k}, \quad \pi_{k}=\frac{N_{k}}{N}, \quad \mu_{k}=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{n k} \mathbf{x}_{n} \\
\Sigma_{k}=\frac{1}{N_{k}} \sum_{n=1}^{N} \gamma_{n k}\left(\mathbf{x}_{n}-\mu_{k}\right)\left(\mathbf{x}_{n}-\mu_{k}\right)^{T}
\end{gathered}
$$

## Demo

Implementing EM for GMMs
from sklearn.mixture import GaussianMixture
\# Create a Gaussian Mixture Model with 4 components
gmm = GaussianMixture(n_components=4, random_state=42, )
\# Fit the model to the data
gmm.fit(bikes[['Speed', 'Length']])
\# Get the cluster labels
bikes['scikit gmm'] = gmm.predict(bikes[['Speed', 'Length']]).astype(str)
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-20.jpg?height=1319&width=1532&top_left_y=489&top_left_x=1445)

**Image Description:** The image is a contour plot displaying four distinct clusters, represented as ellipses. The x-axis ranges from 5 to 20, while the y-axis extends from 0 to 30. Each colored point indicates data distribution across these clusters, with color intensity represented by a gradient on the right, indicating density values from approximately 0.005 to 0.035. The color coding distinguishes between different clusters, labeled by numbers (0, 1, 3). The background is a gradient from dark purple to lighter shades, signifying lower to higher densities of points, respectively.


## Linear Regression

- Gaussian Mixture Modek689302
- Linear Regression
- Basis Functions
- Vectorizing Calculations
- Error Function
- Error Function Minimization
- Geometric Interpretation
- Evaluation


## Linear Regression Outline

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-22.jpg?height=1485&width=3024&top_left_y=331&top_left_x=187)

**Image Description:** The image is a conceptual diagram illustrating the components of a machine learning workflow, divided into four quadrants: "Learning Problem" (P), "Model Design" (M), "Predict & Evaluate" (E), and "Optimization" (O). Each quadrant contains relevant icons or symbols representing key concepts, such as regression analysis and evaluation metrics. Arrows connect the quadrants, indicating the iterative nature of the workflow. The "Model Design" section includes the formula for Linear Regression: 

$$ w' = \arg\min E(w) $$ 

along with contextual descriptions for each quadrant, enhancing understanding of the overall process.


## Learning Problem

## LEARNING PROBLEM

Supervised learning of scalar target values
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-23.jpg?height=737&width=881&top_left_y=327&top_left_x=1029)

**Image Description:** The image is a stylized diagram featuring a blue, abstract shape with a triangular arrow pointing upwards to the right. The arrow contains a yellow letter "L." In the lower section, there is a circular icon depicting a bar graph with a magnifying glass overlay, symbolizing data analysis or research. The overall design conveys themes of growth and analytics, integrating both geometric elements and graphical representation of data trends. The diagram does not include specific axes or labels.


## Regression

Estimating relationship between x and $t$.

- $t$ is a quantitative value
- We will soon see x can be almost anything ...
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-24.jpg?height=753&width=2272&top_left_y=1029&top_left_x=514)

**Image Description:** The image presents a diagram illustrating a model within a mathematical framework. The diagram includes a labeled green region representing the "Domain," alongside a blue arrow pointing to the notation $y(\tilde{x}, \tilde{w})$, suggesting a mapping from the domain to a corresponding model output. Additionally, there is a horizontal line with labeled endpoints at $-\infty$ and $\infty$, likely representing a range or output space where the variable $t$ is defined. The overall structure visually conveys the relationship between the domain and the model output.

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-24.jpg?height=797&width=1090&top_left_y=352&top_left_x=2232)

**Image Description:** The image depicts a scatter plot with a linear regression line. The x-axis is labeled "X" and represents the independent variable, while the y-axis is labeled "t" and represents the dependent variable. The data points are represented as black dots, showing a positive correlation between the variables. The regression line, drawn in black, indicates the general trend of the data points. The axes intersect at the origin, suggesting a typical Cartesian coordinate system.



## Model Design

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-25.jpg?height=954&width=2948&top_left_y=327&top_left_x=191)

**Image Description:** The image presents a conceptual diagram illustrating the relationship between the learning problem and model design in supervised learning. It features two main sections labeled "LEARNING PROBLEM" and "MODEL DESIGN." The left section contains an icon representing data, while the right section features an icon symbolizing model architecture. An arrow flows from the learning problem to model design, emphasizing the workflow. Below the model design section, a linear regression formula is provided: $$ y(\mathbf{x}, \mathbf{w}) = \mathbf{x}^T \mathbf{w} $$ indicating the relationship between input features and target values.


## Supervised Linear Regression

- Goal: Predict one or more continuous target variable $t$ given the vector $\overrightarrow{\mathrm{x}} \in \mathbb{R}^{D}$ of input variables.

$$
\left\{\left(x_{n}\right)\right\}_{n=1}^{N} \stackrel{\vec{w}}{\longrightarrow}\left\{\left(t_{n}\right)\right\}_{n=1}^{N}
$$
- We formulate a function $y(\overrightarrow{\mathrm{x}}, \vec{w})$ whose values for new input $\overrightarrow{\mathrm{x}}$ constitute the predictions for the corresponding values of $t$.
- $\vec{w}$ is a vector of parameters that can be learned from the training data.

Visualizing Dataset and a Linear Regression Function
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-26.jpg?height=882&width=1145&top_left_y=552&top_left_x=2079)

**Image Description:** The image is a scatter plot depicting a dataset of points indicating the relationship between two variables: the horizontal axis represents the input variable \( x \) (ranging from 0 to 10), while the vertical axis represents the output variable \( t \) or \( y \). Data points are represented as blue dots, showing a linear trend. A red dashed line represents the linear regression prediction \( y(x) \), illustrating the model's fitted relationship with the data. The title indicates the focus on normalizing the dataset and the linear regression function.


## The Simplest Linear Regression Model

The simplest model is of the following form:
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-27.jpg?height=1269&width=3211&top_left_y=518&top_left_x=106)

**Image Description:** The image consists of a diagram illustrating a linear regression function. The axes are labeled as "x (input)" and "output (y(x))", showing a scatter plot with data points represented as blue dots. A red dashed line represents the regression line, with its intercept and slope visually indicated. The equation \( y(\mathbf{x}, \mathbf{w}) = w_0 + w_1 x_1 + \ldots + w_D x_D \) describes the predicted output as a linear combination of parameters \( w_0, w_1, \ldots, w_D \), showcasing the concept of linear relationships in regression analysis.


## The Simplest Linear Regression Model

- The simplest model is of the following form:
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-28.jpg?height=1200&width=3067&top_left_y=514&top_left_x=106)

**Image Description:** The slide contains a 3D scatter plot illustrating a hyperplane in a high-dimensional space \( \mathbb{R}^{D+1} \). The axes represent variables \( x_1, x_2, \) and \( x_3 \), with values plotted in three-dimensional coordinates. A shaded region depicts the area representing predictions made by a linear combination of parameters \( w_0, w_1, \ldots, w_D \), highlighting how the predicted values form a hyperplane. An equation is presented as \( y(\mathbf{x}, \mathbf{w}) = w_0 + w_1 x_1 + \ldots + w_D x_D \). The notation indicates each \( x_d \) is scaled by its corresponding parameter \( w_d \).


Interactive link

## :

Which of the following is a linear regression model?

## Basis Functions

- Gaussian Mixture Modeß689302
- Linear Regression
- Basis Functions
- Vectorizing Calculations
- Error Function
- Error Function Minimization
- Geometric Interpretation
- Evaluation


## Linear Functions From Slido

These are all linear models with different basis functions.

We will now see what basis functions are...
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-31.jpg?height=1469&width=1770&top_left_y=318&top_left_x=1309)

**Image Description:** The image features a 2x2 grid of plots, each illustrating different regression techniques. 

1. **Top Left: Regular Linear Regression** - Displays a scatter plot of data points labeled "Data," with a linear fit shown as a red line. The axes are labeled 'X' (horizontal) and 'Y' (vertical).
   
2. **Top Right: Polynomial Basis Function** - Features a scatter plot with quadratic fit, labeled similarly, showing a more complex curve.

3. **Bottom Left: Gaussian Basis Function** - Shows data points with a smooth Gaussian fit, illustrating non-linear regression.

4. **Bottom Right: Fourier Basis Function** - Displays a scatter plot with a Fourier fit, capturing periodic behavior in the data.

Each plot effectively visualizes different regression methods.


## What Does It Mean To Be a Linear Model?

In what sense are the previous plots linearly modeled?

$$
y(\overrightarrow{\mathrm{x}}, \vec{w})=w_{0}+\sum_{j=1}^{M-1} w_{j} \phi_{j}(\mathrm{x})
$$

Are linear models linear in the

1. Features?
2. Parameters?
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-33.jpg?height=481&width=426&top_left_y=561&top_left_x=229)

**Image Description:** This image appears to be a simple icon representing a checklist. It features a rectangular outline with rounded corners. Within the rectangle, there is a checkmark symbol indicating completion or approval, and a circular button with a solid fill suggesting an option selection. The background is light blue, and the overall design is minimalistic and clean, suitable for visual communication in academic or organizational contexts concerning task management or verification processes.


## Are linear models linear in the

## What Does It Mean To Be a Linear Model?

In what sense are the previous plots linearly modeled?

$$
y(\overrightarrow{\mathrm{x}}, \vec{w})=w_{0}+\sum_{j=1}^{M-1} \widehat{w_{j} \phi_{j}(\mathrm{x})}
$$

Linear in the
Parameters

Are linear models linear in the
Feature Functions

1. Features?
2. Parameters?

## Basis Functions

- We can extend the class of models by considering linear combinations of fixed nonlinear functions of the input variables, of the form:

$$
y(\overrightarrow{\mathrm{x}}, \vec{w})=w_{0}+\sum_{j=1}^{M-1} w_{j} \phi_{j}(\mathrm{x})
$$
where $\phi_{j}(\mathrm{x})$ are known as basis functions.

## More on Basis Function

$\checkmark$ Polynomial basis: $\phi_{j}(x)=x^{j}$

- Radial Basis Functions (RBF) such as Gaussian: $\phi_{j}(x)=e^{-\frac{\left(x-\mu_{j}\right)^{2}}{2 \sigma^{2}}}$
- Sigmoidal basis: $\phi_{j}(x)=\sigma\left(\frac{x-\mu_{j}}{\sigma}\right)$ where $\sigma(a)=\frac{1}{1+e^{-a}}$
- $\tanh (a)=2 \sigma(2 a)-1$
- Sinusoidal basis: $\phi_{j}(x)=\sin x_{j}$ or $\phi_{j}(x)=\cos x_{j}$


## Basis Functions as Features

Why did we care about basis functions before deep learning?

Fixed feature extraction was the norm before deep learning: raw input x $\rightarrow$ hand-designed features $\phi_{j}(\mathrm{x})$.
Goal: Pick expressive features so a simple linear model could succeed.
Popular choices: polynomials, radial-basis functions, Fourier terms, kernels, etc.
Heavy manual engineering.
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-37.jpg?height=1264&width=1914&top_left_y=348&top_left_x=1331)

**Image Description:** The image is a graph titled "Representative Basis Functions from Different Families." It features three distinct curves representing different mathematical functions plotted against the x-axis:

1. A blue solid line labeled "Polynomial" depicting \( \phi_{\text{poly}}(x) = x^3 \).
2. A green dashed line labeled "Gaussian RBF" showing \( \phi_{\text{RBF}}(x) = e^{-x^2} \).
3. An orange dashed line labeled "Fourier" representing \( \phi_{\text{Fourier}}(x) = \sin(3\pi x) \).

The x-axis is labeled "X," while the y-axis shows the function values \( \phi(x) \). The range of the y-values includes both positive and negative values, highlighting the diversity in the behavior of the functions.


## How Would Basis

3689302 Functions Improve Predictions?

Using polynomial basis functions of degree 5, we redefined the linear regression equation as:

$$
y(\overrightarrow{\mathrm{x}}, \vec{w})=\sum_{j=0}^{5} w_{j} \mathrm{x}^{j}
$$

Looking at the Mean Squared Error (MSE) between targets and predictions, the polynomial fit has a better performance.
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-38.jpg?height=1103&width=2089&top_left_y=331&top_left_x=1190)

**Image Description:** The image is a diagram comparing linear and polynomial regression. It features a scatter plot with data points represented by blue dots, plotted against the x-axis and y-axis labeled "X" and "Y," respectively. A solid orange line indicates the linear fit, while a dashed green curve represents the polynomial fit (degree=5). The mean squared error (MSE) values for both fits are noted: MSE for linear fit is 0.511 and for polynomial fit is 0.013. The polynomial fit closely follows the data's variation, unlike the linear fit.


$$
\begin{aligned}
& y(x)=0.000+4.355 x^{1}-0.203 x^{2}-14.426 x^{3}+0.164 x^{4}+9.368 x^{5} \\
& y(x)=0.120-0.320 x
\end{aligned}
$$

## Vectorizing Calculations

- Gaussian Mixture Modeß689302
- Linear Regression
- Basis Functions
- Vectorizing Calculations
- Error Function
- Error Function Minimization
- Geometric Interpretation
- Evaluation


## Vectorizing Calculations

- Rewriting the equation for linear regression:

$$
y(\overrightarrow{\mathrm{x}}, \vec{w})=w_{0}+w_{1} x_{1}+\ldots+w_{D} x_{D}=w_{0}+\sum_{d=1}^{D} w_{d} x_{d}
$$
- If we assume that there is an $x_{0}=1$, then the $t_{D}$ vo terms above can
$$
y(\overrightarrow{\mathrm{x}}, \vec{w})=w_{0}+w_{1} x_{1}+\ldots+w_{D} x_{D}=\sum_{d=0}^{D} w_{d} x_{d}
$$

where $\overrightarrow{\mathrm{x}}=\left(x_{0}, x_{1}, x_{2}, \ldots, x_{D}\right)^{T}$ and $\vec{w}=\left(w_{0}, w_{1}, w_{2}, \ldots, w_{D}\right)^{T}$

Augmented input with $x_{0} \quad x_{0}=1$

## Vectorizing Calculations

$$
y(\overrightarrow{\mathrm{x}}, \vec{w})=w_{0}+w_{1} x_{1}+\ldots+w_{D} x_{D}=\sum_{d=0}^{D} w_{d} x_{d}
$$
where $\overrightarrow{\mathrm{x}}=\left(x_{0}, x_{1}, x_{2}, \ldots, x_{D}\right)^{T}$ and $\vec{w}=\left(w_{0}, w_{1}, w_{2}, \ldots, w_{D}\right)^{T}$
\$\$\begin{aligned}

y(\overrightarrow{\mathrm{x}}, \vec{w}) \& =\vec{w} \cdot \overrightarrow{\mathrm{x}} <br>
\& =\vec{w}^{T} \overrightarrow{\mathrm{x}}=\overrightarrow{\mathrm{x}}^{T} \vec{w}

\end{aligned}\$\$

## Matrix Notation

Assuming we have $n=1, \ldots, N$ samples in our dataset, we can write:
$\left\{\begin{array}{c}y\left(\overrightarrow{\mathrm{x}_{1}}, \vec{w}\right)=w_{0}+w_{1} x_{11}+w_{2} x_{12}+\ldots+w_{D} x_{1 D} \\ y\left(\overrightarrow{\mathrm{x}_{2}}, \vec{w}\right)=w_{0}+w_{1} x_{21}+w_{2} x_{22}+\ldots+w_{D} x_{2 D} \\ \ldots \\ y\left(\overrightarrow{\mathrm{x}_{\mathrm{N}}}, \vec{w}\right)=w_{0}+w_{1} x_{N 1}+w_{2} x_{N 2}+\ldots+w_{D} x_{N D}\end{array}\right.$
$\left\{\begin{aligned} y\left(\overrightarrow{\mathrm{x}_{1}}, \vec{w}\right)=\overrightarrow{\mathbf{x}_{1}^{T}} \vec{w} & \text { where } \overrightarrow{\mathbf{x}_{1}^{T}}=\left[1, x_{11}, x_{12}, \ldots, x_{1 D}\right] \\ y\left(\overrightarrow{\mathrm{x}_{2}}, \vec{w}\right)=\overrightarrow{\mathbf{x}_{2}^{T}} \vec{w} & \text { where } \overrightarrow{\mathbf{x}_{2}^{T}}=\left[1, x_{21}, x_{22}, \ldots, x_{2 D}\right] \\ \cdots & \\ y\left(\overrightarrow{\mathrm{x}_{\mathrm{N}}}, \vec{w}\right)=\overrightarrow{\mathbf{x}_{N}^{T}} \vec{w} & \text { where } \overrightarrow{\mathbf{x}_{N}^{T}}=\left[1, x_{N 1}, x_{N 2}, \ldots, x_{N D}\right]\end{aligned}\right.$

## Matrix Notation

$$
\begin{aligned}
& \left\{\begin{array}{ll}
y\left(\overrightarrow{x_{1}}, \vec{w}\right)=\overrightarrow{x_{1}^{T}} \vec{w} & \text { where } \overrightarrow{x_{1}^{T}}=\left[1, x_{11}, x_{12}, \ldots, x_{1 D}\right] \\
y\left(\overrightarrow{x_{2}}, \vec{w}\right)=\overrightarrow{x_{2}^{T}} \vec{w} & \text { where } \overrightarrow{x_{2}^{T}}=\left[1, x_{21}, x_{22}, \ldots, x_{2 D}\right] \\
\ldots\left(\overrightarrow{x_{N}}, \vec{w}\right)=\overrightarrow{x_{N}^{T}} \vec{w} & \text { where } \overrightarrow{x_{N}^{T}}=\left[1, x_{N 1}, x_{N 2}, \ldots, x_{N D}\right]
\end{array} \quad \mathbb{Y}=\left\{y\left(\overrightarrow{x_{1}}, \vec{w}\right), y\left(\overrightarrow{x_{2}}, \vec{w}\right), \ldots, y\left(\overrightarrow{x_{N}}, \vec{w}\right)\right\} \in \mathbb{R}^{N}\right. \\
& \left\{\begin{array}{l}
y\left(\overrightarrow{x_{1}}, \vec{w}\right)=\left[1, x_{11}, x_{12}, \ldots, x_{1 D}\right] \vec{w} \\
y\left(\overrightarrow{x_{2}}, \vec{w}\right)=\left[1, x_{21}, x_{22}, \ldots, x_{2 D}\right] \vec{w} \\
\cdots \\
y\left(\overrightarrow{x_{N}}, \vec{w}\right)=\underbrace{\left[1, x_{N 1}, x_{N 2}, \ldots, x_{N D}\right]}_{\mathbb{Y}(\mathbb{X}, \vec{w})=\mathbb{X} \longrightarrow \vec{w}}] \\
\mathbb{X}=\left\{\overrightarrow{x_{1}}, \overrightarrow{x_{2}}, \ldots, \overrightarrow{x_{N}}\right\} \in \mathbb{R}^{N \times(D+1)} \\
\text { Design matrix }
\end{array}\right.
\end{aligned}
$$

## Error Function

- Gaussian Mixture Modeß689302
- Linear Regression
- Basis Functions
- Vectorizing Calculations
- Error Function
- Error Function Minimization
- Geometric Interpretation
- Evaluation


## Optimization

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-45.jpg?height=1476&width=3024&top_left_y=340&top_left_x=187)

**Image Description:** The image is a diagram illustrating the components of a machine learning framework. It consists of three sections labeled "LEARNING PROBLEM," "MODEL DESIGN," and "OPTIMIZATION," each connected by arrows. The "LEARNING PROBLEM" section features icons representing data and targets. The "MODEL DESIGN" section includes a description of linear regression with the equation \( y(\mathbf{x}) = \mathbf{x}^T \mathbf{w} \). The "OPTIMIZATION" section features the equation:

$$
E(w) = \frac{1}{N} \sum_{n=1}^{N} (t_n - y(\mathbf{x}_n, w))^2 \quad w^* = \arg\min E(w)
$$

The layout is visually clear with distinct color coding for each section.


## Error Function

- How to calculate the weights $\vec{w}$ ?
- We do this by minimizing an error function that measures the misfit between the function $\mathbb{Y}(\mathbb{X}, \vec{w})$ and targets, for any given value of $\vec{w}$, and the training data $\mathbb{X}$.
- One simple error function: Sum of the squares of the differences between targets and the predictions.
- Non-negative quantity.

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}
$$
- Only 0 if all predictions are equal to targets.

The factor of $\frac{1}{2}$ is included for later convenience.

## Error Function Visualization

Residuals are defined as $\mathrm{e}_{\mathrm{n}}=\left(t_{n}-y\left(x_{n}, w\right)\right)$ and they can be + or -

The error function $E(w)$ combines the squares of the residuals to measure the goodness of the fit.

Linear Regression plot with Residuals
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-47.jpg?height=1319&width=1981&top_left_y=323&top_left_x=1248)

**Image Description:** The image is a scatter plot depicting data points and a fitted line representing a relationship between the variables \( x \) and \( t/y(x, w) \). The x-axis ranges from 0 to 10, while the y-axis displays values from approximately 7 to 23. Data points are marked with blue circles, with vertical dashed lines indicating deviations from the fitted line. The fitted line, shown in red, illustrates the trend of the dataset. Annotations next to data points indicate numerical residuals.


## Error Function Minimization

- Gaussian Mixture Modeß689302
- Linear Regression
- Basis Functions
- Vectorizing Calculations
- Error Function
- Error Function Minimization
- Geometric Interpretation
- Evaluation


## Error Function Minimization

- $E(w)$ is a quadratic function.

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}
$$

Therefore, $\frac{\partial E(w)}{\partial \mathrm{w}}$ is a linear function of $w$ and hence solving $\frac{\partial E(w)}{\partial \mathrm{w}}=0$ has one and only one solution. Let's now derive this solution:

$$
=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-w_{0}-\sum_{j=1}^{D} w_{j} x_{n j}\right)^{2} \underset{\substack{\text { Finding the } \\ \text { optimum solution }}}{\longrightarrow} \frac{\partial E(w)}{\partial \mathrm{w}}=0
$$

## Error Function Minimization

$$
\left[\begin{array}{rl}
\frac{\partial E(w)}{\partial \mathrm{w}_{0}}=0 & \longrightarrow \frac{\partial\left(\frac{1}{2} \sum_{n=1}^{N}\left(\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)\right)^{2}\right.}{\partial \mathrm{w}_{0}}=0 \\
\begin{array}{l}
\text { Sum rule } \\
\partial \sum^{f} f=\sum \partial f
\end{array} & =\frac{\frac{1}{2} \sum_{n=1}^{N} \partial\left(\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)^{2}\right)}{\partial w_{0}} \\
\begin{array}{l}
\text { Chain rule } \\
\partial f^{2}=2 f \partial f
\end{array} & =\frac{1}{2} \sum_{n=1}^{N} 2\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)\left(\frac{\partial\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)}{\partial w_{0}}\right) \\
\begin{array}{l}
\text { Separating } \\
\text { the terms }
\end{array} & =\sum_{n=1}^{N}\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)(-1)=-\left(\sum_{n=1}^{N} t_{n}-n w_{0}-\sum_{n=1}^{N}\left(\sum_{d=1}^{D} w_{d} x_{n d}\right)\right) \\
\frac{\partial E(w)}{\partial \mathrm{w}_{0}}=0 \longrightarrow w_{0}=\frac{1}{n}\left(\sum_{n=1}^{N} t_{n}-\sum_{n=1}^{N}\left(\sum_{d=1}^{D} w_{d} x_{n d}\right)\right)
\end{array}\right.
$$

## Error Function Minimization

$$
\begin{aligned}
\frac{\partial E(w)}{\partial \mathrm{w}_{\mathrm{k}}}=0 \longrightarrow & \frac{\partial\left(\frac{1}{2} \sum_{n=1}^{N}\left(\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)\right)^{2}\right.}{\partial \mathrm{w}_{\mathrm{k}}}=0 \\
\begin{aligned}
\text { Sum rule } & \\
\partial \sum^{2} f=\sum \partial f & =
\end{aligned} & \frac{1}{2} \sum_{n=1}^{N} \partial\left(\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)^{2}\right) \\
\begin{array}{l}
\text { Chain rule } \\
\partial f^{2}=2 f \partial f
\end{array} & =\frac{1}{2} \sum_{n=1}^{N} 2\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)\left(\frac{\partial\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)}{\partial w_{k}}\right) \\
\text { Reordering } & =\sum_{n=1}^{N}\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)\left(-x_{n k}\right) \\
\frac{\partial E(w)}{\partial \mathrm{w}_{\mathrm{k}}} & =-\sum_{n=1}^{N} x_{n k}\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)
\end{aligned}
$$

## Error Function Minimization

$$
\begin{aligned}
& \frac{\partial E(w)}{\partial \mathrm{w}_{\mathrm{k}}}=-\sum_{n=1}^{N} x_{n k}\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right) \\
& \sum_{n=1}^{N} x_{n k} t_{n}=\sum_{n=1}^{N} x_{n k}\left(w_{0}+\sum_{d=1}^{D} w_{d} x_{n d}\right) \longrightarrow \underbrace{\sum_{n=1}^{N} x_{n k} t_{n}}_{\mathbb{X}^{T} \mathbb{X} u}=\underbrace{w_{0} \sum_{n=1}^{N} x_{n k}+\sum_{d=1}^{D} w_{d} \sum_{n=1}^{N} x_{n k} x_{n d}}_{\substack{\text { Normal equations for } \\
\text { the least squares } \\
\text { problem }{ }^{T} \mathbb{X} w}} \\
& \left(\mathbb{X}^{T} \mathbb{X}\right)^{-1} \mathbb{X}^{T} T=w \quad W^{*}=\left(\mathbb{X}^{T} \mathbb{X}\right)^{-}
\end{aligned}
$$

## Solving Normal Equation Runtime

$$
\mathbb{X}=\left\{\overrightarrow{\mathrm{x}_{1}}, \overrightarrow{\mathrm{x}_{2}}, \ldots, \overrightarrow{\mathrm{x}_{\mathrm{N}}}\right\}=\underbrace{\left[\begin{array}{cccc}
1 & x_{11} & \ldots & x_{1 D} \\
1 & x_{21} & \ldots & x_{2 D} \\
\ldots & \ldots & \ldots & \ldots \\
1 & x_{N 1} & \ldots & x_{N D}
\end{array}\right]}_{D+1}\} N \quad \underset{\mathbb{X}^{T} \mathbb{X} \in \mathbb{R}^{(D+1) \times(D+1)}}{\in \mathbb{R}^{N \times(D+1)}}
$$

Forming $\mathbb{X}^{T} \mathbb{X}: \mathcal{O}\left(N^{2} D^{2}\right)$
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-53.jpg?height=227&width=205&top_left_y=1224&top_left_x=161)

**Image Description:** The image depicts a stopwatch, which is circular with a prominent starting button at the top. The design is symmetrical, featuring a simple numeral '1' at the center, indicating the function of measuring elapsed time. This symbol is often used in contexts related to timing events, experiments, or performance measurements, emphasizing efficiency and precision in time tracking.


Time complexity

Forming $X^{T} T: \mathcal{O}\left(N^{2} D\right)$
Forming $\left(\mathbb{X}^{T} \mathbb{X}\right)^{-1}: \mathcal{O}\left(D^{3}\right)$
Calculating $\left(\mathbb{X}^{T} \mathbb{X}\right)^{-1} \mathbb{X}^{T} T: \mathcal{O}\left(D^{3}\right)$
$\longrightarrow \mathcal{O}\left(N^{2} D^{2}+D^{3}\right.$,

## Geometric Interpretation

- Gaussian Mixture Modeß689302
- Linear Regression
- Basis Functions
- Vectorizing Calculations
- Error Function
- Error Function Minimization
- Geometric Interpretation
- Evaluation


## [Linear Algebra] Span

- The set of all possible linear combinations of the columns of $\mathbb{X}$ is called the span of $\mathbb{X}$ (denoted span( $\mathbb{X}$ )), also called the column space.
- Intuitively, this is all vectors you can "reach" using the columns of $\mathbb{X}$.
- If each column of $\mathbb{X}$ has length $D, \operatorname{span}(\mathbb{X})$ is a subspace of $\mathbb{R}^{D}$.
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-55.jpg?height=1103&width=1476&top_left_y=620&top_left_x=1854)

**Image Description:** The image depicts a geometric representation of a subspace in a vector space, specifically illustrating the concept of a subspace spanned by vectors in \( \mathbb{R}^D \). The diagram features a shaded region representing the subspace, with labeled axes indicating two vectors, \( X_{.,2} \) and \( X_{.,1} \), which are part of the spanning set. The axes are oriented in a three-dimensional space, emphasizing the dimensionality and orientation of the subspace. The text annotations reinforce the subspace concept and its relation to \( \mathbb{R}^D \).



## [Linear Algebra] Matrix-Vector Multiplication

Approach 1: So far, we've thought of our model as horizontally stacked predictions per datapoint:

$$
\left[\begin{array}{l}
\mid \\
\mathbb{Y} \\
\mid
\end{array}\right]=\left[\begin{array}{lcl}
- & \mathbf{x}_{1}^{T} & - \\
- & \mathbf{x}_{2}^{T} & - \\
& \vdots & \\
- & \mathbf{x}_{N}^{T} & -
\end{array}\right]\left[\begin{array}{l}
W \\
\end{array}\right] \quad \begin{aligned}
& \text { Matrix sizes } \\
& N \times 1=(N \times(D+1))((D+1) \times 1)
\end{aligned}
$$

- Approach 2: However, it is helpful sometimes to think of matrix-vector multiplication as performed by columns. We can also think of $\mathbb{Y}$ as a linear combination of feature vectors, scaled by parameters.
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-56.jpg?height=477&width=2391&top_left_y=1386&top_left_x=854)

**Image Description:** The image presents a mathematical equation in matrix form. It represents a linear combination of variables. The left side shows a matrix \( Y \) equal to a product of a matrix of input features \( X \) and a weight vector \( W \). It is detailed as:

$$
Y = \begin{bmatrix}
X_{:,0} & \cdots & X_{:,D}
\end{bmatrix} W = w_0 X_{:,0} + w_1 X_{:,1} + \cdots + w_D X_{:,D}
$$ 

Here, \( X_{:,i} \) denotes individual columns of the input feature matrix, and \( w_i \) are the corresponding weights for each feature.



## Prediction Is a Linear Combination of Columns

Our prediction of $\mathbb{Y}(\mathbb{X}, \vec{w})=\mathbb{X} \vec{w}$ is a linear combination of columns of $\mathbb{X}$.

Interpret: Our linear prediction $\mathbb{Y}$ will be in $\operatorname{span}(\mathbb{X})$, even if ground-truth values $t$ are not.

Goal: Find vector of $\mathbb{Y}$ in $\operatorname{span}(\mathbb{X})$ that is closest to t .
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-57.jpg?height=1096&width=1489&top_left_y=602&top_left_x=1845)

**Image Description:** The image is a geometric diagram illustrating a subspace of \(\mathbb{R}^D\) spanned by a set \(X\). It features a three-dimensional green parallelohedron shape representing the span of \(X\) with axis labels \(X_{:,1}\) and \(X_{:,2}\) indicated by black arrows. A vector \(Y\) extends into the subspace, depicted with downward arrows, while a red arrow labeled \(t\) emerges from it, showing a projection or direction within the span. The diagram visually represents the relationship between the vector space and specific vectors within it.

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-58.jpg?height=452&width=541&top_left_y=561&top_left_x=178)

**Image Description:** The image is a graphic representation of stylized clouds, featuring a larger cloud with a soft purple outline and a smaller cloud positioned beneath it. The clouds have a smooth, rounded shape, and the larger cloud has a subtle lighter purple shading that adds depth. This type of illustration is often used in contexts related to weather, meteorology, or cloud computing to symbolize concepts like data storage or atmospheric conditions. The design is simple and emphasizes a modern aesthetic with clear outlines and minimal detail.


# What's the geometry word for 'closest point in a subspace'? 

## Finding Optimum Predictions ( $\mathbb{Y}$ )

To minimize distance between vector of $\mathbb{Y}$ and t , we should minimize the length of the residual vector.
$\vec{e}$ is minimized if it is the orthogonal projection of $t$ on the $\operatorname{span}(\mathbb{X})$.

$$
\left.\begin{array}{l}
\text { Length of the residual } \\
\text { vector } \\
L_{2}-r \omega r m(e)=\|{ }^{-}{ }_{1}{ }_{2} \\
\quad=\sqrt{\sum_{d=0}^{D} e_{d}^{2}}
\end{array}\right]
$$

## Geometry of Least Squares in Plotly

Interactive link
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-60.jpg?height=1247&width=1613&top_left_y=446&top_left_x=1360)

**Image Description:** The image depicts a 3D vector space representation, featuring three vectors in a Cartesian coordinate system. The axes are labeled with 'x', 'y', and 'z'. The vector \( \mathbf{y} \) is shown as a blue arrow originating from the origin and represents the output \( \mathbf{Y} = \mathbf{Xw} \). Two input vectors \( \mathbf{x_1} \) (black) and \( \mathbf{x_2} \) (green) are also depicted. The grid indicates the plane of interaction where these vectors lie, enhancing the visualization of linear transformations in a multidimensional space.


## [Linear Algebra] Orthogonality

- A vector $\vec{v}$ is orthogonal to the $\operatorname{span}(\mathbb{X})$ if and only if it is orthogonal to all columns of $\mathbb{X}$.

Assuming $\vec{v} \in \mathbb{R}^{N}$ and $\mathbb{X}=\left[\begin{array}{ccc}\mid & & \mid \\ \mathbb{X}_{i, 0} & \cdots & \mathbb{X}_{i, D} \\ \mid & & \mid\end{array}\right] \in \mathbb{R}^{N \times D}$
$\vec{v}$ orthogonal to each column means:

We will use this shortly

$$
\begin{gathered}
\mathbb{X}_{:, 0}^{T} \vec{v}=0 \\
\mathbb{X}_{:, 1}^{T} \vec{v}=0 \\
\vdots \\
\mathbb{X}_{:, D}^{T} \vec{v}=0
\end{gathered} \longrightarrow\left[\begin{array}{c}
\mathbb{X}_{:, 0}^{T} \vec{v} \\
\mathbb{X}_{:, 1}^{T} \vec{v} \\
\vdots \\
\mathbb{X}_{:, D}^{T} \vec{v}
\end{array}\right]=\left[\begin{array}{c}
0 \\
0 \\
\vdots \\
0
\end{array}\right] \longmapsto \mathbb{X}^{T} \vec{v}=0
$$

## Going Back to Our Error Function

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}=\frac{1}{2}\left(\|\vec{e}\|_{2}\right)^{2}
$$

Equivalently, vector $\vec{e}$ is orthogonal to the span( $\mathbb{X}$ ), meaning that:
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-62.jpg?height=966&width=3075&top_left_y=901&top_left_x=0)

**Image Description:** The image features an academic lecture slide that presents the "Normal Equation" in a linear regression context. It contains three main equations. The first introduces the concept of residuals, while the second incorporates the formula for the response variable \( Y \). The third equation rearranges terms. A central blue starburst highlights the equation \( w^* = (X^TX)^{-1}X^Tt \), indicating the condition for invertibility of \( X^TX \). The text is formatted in different colors for emphasis, enhancing clarity and understanding of key concepts.


## Evaluation

- Gaussian Mixture Modeß689302
- Linear Regression
- Basis Functions
- Vectorizing Calculations
- Error Function
- Error Function Minimization
- Geometric Interpretation
- Evaluation


## Predict and Evaluate

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-64.jpg?height=1490&width=3026&top_left_y=324&top_left_x=188)

**Image Description:** The image is a flowchart diagram with four quadrants labeled "Learning Problem," "Model Design," "Predict & Evaluate," and "Optimization." Each quadrant contains icons and brief descriptions. The "Learning Problem" quadrant addresses supervised learning of scalar target values. The "Model Design" section focuses on linear regression with basis functions, including notation for \( y = \mathbf{x}^T \mathbf{w} \). The "Predict & Evaluate" quadrant emphasizes evaluation metrics, while "Optimization" presents the equation for deriving the direct solution, represented as \( E(w) = \frac{1}{n} \sum_{i=1}^{n} (y - \mathbf{x}_i^T \mathbf{w})^2 \) and \( w' = \arg\min E(w) \).


## Evaluation Visualization

Residual plot shows the trend of the residuals $\mathrm{e}_{\mathrm{n}}=\left(t_{n}-y\left(x_{n}, w\right)\right)$ with respect to predictions $y\left(x_{n}, w\right)$.

What a good residual plot looks like:
Points scattered randomly around 0 (no pattern).
Roughly constant vertical spread (homoscedasticity).
No obvious trend with fitted values or with predictors.
![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-65.jpg?height=783&width=1000&top_left_y=561&top_left_x=1267)

**Image Description:** The image displays a scatter plot illustrating a simple linear regression analysis. The x-axis represents "Median Income (MedInc)" while the y-axis shows "Median House Value (Y)." Individual data points are represented as red dots, indicating the relationship between income and house value. A fitted regression line, depicted in blue, shows the trend, suggesting that as median income increases, median house values also increase. The title of the diagram is "Simple Linear Fit: y ~ MedInc."

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-65.jpg?height=822&width=1022&top_left_y=527&top_left_x=2266)

**Image Description:** The diagram is a scatter plot titled "Residuals vs Fitted" illustrating the relationship between residuals and fitted values in a regression analysis. The x-axis represents "Fitted values," ranging from approximately 1.0 to 4.0, while the y-axis signifies "Residuals," ranging from about -3 to 3. Data points are depicted as red dots, indicating the distribution of residuals. A dashed line at zero represents the "Zero Residual Line," emphasizing the presence of heteroscedasticity, indicated by the fan shape of the plotted residuals across different fitted values. This suggests variability in the residuals as the fitted values increase.

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-66.jpg?height=465&width=541&top_left_y=548&top_left_x=170)

**Image Description:** The image depicts a simplified graphic representation of clouds, characterized by smooth curves and a purple outline. The design features two clouds: a larger one positioned to the left and a smaller one on the right, both with a light purple fill. This graphic emphasizes atmospheric elements, easily recognizable in discussions related to weather or environmental science. The overall aesthetic is minimalist, focusing on the shapes without detailed texture or color gradients.


# When you see a fan shape in the residual plot, what comes to mind? 

## Evaluation - Metrics

## Evaluation - Metrics

## Mean Squared Error (MSE)

## Evaluation - Metrics

## Mean Squared Error (MSE)

## Root Mean Squared Error (RMSE)

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
$\stackrel{i}{\sim}\left(\sum_{n=1}^{N}(c \ldots-\gamma(c \ldots \ldots n))^{2}\right)$
$\sqrt{\frac{\lambda}{N}\left(\sum_{n=1}^{N}\left(c_{n-1}-\mu\left(x_{n}, \ldots\right)\right)^{2}\right)}$

N(Noves) the metric
$\sqrt{\text { vis }}$ doack to the original unit of the data compared to MSE

## Evaluation - Metrics

## Mean Squared Error (MSE)

## Root Mean Squared Error (RMSE)

## R-Squared ( $\mathrm{R}^{2}$ ) Score

## Mean Squared Error (MSE) <br> Root Mean Squared Error (RMSE)

R-Squared ( $\mathbf{R}^{2}$ ) Score

## Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)
R-Squared ( $R^{2}$ ) Score

## Mean Squared Error (MSE) <br> Root Mean Squared Error (RMSE) <br> R-Squared ( $\mathbf{R}^{2}$ ) Score

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-70.jpg?height=176&width=409&top_left_y=399&top_left_x=2453)

**Image Description:** The image presents a mathematical formula consisting of two equations. The first equation, representing the Mean Squared Error, is given as:

$$ \frac{1}{N} \sum_{n=1}^{N} (t_n - y(x_n, W))^2 $$

The second equation shows the calculation of the coefficient of determination (R-squared):

$$ 1 - \frac{\sum_{n=1}^{N} (t_n - y(x_n, W))^2}{\sum_{n=1}^{N} (t_n - \bar{t})^2} $$

Both equations utilize summation, where \( t_n \) are target values, \( y(x_n, W) \) is the predicted value, \( N \) is the total number of samples, and \( \bar{t} \) represents the mean of target values.

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-70.jpg?height=188&width=397&top_left_y=684&top_left_x=2457)

**Image Description:** The image contains two equations related to the calculation of loss in a regression model. The first equation expresses the Mean Squared Error (MSE) as 

$$
\frac{1}{N} \sum_{n=1}^{N} (t_n - y(x_n, w))^2.
$$

The second equation shows the calculation of R-squared (coefficient of determination), represented as 

$$
1 - \frac{\sum_{n=1}^{N} (t_n - y(x_n, w))^2}{\sum_{n=1}^{N} (t_n - \bar{t})^2}.
$$ 

Both equations involve summation over \(N\) data points, where \(t_n\) is the target value, \(y(x_n, w)\) is the model prediction, and \(\bar{t}\) is the mean of the target values.

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-70.jpg?height=163&width=397&top_left_y=1003&top_left_x=2457)

**Image Description:** The image contains three mathematical expressions related to a loss function, commonly used in machine learning. The first line shows the mean squared error formula:

$$
\frac{1}{N} \sum_{n=1}^{N} (t_n - y(x_n, w))^2
$$

The second line shows the root mean squared error:

$$
\sqrt{\frac{1}{N} \left( \sum_{n=1}^{N} (t_n - y(x_n, w))^2 \right)}
$$

The third line appears to normalize the error:

$$
1 - \frac{\sum_{n=1}^{N} (t_n - y(x_n, w))^2}{\sum_{n=1}^{N} (t_n - y(x_n, w))^2}
$$

This provides a comparative evaluation of model predictions.


## Visualizing the Sum of Squared Error of Regression Model

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-71.jpg?height=1230&width=2455&top_left_y=357&top_left_x=382)

**Image Description:** The image is a diagram illustrating a regression model. It features a Cartesian coordinate system where the x-axis represents the input variable \( \mathbf{x} \) and the y-axis represents the output variable \( y \). A diagonal line indicates the regression line, representing the predicted relationship between \( x \) and \( y \). Black dots signify data points, and red dashed lines illustrate the residuals (errors) between the predicted values and actual values, enclosed in small boxes. The text emphasizes the goal of regression: to minimize the total area of these boxes.


## Visualizing the Sum of Squared Error of Intercept Model

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-72.jpg?height=1230&width=2459&top_left_y=357&top_left_x=382)

**Image Description:** The diagram depicts a one-dimensional regression model on a real number line. The horizontal axis represents the variable \( \mathbf{x} \) with corresponding output values \( y \) indicated on the vertical axis. It includes discrete black points representing data observations. Red vertical lines illustrate the residuals or errors between the predicted values (represented by blue lines) and the actual data points. The notation \( y(\mathbf{x}, \mathbf{w}) = \bar{t}_n \) signifies the regression function, incorporating parameters \( \mathbf{w} \) and observed target values \( \bar{t}_n \).


## R²: Quality of the Fit Relative to Intercept Model

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-73.jpg?height=762&width=1319&top_left_y=357&top_left_x=55)

**Image Description:** The image presents a mathematical equation accompanied by a diagram. The equation is expressed as \( y(\boldsymbol{x}, \boldsymbol{w}) = \boldsymbol{x}^T \boldsymbol{w} \), illustrating a linear function. The diagram features two black circles representing data points, with red vertical lines denoting the "margin" between the points and a decision boundary (represented by a dashed line). Dotted blue rectangles surround the circles, likely indicating the boundaries of different classes or groups within a classification context. The overall structure emphasizes a linear classification scenario in machine learning.

![](https://cdn.mathpix.com/cropped/2025_10_01_bc10b768cee6b2ce1858g-73.jpg?height=507&width=1438&top_left_y=357&top_left_x=1671)

**Image Description:** The image is a diagram illustrating a mathematical function, represented as \( y(\mathbf{\bar{x}}, \mathbf{\bar{w}}) = \bar{t}_n \). It features a horizontal line with several vertical lines drawn at specific intervals. Each vertical line has a black circle at the bottom endpoint, indicating a point of interest. The vertical lines are connected to the horizontal line with dashed blue rectangles. This arrangement suggests a relationship between the variables represented by the horizontal and vertical axes, likely related to statistical or mathematical modeling.


$$
R^{2}=\frac{\Delta \text { in area }}{\text { Constant Model Area }}=1-\frac{\sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}}{\sum_{n=1}^{N}\left(t_{n}-\overline{t_{n}}\right)^{2}}
$$
unitless and only compares performance relative to mean baseline.

## Evaluation - Metrics

## Mean Squared Error (MSE)

## Root Mean Squared Error (RMSE)

## R-Squared ( $\mathrm{R}^{2}$ ) Score

## Mean Absolute Error (MAE)

| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{m=1}^{N}\left(\epsilon_{m}-\nu\left(x_{m}, w\right)^{2}\right)\right.$ |
| :--- | :--- |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{m=1}^{N}\left(t_{m}-\nu\left(x_{m}, w\right)\right)^{2}\right)}$ |
| R-Squared ( $\mathbf{R}^{\mathbf{2}}$ ) Score | $1-\frac{\sum_{m-1}^{n}\left(\epsilon_{m}-x\left(x_{m}, n\right)\right)^{2}}{\left.\sum_{m=1}^{n}\left(\epsilon_{m}\right)-E_{m}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \mid c_{m}-\nu c_{m}, \ldots p 1\right)$ |
| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{m=1}^{N}\left(t_{m}-\nu\left(x_{m}, w\right)\right)^{2}\right)$ |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{m=1}^{N}\left(t_{m}-y\left(x_{m}, w\right)\right)^{2}\right)}$ |
| R-Squared ( $\mathbf{R}^{\mathbf{2}}$ ) Score | $1-\frac{\sum_{m-1}^{N}\left(t_{m}-y\left(x_{m \cdot w}\right)\right)^{2}}{\sum_{m-1}^{N}\left(t_{m}-t_{m}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \mid t_{m}-\nu c x_{m}, w D 1\right)$ |
| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{m=1}^{N}\left(t_{m}-\nu c_{m \cdot n} c^{2}\right)\right.$ |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{m=1}^{N}\left(t_{m}-\nu r\left(x_{m}, w\right)\right)^{2}\right)}$ |
| R-Squared ( $\mathbf{R}^{\mathbf{2}}$ ) Score | $1-\frac{\sum_{m-1}^{n}\left(\epsilon_{m}-x\left(x_{m}, n\right)\right)^{2}}{\left.\sum_{m=1}^{n}\left(\epsilon_{m}\right)-E_{m}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{m=1}^{N}\left\|\epsilon_{m}-v\right\| x_{m}, w \mid\right)$ |

## Mean Squared Error (MSE) <br> Root Mean Squared Error (RMSE)

## R-Squared ( $\mathbf{R}^{2}$ ) Scare

Mean Absolute Error (MAE)

In the same unit as the data; similar to MSE but differs in how the penalization applies.

## Evaluation - Metrics

## Mean Squared Error (MSE)

## Root Mean Squared Error (RMSE)

R-Squared ( $\mathrm{R}^{2}$ ) Score

Mean Absolute Error (MAE)

Mean Absolute Percentage Error (MAPE)

| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{m=1}^{N}\left(x_{m}-y\left(x_{m}, \ldots p\right)^{2}\right)\right.$ |
| :--- | :--- |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{m=1}^{N}\left(c_{m}-\nu\left(x_{m} \ldots m\right)\right)^{2}\right)}$ |
| R-Squared ( $\mathbf{R}^{\mathbf{2}}$ ) Score | $1-\frac{\sum m_{-1}\left(t_{m}-y \mid\left(x_{m}, m\right)\right)^{2}}{\left.\sum m_{-1}\left(t_{m}\right)-\frac{t_{m}}{t_{m}}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \mid x_{m}-\nu\left(x_{m}, \ldots, D\right)\right)$ |
| Mean Absolute Percentage Error (MAPE) | $\frac{1}{N}\left(\sum_{n=1}^{N} \frac{x_{n}-x_{n}\left(x_{n}, w\right)}{x_{n}}\right) \times 100$ |
| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{n=1}^{N}\left(\varepsilon_{n}-y\left(x_{n}, w\right)\right)^{2}\right)$ |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{m=1}^{N}\left(t_{m}-\nu\left(x_{m}+w\right)\right)^{2}\right)}$ |
| R-Squared ( $\mathbf{R}^{\mathbf{2}}$ ) Score | $1-\frac{\sum_{m=1}^{M}\left(t_{m}-y\left(x_{m}, m\right)\right)^{2}}{\left.\sum_{m=1}^{N}\left(t_{m}\right)-E_{m}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{n=1}^{N}\left\|t_{n}-\nu\right\|\left(x_{n}, w\right) \mid\right)$ |
| Mean Absolute Percentage Error (MAPE) | $\frac{1}{N}\left(\sum_{n=1}^{N} \frac{t_{m}-y\left(x_{m}, w\right)}{t_{m}}\right) \times 100$ |
| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{m=1}^{N}\left(x_{m}-\nu\left(x_{m}, \ldots p\right)^{2}\right)\right.$ |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{m=1}^{N}\left(c_{m}-\nu\left(x_{m} \ldots m\right)\right)^{2}\right)}$ |
| R-Squared ( $\mathbf{R}^{\mathbf{2}}$ ) Score | $1-\frac{\sum m_{-1}\left(t_{m}-3\right)\left(x_{m} m p\right)^{2}}{\sum \sum_{m-1}\left(t_{m}\right)-\left(t_{m}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \mid x_{m}-\nu\left(x_{m}, \ldots, D\right)\right)$ |
| Mean Absolute Percentage Error (MAPE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \frac{x_{m}-N\left(x_{m}, W\right)}{x_{m}}\right) \times 100$ |
| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{m=1}^{m}\left(m_{m}-\nu\left(x_{m}, \ldots p\right)^{\bar{x}}\right)\right.$ |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{n=1}^{N}\left(t_{n}-\nu\left(x_{n} \ldots m\right)\right)^{2}\right)}$ |
| R-Squared ( $\mathbf{R}^{\mathbf{2}}$ ) score | $1-\frac{\sum m_{-1}\left(t_{m}-3 \cdot\left(x_{m}, \ldots\right)\right)^{2}}{\left.\sum x_{m}^{m}-1\left(t_{m}\right)-\frac{\left(x_{m}\right)}{x_{m}}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \mid x_{m=}-\nu\left(x_{m . . m p}\right)\right.$ |
| Mean Absolute Percentage Error (MAPE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \frac{c_{m n}-N\left(x_{m n}, W\right)}{N_{m n}}\right) \times 100$ |
| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{m=1}^{m}\left(x_{m}-\nu\left(x_{m}, \ldots p\right)^{z}\right)\right.$ |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{n=1}^{N}\left(c_{n m}-\nu\left(x_{n \ldots} \omega\right)\right)^{2}\right)}$ |
| R-Squared ( $R^{2}$ ) Score | $1-\frac{\sum m_{-1}\left(t_{m}-y \mid\left(x_{m} m p\right)^{2}\right.}{\left.\sum m_{-1}^{m}\left(t_{m}\right)-\varepsilon_{m}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \mid x_{m}-\nu\left(x_{m}, \ldots, 1\right)\right.$ |
| Mean Absolute Percentage Error (MAPE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \frac{E_{m}-s\left(x_{m}, W\right)}{x_{m}}\right) \times 100$ |

## Lecture 6

## Linear Regression (1)

Credit: Joseph E. Gonzalez and Narges Norouzi
Reference Book Chapters: Chapter 1.[2.1-2.3], Chapter 4.[1.1, 1.4]

