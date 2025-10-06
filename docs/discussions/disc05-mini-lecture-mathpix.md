---
course: CS 189
semester: Fall 2025
type: discussion
title: Discussion 5
source_type: slides
source_file: Discussion Mini Lecture 05.pptx
processed_date: '2025-10-04'
processor: mathpix
---

## Discussion Mini Lecture 5

## More on Regularization

Statistical Justification of Regularization \& Bias-Variance Trade-Off

## CS 189/289A, Fall 2025 @ UC Berkeley

Sara Pohland

1. Regularization

Concepts Covered
a) Review
b) Statistical Justification
2. Bias \& Variance
a) Decomposition for LS
b) Regularization Trade-off

## Review of Regularized Least Squares

1. Regularization
a) Review
b) Statistical Justification
2. Bias \& Variance
a) Decomposition for $L S$
b) Regularization Trade-off

## What's wrong with linear regression?

- Sometimes $\mathbb{X}^{T} \mathbb{X}$ is "ill-conditioned."
- If some features are highly correlated, $\mathbb{X}^{T} \mathbb{X}$ becomes nearly singular (noninvertible), meaning that some of its eigenvalues are close to zero.
- This means the eigenvalues of $\left(\mathbb{X}^{T} \mathbb{X}\right)^{-1}$ are very large for those features.
- We then say our solution is "unstable" because small changes in our data will lead to large changes in our optimal solution $\widehat{w}$.
- Our solution is sensitive to noisy data.
- Ordinary least squares tries to minimize training error, but our training data might be noisy and not follow the underlying linear trend.
- To account for noise, we will assign large positive weights to some features and large negative weights to others to "explain away" the randomness.
- Now we're not learning the general trend; we're fitting to noise.


## How can we address these limitations?

Let's update our ordinary least square problem with a regularization term that penalizes the size of the weights:

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathbf{x}_{n}, w\right)\right)^{2}+\lambda \operatorname{Reg}[w]
$$
$\lambda>0$ is a hyperparameter we choose (usually through validation)

## How do we choose this regularization?

Usually, we choose one of our three favorite norms:

1) $L_{1}(w):\|w\|_{1}=\sum_{d}\left|w_{d}\right|$ promotes sparsity by pushing some coefficients exactly to zero
2) $L_{2}(w):\|w\|_{2}=\sqrt{\sum_{d} w_{d}{ }^{2}}$ shrinks all coefficients smoothly toward zero, but almost never exactly to zero
3) $L_{\infty}(w):\|w\|_{\infty}=\max _{d}\left|w_{d}\right|$ bounds the maximum coefficient, preventing any one from dominating

## $L_{2}$ Regularization $\rightarrow$ Ridge Regression

Often, we add $L_{2}$ regularization to our least squares problem:

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathbf{x}_{n}, w\right)\right)^{2}+\frac{\lambda}{2}\|w\|_{2}^{2} \quad \begin{aligned}
& \text { We call this } \\
& \text { Ridge } \\
& \text { regression }!
\end{aligned}
$$

As before, $E(w)$ is a convex quadratic function, so we can find the minimum by taking the derivative with respect to w and setting it equal to 0 . Doing so gives us the normal equation:

$$
\mathbb{X}^{T} \mathrm{t}=\left(\mathbb{X}^{T} \mathbb{X}+\lambda I\right) w
$$

If $\mathbb{X}^{T} \mathbb{X}+\lambda I$ is invertible, then the optimal weight is

$$
\widehat{w}=\left(\mathbb{X}^{T} \mathbb{X}+\lambda I\right)^{-1} \mathbb{X}^{T} \mathrm{t}
$$

## $L_{1}$ Regularization $\rightarrow$ LASSO

It's also common to add $L_{1}$ regularization to our LS problem:

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathbf{x}_{n}, w\right)\right)^{2}+\frac{\lambda}{2}\|w\|_{1} \quad \quad \text { We call this } \quad \text { LASSO! }
$$

Now, $E(w)$ is still a convex function, but it is not differentiable for all $\mathbf{w}$, so we cannot find the minimum by taking the derivative with respect to w and setting it equal to 0 .

LASSO is useful but does not admit a closed-form solution!

## Statistical Interpretation of Regularization

1. Regularization
a) Review
b) Statistical Justification
2. Bias \& Variance
a) Decomposition for LS
b) Regularization Trade-off

## Goal of Linear Regression

Goal: Find a function of the form $y(\mathbf{x}, w)=\mathbf{x}^{T} w$ that best estimates the true input-output relationship of our population, $h(\mathbf{x})$.

ML Approach: Estimate this function using data sampled from that population: $\mathcal{D}=\left\{\left(x_{n}, t_{n}\right)\right\}_{n=1}^{N}$, where $x_{n}$ is the $n$th sample point with corresponding target/output $t_{n}$.

Question: How do we find the best parameter of this function, $\widehat{w}$ ?

## Maximum a Posteriori (MAP) Estimation

Suppose we have some prior belief about $\mathrm{w}, p(\mathrm{w})$. We get some observations $\mathrm{t}=\left(t_{1}, \ldots t_{N}\right)$, and then we update our a posteriori belief:

$$
p(\mathrm{w} \mid \mathrm{t})
$$

We want to find the value of $w$ that maximizes this probability:

$$
\widehat{w}_{\text {MAP }}=\underset{w}{\operatorname{argmax}} p(\mathbf{w} \mid \mathbf{t})
$$

How do we estimate this posterior probability? Bayes' theorem!

$$
p(\mathrm{w} \mid \mathrm{t})=\frac{p(\mathrm{t} \mid \mathrm{w}) p(\mathrm{w})}{p(\mathrm{t})} \propto p(\mathrm{t} \mid \mathrm{w}) p(\mathrm{w})
$$

## Likelihood of Observed Target

What is the likelihood of observing targets $\mathrm{t}=\left(t_{1}, \ldots t_{N}\right)$, given that our weight vector is w (i.e., what is $p(\mathrm{t} \mid \mathrm{w})$ )?

Assume the target $t_{n}$ is the true signal plus noise:

$$
t_{n}=y\left(\mathbf{x}_{n}, w\right)+\epsilon_{n},
$$
where $\epsilon_{n} \sim \mathcal{N}\left(0, \sigma^{2}\right)$ is zero-mean Gaussian random noise.
Then, $t_{n} \sim \mathcal{N}\left(y\left(\mathbf{x}_{n}, w\right), \sigma^{2}\right)$ is a Gaussian r.v with mean $y\left(\mathbf{x}_{n}, w\right)$ :
$$
p\left(t_{n} \mid \mathrm{w}\right)=\mathcal{N}\left(t_{n} ; y\left(\mathbf{x}_{n}, w\right), \sigma^{2}\right)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\frac{\left(\mathrm{w}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}}{2 \sigma^{2}}\right)
$$

## Likelihood of Observed Target

What is the likelihood of observing targets $\mathrm{t}=\left(t_{1}, \ldots t_{N}\right)$, given that our weight vector is w (i.e., what is $p(\mathrm{t} \mid \mathrm{w})$ )?

Assume the target $t_{n}$ is the true signal plus noise:

$$
t_{n}=y\left(\mathbf{x}_{n}, w\right)+\epsilon_{n},
$$
where $\epsilon_{n} \sim \mathcal{N}\left(0, \sigma^{2}\right)$ is zero-mean Gaussian random noise.
If our noise signals $\epsilon_{1}, \ldots \epsilon_{\mathrm{N}}$ are independent, then
$$
p(t \mid \mathrm{w})=\prod_{n=1}^{N} \mathcal{N}\left(t_{n} ; y\left(\mathbf{x}_{n}, w\right), \sigma^{2}\right)
$$

## Prior Assumption about Weight/Function

What is our prior belief about the weight vector (i.e., what is $p(\mathrm{w})$ )?
Before observing our data, let's assume w is distributed around 0 .
There are two very common zero-mean distributions...

## Prior Assumption about Weight/Function

![](https://cdn.mathpix.com/cropped/2025_10_04_69a18d30d8205fd7a6ecg-15.jpg?height=1502&width=1298&top_left_y=348&top_left_x=174)

**Image Description:** The image features a lecture slide on the Gaussian distribution, labeled as \( N(0, \sigma^2) \). It includes a graph depicting the probability density function \( p(w) \) with the x-axis representing the variable \( w \), ranging approximately from -4 to 4, and the y-axis representing the probability density, peaking at 0.40. The curve of the Gaussian is bell-shaped, centered at 0, indicating the distribution of values for \( w \) when \( \sigma = 1 \). The equation for the Gaussian distribution is presented above the graph as follows: 

$$
p(w) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{w^2}{2\sigma^2}\right)
$$

(2 Laplace: $\operatorname{Laplace}(0, \mathrm{~b})$

$$
\mathrm{p}(\mathrm{w})=\frac{1}{2 \mathrm{~b}} \exp \left(-\frac{|\mathrm{w}|}{\mathrm{b}}\right)
$$
![](https://cdn.mathpix.com/cropped/2025_10_04_69a18d30d8205fd7a6ecg-15.jpg?height=1009&width=1188&top_left_y=850&top_left_x=1951)

**Image Description:** The image is a graph of a Laplace distribution. The x-axis represents the variable \( W \), ranging from -4 to 4, while the y-axis indicates the probability density function (PDF), showing values from 0 to approximately 0.7. The curve is peaked at \( W = 0 \) and exhibits exponential decay on either side. The line is colored orange and is labeled as "Laplace (b=σ/2)." The plot includes grid lines for clarity.


## Prior Assumption about Weight/Function

(1) Gaussian: $\mathcal{N}\left(0, \sigma^{2}\right)$

$$
\begin{gathered}
p(\mathrm{w})=\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\frac{\mathrm{w}^{2}}{2 \sigma^{2}}\right) \\
p(t \mid \mathrm{w})=\prod_{\mathrm{n}=1}^{\mathrm{N}} \mathcal{N}\left(t_{n} ; y\left(\mathbf{x}_{n}, w\right), \sigma^{2}\right) \\
\widehat{w}_{M A P}=\underset{w}{\operatorname{argmax}} p(t \mid \mathrm{w}) p(\mathrm{w})
\end{gathered}
$$

Ridge regression!
(2) Laplace: Laplace( $0, \mathrm{~b}$ )

$$
\begin{gathered}
p(\mathrm{w})=\frac{1}{2 \mathrm{~b}} \exp \left(-\frac{|\mathrm{w}|}{\mathrm{b}}\right) \\
p(t \mid \mathrm{w})=\prod_{\mathrm{n}=1}^{\mathrm{N}} \mathcal{N}\left(t_{n} ; y\left(\mathbf{x}_{n}, w\right), \sigma^{2}\right) \\
\widehat{w}_{M A P}=\underset{w}{\operatorname{argmax}} p(t \mid \mathrm{w}) p(\mathrm{w})
\end{gathered}
$$

Lasso!
We show this in Problem 1 on the Disc. 5 worksheet.

## Prior Assumption about Weight/Function

What if we don't have any prior belief about our function (i.e., we don't have an initial guess for $p(\mathrm{w}))$ ?

Then we can go back to doing maximum likelihood estimation:

$$
\widehat{w}_{M L E}=\underset{w}{\operatorname{argmax}} p(\mathrm{t} \mid \mathrm{w})
$$

We then recover the ordinary least squares (OLS) solution!

## Bias-Variance Decomposition for Least Squares

1. Regularization
a) Review
b) Statistical Justification
2. Bias \& Variance
a) Decomposition for LS
b) Regularization Trade-off

## Goal of Linear Regression

Goal: Find a function of the form $y(\mathbf{x}, w)=\mathbf{x}^{T} w$ that best estimates the true input-output relationship of our population, $h(\mathbf{x})$.

Risk-Analysis Approach: Find the function that minimizes our risk/expected loss. For linear regression, we're interested in the expected squared error for an input-output pair ( $\mathrm{x}, t$ ):

$$
\left.\mathrm{E}[(t-y(\mathbf{x}, w)))^{2}\right]
$$

Assumption: Suppose that our observation $t$ is the sum of our true function and random noise:

$$
t=h(\mathbf{x})+\epsilon
$$

## Decomposing the Risk

Let's take a closer look at this risk function...

$$
\begin{aligned}
& \begin{aligned}
\left.\mathrm{E}[(t-\boldsymbol{y}(\mathbf{x}, \boldsymbol{w})))^{2}\right] & =\mathrm{E}\left[t^{2}-2 t \boldsymbol{y}(\mathbf{x}, \boldsymbol{w})+\boldsymbol{y}(\mathbf{x}, \boldsymbol{w})^{2}\right] \\
& =\mathrm{E}\left[t^{2}\right]-2 \mathrm{E}[t \boldsymbol{y}(\mathbf{x}, \boldsymbol{w})]+\mathrm{E}\left[\boldsymbol{y}(\mathbf{x}, \boldsymbol{w})^{2}\right] \overbrace{}^{\begin{array}{c}
\text { linearity of } \\
\text { expectation }
\end{array}} \\
& =\mathrm{E}\left[t^{2}\right]-2 \mathrm{E}[t] \mathrm{E}[\boldsymbol{y}(\mathbf{x}, \boldsymbol{w})]+\mathrm{E}\left[\boldsymbol{y}(\mathbf{x}, \boldsymbol{w})^{2}\right] \underbrace{\begin{array}{c}
\text { independenc } \\
\text { e of samples }
\end{array}}_{\operatorname{Var}(t)+\mathrm{E}[t]^{2}} \overbrace{\operatorname{Var}(\boldsymbol{y}(\mathbf{x}, \boldsymbol{w}))+\mathrm{E}[\boldsymbol{y}(\mathbf{x}, \boldsymbol{w})]^{2}}^{\begin{array}{c}
\text { def'n of } \\
\text { variance }
\end{array}}
\end{aligned} \\
& =(\mathrm{E}[t]-\mathrm{E}[\boldsymbol{y}(\mathbf{x}, \boldsymbol{w})])^{2}+\operatorname{Var}(t)+\operatorname{Var}(\boldsymbol{y}(\mathbf{x}, \boldsymbol{w}))<\begin{array}{c}
\text { combinin } \\
\mathrm{g} \text { terms }
\end{array} \\
& =(h(\mathbf{x})-\mathrm{E}[y(\mathbf{x}, w)])^{2}+\operatorname{Var}(\epsilon)+\operatorname{Var}(y(\mathbf{x}, w)) \underset{\substack{\text { target } \\
\text { assumption }}}{\text { target }}
\end{aligned}
$$

## What are these terms?

![](https://cdn.mathpix.com/cropped/2025_10_04_69a18d30d8205fd7a6ecg-21.jpg?height=1106&width=2740&top_left_y=374&top_left_x=110)

**Image Description:** The image displays an equation and accompanying explanatory text related to the error in a predictive model. The equation is formatted as follows:

$$ E[(t - y(x,w))^2] = (h(x) - E[y(x,w)])^2 + Var(\epsilon) + Var(y(x,w)) $$

The diagram illustrates the components of the mean squared error, with arrows pointing to definitions: "squared bias of model," "random noise," and "variance of model." Each term is clearly delineated, showing the relationship between model bias, prediction variance, and noise in the context of predictions.


## What are these terms?

Noise - variability in our observations due to random noise.

Dataset 1
![](https://cdn.mathpix.com/cropped/2025_10_04_69a18d30d8205fd7a6ecg-22.jpg?height=1022&width=1604&top_left_y=773&top_left_x=85)

**Image Description:** The image is a scatter plot with blue dots representing observations labeled as \(D_1 = \{(x_n, t_n)\}_{n=1}^N\). The x-axis is labeled \(X\) and represents the input variable, while the y-axis represents the output variable. A smooth blue curve illustrates the true function, denoted as \(h(x)\), which fits the scattered data points. Arrows indicate the observations and the true function, emphasizing their relationship. The overall purpose of the diagram is to depict the approximation of a function based on discrete data points in a regression context.


Dataset 2
![](https://cdn.mathpix.com/cropped/2025_10_04_69a18d30d8205fd7a6ecg-22.jpg?height=1009&width=1540&top_left_y=782&top_left_x=1684)

**Image Description:** The image presents a scatter plot diagram depicting observations of data points represented by blue dots. The x-axis is labeled with the variable \( x \), ranging from 0 to 1, while the y-axis is implicitly determined by the curve. A sinusoidal trend line, indicated by a blue curve, illustrates the relationship between the observations. The formula \( D_2 = \{(x_n, t_n)\}_{n=1}^N \) is included, indicating a dataset of paired observations. An arrow points to the term "observations," signifying the significance of the data points in the context of the graph.


## What are these terms?

Bias - expected deviation between predicted value and true value.

Dataset 1 Dataset 2
![](https://cdn.mathpix.com/cropped/2025_10_04_69a18d30d8205fd7a6ecg-23.jpg?height=1047&width=3144&top_left_y=769&top_left_x=80)

**Image Description:** The image consists of two diagrams depicting regression functions. Each diagram features a scatter plot with blue dots representing data points along the x-axis labeled \( x \) (ranging from -1 to 1) and the y-axis representing the predicted values. A purple line models the predicted function \( h(x) \), while the true function is indicated in each diagram. The notation \( y(x, \theta_1) \) and \( y(x, \theta_2) \) highlights predictions at different parameters. The arrows denote the relationship of the predictions to the true function. The diagrams are illustrative of predictive modeling variations.


## What are these terms?

## Variance - variability in model across different training datasets.

Dataset 1
![](https://cdn.mathpix.com/cropped/2025_10_04_69a18d30d8205fd7a6ecg-24.jpg?height=1043&width=1608&top_left_y=769&top_left_x=85)

**Image Description:** The image depicts a scatter plot illustrating the relationship between a true function \( h(x) \) and a predicted function \( y(x, w_1) \). The x-axis represents the variable \( x \), while the y-axis indicates the output values. Blue dots correspond to data points, the pink curve represents the predicted function, and the purple curve signifies the true function. Arrows point to the respective curves, labeling them accordingly. This diagram visually compares the accuracy of a model's predictions against the actual function.


Dataset 2
![](https://cdn.mathpix.com/cropped/2025_10_04_69a18d30d8205fd7a6ecg-24.jpg?height=1013&width=1540&top_left_y=782&top_left_x=1684)

**Image Description:** The image is a scatter plot displaying a predictive function with data points represented as blue circles. The x-axis indicates the independent variable, labeled as \( x \), and ranges from 0 to 1. The y-axis, labeled \( y \), corresponds to the dependent variable. There are two curves: one in blue and another in magenta, representing different predictive models. An arrow points to the magenta curve, which is indicated as the "predicted function." The diagram illustrates the relationship between the input variable \( x \) and the predicted output \( y \).


## What is the impact of regularization?

$$
\left.\mathrm{E}[(t-y(\mathbf{x}, w)))^{2}\right]=\underbrace{(h(\mathbf{x})-\mathrm{E}[y(\mathbf{x}, w)])^{2}}+\underbrace{\operatorname{Var}(\epsilon)}+\underbrace{\operatorname{Var}(y(\mathbf{x}, w))}
$$
![](https://cdn.mathpix.com/cropped/2025_10_04_69a18d30d8205fd7a6ecg-25.jpg?height=801&width=1985&top_left_y=680&top_left_x=870)

**Image Description:** The slide contains a flowchart depicting the relationship between the squared difference between true output and expected prediction, variance of noise, and variance of prediction. The flowchart has three main components: the top component labeled "squared difference between true output and expected prediction," which points down to "often increases" on the left, "no impact" in the center, and "decreases" on the right. Arrows connect each component, illustrating the causal relationships between them. The text is formatted in a clear, bold font, enhancing readability.


## Bias-Variance Trade-off for Regularization

1. Regularization
a) Review
b) Statistical Justification
2. Bias \& Variance
a) Decomposition for LS
b) Regularization Trade-off

## The Classic Bias-Variance Trade-Off

![](https://cdn.mathpix.com/cropped/2025_10_04_69a18d30d8205fd7a6ecg-27.jpg?height=1201&width=2927&top_left_y=386&top_left_x=212)

**Image Description:** The diagram is a graphical representation of bias-variance tradeoff in machine learning. It features a Cartesian plane with "Model Complexity" on the x-axis and "Prediction Error" on the y-axis. Two curves represent training and test sample errors, showing a U-shape. The areas labeled "underfitting" and "overfitting" illustrate regions of low and high bias and variance, respectively. A green "sweet spot" area indicates optimal model complexity minimizing prediction error, balancing bias and variance effectively.

* This plot comes from chapter 2 of Hastie, Tibshirani, and Freedman's Elements of Statistical Lear ning. There are nuances to this diagram as described a bit provocatively in this blog post.

## Bias-Variance Trade-Off for Regularization

![](https://cdn.mathpix.com/cropped/2025_10_04_69a18d30d8205fd7a6ecg-28.jpg?height=1455&width=2889&top_left_y=395&top_left_x=229)

**Image Description:** The diagram illustrates the bias-variance tradeoff in machine learning. It features two curves: one represents prediction error for a training sample (decreasing blue curve) and the other for a test sample (increasing red curve). The x-axis indicates model complexity, labeled from "Low" on the left to "High" on the right, representing less regularization as it increases. The y-axis shows prediction error, ranging from high to low. The highlighted "sweet spot" indicates optimal model complexity where prediction error is minimized, with regions of "underfitting" and "overfitting" marked on either side.


## Discussion Mini Lecture 5

## More on Regularization

Contributors: Sara Pohland

## Additional Resources

1. Regularization

- Deep Learning Foundations and Concepts - Chapter 4.1.6
- Sara's notes on Machine Learning - Section 10

2. Bias \& Variance

- Deep Learning Foundations and Concepts - Chapter 4.3

