---
course: CS 189
semester: Fall 2025
type: lecture
title: Density Estimation and GMM
source_type: slides
source_file: Lecture 05 -- Density Estimation and GMM.pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Lecture 5

## Density Estimation and Gaussian Mixture Models

Introducing Maximum Likelihood Estimation, and Gaussian Mixture Models

EECS 189/289, Fall 2025 @ UC Berkeley
Joseph E. Gonzalez and Narges Norouzi

# III Join at slido.com頑든 \#1924089 

## In The Last Lecture We...

- Introduced $\mathbf{k}$-means clustering as an unsupervised clustering algorithm.
- Wanted a better way to capture uncertainty in cluster predictions.
- Reviewed basic ideas in probability.
- Explored Bayesian updating to determine the false-positive rate needed for a wake-word detector.

Listening!
![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-03.jpg?height=368&width=381&top_left_y=1347&top_left_x=1777)

**Image Description:** This image depicts a smart speaker, specifically a spherical device with a mesh fabric covering. It features a smooth, rounded base with a contrasting lower section that is likely made of plastic. A subtle blue light ring is visible along the base, indicating its operational status. The speaker's design is minimalist and modern, suggesting usability in various environments such as homes or offices. There are no visible buttons or interfaces, pointing towards voice activation as the primary mode of interaction.

![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-03.jpg?height=805&width=1103&top_left_y=1007&top_left_x=2219)

**Image Description:** The image consists of a 3D scatter plot and two images of a cyclist. The scatter plot, with axes labeled X, Y, and Z, displays data points colored in red and blue. It illustrates clustering or segmentation results, prompting a question about appropriate color coding for certain data points. The side images show the original and segmented versions of the cyclist, depicting how K-means compression alters image representation. This visualizes the effectiveness of clustering in image processing.


- Expectation and Variance
- Density Estimation
- Basic Probability Distributions
- Maximum Likelihood

Roadmap

- Bias
- Multivariate Normal
- Gaussian Mixture Models

Questions

- Expectation and Variance
- Density Estimation
- Basic Probability Distributions

Expectation and Variance

- Maximum Likelihood
- Bias
- Multivariate Normal
- Gaussian Mixture Models

Questions

## Expectations

The expectation of a random variable $X$ is the the weighted average with respect to a probability distribution $p(x)$ :

$$
\mathbb{E}[X]=\sum_{x} x p(x)
$$

Expectation is an operator from a random variable to a value.
The conditional expectation of a random variable is a weighted average with respect to the conditional distribution:

$$
\mathbb{E}[X \mid y]=\sum_{x} x p(x \mid y)
$$

## Functions of Random Variables

The function of a random variable is also a random variable:

$$
Y=f(X)
$$

The expectation of a function $f(X)$ is the weighted average with respect to a probability distribution $p(x)$ :

$$
\mathbb{E}[f]=\mathbb{E}[f(X)]=\sum_{x} f(x) p(x)
$$

The expectation of a function of multiple variables is defined over the joint distribution:

$$
\mathbb{E}[f(X, Y)]=\sum_{x, y} f(x, y) p(x, y)
$$

## Linearity of Expectation

For any two random variables $X$ and $Y$ :

$$
\mathbb{E}[a X+b Y+c]=a \mathbb{E}[X]+b \mathbb{E}[Y]+c
$$

- Does not require any assumptions about $p(X, Y)$.

$$
\begin{aligned}
\mathbb{E}[a X+b Y+c] & =\sum_{x} \sum_{y}(a x+b y+c) p(x, y) \\
& =\sum_{x} \sum_{y}(a x p(x, y)+b y p(x, y)+c p(x, y)) \\
& =a \sum_{x} x \sum_{y} p(x, y)+b \sum_{y} y \sum_{x} p(x, y)+c \sum_{x} \sum_{y} p(x, y) \\
& =a \mathbb{E}[X]+b \mathbb{E}[Y]+c
\end{aligned}
$$

## Variance

The variance of a random variable defines its spread:

$$
\operatorname{var}[X]=\mathbb{E}\left[(X-\mathbb{E}[X])^{2}\right]
$$

Using linearity of expectation, we can rewrite the variance as:

$$
\operatorname{var}[X]=\mathbb{E}\left[X^{2}\right]-\mathbb{E}[X]^{2}
$$

What is $\operatorname{var}[a X+b]$ ?

$$
\begin{array}{ll}
\quad \operatorname{var}[a X+b]=\underbrace{\mathbb{E}\left[(a X+b)^{2}\right]} & -\underbrace{\mathbb{E}[a X+b]^{2}} \\
\mathbb{E}\left[a^{2} X^{2}+2 a b X+b^{2}\right] & -(a \mathbb{E}[X]+b)^{2} \\
a^{2} \mathbb{E}\left[X^{2}\right]+2 a b \mathbb{E}[X]+b^{Z} & -a^{2} \mathbb{E}[X]^{2}-2 a b \mathbb{E}[X]-b^{Z} \\
\operatorname{var}[a X+b]=a^{2}\left(\mathbb{E}\left[X^{2}\right]-\mathbb{E}[X]^{2}\right)=a^{2} \operatorname{var}[X]
\end{array}
$$

## Covariance

The covariance of two random variables defines the degree to which they vary together:

$$
\operatorname{cov}[X, Y]=\mathbb{E}[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])]=\mathbb{E}[X Y]-\mathbb{E}[X] \mathbb{E}[Y]
$$

The covariance of two random vectors $\boldsymbol{x}$ and $\boldsymbol{y}$ is the matrix:

$$
\operatorname{cov}[\boldsymbol{x}, \boldsymbol{y}]=\mathbb{E}\left[(\boldsymbol{x}-\mathbb{E}[\boldsymbol{x}])\left(\boldsymbol{y}^{T}-\mathbb{E}\left[\boldsymbol{y}^{T}\right]\right)\right]=\mathbb{E}\left[\boldsymbol{x} \boldsymbol{y}^{T}\right]-\mathbb{E}[\boldsymbol{x}] \mathbb{E}\left[\boldsymbol{y}^{T}\right]
$$

We will return to covariance in greater detail in future lectures.

## Probability Density Functions

For continuous random variables, we use a density function $p(x)$ where the probability that $x$ falls in the interval ( $a, b$ ) is given by:

$$
p(x \in(a, b))=\int_{a}^{b} p(x) \mathrm{d} x
$$

Just as with discrete distribution, we have the following two properties:

$$
p(x) \geq 0 \quad \text { and } \quad \int_{-\infty}^{\infty} p(x) \mathrm{d} x=1
$$

We define the cumulative distribution function:

$$
p(X \leq z)=P(z)=\int_{-\infty}^{z} p(x) \mathrm{d} x
$$

The earlier discrete definitions apply but with integration replacing summation.

- Expectation and Variance
- Density Estimation
- Basic Probability Distributions
- Maximum Likelihood

Density Estimation

- Bias
- Multivariate Normal
- Gaussian Mixture Models

Questions

## Density Estimation

Density estimation is the process of trying to infer the probability distribution $\boldsymbol{p}(\boldsymbol{X})$ given observations (data) $\mathcal{D}=\left\{x_{1}, \ldots, x_{N}\right\}$.
There are infinitely many density functions from which our data could have been sampled.

- Could be any function that assigns non-zero probability to the data.
- The choice of function is a modeling decision (inductive bias!).

Density estimation is a fundamental technique in machine learning and is part of most modern methods.

## Estimating the Parameters of a Distributionalize <br> 1924089

Given a collection of data $\mathcal{D}=\left\{x_{1}, \ldots, x_{N}\right\}$, there are several ways to estimate the "best" parameters for a given distribution.

- Maximum Likelihood Estimation (MLE): Choose the parameters that maximize the likelihood of the data. (Frequentist)

$$
w_{\text {MLE }}^{*}=\arg \max _{w} p(\mathcal{D} \mid w)
$$
- Maximum A Posteriori (MAP) Estimation: Choose the parameters that maximize the posterior given the data and the prior. (Bayesian)
$$
w_{\mathrm{MAP}}^{*}=\arg \max _{w} p(\mathcal{D} \mid w) p(w)
$$

Both methods rely on computing the likelihood of the data.

- We will focus on MLE today.


## The Likelihood Function

The likelihood function describes the probability of the observed data $\mathcal{D}=\left\{x_{1}, \ldots, x_{N}\right\}$ under a particular distribution.

If we assume each observation $X_{n} \sim p(x \mid w)$ is an Independent and Identically Distributed (IID) sample from some parametric distribution $p(x \mid w)$ parameterized by $w$, then the likelihood is:
![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-15.jpg?height=652&width=1511&top_left_y=1003&top_left_x=1054)

**Image Description:** The image presents a mathematical equation in LaTeX format, which is:

$$ p(\mathcal{D} | w) = \prod_{n=1}^{N} p(x_n | w) $$

This equation represents the likelihood of data \(\mathcal{D}\) given parameters \(w\), indicating that the individual data points \(x_n\) are independent and identically distributed (i.i.d.). The labels "Independent" and "Identically" suggest a focus on the assumptions regarding the statistical distribution of the data points.


## The Log Likelihood Function

In practice, we will typically work with the log likelihood function:

$$
p(\mathcal{D} \mid w)=\prod_{n=1}^{N} p\left(x_{n} \mid w\right) \Rightarrow \ln p(\mathcal{D} \mid w)=\sum_{n=1}^{N} \ln p\left(x_{n} \mid w\right)
$$

We take the log of the likelihood function for several reasons:

- It's easier to manipulate summations than products (e.g., calculus).
- The log probabilities and summation are more numerically stable.

Does this change our optimization problem?

- No - $\log$ is monotonically increasing function of its arguments:

$$
\arg \max _{w} p(\mathcal{D} \mid w)=\arg \max _{w} \ln p(\mathcal{D} \mid w)
$$

## Maximum Likelihood Estimation

Maximum likelihood estimation has two steps:

1. Modeling: Define the log likelihood function of the data: $\ln p(\mathcal{D} \mid w)$
2. Optimization: Compute the parameters that maximize the log-likelihood:

$$
\theta_{\mathrm{ML}}=\arg \max _{\theta} \log p(\mathcal{D} \mid w)
$$

Maximum likelihood estimation is a core technique in ML and is the basic objective for everything from the logistic regression to ChatGPT.

## Modeling: Bernoulli Distribution

![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-18.jpg?height=319&width=286&top_left_y=0&top_left_x=2619)

**Image Description:** This image depicts a metallic top, commonly used in games and demonstrations of rotational motion. The top features a circular disk at the top, with a conical base and a pointed tip that allows it to spin. The design emphasizes the shape necessary for stable rotation, illustrating principles of physics related to angular momentum and stability. The reflective surface highlights its material properties, serving as a visual representation of dynamics concepts in a lecture on rotational motion.


The distribution of binary random variable $X \in\{0,1\}$ defined by:

$$
X \sim \operatorname{Bern}(x \mid \mu) \quad \Longrightarrow \quad p(x \mid \mu)=\mu^{x}(1-\mu)^{1-x}
$$
where the parameter $0 \leq \mu \leq 1$.
\$\$\begin{gathered}

\mathbb{E}[X]=0(1-\mu)+1 \mu=\mu <br>
\operatorname{var}[X]=\mathbb{E}\left[X^{2}\right]-\mathbb{E}[X]^{2}=\mu-\mu^{2}=\mu(1-\mu) <br>
\mathbb{E}\left[X^{2}\right]=o^{2}(1-\mu)+1^{2} \mu=\mu

\end{gathered}\$\$

Note that for $\mu \notin\{0,1\}, X$ never takes the expected value!
The Bernoulli distribution is used extensively in classification.

## The MLE for IID Bernoulli Samples (Part 1)

## Example: Bernoulli Distribution

We observe $N$ samples $\mathcal{D}=\left\{x_{1}, \ldots, x_{N}\right\}$ from a Bernoulli Distribution.

1. Modeling: Let's assume IID Bernoulli trials:

$$
\ln p(\mathcal{D} \mid \mu)=\sum_{n=1}^{N} \ln \left(\mu^{x_{n}}(1-\mu)^{\left(1-x_{n}\right)}\right)=\sum_{n=1}^{N} x_{n} \ln (\mu)+\left(1-x_{n}\right) \ln (1-\mu)
$$

Since each $x_{n}$ is either zero or one we can rewrite this expression as:

$$
\ln p(\mathcal{D} \mid \mu)=n_{1} \ln (\mu)+n_{0} \ln (1-\mu)
$$

Where $n_{0}$ and $n_{1}$ are the number of observed zeros and ones, respectively.

- The counts $n_{0}$ and $n_{1}$ are called sufficient statistics because they fully describe the data.


## The MLE for IID Bernoulli Samples（Part 2）${ }^{\text {細教。 }}$ <br> 1924089

2．Optimization：Maximize the log likelihood function：

$$
\arg \max _{\mu} n_{1} \ln (\mu)+n_{0} \ln (1-\mu)
$$

Here we use the critical－points approach to optimization：
1．Take the derivative $\frac{\partial}{\partial \mu} \ln p(\mathcal{D} \mid \mu)$ ：

$$
\frac{\partial}{\partial \mu}\left(n_{1} \ln (\mu)+n_{0} \ln (1-\mu)\right)=\frac{n_{1}}{\mu}-\frac{n_{0}}{1-\mu}
$$

2．Solve for the critical point $\frac{\partial}{\partial \mu} \ln p(\mathcal{D} \mid \mu)=0$

$$
\frac{n_{1}}{\mu}-\frac{n_{0}}{1-\mu}=0 \Rightarrow n_{1}(1-\mu)=n_{0} \mu \Rightarrow \mu_{M L}=\frac{n_{1}}{n_{0}+n_{1}}
$$

3．Verify that critical points are maxima $\frac{\partial^{2}}{\partial \mu^{2}} \ln p(\mathcal{D} \mid \mu)<0$
$\alpha$
How can we mathematically formulate the problem of next-token prediction as a Maximum Likelihood Estimation (MLE) problem?

## The MLE for ChatGPT

We model the likelihood of all the documents on the internet:

$$
=\sum_{\text {Doc } \in \text { Internet }}^{\ln p(\text { Internet } \mid w)} \sum_{\text {Word } \in \text { Doc }} \ln p(\text { Word } \mid \text { Words before it in Doc, } w)
$$

Pre-Training: billions of dollars spent annually computing an MLE:

$$
\arg \max _{w} \ln p(\text { Interent } \mid w)
$$

## Normal (Gaussian) Distribution

If a single real-valued variable $X$ is normally distributed, then:

$$
X \sim \mathcal{N}\left(x \mid \mu, \sigma^{2}\right) \quad \Rightarrow \quad p\left(x \mid \mu, \sigma^{2}\right)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right)
$$

Where the parameters $\mu$ and $\sigma^{2}$ are called the mean and variance.

$$
\mathcal{N}\left(x \mid \mu,=0 \sigma^{2}=1\right)
$$

With some calculus you can show $\mathbb{E}[X]=\mu$ and $\operatorname{var}[X]=\sigma^{2}$.

## The MLE for IID Gaussian Samples (Part 1) <br> 1924089

## Example: Gaussian Distribution

We observe $N$ continuous real valued samples $\mathcal{D}=\left\{x_{1}, \ldots, x_{n}\right\}$.

1. Modeling: Let's assume IID Gaussian trials:

$$
\begin{gathered}
\ln p\left(\mathcal{D} \mid \mu, \sigma^{2}\right)=\sum_{n=1}^{N} \ln \left(\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\frac{1}{2 \sigma^{2}}\left(x_{n}-\mu\right)^{2}\right)\right) \\
=-\frac{N}{2} \ln (2 \pi)-\frac{N}{2} \ln \left(\sigma^{2}\right)-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(x_{n}-\mu\right)^{2}
\end{gathered}
$$

2. Optimization: Maximize the log likelihood function:

$$
\arg \max _{\mu, \sigma^{2}}-\frac{N}{2} \ln (2 \pi)-\frac{N}{2} \ln \left(\sigma^{2}\right)-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(x_{n}-\mu\right)^{2}
$$

## The MLE for IID Gaussian Samples (Part 2)

Maximize the log likelihood function:

$$
\arg \max _{\mu, \sigma^{2}}-\frac{N}{2} \ln (2 \pi)-\frac{N}{2} \ln \left(\sigma^{2}\right)-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(x_{n}-\mu\right)^{2}
$$

Compute derivatives with respect to $\mu$ and $\sigma^{2}$ and solve for the critical-points.

$$
\frac{\partial}{\partial \mu} \ln p\left(\mathcal{D} \mid \mu, \sigma^{2}\right)=\frac{1}{\sigma^{2}} \sum_{n=1}^{N}\left(x_{n}-\mu\right)
$$

Solving for $\mu_{\mathrm{ML}}$ :

$$
\frac{1}{\sigma^{2}} \sum_{n=1}^{N}\left(x_{n}-\mu\right)=0 \Longrightarrow \mu_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} x_{n}
$$

## The MLE for IID Gaussian Samples (Part 2)

Computing the derivative with respect to the variance parameter $v=\sigma^{2}$

$$
\frac{\partial}{\partial v} \ln p(\mathcal{D} \mid \mu, v)=\frac{\partial}{\partial v}\left(-\frac{N}{2} \ln (v)-\frac{1}{2 v} \sum_{n=1}^{N}\left(x_{n}-\mu\right)^{2}\right)=-\frac{N}{2}\left(\frac{1}{v}\right)+\frac{1}{2 v^{2}} \sum_{n=1}^{N}\left(x_{n}-\mu_{\mathrm{ML}}\right)^{2}
$$

Set equal to zero and solving for $v_{\text {ML }}=\sigma_{\text {ML }}^{2}$ :

$$
\begin{gathered}
-\frac{N}{2}\left(\frac{1}{v}\right)+\frac{1}{2 v^{2}} \sum_{n=1}^{N}\left(x_{n}-\mu_{\mathrm{ML}}\right)^{2}=0 \\
\frac{v^{2}}{N} * \quad N\left(\frac{1}{v}\right)=\frac{1}{v^{2}} \sum_{n=1}^{N}\left(x_{n}-\mu_{\mathrm{ML}}\right)^{2} \\
v_{\mathrm{ML}}=\sigma_{\mathrm{ML}}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left(x_{n}-\mu_{\mathrm{ML}}\right)^{2}
\end{gathered}
$$

## Recap of the MLE for IID Gaussian Sample

We used the method of Maximum Likelihood to estimate the mean and variance parameters of a Gaussian from data:

$$
\begin{gathered}
\mu_{\mathrm{ML}}=\frac{1}{N} \sum_{n=1}^{N} x_{n} \\
\sigma_{\mathrm{ML}}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left(x_{n}-\mu_{\mathrm{ML}}\right)^{2}=\frac{1}{N}\left(\sum_{n=1}^{N} x_{n}^{2}\right)-\mu_{\mathrm{ML}}^{2}
\end{gathered}
$$

What are the sufficient statistics?
Do they match our intuition for the mean and variance.
Are they good estimates?

- Expectation and Variance
- Density Estimation
- Basic Probability Distributions
- Maximum Likelihood
- Bias
- Multivariate Normal
- Gaussian Mixture Models

Questions

## Bias of an Estimate

How close is the true (unknown) parameter value to our estimate in expectation?

Before we observe our data, the value of our estimate $\theta_{\mathrm{ML}}$ is a random variable.

We can compute the bias of the estimation procedure as:

$$
\operatorname{Bias}\left(\theta_{\mathrm{ML}}\right)=\mathbb{E}\left[\theta_{\mathrm{ML}}\right]-\theta
$$

Relative to the true (but unknown) $\theta$.

## Bias of the MLE for the Gaussian Distribution

Assuming the data are drawn IID from $\mathcal{N}\left(x \mid \mu, \sigma^{2}\right)$ treating $X_{n}$, $\sqrt{\iota_{\text {ML }}}$, and $\overbrace{\text { EA }}^{2}$ as andom variables

$$
\mathbb{E}\left[\mu_{\mathrm{ML}}\right]=\frac{1}{N} \sum_{n=1}^{N} \mathbb{E}\left[X_{n}\right]=\frac{1}{N} \sum_{n=1}^{N} \mu=\mu
$$

Now looking at the variance:

## Variance

$$
\begin{aligned}
\mathbb{E}\left[\sigma_{\mathrm{ML}}^{2}\right]= & \frac{1}{N} \sum_{n=1}^{N} \mathbb{E}\left[X^{2}\right]-\mathbb{E}\left[\mu_{\mathrm{M}}^{2}\right] \\
& =\frac{1}{N} \sum_{n=1}^{N}\left(\sigma^{2}+u^{2}\right)-\left(\operatorname{var}\left(u_{n m}\right)+\mathbb{E}\left[\mu_{\mathrm{MI}}\right]^{2}\right) \\
& \operatorname{var}\left(\mu_{\mathrm{ML}}\right)=\operatorname{var}\left(\frac{1}{N} \sum_{n=1}^{N} X_{n}\right)=\frac{1}{N^{2}}(\underbrace{\left(\sum_{n=1}^{N} \operatorname{var}\left(X_{n}\right)\right)}_{\text {Independence of } X_{n}}=\frac{1}{N^{2}} N \sigma^{2}=\frac{\sigma^{2}}{N}
\end{aligned}
$$

## Bias of the MLE for the Gaussian Distribution

![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-31.jpg?height=99&width=3097&top_left_y=544&top_left_x=114)

$$
\mathbb{E}\left[\mu_{\mathrm{ML}}\right]=\frac{1}{N} \sum_{n=1}^{N} \mathbb{E}\left[X_{n}\right]=\frac{1}{N} \sum_{n=1}^{N} \mu=\mu
$$

Now looking at the variance:

$$
\begin{aligned}
\mathbb{E}\left[\sigma_{\mathrm{ML}}^{2}\right]= & \frac{1}{N} \sum_{n=1}^{N} \mathbb{E}\left[X^{2}\right]-\mathbb{E}\left[\mu_{\mathrm{M}}^{2}\right] \\
& =\frac{1}{N} \sum_{n=1}^{N}\left(\sigma^{2}+\mu^{2}\right)-\left(\operatorname{var}\left(\mu_{\mathrm{ML}}\right)+\mathbb{E}\left[\mu_{\mathrm{ML}}\right]^{2}\right) \\
& =\left(\sigma^{2}+\mu^{2}\right)-\left(\frac{\sigma^{2}}{N}+\mu^{2}\right)=\frac{\sigma^{2}\left(\frac{N-1}{N}\right)}{\text { Biased }}
\end{aligned}
$$

## Variance

Identity
$\operatorname{var}\left(\mu_{\mathrm{ML}}\right)=\mathbb{E}\left[\mu_{\mathrm{ML}}^{2}\right]-\mathbb{E}\left[\mu_{\mathrm{ML}}\right]^{2}$

## Sample Mean Variance <br> $\operatorname{var}\left(\mu_{\mathrm{ML}}\right)=\bar{N}$

How do we derive the formula for an unbiased estimator of the Gaussian variance?

- Expectation and Variance
- Density Estimation
- Basic Probability Distributions
- Maximum Likelihood


## Multivariate Normal

- Bias
- Multivariate Normal
- Gaussian Mixture Models

Questions

## Multivariate Gaussian Distribution

The probability density function for a multivariate Gaussian distribution in $D$ dimensions is give by:

$$
\mathcal{N}(\mathbf{x} \mid \mu, \Sigma)=\frac{1}{(2 \pi)^{D / 2}} \frac{1}{|\Sigma|^{1 / 2}} \exp \left(-\frac{1}{2}(\mathbf{x}-\mu)^{T} \Sigma^{-1}(\mathbf{x}-\mu)\right)
$$
where the parameters $\mu \in \mathbb{R}^{D}$ and $\Sigma \in \mathbb{R}^{D x D}$ are the mean vector and covariance matrix.
![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-34.jpg?height=838&width=1600&top_left_y=1024&top_left_x=1435)

**Image Description:** The image depicts a three-dimensional surface plot representing a function of two variables, \( x \) and \( y \). The axes are labeled \( x \), \( y \), and \( z \). The \( z \)-axis shows values ranging from 0 to 0.05, while the \( x \) and \( y \)-axes range from -5 to 5. The plot features a smooth, bell-shaped surface, with color gradation indicating the height: darker shades of blue represent lower values, and brighter yellows indicate higher values at the peak. A color bar on the right illustrates this gradient.


## Demo

## Gaussians

```
mu = np.array([1, 0])
Sigma = np.array([[3, 0.4], [0.4, 2]])
from scipy.stats import multivariate_normal
normal = multivariate_normal(mean=mu, cov=Sigma)
normal.pdf(np.array([[1, 0.5]]))
```

![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-35.jpg?height=724&width=779&top_left_y=939&top_left_x=1403)

**Image Description:** The image depicts a 3D surface plot illustrating a cone-shaped function. The axes are labeled with \( x \), \( y \), and \( z \), where \( x \) and \( y \) range from -5 to 5, and \( z \) represents the height of the function. The surface transitions from dark purple at lower values to bright yellow at its peak, indicating increasing values of the function as it approaches the apex of the cone. There are contour lines on the base that represent varying elevations of the function in the \( x-y \) plane.

![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-35.jpg?height=783&width=996&top_left_y=917&top_left_x=2258)

**Image Description:** The image is a contour plot representing a two-dimensional function. The axes are labeled with values ranging from -5 to 5 for both the x-axis and y-axis. The contours indicate levels of equal function value, with the innermost contour being yellow and the outer contours transitioning from green to dark blue, representing decreasing function values. A color bar on the right provides a gradient scale from yellow (higher values) to dark blue (lower values), indicating the intensity of the function across the plotted area.


- Expectation and Variance
- Density Estimation
- Basic Probability Distributions
- Maximum Likelihood


## Gaussian Mixture Models

- Bias
- Multivariate Normal
- Gaussian Mixture Models


## Questions

## Gaussian Mixture Model (GMM)

The Gaussian mixture model defines the probability of sampling a $D$-dimensional vector $\mathbf{x}$ as a weighted combination of Gaussians:

- Example: $K=3$
![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-37.jpg?height=677&width=3093&top_left_y=876&top_left_x=25)

**Image Description:** The image consists of two graphs depicting probability density functions of Gaussian distributions. 

1. The left graph shows three overlapping distributions in different colored curves (red, green, and blue) representing mixtures: $\pi_1, \mu_1, \sigma_1^2$, $\pi_2, \mu_2, \sigma_2^2$, and $\pi_3, \mu_3, \sigma_3^2$. The x-axis represents the variable $x$, while the y-axis represents the probability density. The colored dots indicate sample data points. 
2. The right graph consolidates the distributions into a single mixture model, with a darker purple curve illustrating the resultant combined density.



## Gaussian Mixture Model (GMM)

The Gaussian mixture model defines the probability of sampling a $D$-dimensional vector $\mathbf{x}$ as a weighted combination of Gaussians:

$$
p\left(\mathbf{x} \mid \pi, \mu_{1}, \Sigma_{1}, \ldots, \mu_{K}, \Sigma_{\mathrm{K}}\right)=\sum_{k=1}^{K} \pi_{k} \mathcal{N}\left(\mathbf{x} \mid \mu_{k}, \Sigma_{k}\right)
$$
where $\sum_{k=1}^{K} \pi_{k}=1$ is a probability distribution over clusters.
- Example: $K=3$
![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-38.jpg?height=592&width=1566&top_left_y=1207&top_left_x=1097)

**Image Description:** The image is a multivariate Gaussian mixture model diagram. It presents three Gaussian distributions, each represented by distinct colors: blue, red, and green. The x-axis ranges from -4 to 7 and denotes the variable of interest. The y-axis shows probability density, ranging from 0 to 0.3. The parameters for each Gaussian are labeled: π for weights, μ for means, and σ² for variances. The blue curve corresponds to the first distribution, the red to the second, and the green to the third, illustrating the composite nature of the mixture model.


## Demo

Gaussian Mixture Model
from sklearn.mixture import GaussianMixture
\# Create a Gaussian Mixture Model with 4 components
gmm = GaussianMixture(n_components=4, random_state=42, )
\# Fit the model to the data
gmm.fit(bikes[['Speed', 'Length']])
\# Get the cluster labels
bikes['scikit gmm'] = gmm.predict(bikes[['Speed', 'Length']]).astype(str)

1924089
![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-39.jpg?height=1319&width=1532&top_left_y=489&top_left_x=1445)

**Image Description:** The image is a contour plot displaying the distribution of data points grouped into clusters. The x-axis ranges from 5 to 20, while the y-axis ranges from 0 to 30. Each cluster is indicated by a series of concentric ellipses, with color gradients representing density levels: red indicates high density, transitioning through orange, green, and blue to purple, which represents low density. A color bar on the right provides a scale for density values from 0 to 0.035, facilitating interpretation of the clustering patterns.


## The GMM is a Latent Variable Model

We introduce a latent (unobserved) assignment variable $z$ for each $\mathbf{x}$.

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

If we marginalize over $z$ we obtain the GMM:

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
![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-41.jpg?height=448&width=175&top_left_y=790&top_left_x=2900)

**Image Description:** The image depicts a simple directed diagram. It features two circular nodes labeled "z" and "x," colored in yellow with bold blue lettering. A thick blue arrow points downward from the "z" node to the "x" node, indicating a directional relationship or transition from one variable to another. This visualization suggests a hierarchical or causal relationship, where "z" affects or leads to "x." The diagram conveys a clear, linear flow of information or influence between the two entities.

conditional structure of a probability model.
We can sample data from a graphical model using ancestor sampling.
- Sample all the variables that have no parents: $p(z)$
- Continue sampling variables after all their parents have been sampled: $p(\mathbf{x} \mid z)$


## Demo

Sampling from a GMM

```
# Ancestor Sampling to create a synthetic dataset
np.random.seed(42)
N = 100
mu = np.array([-1, 2, 5])
pi = np.array([0.2, 0.5, 0.3])
Sigma = np.array([0.2, 0.5, .1])
0.0s
```

```
z = np.random.choice(len(mu), size=N, p=pi)
x = np.random.normal(mu[z], np.sqrt(Sigma[z]))
```

![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-42.jpg?height=250&width=231&top_left_y=856&top_left_x=922)

**Image Description:** The image appears to be a snippet from a coding or programming environment, illustrating a function or command named `log_like`. It likely shows parameters related to logarithmic likelihood calculations, along with a check mark indicating successful execution and a time metric of 0.0 seconds. The design suggests that it is part of an academic lecture on statistical modeling or computational methods. The aesthetic includes a dark background typical of coding interfaces, emphasizing the text.


Synthetic Dataset from GMM (Log Likelihood: 181.82)
![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-42.jpg?height=731&width=1777&top_left_y=1022&top_left_x=1223)

**Image Description:** The image is a statistical graphical representation. It features a 2D plot with the x-axis labeled "x" ranging from -4 to 8 and the y-axis labeled "y" displaying probability density values. The plot contains three overlaid probability density functions (PDFs) represented by smooth curves: one in blue, one in red, and one in green. Additionally, there are scatter points: blue points correspond to the blue PDF, red points to the red PDF, and green points to the green PDF, illustrating data distribution associated with each PDF. The background is a light color, enhancing clarity.


True Cluster

- 0
- 1
- 2
- p0
- p 1
$-\mathrm{p} 2$


## Latent Variable Posteriors

We are interested in modeling the distribution (the uncertainty) over the cluster assignment $z$.

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
![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-45.jpg?height=665&width=2727&top_left_y=620&top_left_x=344)

**Image Description:** The image depicts a flow diagram illustrating the relationship between model parameters and cluster assignments in a clustering algorithm. It features a circular arrow connecting two labeled sections: “Model Parameters” on the left, which includes components \( \pi_k \) and \( \{ \mu_k, \Sigma_k \}_{k=1}^K \), and “Cluster Assignments” on the right, denoted by \( \{ z_n \}_{n=1}^N \). This representation emphasizes the iterative nature of the model's training process, where parameters adjust based on cluster assignments and vice versa.


If we knew the cluster assignments, we could easily estimate the model parameters.

## How can we solve this cyclic dependency?

## The EM Algorithm

The Expectation-Maximization (EM) algorithm is a standard technique for fitting latent variable models like the Gaussian Mixture Model.

1. Initialize the model parameters $\theta$.
2. Iterate until convergence.

- E-Step: Compute expected log-likelihood
![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-46.jpg?height=257&width=605&top_left_y=739&top_left_x=1973)

**Image Description:** The image features a speech bubble graphic with a blue background. Inside the bubble, the text includes the phrases "Easy to optimize" and "joint probability," with "joint probability" emphasized. The font is bold and large, enhancing the focus on the concept. The overall design suggests a simplified and accessible approach to understanding joint probability in the context of optimization, likely aimed at an educational audience.


$$
Q\left(\theta^{\prime}\right)=\sum_{Z} \ln \left(p\left(X, Z \mid \theta^{\prime}\right)\right) p(Z \mid X, \theta)
$$

Current distribution
over the latent Z

- M-Step: Maximize the expected log-likelihood.

$$
\theta=\arg \max _{\theta^{\prime}} Q\left(\theta^{\prime}\right)
$$

Updates distribution over Z.

## The EM Algorithm: E-step

E-Step: We don't know the value of $\left\{z_{n}\right\}_{n=1}^{\boldsymbol{N}}$ but we can compute

$$
\gamma_{n k}=p\left(z_{n}=k \mid \mathbf{x}_{n}, \pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}\right)=\frac{p\left(\mathbf{x}_{n} \mid z_{n}\right) p\left(z_{n}\right)}{\sum_{k^{\prime}=1}^{K} p\left(\mathbf{x}_{n} \mid k^{\prime}\right) p\left(k^{\prime}\right)}=\frac{\mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{z_{n}}, \Sigma_{z_{n}}\right) \pi_{z_{n}}}{\sum_{k^{\prime}=1}^{K} \mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{k^{\prime}}, \Sigma_{k^{\prime}}\right) \pi_{k^{\prime}}}
$$
- $\gamma_{n k}$ are "soft" assignments to each class.

Using the $\gamma_{n k}$ we can compute the expected log likelihood:

$$
\begin{aligned}
Q\left(\pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}\right) & =\sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k} \ln \left(p\left(\mathbf{x}_{n}, z_{n}=k \mid \pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}\right)\right)^{N} \\
& =\sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k}\left(\ln \left(\pi_{k}\right)+\ln \left(\mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{k}, \Sigma_{k}\right)\right)\right)
\end{aligned}
$$

## The EM Algorithm: M-Step ( $\pi_{k}$ )

M-Step: Compute the parameters that maximize $Q$ :

$$
Q\left(\pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}\right)=\sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k}\left(\ln \left(\pi_{k}\right)+\right.
$$

Optimizing with respect to $\pi_{k}$ :

$$
\begin{gathered}
\frac{\partial}{\partial \pi_{k}} \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k}\left(\ln \left(\pi_{k}\right)+\ln \left(\mathcal{N}\left(\mathbf{x}_{n} \mid \mu_{k}, \Sigma_{k}\right)\right)\right)+\lambda\left(\sum_{k=1}^{K} \pi_{k}-1\right)=\sum_{n=1}^{N} \frac{\gamma_{n k}}{\pi_{k}}+\lambda=0 \\
\pi_{k}=-\frac{1}{\lambda} \sum^{N} \gamma_{n k}
\end{gathered}
$$

$$
\begin{aligned}
& 1=\sum_{k=1}^{K} \pi_{k}=-\frac{1}{\lambda} \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k}=-\frac{1}{\lambda} \sum_{n=1}^{N} 1 \\
& =-\frac{N}{\lambda} \Rightarrow \lambda=-N
\end{aligned}
$$

$$
N_{k}=\sum_{n=1}^{N} \gamma_{n k}
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

1. Initialize the model parameters $\pi,\left\{\mu_{k}, \Sigma_{k}\right\}_{k=1}^{K}$ (often using k-means)
2. Iterate until convergence

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

Implement GMMs
from sklearn.mixture import GaussianMixture
\# Create a Gaussian Mixture Model with 4 components
gmm = GaussianMixture(n_components=4, random_state=42, )
\# Fit the model to the data
gmm.fit(bikes[['Speed', 'Length']])
\# Get the cluster labels
bikes['scikit gmm'] = gmm.predict(bikes[['Speed', 'Length']]).astype(str)

1924089
![](https://cdn.mathpix.com/cropped/2025_10_01_941a8e9b9882dfb49e2dg-52.jpg?height=1319&width=1532&top_left_y=489&top_left_x=1445)

**Image Description:** The image is a contour plot displaying multiple data clusters. The x-axis ranges from 5 to 20, while the y-axis extends from 0 to 30. Each cluster is represented by a set of colored ellipses with varying densities, indicated by a color gradient from green to red. The clusters are marked by scattered points in corresponding colors (green, red, purple) based on their density. The color bar on the right quantifies the density values, ranging from approximately 0.005 to 0.035, representing the concentration of points within each cluster.


## Lecture 5

## Density Estimation and Gaussian Mixture Models

Credit: Joseph E. Gonzalez and Narges Norouzi
Reference Book Chapters:

- Probability: Chapter 2.[1-2], 2.3.[1-3] (we will return to 2.4 onward later)
- Distributions: Chapter 3.1, 3.2.1 (the rest of 3 is more advanced Gaussians)
- Clustering: Chapter 15.1 (k-means), 15.2 (Gaussian Mixture Models)

