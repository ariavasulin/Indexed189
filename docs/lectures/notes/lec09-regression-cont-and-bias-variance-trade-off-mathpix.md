---
course: CS 189
semester: Fall 2025
type: lecture
title: Regression (Cont.) and Bias-Variance Trade-off
source_type: slides
source_file: Lecture 09 -- Regression (Cont.) and Bias-Variance Trade-off.pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Lecture 9

## Regression (Cont.) and Bias-Variance Trade-off

More on probabilistic view of regression and bias-variance trade-off

EECS 189/289, Fall 2025 @ UC Berkeley
Joseph E. Gonzalez and Narges Norouzi

# III Join at slido.com <br> '1-1 \#2312298 

## Roadmap

- MLE Recap: Least Squares as Maximum Likelihood
- Choosing Different Noise Models
- Prior Beliefs
- Bias-Variance Trade-off


## MLE Recap: Least Squares as Maximum Likelihood

- MLE Recap: Least Squares as Maximum Likelihood
- Choosing Different Noise Models
- Prior Beliefs
- Bias-Variance Trade-off


## Least Squares $\cong$ Maximum Likelihood

$$
p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)=\prod_{n=1}^{N} \mathcal{N}\left(t_{n} \mid w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right), \sigma^{2}\right)
$$

The optimum value of $w$ in the above equation equals the optimum value of the log of the function (because log is a monotonically increasing function)

$$
\begin{array}{rlr}
\ln \left(p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)\right) & =\sum_{n=1}^{N} \ln \left(\mathcal{N}\left(t_{n} \mid \sqrt{w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)} \sigma^{2}\right)\right) & \\
& \left.=\sum_{n=1}^{N} \ln \left(\frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-\frac{1}{2 \sigma^{2}}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right.}\right)^{2}\right) & \begin{array}{l}
\text { Writing the equation for a } \\
\text { Normal distribution }
\end{array} \\
& =\sum_{n=1}^{N}\left(\ln \left(\frac{1}{\sqrt{2 \pi \sigma^{2}}}\right)-\frac{1}{2 \sigma^{2}}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2}\right) & \text { Applying } \log a \times b=\log a+\log b \\
\ln \left(p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)\right) & =-\frac{N}{2} \ln \left(2 \pi \sigma^{2}\right)-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2} & =-\frac{N}{2} \ln \left(2 \pi \sigma^{2}\right)-\frac{1}{2} E_{D}(w)
\end{array}
$$

## Least Squares $\cong$ Maximum Likelihood

$$
\begin{aligned}
& p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)=\prod_{n=1}^{N} \mathcal{N}\left(t_{n} \mid w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right), \sigma^{2}\right) \\
& \ln \left(p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)\right)=-\frac{N}{2} \ln \left(2 \pi \sigma^{2}\right)-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right)\right)^{2}=-\frac{N}{2} \ln \left(2 \pi \sigma^{2}\right)-\frac{1}{2} E_{D}(w) \\
& \nabla_{\mathrm{w}} \ln \left(p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)\right)=-\frac{1}{\sigma^{2}} \nabla_{\mathrm{w}} \mathrm{E}_{\mathrm{D}}(\mathrm{w}) \longrightarrow w_{M L}=\left(\Phi(\mathbb{X})^{T} \Phi(\mathbb{X})\right)^{-1} \Phi(\mathbb{X})^{T} \mathrm{t} \\
& \Phi(\mathbb{X})=\left[\begin{array}{c}
\phi\left(\mathrm{x}_{1}\right)^{T} \\
\phi\left(\mathrm{x}_{2}\right)^{T} \\
\cdots\left(\mathrm{x}_{\mathrm{N}}\right)^{T}
\end{array}\right]
\end{aligned}
$$

The least-squares solution is the MLE under Gaussian noise.

## Choosing Different Noise Models

- MLE Recap: Least Squares as Maximum Likelihood
- Choosing Different Noise Models
- Prior Beliefs
- Bias-Variance Trade-off


## Noise Model $\Longleftrightarrow$ Error Function

## Zero-mean Gaussian Noise

$p(\epsilon)=\frac{1}{\sqrt{2 \pi} \sigma} \exp \left(-\frac{\epsilon^{2}}{2 \sigma^{2}}\right)$
$-\log p(\epsilon)=\frac{\epsilon^{2}}{2 \sigma^{2}}+$ const
$\Rightarrow$ Sum of the squares error
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-08.jpg?height=779&width=916&top_left_y=1071&top_left_x=416)

**Image Description:** The diagram displays probability density functions (PDFs) for Gaussian and Laplace distributions. The x-axis represents the value of the random variable, spanning from -4 to 4, while the y-axis indicates the probability density, ranging from 0 to 1. The Gaussian PDF is depicted with a blue line, showing a bell-shaped curve centered at 0, with a standard deviation that influences its width. The Laplace PDF is represented by an orange line, featuring a sharper peak and wider tails compared to the Gaussian. This visual comparison illustrates differences in distribution characteristics.


## Zero-mean Laplacian Noise

$p(\epsilon)=\frac{1}{2 b} \exp \left(-\frac{|\epsilon|}{b}\right)$
$-\log p(\epsilon)=\frac{|\epsilon|}{b}+$ const
$\Rightarrow$ Sum of the Absolute error
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-08.jpg?height=770&width=885&top_left_y=1080&top_left_x=1824)

**Image Description:** The image is a diagram showing two loss functions: Mean Squared Error (MSE) and Mean Absolute Error (MAE). The graph has an x-axis representing the error values, ranging from -4 to 4, and a y-axis representing the corresponding loss values, ranging from 0 to 6. The blue curve represents MSE, which has a parabolic shape, while the orange line represents MAE, which is linear and intersects the MSE curve at the origin (0,0). A legend in the top right identifies MSE with a solid blue line and MAE with a solid orange line.


- MLE Recap: Least Squares as Maximum
Likelihood
- Choosing Different Noise Models
- Prior Beliefs
- Bias-Variance Trade-off


## Recall: Beliefs and Priors

In the coin flip example, assume the true probability of head is $\theta$.

Frequentist: Use data only (MLE).

$$
\hat{\theta}=\frac{\text { \#heads }}{\text { \# flips }}
$$

Flip twice, see HH , then:

$$
\hat{\theta}_{M L E}=\frac{2}{2}=1
$$

Overconfident with $n=2!!!$

Bayesian: Combine data with belief.

- Prior belief: $h_{0}=t_{0}=5$
- See HH:
- Posterior $p(\theta \mid H H) \propto p(\theta) \times$ likelihood


## Strong prior ensures alignment with beliefs

## What About Regression? MLE for Weights: No Prior

- In regression, MLE finds weights w by maximizing likelihood of data under noise model.
- Problem: with few/noisy data points $\rightarrow$ can fit extreme weights.
- No mechanism to encode "weights should be small/reasonable."
- Why small?
- With inputs in SD units we don't expect huge effects unless data is persuasive.
- Shrinking $w$ lowers prediction variance, especially when features are correlated.


## Belief about Parameters

- Just like coins: we can encode a prior belief about weights $w$.
- Common prior: $w_{j} \sim \mathcal{N}\left(0, \sigma_{w}^{2}\right)$
- Centered at $0 \rightarrow$ no strong effect unless data supports it.
- Variance $\sigma_{w}^{2}$ controls how strongly we expect small weights.
- This prior = belief that simpler models are preferred.

Likelihood function
$p\left(\mathrm{t} \mid \mathbb{X}, w, \sigma^{2}\right)=\prod_{n=1}^{N} \mathcal{N}\left(t_{n} \mid w^{T} \boldsymbol{\phi}\left(\mathrm{x}_{\mathrm{n}}\right), \boldsymbol{\sigma}^{2}\right)$

Prior (small, centered)

$$
\mathrm{p}\left(\mathrm{w} \mid \tau^{2}\right)=\frac{1}{\left(2 \pi \tau^{2}\right)^{\frac{M}{2}}} e^{-\frac{\|w\|^{2}}{2 \tau^{2}}}
$$

Posterior (from Bayes rule) up to a constant:

$$
p(\mathrm{w} / \mathrm{t}) \propto
$$

## What Is MAP?

Bayes Rule
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-13.jpg?height=639&width=2111&top_left_y=497&top_left_x=250)

**Image Description:** The image contains a mathematical equation showing the relationship of conditional probability in a context of Bayesian inference. The equation is expressed as:

$$
p(w | t) \propto \prod_{n=1}^{N} \mathcal{N}(t_n | w^T \phi(x_n), \sigma^2) \cdot \frac{1}{(2\pi)^{M/2}} e^{-\frac{||w||^2}{2\sigma^2}}
$$

It is dissected into components labeled "Likelihood" and "Prior," indicating their roles in the equation's context. The diagram visually emphasizes the contributions of likelihood and prior in Bayesian modeling.


Maximum A Posteriori (MAP):

$$
\widehat{w}_{M A P}=\underset{w}{\operatorname{argmax}} p(\mathrm{w} / \mathrm{t})=\underset{w}{\operatorname{argmax}}[p(\mathrm{t} \mid \mathrm{w}) \times p(w)]
$$

## What Is MAP?

Maximum A Posteriori (MAP):

$$
\widehat{w}_{M A P}=\underset{w}{\operatorname{argmax}} p(\mathrm{w} / \mathrm{t})=\underset{w}{\operatorname{argmax}}[p(\mathrm{t} \mid \mathrm{w}) \times p(w)]
$$

Taking $-\log$ of the objective turns maximization problem to a minimization problem:

$$
\widehat{w}_{M A P}=\underset{w}{\operatorname{argmin}}[-\ln p(\mathrm{t} \mid \mathrm{w})-\ln p(w)]
$$

## Plugging in Distributions

$$
\begin{aligned}
\widehat{W}_{M A P} & =\underset{w}{\operatorname{argmin}}[\underbrace{-\ln p(\mathrm{t} \mid \mathrm{w})} \underbrace{-\ln p(w)}_{-\ln \frac{1}{\left(2 \pi \tau^{2}\right)^{\frac{M}{2}}}}] \\
& -\ln \prod_{n=1}^{N} \mathcal{N}\left(t_{n} \mid w^{T} \phi\left(\mathrm{x}_{\mathrm{n}}\right), \sigma^{2}\right) \\
= & \underbrace{}_{\frac{1}{2 \sigma^{2}} \sum_{n}\left(t_{n}-y(\phi(\mathrm{x}), w)\right)^{2}}={ }_{\frac{\|w\|^{2}}{2 \tau^{2}}}+\text { const } \\
\widehat{W}_{M A P} & =\underset{w}{\operatorname{argmin}}\left[\frac{1}{2 \sigma^{2}} \sum_{n}\left(t_{n}-y(\phi(\mathrm{x}), w)\right)^{2}+\frac{\|w\|^{2}}{2 \tau^{2}}\right.
\end{aligned}
$$

## Does This Looks Like a Ridge Regression?

$$
\widehat{w}_{M A P}=\underset{w}{\operatorname{argmin}}\left[\frac{1}{2 \sigma^{2}} \sum_{n}\left(t_{n}-y(\phi(\mathrm{x}), w)\right)^{2}+\frac{\|w\|^{2}}{2 \tau^{2}}\right.
$$

Multiply by $\sigma^{2}$

$$
\widehat{w}_{M A P}=\underset{w}{\operatorname{argmin}}[\underbrace{\frac{1}{2} \sum_{n}\left(t_{n}-y(\phi(\mathrm{x}), w)\right)^{2}}_{\frac{1}{2} E_{D}(w)}+\underbrace{\frac{\sigma^{2}\|w\|^{2}}{2 \tau^{2}}}_{E_{w}(w)} \lambda
$$

Least-Squares + Gaussian Prior $\Rightarrow$ Ridge Regression

# What is the relationship between MAP and ridge regression? 

## Bias-Variance Trade-off

- MLE Recap: Least Squares as Maximum Likelihood
- Choosing Different Noise Models
- Prior Beliefs
- Bias-Variance Trade-off


## Fundamental Challenges in Learning?

- Fit the Data
- Provide an explanation for what we observe
- Generalize to the World
- Predict the future
- Explain the unobserved

Is this cat grumpy or are we overfitting to human faces?
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-19.jpg?height=1378&width=1026&top_left_y=489&top_left_x=2304)

**Image Description:** This image is a photograph of a cat known for its distinct grumpy expression. The cat features white fur with brown patches, bright blue eyes, and a flat face with prominent whiskers. The background is a solid, light blue, which contrasts with the cat's fur color. The cat is positioned with its front paws visible, resting on a surface, and its expression conveys a sense of displeasure or annoyance. The overall composition is well-lit, highlighting the cat's features effectively.


## Fundamental Challenges in Learning?

- Bias: The expected deviation between the predicted value and the true value.
- Variance: Two sources
- Noise: The variability of the random noise in the process we are trying to model.
- Model Variance: The variability in the predicted value across different training datasets.


## Bias

The expected deviation between the predicted value and the true value

- Depends on both the:
- Choice of $f$
- Learning procedure
- Under-fitting

All possible functions
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-21.jpg?height=911&width=1302&top_left_y=880&top_left_x=1958)

**Image Description:** The image presents a diagram illustrating the concept of bias in a function \( f_w \). It features an oval shape labeled "Possible \( w \) values" containing the notation \( f_w \) in yellow. A yellow point represents "Bias," which is connected by an arrow to a black point labeled "True Function." This visual emphasizes the relationship between the possible values of \( w \) and their impact on bias regarding the true function being modeled. The overall layout suggests a flow from potential model parameters to their influence on bias.


![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-21.jpg?height=716&width=1192&top_left_y=1092&top_left_x=569)

**Image Description:** The image is a scatter plot displaying the relationship between two variables. The x-axis ranges from approximately -10 to 10, while the y-axis ranges from 0 to 40. Blue dots represent individual data points, indicating a positive correlation between the variables. A linear regression line, depicted in orange, illustrates the trend, with a slope indicating the strength of the relationship. The line appears to fit through the data, showing a general upward trend as x increases. The title "Fitting" suggests the focus on regression analysis or model fitting.


## Noise

The variability of the random noise in the process we are trying to model.

- Measurement variability
- Stochasticity
- Missing information


## Beyond our control (usually)

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-22.jpg?height=1043&width=1638&top_left_y=629&top_left_x=1543)

**Image Description:** The image depicts a scatter plot overlaid with a smooth curve. The x-axis ranges from -10 to 10, while the y-axis ranges approximately from 0 to 50. Blue dots represent individual data points scattered across the plot. An orange line indicates a fitted curve, suggesting a non-linear relationship between the x and y values. The curve reflects fluctuating trends in the data, possibly indicating periodic behavior. This visual representation is commonly used in statistical analysis to show the relationship between variables and the goodness of fit for the model.


## Model Variance

Variability in the predicted value across different training datasets.

- Sensitivity to variation in the training data
- Poor generalization
- Overfitting
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-23.jpg?height=971&width=1698&top_left_y=782&top_left_x=1522)

**Image Description:** The image is a scatter plot overlaid with multiple smoothing curves. The x-axis ranges from -10 to 10, while the y-axis displays values from approximately -20 to 20. Blue points represent data observations, and orange lines indicate fitted smoothing functions, illustrating trends in the data. The varying smooth curves demonstrate how different models can fit the same dataset, showcasing the relationship between the independent and dependent variables. The presence of oscillation suggests complexity in the underlying data pattern.



# Which of the following models would have high bias? 

# Which of the following models would have high variance? 

## Analysis of Squared Error

- For the test point X the expected error:
- Random variables are red

Assume noisy observations
$\rightarrow t$ is a random variable

True
Function
$t=h(\mathrm{x})+\epsilon$

Noise term:

$$
\begin{gathered}
\mathrm{E}(\epsilon)=0 \\
\operatorname{Var}(\epsilon)=\sigma^{2}
\end{gathered}
$$

$$
\mathrm{E}\left[\left(t-f_{w}(\mathrm{x})\right)^{2}\right]
$$

Can be any parametric
function

Assume training data is random
$\rightarrow w$ is a random variable

## Analysis of Squared Error

Goal:

$$
\mathrm{E}\left[\left(t-f_{w}(\mathrm{x})\right)^{2}\right]
$$
"Noise" $+(\text { Bias })^{2}+$ Model Variance
$$
\mathrm{E}\left[\left(t-f_{w}(\mathrm{x})\right)^{2}\right]=\mathrm{E}[(t-\underbrace{h(\mathrm{x})+h(\mathrm{x})}_{\text {Subtracting and adding } h(\mathrm{x})}-f_{w}(\mathrm{x}))^{2}]
$$

Useful Equations:

$$
\begin{gathered}
t=h(\mathrm{x})+\epsilon \\
\mathrm{E}(\epsilon)=0 \\
\operatorname{Var}(\epsilon)=\sigma^{2} \\
\hline
\end{gathered}
$$

$$
\mathrm{E}\left[\left(t-f_{w}(\mathrm{x})\right)^{2}\right]=\mathrm{E}[(\underbrace{t-h(\mathrm{x})}_{a}+\underbrace{h(\mathrm{x})-f_{w}(\mathrm{x})}_{b})^{2}]
$$

$$
\begin{aligned}
& \text { Expanding in terms of } a \text { and } b \text { : } \\
& a^{2} \\
& (a+b)^{2}=a^{2}+b^{2}+2 a b \\
& b^{2} \\
& =\mathrm{E}\left[(t-h(\mathrm{x}))^{2}\right]+\mathrm{E}\left[\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)^{2}\right] \\
& +2 \mathrm{E}[\underbrace{(\underbrace{t-h(\mathrm{x})}_{\epsilon})}_{\epsilon} \underbrace{}_{2 a b}\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)] \\
& \text { Useful Equations: } \\
& t=h(\mathrm{x})+\epsilon \\
& \mathrm{E}(\epsilon)=0 \\
& +2 \mathrm{E}\left[\epsilon\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)\right] \\
& \operatorname{Var}(\epsilon)=\sigma^{2}
\end{aligned}
$$

$$
\mathrm{E}\left[\left(t-f_{w}(\mathrm{x})\right)^{2}\right]=\mathrm{E}[(\underbrace{t-h(\mathrm{x})}_{a}+\underbrace{h(\mathrm{x})-f_{w}(\mathrm{x})}_{b})^{2}]
$$

Expanding in terms of $a$ and $b$ :

$$
(a+b)^{2}=a^{2}+b^{2}+2 a b
$$

$$
\begin{aligned}
= & \mathrm{E}\left[(t-h(\mathrm{x}))^{2}\right]+\mathrm{E}\left[\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)^{2}\right] \\
& +2 \mathrm{E}\left[\epsilon\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)\right]
\end{aligned}
$$

Independence of $\epsilon$ and $w$

$$
+2 \underbrace{\mathrm{E}[\epsilon]}_{0} \mathrm{E}\left[\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)\right]
$$

Useful Equations:

$$
\begin{gathered}
t=h(\mathrm{x})+\epsilon \\
\mathrm{E}(\epsilon)=0 \\
\operatorname{Var}(\epsilon)=\sigma^{2} \\
\hline
\end{gathered}
$$

$$
\mathrm{E}\left[\left(t-f_{w}(\mathrm{x})\right)^{2}\right]
$$

## "Noise" Term

Model
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-31.jpg?height=511&width=1200&top_left_y=1135&top_left_x=238)

**Image Description:** The image features a mathematical expression representing an expected value function. It shows the difference between a true value function \( h(x) \) and a predicted value function \( f_W(x) \), squared, with the expectation denoted by \( E \). The left side highlights "True Value" and the right side "Pred. Value" in speech bubbles. The equation is formatted as:

$$
E\left[ (h(x) - f_W(x))^2 \right]
$$

This formula is commonly used in statistical learning or regression analysis to measure prediction error.

Estimation Error

Useful Equations:

$$
\begin{gathered}
t=h(\mathrm{x})+\epsilon \\
\mathrm{E}(\epsilon)=0 \\
\operatorname{Var}(\epsilon)=\sigma^{2} \\
\hline
\end{gathered}
$$

$$
\begin{aligned}
& \mathrm{E}\left[\left(t-f_{w}(\mathrm{x})\right)^{2}\right] \\
& =\mathrm{E}\left[(t-h(\mathrm{x}))^{2}\right]+\sigma^{2} \\
& \underbrace{\mathrm{E}\left[\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)^{2}\right]}_{\begin{array}{c}
\text { We need to } \\
\text { calculate this term }
\end{array}} \text { "Noise" Term } \\
& \begin{array}{l}
\text { Model } \\
\text { Estimation }
\end{array}
\end{aligned}
$$

Useful Equations:

$$
\begin{gathered}
t=h(\mathrm{x})+\epsilon \\
\mathrm{E}(\epsilon)=0 \\
\operatorname{Var}(\epsilon)=\sigma^{2} \\
\hline
\end{gathered}
$$

$$
\begin{aligned}
& \mathrm{E}\left[\left(t-f_{w}(\mathrm{x})\right)^{2}\right] \\
& =\mathrm{E}\left[(t-h(\mathrm{x}))^{2}\right]+\sigma^{2} \\
& \underbrace{\mathrm{E}\left[\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)^{2}\right]}_{\begin{array}{c}
\text { We need to } \\
\text { calculate this term }
\end{array}} \text { "Noise" Term } \\
& \begin{array}{l}
\text { Estimation } \\
\text { Error }
\end{array}
\end{aligned}
$$

Useful Equations:

$$
\begin{gathered}
t=h(\mathrm{x})+\epsilon \\
\mathrm{E}(\epsilon)=0 \\
\operatorname{Var}(\epsilon)=\sigma^{2} \\
\hline
\end{gathered}
$$

$\mathrm{E}\left[\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)^{2}\right]=$ Next we will show....
$\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right)^{2}+\mathrm{E}\left[\left(\mathrm{E}\left[f_{w}(\mathrm{x})\right]-f_{w}(\mathrm{x})\right)^{2}\right]$
(Bias)² Model Variance
-How?
-Adding and subtracting what?

$$
\begin{aligned}
& \mathrm{E}\left[\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)^{2}\right]= \\
& \quad \mathrm{E}[\underbrace{\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right.}_{c}+\underbrace{\mathrm{E}\left[f_{w}(\mathrm{x})\right]-f_{w}(\mathrm{x})}_{b})^{2}]
\end{aligned}
$$

Expanding in terms of $a$ and $b$ :

$$
(a+b)^{2}=a^{2}+b^{2}+2 a b
$$

$$
\begin{gathered}
\mathrm{E}\left[\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right)^{2}\right]+\mathrm{E}\left[\left(\mathrm{E}\left[f_{w}(\mathrm{x})\right]-f_{w}^{2}(\mathrm{x})\right)^{2}\right]+ \\
2 \mathrm{E}\left[\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right)\left(\mathrm{E}\left[f_{w}(\mathrm{x})\right]-f_{w}(\mathrm{x})\right)\right] \\
2 a b
\end{gathered}
$$

$\mathrm{E}\left[\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)^{2}\right]=$
$\mathrm{E}\left[\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right)^{2}\right]+\mathrm{E}\left[\left(\mathrm{E}\left[f_{w}(\mathrm{x})\right]-f_{w}(\mathrm{x})\right)^{2}\right]+$
$2 \mathrm{E}\left[\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right)\left(\mathrm{E}\left[f_{w}(\mathrm{x})\right]-f_{w}(\mathrm{x})\right)\right]$
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-36.jpg?height=383&width=2186&top_left_y=761&top_left_x=306)

**Image Description:** The image depicts an equation excerpt involving a variable \( W(x) \) and its relationship to constants and derivatives. Centered in the bottom half, the word "constant" is labeled, suggesting that the equation derives a value that is invariant across the function's domain. The equation features multiple elements, including parentheses and multiplication, indicating interactions between different variable functions of \( x \). The notation appears to involve calculus concepts such as differentiation or integration, though the specifics of the relationship are not fully visible in this excerpt.


$$
2\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right) \mathrm{E}\left[\left(\mathrm{E}\left[f_{w}(\mathrm{x})\right]-f_{w}(\mathrm{x})\right)\right]
$$
$\mathrm{E}\left[\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)^{2}\right]=$
$\mathrm{E}\left[\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right)^{2}\right]+\mathrm{E}\left[\left(\mathrm{E}\left[f_{w}(\mathrm{x})\right]-f_{w}(\mathrm{x})\right)^{2}\right]+$
$2\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right) \mathrm{E}[(\underbrace{\mathrm{E}\left[f_{w}(\mathrm{x})\right]}_{\text {constant }}-f_{w}(\mathrm{x}))]$
$2\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right)(\underbrace{\mathrm{E}\left[f_{w}(\mathrm{x})\right]-\mathrm{E}\left[f_{w}(\mathrm{x})\right]}_{\mathbf{0}})$
\$\$\begin{aligned}

\& \mathrm{E}\left[\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)^{2}\right]= <br>
\& \mathrm{E}[\underbrace{\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right.}_{constant})^{2}]+\mathrm{E}\left[\left(\mathrm{E}\left[f_{w}(\mathrm{x})\right]-f_{w}(\mathrm{x})\right)^{2}\right]

\end{aligned}\$\$

$$
\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right)^{2}+
$$

$$
\begin{array}{cc}
\mathrm{E}\left[\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)^{2}\right]= & \\
\begin{array}{cc}
\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right)^{2} & +\mathrm{E}\left[\left(\mathrm{E}\left[f_{w}(\mathrm{x})\right]-f_{w}(\mathrm{x})\right)^{2}\right] \\
(\text { Bias })^{2} & \text { Model Variance }
\end{array}
\end{array}
$$

$$
\begin{aligned}
& \mathrm{E}\left[\left(t-f_{w}(\mathrm{x})\right)^{2}\right] \\
& =\mathrm{E}\left[(t-h(\mathrm{x}))^{2}\right]+\sigma^{2} \\
& \underbrace{\mathrm{E}\left[\left(h(\mathrm{x})-f_{w}(\mathrm{x})\right)^{2}\right]}_{\begin{array}{c}
\text { We now have } \\
\text { calculated this }
\end{array}} \text { "Noise" Term }_{\text {Estimation }}^{\text {Error }}
\end{aligned}
$$

Useful Equations:

$$
\begin{array}{|c|}
\hline t=h(\mathrm{x})+\epsilon \\
\mathrm{E}(\epsilon)=0 \\
\operatorname{Var}(\epsilon)=\sigma^{2} \\
\hline
\end{array}
$$

$$
\begin{aligned}
& \mathrm{E}\left[\left(t-f_{w}(\mathrm{x})\right)^{2}\right] \\
& =\mathrm{E}\left[(t-h(\mathrm{x}))^{2}\right]+\sigma^{2} \\
& \quad\left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right)^{2}+\text { (Bias) }^{2} \\
& \quad \mathrm{E}\left[\left(\mathrm{E}\left[f_{w}(\mathrm{x})\right]-f_{w}(\mathrm{x})\right)^{2}\right] \quad \begin{array}{l}
\text { Model Term } \\
\text { Variance }
\end{array} \begin{array}{|c|}
\hline \begin{array}{c}
\text { Useful Equations: } \\
t=h(\mathrm{x})+\epsilon \\
\mathrm{E}(\epsilon)=0 \\
\operatorname{Var}(\epsilon)=\sigma^{2}
\end{array}
\end{array}
\end{aligned}
$$

$$
\begin{aligned}
& \mathrm{E}\left[\left(t-f_{w}(\mathrm{x})\right)^{2}\right] \\
& =\mathrm{E}\left[(t-h(\mathrm{x}))^{2}\right]+\sigma^{2} \\
& \left(h(\mathrm{x})-\mathrm{E}\left[f_{w}(\mathrm{x})\right]\right)^{2}+\text { "Noise" Term } \\
& \mathrm{E}\left[\left(\mathrm{E}\left[f_{w}(\mathrm{x})\right]-f_{w}(\mathrm{x})\right)^{2}\right] \quad \begin{array}{c}
\text { Model } \\
\text { Variance }
\end{array}
\end{aligned}
$$

## Bias Variance Plot

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-43.jpg?height=1345&width=2246&top_left_y=386&top_left_x=761)

**Image Description:** The diagram illustrates the bias-variance tradeoff in machine learning. It shows two curves representing "Test Error" (black) and "Bias² + Variance" (purple), plotted against "Model Complexity" on the horizontal axis. The vertical axis denotes the error values, with a point labeled as "Optimal Value" indicating the ideal model complexity that minimizes test error. The curve representing bias (blue) decreases as model complexity increases, while variance (purple) rises. The intersection of these curves signifies the point of minimal error.


## More Data Supports More Complexity

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-44.jpg?height=1358&width=2400&top_left_y=382&top_left_x=752)

**Image Description:** The diagram is a graphical representation of the bias-variance tradeoff in machine learning. It features two primary curves: one representing bias squared (green) and the other representing variance (blue), both plotted against decreasing model complexity on the x-axis. The y-axis indicates the test error. A horizontal black line shows the optimal value of test error, while vertical dashed lines denote the transition points. The curves illustrate how test error changes with model complexity, indicating the balance between bias and variance. The optimal point minimizes total error.


## How Do We Control Model Complexity?

- So far:
- Number of features
- Choices of features
- Regularization
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-45.jpg?height=1366&width=2127&top_left_y=501&top_left_x=1203)

**Image Description:** The diagram illustrates the relationship between model complexity and errors in machine learning. The x-axis represents "Decreasing Model Complexity," while the y-axis indicates "Test Error." It features three curves: a black curve labeled "Test Error," a purple curve indicating "Variance," and a blue curve representing "(Bias)²." The curves depict how test error decreases with increased complexity, while variance increases, and bias decreases. An optimal value is indicated by a dashed vertical line, highlighting the trade-off between bias and variance as model complexity changes.



## Determining the Optimal $\lambda$

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-46.jpg?height=1434&width=2076&top_left_y=0&top_left_x=1254)

**Image Description:** The image is a graph showing the relationship between error and the parameter \(\lambda\). The x-axis is labeled "increasing \(\lambda\)" and the y-axis is labeled "Error." It features three curves: 

1. Validation Error (purple) decreases with increasing \(\lambda\).
2. Test Error (black) decreases, reaching a minimum at an optimal point.
3. Variance (blue) increases with \(\lambda\), representing the bias-variance tradeoff. 

A vertical dashed line indicates the optimal value of \(\lambda\). A QR code is present on the top right.


- Value of $\lambda$ determines bias-variance tradeoff
- Larger values $\square$ more regularization $\square$ more bias $\square$ less variance Determined through validation


## Dataset Example

## Data generation function:

$$
h(x)=\sin (2 \pi x), x \sim \operatorname{Uniform}(0,1)
$$

Generate $L$ independent datasets ( $L=100$ ), each with $N=25$ noisy observations.

$$
\begin{aligned}
t_{n} & =h\left(\mathrm{x}_{\mathrm{n}}\right)+\epsilon_{n}, \\
\epsilon_{n} & \sim \mathcal{N}(0,0.32)
\end{aligned}
$$

Model: linear combination of $M=$ 24 Gaussian basis functions plus bias.
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-47.jpg?height=1829&width=1596&top_left_y=38&top_left_x=1513)

**Image Description:** The image consists of four panels depicting simulated fits of data points against a ground truth curve. Each panel features a line plot labeled "Simulated Fits" on the left and a corresponding plot titled "Average vs. Ground Truth" on the right. The x-axis represents the independent variable (normalized values from 0 to 1), while the y-axis indicates the dependent variable (output values). The plots illustrate the variability of simulated data fits across different parameter sets, with overlaid average lines comparing simulated fits to the underlying ground truth, characterized by the similarity in shape and alignment.

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-47.jpg?height=311&width=252&top_left_y=0&top_left_x=3078)

**Image Description:** The image is a QR code, a type of two-dimensional barcode used for encoding data. It consists of a matrix of black squares arranged on a white grid, which can be scanned by devices to retrieve encoded information. The code may link to a URL, text, or other data types. Its design follows specific patterns for alignment and error correction, commonly used in various applications, such as marketing and information sharing.

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-48.jpg?height=575&width=805&top_left_y=42&top_left_x=161)

**Image Description:** The image presents a line plot titled "Simulated Fits (lnλ=3.0)". The horizontal axis represents the variable \( x \), ranging from 0 to 1, while the vertical axis denotes the function \( f(x) \), with values approximately between -0.2 and 0.6. Multiple light blue curves illustrate the simulated fits across the range, displaying variations and trends in \( f(x) \) as \( x \) changes. The plot suggests a nonlinear relationship characteristic of simulations in statistical modeling.

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-48.jpg?height=579&width=775&top_left_y=42&top_left_x=956)

**Image Description:** The slide contains a line graph titled "Average vs. Ground Truth." The x-axis ranges from 0 to 1, representing a continuous variable. The y-axis ranges approximately from -0.75 to 0.75, indicating values of the average fit and ground truth. The graph features two curves: a solid blue line representing "Average fit" and a dashed orange line representing "Ground truth." The average fit appears smoother, while the ground truth exhibits more fluctuation, allowing comparison of modeled versus actual values. A legend differentiates the two lines.

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-48.jpg?height=609&width=796&top_left_y=629&top_left_x=161)

**Image Description:** The image is a line plot titled "Simulated Fits (lnλ=1.0)". The x-axis is labeled as "x" and ranges from 0 to 1, while the y-axis is labeled as "f(x)" and also ranges approximately from -0.5 to 0.5. The plot features multiple light blue lines representing simulated functions, which exhibit varying patterns of oscillation over the interval. The lines suggest a distribution of fit estimates for the function f(x) based on varying parameters. There are numerous overlapping lines, indicating a range of simulated outcomes.

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-48.jpg?height=609&width=775&top_left_y=629&top_left_x=956)

**Image Description:** The image is a line graph comparing two functions: "Average fit" represented by a solid blue line and "Ground truth" represented by a dashed orange line. The x-axis is labeled as "x" and ranges from 0 to 1, while the y-axis represents the values of the functions, ranging approximately from -1 to 1. The graph illustrates the relationship between the average estimation and the actual ground truth across the given range of x values, highlighting variations and discrepancies between the two lines.

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-48.jpg?height=613&width=796&top_left_y=1250&top_left_x=170)

**Image Description:** The image is a graph depicting simulated fits for a function \( f(x) \). The x-axis represents the variable \( x \), ranging from 0 to 1, while the y-axis represents the function values \( f(x) \), ranging approximately from -0.5 to 0.5. Multiple light blue lines illustrate the variability in the simulated fits, indicating the influence of a parameter set to \( \ln \lambda = -3.0 \). The overall pattern exhibits oscillatory behavior.

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-48.jpg?height=609&width=775&top_left_y=1250&top_left_x=956)

**Image Description:** The image is a line graph comparing two data sets. The x-axis represents values ranging from 0 to 1, labeled as "x". The y-axis ranges from -1.0 to 1.0, indicating the output values. The solid blue line denotes the "Average fit," while the dashed orange line represents the "Ground truth." Both lines depict a sinusoidal waveform oscillating around the x-axis, with peaks and troughs illustrating their relationship over the interval on the x-axis. The legend identifies the two lines for clarity.

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-48.jpg?height=1103&width=1383&top_left_y=416&top_left_x=1794)

**Image Description:** The image is a graph illustrating the Bias-Variance Trade-off. It features a plot with three curves representing Bias² (green), Variance (orange), and Bias² + Variance (blue) against the natural logarithm of λ (ln(λ)) on the x-axis. The y-axis denotes Error, with values ranging from approximately 0.2 to 1. The curves depict the relationship between model complexity (represented by λ) and the error components, illustrating how increasing complexity reduces bias but increases variance. The graph highlights the trade-off between these two sources of error in predictive modeling.


## Select options below.

## Select options below.

## Select options below.

# Regression (Cont.) and Bias-Variance Trade-off 

Credit: Joseph E. Gonzalez and Narges Norouzi
Reference Book Chapters: Chapter 4.2 and 4.3
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-53.jpg?height=1807&width=1889&top_left_y=68&top_left_x=0)

**Image Description:** This image features a cartoon bear wearing glasses, sitting at a table covered with scattered papers. The bear is focused on an open textbook labeled "TEXTBOOK." The scene conveys a study environment, with a pencil resting behind the bear's ear, suggesting engagement in learning. The background is a warm, sepia tone, enhancing the academic atmosphere. The illustration emphasizes themes of education and curiosity, suitable for academic discussions or materials related to learning and study practices.

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-53.jpg?height=302&width=247&top_left_y=0&top_left_x=3083)

**Image Description:** The image is a QR code comprising a pattern of black and white squares arranged on a grid. QR codes typically encode data that can be scanned by smartphones or cameras to link to URLs, text, or other information. This specific QR code includes a numeric sequence "2312298" beneath it, which may represent specific data encoded within the code. The design is abstract and does not feature axes or traditional diagram elements.


Homework!

## HW 1 Updates

- Please check ed for any HW guidance. We have added a walkthrough for submitting your homework to gradescope
- Autograder for problem 9 of part 2 has been adjusted to give partial credit (you should receive updated grades this afternoon):
o 38-40\%: 1 pt
o 40-42: 2 pts
o 43-44: 3 pts
o >44: 5 pts
- There is a separate assignment on Gradescope for just your PDF export - this will be the norm moving forward
- Thank yall for being so patient!


## HW2

## Part 1 (Due Oct 3rd)

- Written problems for regression and MLE
- Short coding portion to explore LMArena (formerly Chatbot Arena) data

Uses concepts from
Logistic regression
lectures next week

## Part 2 (Due Oct 17th)

- Paper questions (tutorial on how to read a paper next week)
o Chatbot Arena
o VibeCheck / Style control
- Coding implementation of Chatbot Arena evaluation and simplified version of VibeCheck


## Data Visualizers for HW2

Gradio - a python backed UI developer created by huggingface In the hw we encourage you to play around with the data you visualize and the styles of the UI

YOU SHOULD VIBE CODE THESE SINCE IT IS EASY TO

## Chatbot Arena Battle Viewer (100k)

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-56.jpg?height=1043&width=1766&top_left_y=323&top_left_x=1301)

**Image Description:** The image is a screenshot of a lecture slide featuring a chat interface, likely discussing a financial scenario involving Mr. Stone. On the left, a chat log shows Mr. Stone indicating he was overcharged. The right side displays a summary of Mr. Stone's situation, including amounts he paid and received. The layout suggests a comparison of conversational elements. Key phrases such as "overcharged" and "received back" indicate a focus on financial transactions and adjustments.

VERIFY

Find these viewers on our Huggingface Page $\rightarrow$

## Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference

Wei-Lin Chiang* ${ }^{1}$ Lianmin Zheng ${ }^{* 1}$ Ying Sheng ${ }^{2}$ Anastasios N. Angelopoulos ${ }^{1}$ Tianle Li ${ }^{1}$ Dacheng Li ${ }^{1}$ Banghua Zhu ${ }^{1}$ Hao Zhang ${ }^{3}$ Michael I. Jordan ${ }^{1}$ Joseph E. Gonzalez ${ }^{1}$ Ion Stoica

Abstract
Large Language Models (LLMs) have unlocked Large Language tith human preferences still address this is-platform for evaluating LLMs based on human preferences. Our methodology a pairwise comparison approach and leverages input from a diverse user base through crowdsourcing. The platform has been operational for several months, amassing over 240 K votes. This paper describes the platform, analyzes the data we have collected so far, and explains the tried-and-true statistical methods we are using for efficient and accurate evaluation and ranking of models. We confirm that the crowdsourced questions are sufficiently diverse and discriminating and that the crowdsourced human votes are in good agreement with those of expert raters. These analyses collectively establish a robust foundation for the credibility of Chatbot Arena. Because of its unique value and openness, Chatbot Arena has emerged as one of the most referenced LLM leaderboards, companies. Our demo is publicly available at https://chat.lmsys.org.

## VibeCheck: Discover \& Quantify Qualitative <br> Differences in Large Language Models

Lisa Dunlap<br>Krishna Mandal<br>Trevor Darrell<br>Jacob Steinhardt<br>Joseph Gonzalez<br>UC Berkeley<br>UC Berkeley<br>\author{ UC Berkeley<br><br>UC Berkeley }

## Abstract

Large language models (LLMs) often exhibit subtle yet distinctive characteristics in their outputs that users intuitively recognize, but struggle to quantify. These vibes" - such as tone, formatting, or writing style - influence user preferences, yet traditional evaluations focus primarily on the singular vibe of correctness. We
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-57.jpg?height=48&width=1008&top_left_y=765&top_left_x=1820) ating and user-aligned. VibeCheck iteratively discovers vibes from model outputs a vibe We validate that the vibes generated by VibeCheck align with those found in
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-57.jpg?height=35&width=1008&top_left_y=910&top_left_x=1820) user conversations with Llama-3-70b vs GPT-4 VibeCheck reveals that Llama has a friendly, funny, and somewhat controversial vibe. These vibes predict model idena tity with $80 \%$ accuracy and human preference with $61 \%$ accuracy. Lastly, we run vibe captioning to pro covers vibes like Command $X$ prefers to add concrete intros and conclusions
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-57.jpg?height=35&width=1008&top_left_y=1177&top_left_x=1820) press on man problems compared to GP1-4o, and GP-4 prefers to focus on the hershi-1.5-Flash. Code and vibe visualizer found at https://bench-mark.org/

## 1 Intro

vibe check : A process by which a group obtains a subjective assessment of another person, place, or thing.

- Urban Dictionary


## Evaluating LMM's in the Wild

## LMArena

An open platform for human preference evals

## LMArena (formerly chatbot Arena)

A platform for holistic LMM evaluation, where real user conversations and pairwise votes are crowdsourced to build a live human preference leaderboard.

Members (pre-company launch): Wei-Lin Chiang, Anastasios Angelopoulos, Lianmin Zheng, Ying Sheng, Lisa Dunlap, Chris Chou, Tianle Li, Evan Frick, Aryan Vichare, Naman Jain, Manish Shetty, Yifan Song, Kelly Tang, Sophie Xie, Connor Chen, Joseph Tennyson, Dacheng Li, Siyuan Zhuang, Valerie Chen, Wayne Chi
Advisors (pre-company launch): Ion Stoica, Joseph Gonzalez, Hao Zhang, Trevor Darrell

## Crowdsourcing user interactions with LMM's

## Direct Chat

1. User selects a model to chat with (LLM, VLM, Text-2-Img, etc)
2. Conversations are recorded to analyze real world interactions

濒
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-60.jpg?height=645&width=483&top_left_y=503&top_left_x=2653)

**Image Description:** The image depicts a parking sign with specific restrictions. It indicates "2 HR PARKING" with operational hours from "8AM - 7PM" on the left side. Beneath this, it notes exceptions for street sweeping on "SAT, SUN & HOLIDAYS" with an area for designations marked "F G". The sign includes a reference to an ordinance labeled "ORD. 3262 N.S. SEC. 10.6" at the bottom. The background consists of a tree trunk, and the sign's text is in green and white, clearly indicating parking regulations.

its sunday, how long can i park here?

Al claude-3-5-sonnet-20241022

Since it's Sunday, you are exempt from the 2 -hour parking restriction that applies during other times (8AM7PM). The sign indicates that Saturdays, Sundays, and holidays are exceptions to the normal 2-hour limit, so you can park there without a time restriction today.

Ask followup..
2 Chat $\checkmark$

## Crowdsourcing user interactions with LMM's

## Battle

1. User input any prompt
2. Two anonymized models give answers side-by-side
3. User vote which is best Pairwise setting allows for more fine grained comparison

Assistant A

Since it's Sunday, you can park there without the 2-hour time
limit. The sign indicates that the 2 -hour parking restriction (8AM-7PM) desn't apply on Saturdays, Sundays, and holidays. You should be able to park there for an unlimited time today. time today.
-

Assistant B

Based on the sign, the 2-hour limit between 8 AM and 7 PM does not apply on Sunday because it is listed as an exception.

Therefore, according to this sign, you can likely park for an unlimited amount of time on cunday
$\theta$ Both are bad
Right is Better $\rightarrow$
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-61.jpg?height=516&width=669&top_left_y=484&top_left_x=2449)

**Image Description:** The image is a parking sign. It features a green and white rectangular design with black text. The sign indicates "2 HR PARKING" from "8AM - 7PM," specifying exceptions for street sweeping, Saturdays, Sundays, holidays, and vehicles with area permits. The letters "F" and "G" are included in a separate box, indicating permit areas. The bottom of the sign references an ordinance number "ORD. 3262 N.S. SEC. 10.6." The sign's layout is structured for clarity and quick reference for parking regulations.

its sunday, how long can i park here?

## Using Battles to Generate a Preference Leaderboard

Bradley-Terry Model (next week): estimate the relative strengths of models (X) through logistic regression ( $\mathbf{Y}=$ winner)

$$
\hat{\beta}=\arg \min _{\beta \in \mathbb{R}^{M}} \frac{1}{n} \sum_{i=1}^{n} \text { BCELoss }\left(\operatorname{sigmoid}\left(X_{i}^{\top} \beta\right), Y_{i}\right)
$$

The Arena Score of a model is it's coefficient in this model ( $\beta$ )
![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-62.jpg?height=1226&width=1179&top_left_y=446&top_left_x=1947)

**Image Description:** The image is a scatter plot displaying confidence intervals on model strength via bootstrapping. The x-axis represents different models, while the y-axis indicates Elo rating, ranging from approximately 1360 to 1480. Each data point is plotted with error bars, illustrating the confidence intervals around the Elo ratings for each model. The data points show a decreasing trend in Elo ratings as the model values increase, indicating variability in model performance. The visualization aids in assessing the robustness of the models based on their performance metrics.


## If you think logistic regression is not valuable

## Chatbot Arena Raises \$100 Million for Platform That Compares AI Models

![](https://cdn.mathpix.com/cropped/2025_10_01_ec6e7c1f271c3a42cf8dg-63.jpg?height=975&width=1506&top_left_y=612&top_left_x=901)

**Image Description:** The image is a digital illustration featuring a circuit-like background with a prominent golden square labeled "AI" at the center. The background consists of interconnected geometric shapes resembling microchip components, with varying shades of black and gold, suggesting a technological theme. This visually conveys concepts related to artificial intelligence, modern computing, and digital networks, emphasizing the central role of AI in contemporary technology advancement. The overall composition is intricate, with a focus on design elements indicative of electronic circuitry.


Chatbot Arena raised \$100 million in seed funding to support its platform that helps users compare the performance of different artificial intelligence models.

The company also said in a Wednesday (May 21) post on social platform $X$ that it plans to relaunch the platform, LMArena, with a new look based on feedback from members of its community.

## Structure of a paper

- Abstract
- Intro
- Related works
- Methods and/or Problem Formulation
- Results
- Limitations and Conclusion


## What to answer in every paper

1. What problem is this paper tackling?
2. What do prior works do and how do they fall short?
3. What is the key insight of this paper? What do they do that prior works do not?
4. What are the inputs and outputs of their method/contribution?
5. What are limitations to this method?

## What to answer in every paper

1. What problem is this paper tacklind? paragraph
2. What do prior works do and how do they fall shorts - 2nd
3. What is the key insight of this paper? What do they do that prior works do not? 4th
4. What are the inputs and outputs of their method/contribution? nd- 4th
5. What are limitations to this method?
