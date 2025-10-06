---
course: CS 189
semester: Fall 2025
type: discussion
title: Discussion 4
source_type: slides
source_file: Discussion Mini Lecture 04.pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Discussion Mini Lecture 4

## GMM \& Ridge Regression

Review of GMM \& Ridge Regression

CS 189/289A, Fall 2025 @ UC Berkeley
Sara Pohland

## Concepts Covered

1. MLE with Gaussians
2. Gaussian Mixture Models
3. Linear Regression
4. Ridge Regression

## Refresher on MLE with Gaussians

1. MLE with Gaussians
2. Gaussian Mixture Models
3. Linear Regression
4. Ridge Regression

## A Motivating 1D Example

Suppose I give you a set of $N 1 \mathrm{D}$ data points, $\mathcal{D}=\left\{x_{1}, \ldots, x_{N}\right\}$, where $x_{n}$ is the caffeine level of the $n$th cup of coffee.

Assume that each $x_{n}$ is independently sampled from a large coffee distributor sourcing from three different production companies.

Coffee Caffeine Dataset
![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-04.jpg?height=592&width=2923&top_left_y=1165&top_left_x=212)

**Image Description:** The image is a scatter plot displaying the distribution of data points concerning caffeine consumption measured in milligrams (mg). The x-axis represents caffeine levels, ranging from 50 mg to 250 mg, while the y-axis lacks a numerical scale, indicating categorical or qualitative distinction among groups. Data points are represented as teal circles, clustered predominantly in two regions: one between 50-125 mg and the other between 175-225 mg, suggesting distinct patterns or behaviors in caffeine consumption. The plot effectively visualizes the relationship and variability of the data regarding caffeine intake.


## A Motivating 1D Example

Suppose I give you a set of $N 1 \mathrm{D}$ data points, $\mathcal{D}=\left\{x_{1}, \ldots, x_{N}\right\}$, where $x_{n}$ is the caffeine level of the $n$th cup of coffee.

Assume that each $x_{n}$ is independently sampled from a large coffee distributor sourcing from three different production companies.

What is the probability of drawing coffee with caffeine level x (i.e., $p(x)$ )?

## The Gaussian Distribution

Let's assume that we sample our coffee data points from a Gaussian distribution: $\mathrm{X}_{\mathrm{n}} \sim \mathcal{N}\left(\mu, \sigma^{2}\right)$, where $\mu=$ mean and $\sigma^{2}=$ variance.

$$
p\left(x_{n} ; \mu, \sigma^{2}\right)=\frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right)
$$

How do we estimate the parameters of our Gaussian, $\mu$ and $\sigma^{2}$ ?
Let's quickly return to what we covered in Discussion 3...

## How do we Estimate these Parameters?

Approach: Maximize the likelihood of our data under the assumed
probability distribution.
Likelihood: $p(\mathcal{D} \mid w)=\prod_{n=1} p_{\text {assumed }}\left(x_{n} ; w\right)$
Optimization problem: $\widehat{w}=\arg \max _{w} p(\mathcal{D} \mid w)$

* This assumes that our data is independent and identically distributed
(IID)- a very common through rarely true assumption.


## How do we Estimate these Parameters?

In this case, $p_{\text {assumed }}$ is the Gaussian distribution, and $w$ represents the parameters of this distribution, $\mu$ and $\sigma^{2}$.

Likelihood: $p\left(\mathcal{D} \mid \mu, \sigma^{2}\right)=\prod_{n=1}^{N} \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right)$
Optimization problem: $\hat{\mu}, \hat{\sigma}^{2}=\arg \max _{\mu, \sigma^{2}} p\left(\mathcal{D} \mid \mu, \sigma^{2}\right)$

## How do we Estimate these Parameters?

In this case, $p_{\text {assumed }}$ is the Gaussian distribution, and $w$ represents the parameters of this distribution, $\mu$ and $\sigma^{2}$.

Likelihood: $p\left(\mathcal{D} \mid \mu, \sigma^{2}\right)=\prod_{n=1}^{N} \frac{1}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right)$
Optimization problem: $\hat{\mu}, \hat{\sigma}^{2}=\arg \max _{\mu, \sigma^{2}} p\left(\mathcal{D} \mid \mu, \sigma^{2}\right)$

Sample mean:

$$
\hat{\mu}=\frac{1}{N} \sum_{n=1}^{N} x_{n} \quad \begin{aligned}
& \text { Sample } \\
& \text { variance: }
\end{aligned} \quad \hat{\sigma}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left(x_{n}-\hat{\mu}\right)^{2}
$$

## MLE with Gaussian Assumption

MLE with Gaussian Assumption
![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-10.jpg?height=1332&width=2170&top_left_y=480&top_left_x=582)

**Image Description:** The image is a histogram with a superimposed density plot. The x-axis represents caffeine intake in milligrams (mg), ranging from 50 to 250 mg. The y-axis shows density values, indicating the probability density of caffeine consumption. The histogram is composed of multiple bars, displayed in light gray, representing the frequency of observations within specified caffeine ranges. A dark curve illustrates the normal distribution fitted to the data, with parameters specified as $P(x) \sim \mathcal{N}(\mu = 123.8, \sigma^2 = 2292.7)$.


## Gaussian Mixture Model (GMM)

## 1. MLE with Gaussians

## 2. Gaussian Mixture Models

## 3. Linear Regression

4. Ridge Regression

## Improving our Gaussian Assumption

We know our coffee distributor sources from $K=3$ production companies, so each point $x_{n}$ corresponds to a cluster $z_{n} \in\{1,2,3\}$.

However, we don't know which production company each cup of coffee was drawn from (i.e., our data is unlabeled).

How might we guess the cluster label $z_{n}$ for each sample point?

## K-Means Clustering

## In Discussion 3, we covered K-means clustering!

Coffee Caffeine Dataset
![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-13.jpg?height=427&width=2884&top_left_y=705&top_left_x=238)

**Image Description:** The image presents a scatter plot divided into two panels. The left panel shows a dense cluster of data points, indicating a high concentration in a particular region of the plot. The right panel displays a more dispersed arrangement of points, suggesting a lower concentration. The x-axis and y-axis are not labeled, but the overall representation implies a comparison of two datasets or conditions. The points are marked by light teal circles, illustrating distribution patterns within the data.


K-means Clusters
![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-13.jpg?height=584&width=2906&top_left_y=1241&top_left_x=216)

**Image Description:** The image is a scatter plot illustrating the distribution of data points clustered into three groups based on caffeine intake measured in milligrams on the x-axis. The x-axis ranges from 50 to 250 mg, while the y-axis is not labeled but shows the density of data points. The clusters are color-coded: Cluster 0 in green (around 75 mg), Cluster 1 in blue (around 125 mg), and Cluster 2 in orange (around 200 mg). Each point represents individual data on caffeine consumption. The plot indicates distinct groupings based on caffeine levels.


## The Gaussian Distribution

Now let's assume that given that a cup of coffee is from production company $k$, the caffeine level of our coffee is distributed according to a Gaussian distribution: $\left(\mathrm{X}_{\mathrm{n}} \mid Z_{n}=k\right) \sim \mathcal{N}\left(\mu_{k}, \sigma_{k}^{2}\right)$.

$$
p\left(x_{n} \mid z_{n}=k\right)=\frac{1}{\sqrt{2 \pi \sigma_{k}^{2}}} \exp \left(-\frac{1}{2 \sigma_{k}^{2}}\left(x-\mu_{k}\right)^{2}\right)
$$

How do we estimate the parameters of our Gaussians?
We can use MLE exactly as we did before for each cluster!

## MLE with Gaussian Assumptions

MLE with Multiple Gaussians
![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-15.jpg?height=1336&width=2170&top_left_y=476&top_left_x=582)

**Image Description:** The image is a probabilistic density plot showing the distribution of caffeine intake in milligrams under different conditions (indicated by colors and legends). The x-axis represents caffeine amounts ranging from 0 to 250 mg, while the y-axis indicates density. Three colored curves are present: blue for \( P(x|z=0) \), green for \( P(x|z=1) \), and orange for \( P(x|z=2) \), each representing normal distributions with specified means and variances. Additionally, shaded bars suggest empirical data distribution, enhancing visualization of the fitted curves.


## Mixture of Multiple Gaussians

Now we have $p(x \mid z=k)$ for $k \in\{1,2,3\}$, but we really want $p(x)$.
To find $p(x)$, recall the law of total probability from Discussion 1:

$$
p(x)=\sum_{k} p(x, z=k)=\sum_{k} \underbrace{p(x \mid z=k)}_{\begin{array}{c}
\text { estimate } \\
\text { with MLE }
\end{array}} \underbrace{p(z=k)}_{\text {??? }}
$$

Let's estimate $p(z=k)$ as the proportion of data points in cluster $k$ :

$$
\pi_{k}=\frac{1}{N} \sum_{n=1}^{N} I\left\{z_{n}=k\right\}
$$

## Mixture of Multiple Gaussians

Mixture Model: Weighted Sum of Gaussians
![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-17.jpg?height=1370&width=2174&top_left_y=450&top_left_x=579)

**Image Description:** The image is a probabilistic density plot illustrating a mixture model using a weighted sum of Gaussian distributions. The x-axis represents caffeine content in milligrams (mg), ranging from 0 to 250 mg. The y-axis shows the density of caffeine consumption. The plot features several colored dashed lines indicating the conditional probabilities \( P(x | z=k) \) for different latent variables \( z \), as well as a black curve representing the overall mixture density \( P(x) = \sum_k P(x | z=k) P(Z=k) \). Additionally, a histogram further visualizes caffeine consumption data.


## Review of the Current Approach

1. Partition data into $K$ disjoint clusters
2. Model the cluster probability as $p(z=k)=\pi_{k}$
![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-18.jpg?height=226&width=988&top_left_y=391&top_left_x=2215)

**Image Description:** The image displays a label emphasizing the "proportion of points in cluster \( k \)." It is likely part of a larger diagram or concept related to clustering in data analysis. The label is formatted in a prominent, bold font, suggesting its significance in the context. An arrow points toward the text, indicating directionality or a connection to an accompanying image or diagram, possibly illustrating clustering or statistical analysis. There are no equations or complex visual elements in this specific image.

3. Assume $p(x \mid z=k)=\mathcal{N}\left(x ; \mu_{k}, \sigma_{k}^{2}\right)$ and use MLE to estimate:

$$
\hat{\mu}_{k}, \hat{\sigma}_{k}^{2}=\arg \max _{\mu_{k}, \sigma_{k}^{2}} p\left(\mathcal{D}_{k} \mid \mu_{k}, \sigma_{k}^{2}\right) \quad \begin{gathered}
\text { data contained in } \\
\text { cluster } k
\end{gathered}
$$

4. Use the law of total probability to get weighted sum of Gaussians:

$$
p(x)=\sum_{k=1}^{K} p(x \mid z=k) p(z=k)
$$

Problem 1a on the Discussion 4 worksheet reviews this.

## Limitations of the Current Approach

This is much better than performing MLE with a single Gaussian, but can we do even better?

Limitations of current approach:

- K-means clustering gives hard assignments that ignore uncertainty
- Point near two clusters is forced into one
- We might not have distinct clusters
- Distributions may overlap
![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-19.jpg?height=975&width=1672&top_left_y=884&top_left_x=1645)

**Image Description:** The diagram is a probability density function (PDF) showing a mixture model. The x-axis represents the variable \( x \), while the y-axis represents the probability density \( P(x) \). Multiple probability curves are represented: \( P(x|z=1) \) (dashed orange), \( P(x|z=2) \) (dashed green), and the overall function \( P(x) \) (solid black curve). Histograms (grey bars) indicate the distribution of data. The region circled in red highlights an area of intersection between these curves, suggesting a point of density overlap.



## Improving the Current Approach

'What if we don't want to force our data points into one of $K$ disjoint clusters before estimating the parameters of our likelihoods?

Could we estimate $\pi_{k}, \mu_{k}, \sigma_{k}^{2}$ for $k=1, \ldots, K$ all together?
Let's return to our expression for the weighted sum of Gaussians:

$$
\begin{aligned}
p(x)= & \sum_{k=1}^{K} p(x \mid z=k) p(z=k) \\
& \mathcal{N}\left(x ; \mu_{k}, \sigma_{k}^{2}\right)
\end{aligned}
$$

## Improving the Current Approach

'What if we don't want to force our data points into one of $K$ disjoint clusters before estimating the parameters of our likelihoods?

Could we estimate $\pi_{k}, \mu_{k}, \sigma_{k}^{2}$ for $k=1, \ldots, K$ all together?
Now let's rewrite this slightly to match our MLE form:

$$
\begin{gathered}
p\left(x_{n} ; \theta\right)=\sum_{k=1}^{K} \mathcal{N}\left(x ; \mu_{k}, \sigma_{k}^{2}\right) \pi_{k} \\
\theta=\left\{\mu_{k}, \sigma_{k}^{2}, \pi_{k}\right\}_{k=1}^{K}
\end{gathered}
$$

## Improving the Current Approach

'What if we don't want to force our data points into one of $K$ disjoint clusters before estimating the parameters of our likelihoods?

Could we estimate $\pi_{k}, \mu_{k}, \sigma_{k}^{2}$ for $k=1, \ldots, K$ all together?
With this assumed distribution, our MLE problem becomes...

$$
\hat{\theta}=\arg \max _{\theta} \prod_{n=1}^{N} \sum_{k=1}^{K} \mathcal{N}\left(x_{n} ; \mu_{k}, \sigma_{k}^{2}\right) \pi_{k}
$$

How do we solve this?
Expectation maximization (EM)!

You'll see EM in Problem 1b on the Discussion 4 worksheet.

## Prediction of the Improved Approach

Gaussian Mixture Model (GMM) with EM
![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-23.jpg?height=1358&width=2166&top_left_y=450&top_left_x=586)

**Image Description:** The diagram illustrates a Gaussian Mixture Model (GMM) applied to caffeine data. The x-axis represents caffeine content in milligrams (mg) ranging from 0 to 250 mg. The y-axis shows the density of data points. The black curve depicts the overall model, while shaded bars indicate histogram frequencies. Colored dashed lines represent individual Gaussian components for different latent variables \( z \) values (0, 1, 2). The equation \( P(x) = \sum_{k} P(x|z = k)P(Z = k) \) summarizing the model is central to the diagram's interpretation.


## Prediction of the Original Approach

![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-24.jpg?height=1417&width=2166&top_left_y=395&top_left_x=586)

**Image Description:** The image is a diagram illustrating a mixture model for caffeine consumption, showing the weighted sum of Gaussian distributions. The x-axis represents caffeine dosage in milligrams (mg), ranging from 0 to 250 mg. The y-axis represents density. Various curves indicate different probability density functions: a blue dashed line for \( P(x | z=0) \), an orange dashed line for \( P(x | z=1) \), and a green dashed line for \( P(x | z=2) \). The black line represents the overall mixture model density \( P(x) = \sum_k P(x | z = k) P(z = k) \), with vertical green lines marking data points.


## Why is the second approach better?

-We don't force each sample point into a single cluster.

- We generate soft assignments that handle class overlap naturally.


## Consider an example with more overlap

MLE with Single Gaussian
![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-26.jpg?height=1358&width=2170&top_left_y=450&top_left_x=582)

**Image Description:** The image is a histogram representing the distribution of a dataset along with a single Gaussian curve fitted to the data. The x-axis is labeled "Value," ranging approximately from -10 to 10, and the y-axis labeled "Density," showing the density of the values. The histogram is displayed in light gray bars, while the Gaussian curve is overlaid in black. The curve represents the estimated probability density function \( P(x) \) of the dataset, illustrating the normal distribution characteristic of the data.


## Consider an example with more overlap

MLE with K-Means Clustering Gaussians
![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-27.jpg?height=1358&width=2166&top_left_y=450&top_left_x=586)

**Image Description:** The diagram is a probability density plot showing the distribution of a variable measured on the x-axis labeled "Value." It includes a histogram (gray) representing the data distribution and three overlaid curves: a dashed blue curve for \( P(x|Z=0) \), a dashed orange curve for \( P(x|Z=1) \), and a solid black curve representing \( P(x) = \sum_k P(x|Z=k)P(Z=k) \). The y-axis is labeled "Density," indicating probability density values. The diagram illustrates Gaussian clustering related to K-means clustering methodology.


## Consider an example with more overlap

Gaussian Mixture Model (GMM) with EM
![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-28.jpg?height=1358&width=2170&top_left_y=450&top_left_x=582)

**Image Description:** The image shows a probability density function represented by a histogram overlaid with a Gaussian Mixture Model (GMM). The x-axis labeled "Value" represents the variable of interest, while the y-axis, labeled "Density," indicates the probability density. The histogram displays the data distribution, with blue and orange dashed lines representing two component distributions, \( P(x|Z=0) \) and \( P(x|Z=1) \), respectively. The solid black line depicts the overall mixture distribution, \( P(x) = \sum_k P(x|Z=k)P(Z=k) \). The figure illustrates the concept of GMM in a statistical context.


1. MLE with Gaussians
2. Gaussian Mixture Models
3. Linear Regression
4. Ridge Regression

## Linear Regression Outline

![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-30.jpg?height=1481&width=3020&top_left_y=340&top_left_x=191)

**Image Description:** The image is a four-quadrant diagram representing components of a machine learning process. The quadrants are labeled "Learning Problem," "Model Design," "Prediction & Evaluation," and "Optimization." Arrows connect the quadrants, indicating the flow of the process. The "Model Design" quadrant includes the formula for linear regression, presented as $$ y(\mathbf{x}, \mathbf{w}) = \mathbf{x}^{T} \mathbf{w} $$ and an equation for optimization: $$ \mathbf{w}^{*} = \arg\min \; E(\mathbf{w}) $$ . The diagram encompasses concepts related to supervised learning and evaluation methodologies.


## Least Squares Optimization Problem

In linear regression, we seek to minimize the error function:

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathbf{x}_{n}, w\right)\right)^{2}
$$

| $\boldsymbol{w} \in \mathbb{R}^{D+1}$ | $t_{n} \in \mathbb{R}$ | $\mathbf{x}_{n} \in \mathbb{R}^{D+1}$ |
| :---: | :---: | :---: |
| weight vector | $n$th target | $n$th sample |
| (including bias) |  | (augnpoired with <br> 1 ) |

$E(w)$ is a convex quadratic function, so we can find the minimum by taking the derivative with respect to w and setting it equal to 0 .

## Normal Equation for Least Squares

As we showed in Lecture 7, doing so gives us the normal equation:

$$
\begin{gathered}
\mathbb{X}^{T} \mathbf{t}=\mathbb{X}^{T} \mathbb{X} \boldsymbol{w} \\
\mathbb{X}=\left[\begin{array}{ccc}
- & \mathbf{x}_{1}^{T} & - \\
- & \mathbf{x}_{2}^{T} & - \\
\vdots & & \\
- & \mathbf{x}_{N}^{T} & -
\end{array}\right] \in \mathbb{R}^{N \times D+1} \quad \mathbf{t}=\left[\begin{array}{c}
t_{0} \\
t_{1} \\
\vdots \\
t_{N}
\end{array}\right] \in \mathbb{R}^{N} \\
\quad \begin{array}{c}
\text { design } \\
\text { matrix }
\end{array}
\end{gathered} \quad \begin{gathered}
\text { target } \\
\text { vector }
\end{gathered}
$$

If $\mathbb{X}^{T} \mathbb{X}$ is invertible, then the optimal weight is $\boldsymbol{w}^{*}=\left(\mathbb{X}^{T} \mathbb{X}\right)^{-1} \mathbb{X}^{T} \mathrm{t}$.

## When is $\mathbb{X}^{T} \mathbb{X}$ invertible?

$\mathbb{X}^{T} \mathbb{X} \in \mathbb{R}^{D+1 \times D+1}$ is invertible $\equiv \mathbb{X}^{T} \mathbb{X}$ is full rank

$$
\begin{aligned}
& R\left(\mathbb{X}^{T} \mathbb{X}\right)=\mathbb{R}^{D+1} \\
& \operatorname{rank}\left(\mathbb{X}^{T} \mathbb{X}\right)=D+1
\end{aligned}
$$

$\mathbb{X}^{T} \mathbb{X}$ has $D+1$ non-zero eigenvalues
$\mathbb{X}$ has $D+1$ non-zero singular values
$\operatorname{rank}(\mathbb{X})=D+1$
$R(\mathbb{X})=\mathbb{R}^{D+1}$
$\mathbb{X}$ is full column rank
$\mathbb{X}$ is injective

## When is $\mathbb{X}^{T} \mathbb{X}$ invertible?

What do these statements actually mean for us as ML Reginefis lead lecture 7 that we can express the design matrix as

$$
\mathbb{X}=\left[\begin{array}{ccc}
\mid & & \mid \\
\mathbb{X}_{:, 0} & \ldots & \mathbb{X}_{:, D} \\
\mid & & \mid
\end{array}\right] \in \mathbb{R}^{N \times D+1} \quad \begin{aligned}
& \text { where } \mathbb{X}_{:, i} \text { is the } i \text { th feature vector } \\
& \text { (i.e., all of the values for that } \\
& \text { feature contained in our dataset) }
\end{aligned}
$$

$\mathbb{X}^{T} \mathbb{X} \in \mathbb{R}^{D+1 \times D+1}$ is invertible $\equiv R(\mathbb{X})=\mathbb{R}^{D+1}$
$\mathbb{X}$ has $D+1$ linearly independent columns
None of our features are redundant (i.e., none are linearly combinations of others)

## When is $\mathbb{X}^{T} \mathbb{X}$ invertible?

When do we end up with redundant features?
A) There may be more features than samples ( $D>N$ )

- Some directions in weight space don't change predictions
- Our model is overparameterized
- We have "too many knobs" and can fit the training data in infinitely many ways
B. Some features may be highly correlated
- With a limited dataset, highly correlated features may end up capturing exactly the same information
- The model can trade off weight between these features arbitrarily while making the same predictions


## What happens when $\mathbb{X}^{T} \mathbb{X}$ is not invertible?

If a feature is redundant (can be expressed as a linear combination of the others), multiple weight vectors map to the samet,preficiticity many weight vectors map to the same prediction!

Well, that gives us a lot of options... How do we choose one?
Generally, we choose the "minimum norm" solution:

$$
w^{*}=\min _{w: \mathbb{X}^{T} \mathrm{t}=\mathbb{X}^{T} \mathbb{X}}\|w\|_{2}
$$

## What happens when $\mathbb{X}^{T} \mathbb{X}$ is not invertible?

How do we solve this optimization problem?
Linear algebra!
But you don't need to know how to solve this problem for the purposes of this class... The important thing to know is the solution:
![](https://cdn.mathpix.com/cropped/2025_10_01_5597f13e6349d71974d7g-37.jpg?height=341&width=1281&top_left_y=944&top_left_x=1454)

**Image Description:** The image presents a mathematical equation illustrating the calculation of a weight vector \( w^* \). The equation is formatted as follows: 

$$
w^* = X^+ t
$$

where \( X^+ \) denotes the pseudoinverse of matrix \( X \). An accompanying annotation indicates "dagg" as a reference to the pseudoinverse operation. The layout emphasizes the relationship between the weight vector and the data matrix, highlighting the use of pseudoinversion in linear regression contexts.


The pseudoinverse is aptly named; it is like the inverse of a matrix, but not quite. But it's nice because it exists for every matrix.

Ridge Regression:
$L_{2}$ Regularization

1. MLE with Gaussians
2. Gaussian Mixture Models
3. Linear Regression
4. Ridge Regression

## What's wrong with linear regression?

- Sometimes $\mathbb{X}^{T} \mathbb{X}$ is "ill-conditioned."
- If some features are highly correlated, $\mathbb{X}^{T} \mathbb{X}$ becomes nearly singular (noninvertible), meaning that some of its eigenvalues are close to zero.
- This means the eigenvalues of $\left(\mathbb{X}^{T} \mathbb{X}\right)^{-1}$ are very large for those features.
- We then say our solution is "unstable" because small changes in our data will lead to large changes in our optimal solution $w^{*}$.
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

1) $L_{1}(\boldsymbol{w}):\|\boldsymbol{w}\|_{1}=\sum_{d}\left|w_{d}\right|$ promotes sparsity by pushing some coefficients exactly to zero
2) $\quad L_{2}(\boldsymbol{w}):\|\boldsymbol{w}\|_{2}=\sqrt{\sum_{d} w_{d}{ }^{2}}$ shrinks all coefficients smoothly toward zero, but almost never exactly to zero
3) $L_{\infty}(\boldsymbol{w}):\|\boldsymbol{w}\|_{\infty}=\max _{d}\left|w_{d}\right|$ bounds the maximum coefficient, preventing any one from dominating

## $L_{2}$ Regularization $\rightarrow$ Ridge Regression

Often, we add $L_{2}$ regularization to our least squares problem:

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathbf{x}_{n}, w\right)\right)^{2}+\frac{\lambda}{2}\|w\|_{2}^{2} \quad \begin{aligned}
& \text { We call this } \\
& \text { ridge } \\
& \text { regression! }
\end{aligned}
$$

As before, $E(w)$ is a convex quadratic function, so we can find the minimum by taking the derivative with respect to w and setting it equal to 0 . Doing so gives us the normal equation:

$$
\mathbb{X}^{T} \mathrm{t}=\left(\mathbb{X}^{T} \mathbb{X}+\lambda I\right) w
$$

If $\mathbb{X}^{T} \mathbb{X}+\lambda I$ is invertible, then the optimal weight is

$$
w^{*}=\left(\mathbb{X}^{T} \mathbb{X}+\lambda I\right)^{-1} \mathbb{X}^{T} \mathrm{t}
$$

## When is $\mathbb{X}^{T} \mathbb{X}+\lambda I$ invertible?

For $\lambda>0, \mathbb{X}^{T} \mathbb{X}+\lambda I$ is always invertible!
This implies that the ridge regression problem always has a unique solution: $w^{*}=\left(\mathbb{X}^{T} \mathbb{X}+\lambda I\right)^{-1} \mathbb{X}^{T} \mathrm{t}$.

## Discussion Mini Lecture 4

## GMM \& Ridge Regression

Contributors: Sara Pohland

## Additional Resources

1. MLE with Gaussians

- Discussion 3

2. Gaussian Mixture Models

- Discussion 3
- Deep Learning Foundations and Concepts - Chapter 15.1, 15.2

3. Linear Regression

- Deep Learning Foundations and Concepts - Chapter 4.1.2, 4.1.3, 4.1.4
- Sara's notes on Machine Learning - Section 9.2

4. Ridge Regression

- Deep Learning Foundations and Concepts - Chapter 4.1.6
- Sara's notes on Machine Learning - Section 10.2

