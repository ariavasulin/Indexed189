---
course: CS 189
semester: Fall 2025
type: pdf
title: Discussion Worksheet 04
source_type: pdf
source_file: Discussion Worksheet 04.pdf
processed_date: '2025-10-08'
processor: mathpix
slug: discussion-worksheet-04-processed
---

## Discussion 4

Note: Your TA will probably not cover all the problems on this worksheet. The discussion worksheets are not designed to be finished within an hour. They are deliberately made slightly longer so they can serve as resources you can use to practice, reinforce, and build upon concepts discussed in lectures, discussions, and homework.

This Week's Cool AI Demo/Video:
https://oasis.decart.ai/starting-point

## 1 Gaussian Mixture Clustering with Kangaroos and Berkeley Students (1D, Two Gaussians)

Consider a Gaussian Mixture Model (GMM) with two components in one dimension. Let $z_{n} \in \{K, B\}$ be the latent (unobserved) class of the $n$-th jumper ( $K=$ Kangaroo and $B=$ Berkeley student) and $x_{n} \in \mathbb{R}$ be the observed jump height (in meters).
Assume that the classes occur with equal probability and that, conditional on the class, jump heights are Gaussian with class-dependent means $\mu_{\mathrm{K}}$ and $\mu_{\mathrm{B}}$ (the average jump heights of Kangaroos and Berkeley Students, respectively), and a shared variance $\sigma^{2}$ (the variability in jump heights).
(a) Write down the following quantities explicitly.
(i) The prior probability that a jumper is a Kangaroo or Berkeley student before observing the jump height: $p\left(z_{n}\right)$.
(ii) The likelihood of the jump height conditioned on whether the jumper is a Kangaroo or Berkeley student: $p\left(x_{n} \mid z_{n}\right)$.
(iii) The the marginal probability that a certain jump height is observed: $p\left(x_{n}\right)$.
(iv) The posterior probability of a jumper being a Kangaroo or Berkeley student after observing the jump height: $p\left(z_{n} \mid x_{n}\right)$.
(b) Recall the K-means objective:

$$
J\left(\left\{r_{n k}\right\},\left\{\mu_{k}\right\}\right)=\sum_{n=1}^{N} \sum_{k=1}^{K} r_{n k}\left\|x_{n}-\mu_{k}\right\|^{2}
$$
where $\mu_{k}$ is the mean of cluster $k$ and $r_{n k} \in\{0,1\}$ is a binary indicator variable that describes which of the $K$ clusters the data point $x_{n}$ is assigned to. The K-means algorithm alternates between updating the cluster assignments and updating the cluster centers.
We modify this algorithm to generate GMMs in the following way:
(1) Update assignments: Replace the hard assignments $r_{n k}$ with soft assignments $\gamma_{n k}$, representing the probability that a point $x_{n}$ belongs to cluster $k$.
(2) Update parameters: Rather than selecting cluster centers that minimize the squared distances of data points to their assigned cluster centers, select the parameters $\theta$ that maximize the $Q$ function:
$$
Q(\theta)=\sum_{n=1}^{N} \sum_{k=1}^{K} \gamma_{n k} \ln p\left(x_{n}, z_{n}=k \mid \theta\right) .
$$

(i) Identify which probability from part (a) corresponds to $\gamma_{n k}$. Verify $\sum_{k} \gamma_{n k}=1$.
(ii) Show that the $Q$ function is actually the expected log-likelihood of our observed data, $X= \left(x_{1}, \ldots, x_{N}\right)$, and the (unobserved) latent vector, $Z=\left(z_{1}, \ldots, z_{N}\right)$ under the posterior probability of $Z$. ${ }_{1}^{1}$.
(iii) Derive the maximum likelihood estimate for $\mu_{k}$ using the expected log-likelihood, $Q(\theta)$, treating $\gamma_{n k}$ as constants.

[^0]
## 2 Closed-Form Solution for $\ell_{2}$-Regularized Least Squares (Ridge)

Consider a data matrix $X \in \mathbb{R}^{N \times D+1}$ with rows $x_{n}^{\top}$, a target vector $t \in \mathbb{R}^{N}$, and a parameter vector $w \in \mathbb{R}^{D+1}$. For a regularization hyperparameter $\lambda \geq 0$, define the ridge objective:

$$
E(w)=\|X w-t\|_{2}^{2}+\lambda\|w\|_{2}^{2} .
$$
(a) Write $E(w)$ in expanded quadratic form $w^{\top} A w-2 b^{\top} w+c$ by identifying $A, b$, and $c$ in terms of $X, t$, and $\lambda$.
(b) Compute the gradient $\nabla_{w} E(w)$ using the identities
$$
\nabla_{w}\|X w-t\|_{2}^{2}=2 X^{\top}(X w-t), \quad \nabla_{w}\|w\|_{2}^{2}=2 w .
$$

(c) Set the gradient to zero and derive the normal equations for ridge regression. Solve for $w$.
(d) Justify why $X^{\top} X+\lambda I$ is invertible for $\lambda>0$ (and discuss when it might fail for $\lambda=0$ ).


[^0]:    ${ }^{1}$ We optimize the joint likelihood because it is much more tractable to compute while still optimizing the marginal likelihood. See this page for more information.

