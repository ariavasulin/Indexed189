---
course: CS 189
semester: Fall 2025
type: pdf
title: Discussion Worksheet 03
source_type: pdf
source_file: Discussion Worksheet 03.pdf
processed_date: '2025-10-08'
processor: mathpix
slug: discussion-worksheet-03-processed
---

## Discussion 3

Note: Your TA will probably not cover all the problems on this worksheet. The discussion worksheets are not designed to be finished within an hour. They are deliberately made slightly longer so they can serve as resources you can use to practice, reinforce, and build upon concepts discussed in lectures, discussions, and homework.

This Week's Cool AI Demo/Video:
https://youaretheassistantnow.com/
https://gemini.google.com/

## 1 MLE with a Linear Constraint on Independent Gaussians

(a) Let $X \sim \mathcal{N}\left(\mu_{X}, \sigma_{X}^{2}\right)$ and $Y \sim \mathcal{N}\left(\mu_{Y}, \sigma_{Y}^{2}\right)$ be independent. You only observe their sum

$$
S:=X+Y .
$$

Find the Maximum Likelihood Estimate (MLE) of $X$ given $S=2$.
(i) Show that, up to a proportionality constant,

$$
p_{X \mid S}(x \mid 2) \propto p_{X}(x) p_{Y}(2-x)
$$
(ii) Use the Gaussian forms to show that maximizing $p_{X \mid S}(x \mid 2)$ with respect to $x$ is equivalent to minimizing
$$
\frac{\left(x-\mu_{X}\right)^{2}}{2 \sigma_{X}^{2}}+\frac{\left(2-x-\mu_{Y}\right)^{2}}{2 \sigma_{Y}^{2}}
$$

(iii) Find the estimate, $\hat{x}_{\text {MLE }}$, that maximizes the likelihood $p_{X \mid S}(x \mid 2)$ using the expression you found in (ii). Simplify to a weighted-average form.
(iv) Consider the special case where $\mu_{X}=\mu_{Y}=0$ and $\sigma_{X}^{2}=\sigma_{Y}^{2}$. How can we simplify the maximum likelihood estimate?

## 2 Binomial MLE with Misclassification

(a) You run $m$ independent experiments. In each experiment, you perform $n$ independent Bernoulli trials with unknown true success probability $p$. However, your measurement device is imperfect:

- A true success is recorded as a success with probability $1-q$ (and as a failure with probability $q$ ).
- A true failure is recorded as a success with probability $r$ (and as a failure with probability $1-r$ ).

Let $X_{i} \sim \operatorname{Bin}(n, p)$ denote the unobserved number of true successes in experiment $i$, and let $Y_{i}$ be the observed number of recorded successes. Recall that the probability mass function for the binomial random variable $X_{i} \sim \operatorname{Bin}(n, p)$ is

$$
p\left(x_{i}\right)=\binom{n}{x_{i}} p^{x_{i}}(1-p)^{n-x_{i}}
$$

Derive the likelihood for the observations $\mathcal{D}=\left\{y_{i}\right\}_{i=1}^{m}$. Hint: Your answer should be a function of the true success probability $p$.
(b) Find the MLE $\widehat{p}$ in closed form. Show all steps.

## 3 Proof of K-means Convergence

Consider the K-means algorithm applied to data points $\left\{x_{n}\right\}_{n=1}^{N} \subset \mathbb{R}^{D}$ with $K$ clusters. Define binary indicator variables $r_{n k} \in\{0,1\}$ with $\sum_{k=1}^{K} r_{n k}=1$ for each $n$, and cluster centers $\left\{\mu_{k}\right\}_{k=1}^{K}$. The K-means objective is

$$
J\left(\left\{r_{n k}\right\},\left\{\mu_{k}\right\}\right)=\sum_{n=1}^{N} \sum_{k=1}^{K} r_{n k}\left\|x_{n}-\mu_{k}\right\|^{2}
$$

The K-means algorithm alternates between

1. Update assignments: Set $r_{n k}=1$ for the $k$ that minimizes $\left\|x_{n}-\mu_{k}\right\|^{2}$ and $r_{n j}=0$ for all $j \neq k$.
2. Update means: Set the cluster centers to

$$
\mu_{k}=\frac{\sum_{n=1}^{N} r_{n k} x_{n}}{\sum_{n=1}^{N} r_{n k}} \quad\left(\text { for clusters with } \sum_{n} r_{n k}>0\right)
$$

Prove that the K-means algorithm converges after a finite number of iterations. (Hint: It is sufficient to show that the objective is monotonically decreasing, and that there are a finite number of iterations).

