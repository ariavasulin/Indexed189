---
course: CS 189
semester: Fall 2025
type: pdf
title: Discussion Worksheet 01
source_type: pdf
source_file: Discussion Worksheet 01.pdf
processed_date: '2025-10-08'
processor: mathpix
slug: discussion-worksheet-01-processed
---

## Discussion 1

Note: Your TA will probably not cover all the problems on this worksheet. The discussion worksheets are not designed to be finished within an hour. They are deliberately made slightly longer so they can serve as resources you can use to practice, reinforce, and build upon concepts discussed in lectures, discussions, and homework.

## Chain Rule:

$$
\frac{\partial}{\partial x} f(g(x)+h(x))=\frac{\partial f}{\partial g} \frac{d g}{d x}+\frac{\partial f}{\partial h} \frac{d h}{d x}
$$

## 1 Calculus

(a) Consider the function

$$
f(x, y)=\sigma(a x+b y)
$$
where $a, b \in \mathbb{R}$ and $\sigma(t)=\frac{1}{1+e^{-t}}$ for $t \in \mathbb{R}$.
(i) Show that $\frac{d \sigma}{d t}=\sigma(t)(1-\sigma(t))$.
(ii) Using the result you showed in part (i) and the chain rule, compute $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$.
(b) For $\mathbf{x}=\left[\begin{array}{c}x_{1} \\ \vdots \\ x_{n}\end{array}\right] \in \mathbb{R}^{n}$, define
$$
r(\mathbf{x})=\sum_{j=1}^{n} x_{j}^{2}
$$

Compute the partial derivative $\frac{\partial r}{\partial x_{i}}$ for a generic coordinate $i \in\{1, \ldots, n\}$.
(c) Let $\mathbf{w} \in \mathbb{R}^{n}$ be a constant vector and define the scalar function

$$
s(\mathbf{x})=\mathbf{w}^{\top} \mathbf{x}=\sum_{j=1}^{n} w_{j} x_{j}
$$

Compute $\frac{\partial s}{\partial x_{i}}$ for a generic coordinate $i \in\{1, \ldots, n\}$.

## 2 Linear Algebra

(a) Prove that $\mathbf{A}^{\top} \mathbf{A}$ is symmetric for any $\mathbf{A} \in \mathbb{R}^{m \times n}$.
(b) Consider the matrix

$$
\mathbf{B}=\left[\begin{array}{ll}
1 & 0 \\
0 & 2 \\
2 & 1
\end{array}\right]
$$

Find the singular values of $\mathbf{B}$.

## 3 Probability Review

An incoming email is spam with prior $p(S)=0.2$ and not spam with $p(\bar{S})=0.8$. Two independent filters flag spam:

$$
p\left(F_{1}=1 \mid S\right)=0.9, \quad p\left(F_{1}=1 \mid \bar{S}\right)=0.1, \quad p\left(F_{2}=1 \mid S\right)=0.8, \quad p\left(F_{2}=1 \mid \bar{S}\right)=0.05
$$
and $F_{1}, F_{2}$ are independent given the class ( $S$ or $\bar{S}$ ).
(a) What is the probability that both filters flag an email as spam?
(b) Given that both filters flag an email, what is the probability of the email being spam? (You can leave your answer as an unsimplified fraction.)

