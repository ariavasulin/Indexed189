---
course: CS 189
semester: Fall 2025
type: pdf
title: oct6
source_type: pdf
source_file: oct6.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: oct6-processed
---

# Dual Spaces 

## Professor K. A. Ribet

![The image features the official seal of the University of California, Berkeley. It displays an open ](https://cdn.mathpix.com/cropped/2025_10_06_da2a3046519ac6ebafc5g-01.jpg?height=213&width=217&top_left_y=427&top_left_x=520)

**Image description:** The image features the official seal of the University of California, Berkeley. It displays an open book at the center, inscribed with the phrase "LET THERE BE LIGHT." Surrounding the book are various decorative elements, including a star and a coat of arms. The seal is circular, with the outer rim featuring the words "UNIVERSITY OF CALIFORNIA" and "BERKELEY," along with the founding year "1868" at the bottom. The gold and blue color theme indicates institutional identity and prestige. The seal's purpose reinforces the university's historical significance and commitment to education.


October 6, 2025

## Announcements

My office hours are Mondays 1:30-3 PM and Thursday 10:30 AM-noon

Last Lunch with Math Professor Ken Ribet is Thursday at noon.

One or two unofficial lunches each week. Feel free to request a day, time, venue.
![The image is a promotional poster for an academic lecture series titled "Lunch with Math." It featur](https://cdn.mathpix.com/cropped/2025_10_06_da2a3046519ac6ebafc5g-02.jpg?height=474&width=371&top_left_y=295&top_left_x=721)

**Image description:** The image is a promotional poster for an academic lecture series titled "Lunch with Math." It features a central photo of Professor Ken Ribet. The background includes circular design elements in blue, orange, and yellow, enhancing visual appeal. Essential details include the time (12:00 PM - 1:00 PM), location (Foothill Dining Hall), and dates (September 18, September 26, October 3, October 9). A QR code is present, likely for reservations or additional information. The overall purpose is to attract attendees to informal discussions on mathematics.


Lunch at Crossroads at 11:40 AM today

## Dual space

It's Monday morning - time to recall what we did at the end of last week.
If $V$ is a vector space over $\mathbf{F}$, then $V^{\prime}=\mathcal{L}(V, \mathbf{F})$ is the space dual to $V$. For some reason, the elements of $V^{\prime}$ are called linear functionals, and they're typically written with Greek letters.

Example: if $V=\mathcal{P}(\mathbf{R})$, the map

$$
f \longmapsto \int_{110}^{155} f(x) d x
$$
is an element of $V^{\prime}$.
Another element is
$$
f \longmapsto f^{(155)}(110) .
$$

## Dual space

If $V$ has dimension $n, V^{\prime}$ has dimension $n$. If $v_{1}, \ldots, v_{n}$ is a basis of $V$, a basis of $V^{\prime}$ is $\varphi_{1}, \ldots, \varphi_{n}$, where

$$
\varphi_{k}: a_{1} v_{1}+\cdots+a_{n} v_{n} \longmapsto a_{k} .
$$

The list $\varphi_{1}, \ldots, \varphi_{n}$ is the "dual basis," the basis of $V^{\prime}$ dual to $v_{1}, \ldots, v_{n}$.
Note that

$$
\varphi_{k}\left(v_{j}\right)=\delta_{k j}
$$
for $j=1, \ldots, n, k=1, \ldots, n$.

Special case where $V=\mathbf{F}^{n}$

If $V=\mathbf{F}^{n}$ with standard basis $e_{1}, \ldots, e_{n}$, then

$$
\varphi_{k}\left(a_{1}, \ldots, a_{n}\right)=a_{k}
$$

## Dot product, ish

If $V$ is a vector space and $V^{\prime}$ is its dual, there is a natural map

$$
V \times V^{\prime} \longrightarrow \mathbf{F}, \quad(v, \varphi) \mapsto \varphi(v)
$$

If we choose a basis $v_{1}, \ldots, v_{n}$ for $V$ and let $\varphi_{1}, \ldots, \varphi_{n}$ be the dual basis, then we can write

$$
v=\sum_{i} a_{i} v_{i}, \quad \varphi=\sum_{j} b_{j} \varphi_{j}
$$

The number $\varphi(v)$ is nothing other than the dot product $a_{1} b_{1}+a_{2} b_{2}+\cdots+a_{n} b_{n}$.

## Dual map

If $T: V \rightarrow W$ is a linear map, there is an induced linear map

$$
\mathcal{L}(W, \mathbf{F}) \rightarrow \mathcal{L}(V, \mathbf{F}), \quad \psi \in \mathcal{L}(W, \mathbf{F}) \longmapsto \psi \circ T \in \mathcal{L}(V, \mathbf{F}) .
$$

This map is called $T^{\prime}$ and is said to be the map dual to $T$. Thus $T^{\prime}$ is a linear map $W^{\prime} \rightarrow V^{\prime}$.

## Properties of dual maps

Here are some basic properties of dual maps (3.120 in LADR):

- $(S+T)^{\prime}=S^{\prime}+T^{\prime}$ for $S, T \in \mathcal{L}(V, W)$;
- $(\lambda T)^{\prime}=\lambda T^{\prime}$ for $T \in \mathcal{L}(V, W)$;
- $(S T)^{\prime}=T^{\prime} S^{\prime}$ for $T \in \mathcal{L}(V, W)$ and $S \in \mathcal{L}(W, X)$.

These are all things that you should check. For example, $(S T)^{\prime}$ is the map from $X^{\prime}$ to $V^{\prime}$ taking $\varphi \in \mathcal{L}(X, \mathbf{F})$ to $\varphi \circ(S T)$. But

$$
\varphi \circ(S T)=(\varphi \circ S) \circ T=T^{\prime}(\varphi \circ S)=T^{\prime}\left(S^{\prime} \varphi\right)
$$

## Matrix of dual map

Let $T: V \rightarrow W$ be a linear map with $V$ and $W$ finite-dimensional. Choose bases $v_{1}, \ldots, v_{n}$ and $w_{1}, \ldots, w_{m}$ of $V$ and $W$. The matrix of $T$ relative to these bases is

$$
\mathcal{M}(T)=\left(a_{i j}\right), \quad T v_{j}=\sum_{i=1}^{m} a_{i j} w_{i} \text { for } j=1, \ldots n .
$$

The dual of $T$ is the map $T^{\prime}: W^{\prime} \rightarrow V^{\prime}, \psi \mapsto \psi \circ T$.
Let $\varphi_{1}, \ldots, \varphi_{n}$ be the basis of $V^{\prime}$ dual to $v_{1}, \ldots, v_{n}$. Analogously, let $\psi_{1}, \ldots, \psi_{m}$ be the basis of $W^{\prime}$ dual to $w_{1}, \ldots, w_{m}$.
Then $T$ and $T^{\prime}$ are represented by matrices $\mathcal{M}(T)$ and $\mathcal{M}\left(T^{\prime}\right)$ of dimensions $m \times n$ and $n \times m$, respectively.

## Formula (3.123)

The matrices $\mathcal{M}(T)$ and $\mathcal{M}\left(T^{\prime}\right)$ are transposes of each other.

## Matrix of dual map

## Formula

The matrices $\mathcal{M}(T)$ and $\mathcal{M}\left(T^{\prime}\right)$ are transposes of each other.
If $\mathcal{M}\left(T^{\prime}\right)=\left(b_{i j}\right)$, then $T^{\prime}\left(\psi_{j}\right)=\psi_{j} T=\sum_{i} b_{i j} \varphi_{i}$ for each
$j=1, \ldots, m$. The formula to be proved is $b_{i j}=a_{j i}$ for each $i$ and $j$. To verify this equality is to check the following equation for each $j$ :

$$
\psi_{j} T \stackrel{?}{=} \sum_{i} a_{j i} \varphi_{i}
$$

The two sides of this equation are linear maps $V \rightarrow W$. We use the fact that two linear maps $V \rightarrow W$ are equal if they agree on the basis vectors $v_{1}, \ldots, v_{n}$.
Thus the formula is equivalent to the equality

$$
\psi_{j}\left(T v_{k}\right) \stackrel{?}{=} \sum_{i} a_{i j} \varphi_{i}\left(v_{k}\right), \quad j=1, \ldots, m, \quad k=1, \ldots, n .
$$

## Matrix of dual map

## Formula

The matrices $\mathcal{M}(T)$ and $\mathcal{M}\left(T^{\prime}\right)$ are transposes of each other.
The left-hand side of the equality to be verified is

$$
\psi_{j}\left(\sum_{i} a_{i k} w_{i}\right)=\sum_{i} a_{i k} \psi_{j}\left(w_{i}\right)=a_{j k}
$$
the point being that $\psi_{j}\left(w_{i}\right)$ is 0 except when $i=j$, when it's 1 .
The right-hand side $\sum_{i} a_{j i} \varphi_{i}\left(v_{k}\right)$ also collapses to a single
term, and for the same reason: $\varphi_{i}\left(v_{k}\right)$ is 0 except when $i=k$ (when it's 1 ). The single term is $a_{j k}$, as for the left-hand side.

## Column rank = row rank

Let $A$ be an $m \times n$ matrix. The matrix $A$ defines a linear map $T: \mathbf{F}^{n} \rightarrow \mathbf{F}^{m}$ whose dual $T^{\prime}$ has matrix $A^{\mathrm{t}}$. The statement that the column ranks of $A$ and $A^{\mathrm{t}}$ are the same is the statement that range $T$ and range $T^{\prime}$ have the same dimension.

## Proposition

If $T$ is a linear map between finite-dimensional vector spaces, then the ranges of $T$ and $T^{\prime}$ have equal dimensions.

After we prove this proposition, we will have a conceptual proof of the coincidence between the row and column ranks of a matrix.

## Annihilator

The annihilator of a subspace $U \subseteq V$ of $V$ is the subspace

$$
U^{0}=\left\{\varphi \in V^{\prime} \mid \varphi_{\mid U}=0\right\} .
$$

This is the space of vectors of $V^{\prime}$ that are "perpendicular to $U^{\prime \prime}$ in the dot product language that relates to $V \times V^{\prime} \longrightarrow \mathbf{F}$.

## Proposition (3.125)

If $U$ has dimension $d$ and $V$ has dimension $n$, then $U^{0}$ has dimension $n-d$.

Proof: Choose a basis $v_{1}, \ldots, v_{d}$ of $U$ and extend it to a basis $v_{1} \ldots, v_{n}$ of $V$. Let $\varphi_{1}, \ldots, \varphi_{n}$ be the basis of $V^{\prime}$ dual to the chosen basis of $V$. If $\varphi=b_{1} \varphi_{1}+\cdots+b_{n} \varphi_{n}$ is an element of $V^{\prime}$, its restriction to $U$ is 0 if and only if $\varphi\left(v_{j}\right)=0$ for $j=1, \ldots, d$. Since $\varphi\left(v_{j}\right)=b_{j}, \varphi$ is in the annihilator of $U$ if and only if $b_{1}=\cdots=b_{d}=0$, i.e., if and only if $\varphi \in \operatorname{span} \varphi_{d+1}, \ldots, \varphi_{n}$. This span has dimension $n-d$; thus $\operatorname{dim} U^{0}=n-d$.

## Annihilator

## Proposition <br> If $V$ has finite dimension, then $\operatorname{dim} U^{0}=\operatorname{dim} V-\operatorname{dim} U$.

This is the same proposition as on the previous slide. Another view of the proof:
By definition, $U^{0}$ is the null space of the restriction map $\varphi \in V^{\prime} \mapsto \varphi_{\mid U} \in U^{\prime}$. This map is surjective; in other words, every linear funtional $\alpha: U \rightarrow \mathbf{F}$ can be extended to a linear functional $V \rightarrow \mathbf{F}$. Indeed, if $X$ is a vector space complement to $U$ in $V$, then an extension of $\alpha$ to $V$ is given by $v=u+x \mapsto \alpha(u)$. Hence rank-nullity implies that $\operatorname{dim} U^{0}=\operatorname{dim} V^{\prime}-\operatorname{dim} U^{\prime}=\operatorname{dim} V-\operatorname{dim} U$.
Given that $\operatorname{dim} U^{0}=\operatorname{dim} V / U$, one might suspect that there is a relation between these two spaces. Can you guess the relation?

## Annihilators

If $V$ is finite-dimensional, then

$$
U=V \Longleftrightarrow \operatorname{dim} U^{0}=0 \Longleftrightarrow U^{0}=0
$$
and
$$
U=\{0\} \Longleftrightarrow \operatorname{dim} U^{0}=\operatorname{dim} V^{\prime} \Longleftrightarrow U^{0}=V^{\prime} .
$$

If you're keeping score at home, this is 3.127 on page 111 of LADR.

## $T$ and $T^{\prime}$ : null spaces and ranges

## Proposition

If $T: V \rightarrow W$ is a linear map, then the null space of $T^{\prime}$ is the annihilator of the range of $T$.

Since range $T \subseteq W$, (range $T)^{0}$ is a subspace of $W^{\prime}$. So is the null space of $T^{\prime}$. Thus the two space being compared are subspaces of the same vector space.

The null space of $T^{\prime}$ consists of all $\varphi \in W^{\prime}$ satisfying $0=\varphi \circ T$. This condition means that $(\varphi T) v=0$ for all $v \in V$. Rewrite this condition as $\varphi(T v)=0$ for all $v \in V$ and then $\varphi W=0$ for all $w \in$ range $T$. Finally, this is now the condition that $\varphi \mid$ range $T=0$, i.e., that $\varphi$ belongs to the annihilator of range $T$.

## $T$ and $T^{\prime}$ : null spaces and ranges

In the context of the proposition on the previous slide, suppose now that $V$ and $W$ have finite dimension.

## Corollary

The dimension of null $T^{\prime}$ is $\operatorname{dim}$ null $T+\operatorname{dim} W-\operatorname{dim} V$.
Proof: Since the nullspace of $T^{\prime}$ is the annihilator of the range of $T, \operatorname{dim}$ null $T^{\prime}=\operatorname{dim} W-\operatorname{rank} T$. By the rank-nullity formula, rank $T=\operatorname{dim} V-\operatorname{dim}$ null $T$. The desired formula follows.

## $T$ and $T^{\prime}$ : null spaces and ranges

## Corollary

The dimension of null $T^{\prime}$ is $\operatorname{dim}$ null $T+\operatorname{dim} W-\operatorname{dim} V$.

## Corollary

The linear map $T$ is onto if and only if its dual $T^{\prime}$ is 1-1.
Proof: The map $T^{\prime}$ is 1-1 if and only if its nullspace has dimension 0 . This is true if and only if

$$
\operatorname{dim} V \stackrel{?}{=} \operatorname{dim} \operatorname{null} T+\operatorname{dim} W .
$$

The rank-nullity formula gives

$$
\operatorname{dim} V=\operatorname{dim} \operatorname{null} T+\operatorname{dim} \operatorname{range} T .
$$

Hence the formula $\operatorname{dim} V \stackrel{?}{=} \operatorname{dim} \operatorname{null} T+\operatorname{dim} W$ is equivalent to the equality $\operatorname{dim} W=\operatorname{dim}$ range $T$, i.e., to the surjectivity of $T$.

## Row rank = column rank

## Corollary

If $T: V \rightarrow W$ is a linear map between finite-dimensional vector spaces, then $T^{\prime}$ and $T$ have equal ranks.

Proof: We proved $\operatorname{dim} \operatorname{null} T^{\prime}=\operatorname{dim} \operatorname{null} T+\operatorname{dim} W-\operatorname{dim} V$, which is eqivalent to

$$
\operatorname{dim} V-\operatorname{dim} n u l l T=\operatorname{dim} W-\operatorname{dim} n u l l T^{\prime} .
$$

Note that $\operatorname{dim} W=\operatorname{dim} W^{\prime}$. Hence the right-hand side is rank $T^{\prime}$, while the left-hand side is rank $T$. (We used rank-nullity on each side.)

## range $T^{\prime}$, annihilator of null $T$

## Corollary

In the context above, the range of $T^{\prime}$ is the annihilator of the null space of $T$.

The two spaces being compared are subspaces of $V^{\prime}$. They have equal dimension: Indeed, the dimension of the annihilator of the null space of $T$ is $\operatorname{dim} V-\operatorname{dim}$ null $T=\operatorname{dim}$ range $T$. Since dim range $T^{\prime}=\operatorname{dim}$ range $T$, the dimensions agree.
It follows that the equality range $T^{\prime}=(\text { null } T)^{0}$ is equivalent to the inclusion range $T^{\prime} \subseteq(\text { null } T)^{0}$. This inclusion is the statement that the range of $T^{\prime}$ annihilates the null space of $T$. Now an element of the range of $T^{\prime}$ is a linear functional $T^{\prime} \psi=\psi \circ T$, where $\psi$ is a linear functional on $W$. The annihilation is the statement $(\psi \circ T) v=0$ if $T v=0$. This is now clear because $(\psi \circ T) v=\psi(T v)=\psi(0)=0$.

