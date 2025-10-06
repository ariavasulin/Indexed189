---
course: CS 189
semester: Fall 2025
type: pdf
title: oct3
source_type: pdf
source_file: oct3.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: oct3-processed
---

# Quotients, dual spaces 

## Professor K. A. Ribet

![The image depicts the official seal of the University of California, Berkeley. It features an open b](https://cdn.mathpix.com/cropped/2025_10_06_5031b93b1267de65ea1dg-01.jpg?height=213&width=217&top_left_y=427&top_left_x=520)

**Image description:** The image depicts the official seal of the University of California, Berkeley. It features an open book at the center, inscribed with the phrase “LET THERE BE LIGHT.” Above the book is a star, and two hands support the book from below. The circular border displays “UNIVERSITY OF CALIFORNIA” at the top and “BERKELEY” at the bottom, with “1868” denoting the year of establishment. The overall color scheme consists of gold and blue, symbolizing the university's identity and heritage. The seal serves to signify the institution’s authority and academic heritage.


October 3, 2025

## Announcements

The problems in §3D have been moved to HW \#6. The bonus problem has not budged.
My office hours are Mondays 1:30-3 PM and Thursday 10:30 AM-noon

Next to last Foothill DC lunch with Prof. Ribet is today at noon
![The image is a promotional poster for an academic event titled "Lunch with Math Professor Ken Ribet.](https://cdn.mathpix.com/cropped/2025_10_06_5031b93b1267de65ea1dg-02.jpg?height=483&width=377&top_left_y=395&top_left_x=720)

**Image description:** The image is a promotional poster for an academic event titled "Lunch with Math Professor Ken Ribet." It features a prominent headshot of Professor Ribet, positioned centrally. The background is a gradient of blue and yellow, enhancing visibility. Key event details are displayed, including the date (September 18, September 26, October 3, October 9), time (12:00 PM - 1:00 PM), and location (Foothill Dining Hall). A QR code is included for further information. The poster effectively communicates event details, inviting participation from the academic community.

U.S. Cardiologist Warns Aging Seniors About Blueberries for Breakfast

## Quotient spaces

If $U$ is a subspace of $V$, we have defined a vector space $V / U$ along with a surjective map ("quotient map")

$$
\pi: V \rightarrow V U .
$$

If $W$ is a vector space, there is a natural linear map $\mathcal{L}(V / U, W) \rightarrow \mathcal{L}(V, W), f: S \longmapsto S \circ \pi$. A linear map $V \rightarrow W$ of the form $S \circ \pi$ is said to factor through $\pi$.

## Proposition

The function $f$ is an injective linear map $\mathcal{L}(V / U, W) \rightarrow \mathcal{L}(V, W)$ whose image is the set of linear maps $V \rightarrow W$ whose restriction to $U$ is 0 .

A paraphrase: a linear map $V \rightarrow W$ factors through $\pi$ if and only if its null space contains $U$.

## Quotients

## Proposition

The function $f$ is an injective linear map $\mathcal{L}(V / U, W) \rightarrow \mathcal{L}(V, W)$ whose image is the set of linear maps $V \rightarrow W$ whose restriction to $U$ is 0 .
"A linear map $V \rightarrow W$ factors through $\pi$ if and only if its null space contains $U$."
On Wednesday, we proved this. If $T: V \rightarrow W$ is identically 0 on $U$, then $T=S \circ \pi$, where $S: V / U \rightarrow W$ is defined by the formula $S(v+U)=T v$. That $T u=0$ for all $u \in U$ ensures that $S$ is well defined. So we're in good shape, and the class on Wednesday ended at a good breakpoint.

## The range and null space of $S$

Let $T: V \rightarrow W$ be a linear map and let $U \subseteq V$ be a subspace that is contained in null $T$. Let $S$ be the unique linear map $V / U \rightarrow W$ such that $T=S \circ \pi$.

## Proposition

The range of $S$ is the range of $T$.
The range of $S$ is the set of all $S(v+U)$, but $S(v+U)=T v$. Thus the range of $S$ consists of all vectors $T v \in W$ and is therefore the range of $T$.

## Proposition

The null space of $S$ is the quotient (null $T$ ) / $U$.
The null space of $S$ is the set of all $v+U \in V / U$ such that $S(v+U)=0$. This is the set of all $v+U$ for which $T v=0$, i.e., the set of all $v+U$ with $v \in$ null $T$. This is the quotient $($ null $T) / U$ of the proposition.

## LADR's map $\tilde{T}$

A slightly different perspective. Start with a linear map $T: V \rightarrow W$, and let $U=$ null $T$. Then $T=S \circ \pi$ for some $S: V /(\operatorname{null} T) \rightarrow W$.
In LADR, the map $S$ in this situation is called $\tilde{T}$.

## Proposition (3.107)

If $U=$ null $T$, the map $\tilde{T}: V / U \rightarrow W$ is injective. Its range is the range of $T$.

This follows from our more general discussion, since the null space of $\tilde{T}$ is $($ null $T) / U=($ null $T) /($ null $T)=0$.

## Functoriality

Starting with $\pi: V \rightarrow V / U$, we produced and studied a linear $\operatorname{map} \mathcal{L}(V / U, W) \rightarrow \mathcal{L}(V, W)$.
More generally, if $\pi: V \rightarrow X$ is some linear map (not necessarily surjective), there's an induced linear map

$$
\mathcal{L}(X, W) \rightarrow \mathcal{L}(V, W), \quad S \mapsto S \circ \pi
$$

Note that the displayed map goes in the opposite direction from $\pi$ in the sense that $\pi$ goes from $V \rightarrow X$ and the induced map goes from an object related to $X$ to an object related to $V$.
Denizens of Evans Hall would write $\pi^{*}$ for the induced map and say that $\mathcal{L}(\bullet, W)$ depends contravariantly on the first argument -. Remember that "contra" means "against or contrary to"; mathematicians interpret this as "opposite of."

## Functoriality

Suppose that $\alpha$ is a linear map $W \rightarrow Y$. Then composition with $\alpha$ gives rise to a linear map $\mathcal{L}(V, W) \rightarrow \mathcal{L}(V, Y)$ :

$$
\alpha_{*}: \mathcal{L}(V, W) \rightarrow \mathcal{L}(V, Y), \quad T \mapsto \alpha \circ T .
$$

The Evans folk like to say that $\mathcal{L}(V, \bullet)$ is covariant in the second variable.

## Dual space

If $V$ and $W$ are vector spaces, we live happily with $\mathcal{L}(V, W)$. It's a space of linear maps. It's a space of $m \times n$ matrices. We're totally comfortable.
Make the choice $W=\mathbf{F}$. It's a special case, so we're still comfortable. It's a space of $1 \times n$ matrices. Chill.
Now say that $\mathcal{L}(V, F)$ is the dual space $V^{\prime}$. Suddenly we're uneasy.
Refer to the linear maps $V \rightarrow \mathbf{F}$ in $\mathcal{L}(V, \mathbf{F})$ as linear functionals. Our palms are sweaty.
Use Greek letters like $\varphi$ for the linear maps $W \rightarrow \mathbf{F}$ and we're quaking.

Welcome to §3F.

## Dual space

The vector space dual to a space $V$ is $V^{\prime}=\mathcal{L}(V, \mathbf{F})$. If $V$ has dimension $n, V^{\prime}$ has dimension $1 \cdot n=n$.

Suppose that $V$ has a basis $v_{1}, \ldots, v_{n}$. Is there a natural basis of $V^{\prime}$ that results from $v_{1}, \ldots, v_{n}$ ?

This question becomes easier if we replace $\mathbf{F}$ by $W$ and say that $w_{1}, \ldots, w_{m}$ is a basis for $W$. Then $\mathcal{L}(V, W)$ is the space of $m \times n$ matrices; it has a basis consisting of the matrices with a 1 in the $k$ th row and $\ell$ th column and 0 s elsewhere (for $k=1, \ldots, m$ and $\ell=1, \ldots, n$ ). The matrix with a 1 in place $\ell, k$ and 0 s elsewhere represents the linear map sending $v_{k}$ to $w_{\ell}$ and all other basis vectors of $V$ to 0 .

Again: a basis of $\mathcal{L}(V, W)$ is given by the linear maps sending $v_{k}$ to $w_{\ell}$ and all other basis vectors of $V$ to 0 . There is one for each $k$ between 1 and $n$ and each $\ell$ between 1 and $m$.

Now take $W=\mathbf{F}$ and use the list of length one "1" as a basis of $\mathbf{F}$. Then $V^{\prime}=\mathcal{L}(V, \mathbf{F})$ has a basis consisting of the $n$ different linear maps gotten by sending some $v_{k}$ to 1 and all other $v_{j}$ to 0 . If $k$ is between 1 and $n$, the linear map taking $v_{k}$ to 1 and the other basis vectors to 0 is called $\varphi_{k}$.

The linear maps $\varphi_{1}, \varphi_{2}, \ldots, \varphi_{n}$ form a basis of $\mathcal{L}(V, \mathbf{F})$ that is called the "dual basis"; it's the basis dual to $v_{1}, \ldots, v_{n}$.

## Dual basis

Again, if $V$ has dimension $n, V^{\prime}=\mathcal{L}(V, \mathbf{F})$ also has dimension $n$.

If $v_{1}, \ldots, v_{n}$ is a basis of $V, \varphi_{1}, \ldots, \varphi_{n}$ is the basis of $V^{\prime}$ that is dual to $v_{1}, \ldots, v_{n}$. Here's a nifty formula:

$$
\varphi_{k}: v_{j} \mapsto \begin{cases}1 & \text { if } j=k \\ 0 & \text { if } j \neq k .\end{cases}
$$

A more compact version:

$$
\varphi_{k}\left(v_{j}\right)=\delta_{k j}
$$

## Dual basis gives coordinates

Now suppose that $v$ is a vector in $V$. Then there are unique scalars $\lambda_{1}, \ldots, \lambda_{n}$ such that $v=\lambda_{1} v_{1}+\cdots+\lambda_{n} v_{n}$.

Formula (3.114)
For $k=1, \ldots, n, \lambda_{k}=\varphi_{k}(v)$.
Compute to see this:

$$
\varphi_{k}(v)=\varphi_{k}\left(\sum_{j} \lambda_{j} v_{j}\right)=\sum_{j} \lambda_{j} \varphi_{k}\left(v_{j}\right)=\sum_{j} \lambda_{j} \delta_{k j}=\lambda_{k} .
$$

## Special case where $V=\mathbf{F}^{n}$

If $V=\mathbf{F}^{n}$ and the basis is the standard basis $e_{1}, \ldots, e_{n}$, then the $\lambda_{j}$ for a vector $v=\left(x_{1}, \ldots, x_{n}\right)$ are the coordinates $x_{1}, x_{2}$, etc. Thus

$$
\varphi_{k}\left(\left(x_{1}, \ldots, x_{n}\right)\right)=x_{k}
$$
for all $k=1, \ldots, n$.

## Dual map

If $T: V \rightarrow W$ is a linear map, there is an induced linear map

$$
T^{*}: \mathcal{L}(W, \mathbf{F}) \rightarrow \mathcal{L}(V, \mathbf{F}), \quad \psi \in \mathcal{L}(W, \mathbf{F}) \longmapsto \psi \circ T \in \mathcal{L}(V, \mathbf{F}) .
$$

This map is called $T^{\prime}$ and is said to be the map dual to $T$. We write

$$
T^{\prime}: W^{\prime} \rightarrow V^{\prime} .
$$

## Dual maps

Axler lists a number of basic properties of dual maps ( 3.120 in LADR):

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

## Formula

The matrices $\mathcal{M}(T)$ and $\mathcal{M}\left(T^{\prime}\right)$ are transposes of each other.

## Matrix of dual map

## Formula

The matrices $\mathcal{M}(T)$ and $\mathcal{M}\left(T^{\prime}\right)$ are transposes of each other.
If $\mathcal{M}\left(T^{\prime}\right)=\left(b_{i j}\right)$, then $T^{\prime}\left(\psi_{j}\right)=\psi_{j} T=\sum_{i} b_{i j} \varphi_{i}$ for each
$j=1, \ldots, m$. The formula to be proved is $b_{i j}=a_{j i}$ for each $i$ and $j$. l.e., the formula states (for each $j$ )

$$
\psi_{j} T \stackrel{?}{=} \sum_{i} a_{j i} \varphi_{i}
$$

The two sides of the desired equality are linear maps $V \rightarrow W$. A key fact is that two linear maps $V \rightarrow W$ are equal if they agree on the basis vectors $v_{1}, \ldots, v_{n}$.
Thus the formula is equivalent to the equality

$$
\psi_{j}\left(T v_{k}\right) \stackrel{?}{=} \sum_{i} a_{j i} \varphi_{i}\left(v_{k}\right)
$$
for all $j=1, \ldots, m$ and $k=1, \ldots, n$.

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

