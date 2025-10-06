---
course: CS 189
semester: Fall 2025
type: pdf
title: sep17
source_type: pdf
source_file: sep17.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: sep17-processed
---

# Fundamental theorem, among other things 

## Professor K. A. Ribet

September 15, 2025

## "Office" Hours

- 732 Evans Hall

Monday, 1:30-3 PM
Thursday, 10:30-noon
![The image shows an academic lecture room featuring rectangular tables arranged in rows, with blue an](https://cdn.mathpix.com/cropped/2025_10_06_e25a3bf6b0c3cb3eb964g-02.jpg?height=331&width=440&top_left_y=128&top_left_x=695)

**Image description:** The image shows an academic lecture room featuring rectangular tables arranged in rows, with blue and green chairs. On the left wall, hexagonal panels display colorful geometric designs, likely illustrating mathematical concepts such as symmetry or tessellations. The right wall contains a whiteboard, typically used for presentations or equations. This setting is designed for collaborative learning and visual engagement, enhancing the teaching of mathematical principles. No diagrams or equations are visible in this image.


We always moved to this room anyway.

Come to office hours, even only once. Even just to introduce yourself. Even only to meet other students and to hear what they have to say.

I have lunch in the DCs at least once per week and at the Faculty Club at least once per week. Joine me!
Check out the official Residential Life "Lunches with Professor Ribet" at noon at Foothill DC on September 18, September 26, October 3 and October 9. There will also be additional lunch gatherings at DCs and the Faculty Club.

Gatherings are optional and not part of Math 110, but I'll continue to list them on slides for those who are interested.
Send me email if you wish to subscribe to email announcements.

- Crossroads lunch today at 12:30 PM
- Official "Lunch with Professor Ribet" tomorrow noonish


## Long linearly independent lists

Proposition (2.38)
If $V$ is a finite-dimensional vector space and $v_{1}, \ldots, v_{\ell}$ is a linearly independent list of length $\ell=\operatorname{dim} V$, then $v_{1}, \ldots, v_{\ell}$ is a basis of $V$.

We proved this on Monday toward the end of class.

## Short spanning lists

## Proposition

Assume that $v_{1}, \ldots, v_{\ell}$ is a spanning list for $V$ and again that $\ell=\operatorname{dim} V$. Then $v_{1}, \ldots, v_{\ell}$ is a basis of $V$.

We can prune this spanning list if necessary to get a basis of $V$ (Theorem 2.30). The resulting basis has length $\operatorname{dim} V$, which happens also to be the length of the unpruned list. Hence no pruning happens.

## Summary

Suppose that $V$ is a finite-dimensional vector space and that $v_{1}, \ldots, v_{\ell}$ is a list of vectors of $V$. Consider the following three statements:
(1) The list is linearly independent.
(2) The list spans $V$.
(3) The length of the list is $\operatorname{dim} V$.

Any two of them implies the third. To say that all three statements are true is to say that the list is a basis of $V$.

## Dimension of a sum

Let $X$ and $Y$ be subspaces of $V$, with $V$ of finite dimension. What is the dimension of $X+Y$ ? If the sum $X+Y$ is direct, then

$$
\operatorname{dim}(X+Y)=\operatorname{dim}(X \oplus Y)=\operatorname{dim} X+\operatorname{dim} Y
$$
by a proposition that we proved on Monday. If the sum $X+Y$ is not assumed to be direct, then $X \cap Y$ may be different from $\{0\}$. In that case, we get a modified version of the formula above (mentioned on Monday):

## Theorem (2.43)

The dimension of $X+Y$ is $\operatorname{dim} X+\operatorname{dim} Y-\operatorname{dim}(X \cap Y)$.
I'll derive this formula from the Fundamental Theorem for Linear Maps.

A linear map (Chapter 3) between $\mathbf{F}$-vector spaces $V$ and $W$ is a function $T: V \rightarrow W$ that takes sums to sums and scalar products to scalar products. (We've seen the formal definition several times.) Some terminology:

- The domain or source of $T$ is the space $V$.
- The null space of $T$, null $T$, is the set of $v \in V$ such that $T v=0$. The null space is a subspace of $V$.
- If null $T$ is finite-dimensional (for example because $V$ is finite-dimensional), the dimension of null $T$ is called the nullity of $T$.
- The range or image of $T$ is the set of all Tv. The range is a subspace of $W$.
- If range $T$ is finite-dimensional (for example because $W$ is finite-dimensional), the dimension of range $T$ is called the rank of $T$.


## Fundamental theorem

## Theorem (3.21)

If $T: V \rightarrow W$ is a linear map and $V$ is finite-dimensional, then range $T$ and null $T$ are also finite-dimensional. Moreover, the dimension of $V$ is the sum of the rank and the nullity of $T$.

You may know this result as the rank-nullity theorem.

## Linear maps and lists

Suppose that $T: V \rightarrow W$ is a linear map and that $v_{1}, \ldots, v_{\ell}$ is a list in $V$. Then $T v_{1}, \ldots, T v_{\ell}$ is a list in $W$.

## Proposition

If $v_{1}, \ldots, v_{\ell}$ spans $V$, then $T v_{1}, \ldots, T v_{\ell}$ spans range $T$.
Proof: Since the list spans $V$, for each $v \in V$, there are $\lambda_{1}, \ldots, \lambda_{\ell} \in \mathbf{F}$ such that $v=\lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}$. Apply $T$ and use linearity to get $T v=\lambda_{1} T v_{1}+\cdots+\lambda_{\ell} T v_{\ell}$.

Now range $T$ is the set of all $T v$ with $v \in V$. Because each $T v$ is a linear combination of $T v_{1}, \ldots, T v_{\ell}$, this list spans range $T$.

## Corollary

If $T$ is surjective, then the image under $T$ of a spanning list for $V$ is a spanning list for $W$.

To say that $T$ is surjective is to say that range $T$ is all of $W$, so that the Corollary follows from the Proposition above it.

## Linear maps and lists

Suppose that $T: V \rightarrow W$ is a linear map and that $v_{1}, \ldots, v_{\ell}$ is a list in $V$.

## Proposition

If $T$ is injective and $v_{1}, \ldots, v_{\ell}$ is linearly independent, then $T v_{1}, \ldots, T v_{\ell}$ is a linearly independent list in $W$.

Proof: Assume that $\lambda_{1} T v_{1}+\cdots+\lambda_{\ell} T v_{\ell}=0$, where the $\lambda_{j}$ are scalars in $\mathbf{F}$. To prove linear independence of the list in $W$ is to show that the scalars $\lambda_{j}$ are all 0 . By linearity, we may write this equation as

$$
0=T\left(\lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}\right) .
$$

Thus $\lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}$ lies in the null space of $T$. The hypothesis that $T$ is $1-1$ ensures that this null space is $\{0\}$. Hence

$$
\lambda_{1} v_{1}+\cdots+\lambda_{\ell} v_{\ell}=0 .
$$

By the linear independence of $v_{1}, \ldots, v_{\ell}$, the scalars $\lambda_{j}$ are all 0 . This is the conclusion that we sought.

## Summary

Suppose that $T: V \rightarrow W$ is a linear map and that $v_{1}, \ldots, v_{\ell}$ is a list in $V$.

- If $T$ is 1-1 and $v_{1}, \ldots, v_{\ell}$ is linearly independent, then $T v_{1}, \ldots, T v_{\ell}$ is linearly independent.
- If $T$ is onto and $v_{1}, \ldots, v_{\ell}$ spans $V$, then $T v_{1}, \ldots, T v_{\ell}$ spans $W$.
- If $T$ is $1-1$ and onto, and if $v_{1}, \ldots, v_{\ell}$ is a basis of $V$, then $T v_{1}, \ldots, T v_{\ell}$ is basis of $W$.
The third item is new, but it follows from the first two items.


## Corollary

If $T: V \rightarrow W$ is a linear map that is $1-1$ and onto, and if $V$ is finite-dimensional, then so is $W$. Moreover, in this case the dimensions of $V$ and $W$ are equal.

## Fundamental theorem

Theorem (3.21)
If $T: V \rightarrow W$ is a linear map and $V$ is finite-dimensional, then range $T$ and null $T$ are also finite-dimensional, and

$$
\operatorname{dim} V=\operatorname{dim} n u l l T+\operatorname{dim} \operatorname{range} T .
$$

## Proof of the rank-nullity theorem

Let $T: V \rightarrow W$ be a linear map, with $V$ finite-dimensional. Let $X=$ null $T$, and let $Y \subseteq V$ be a vector space complement to $X$ in $V$. This means that each $v \in V$ is uniquely a sum $x+y$ with $x \in X$ and $y \in Y$. Because $\operatorname{dim} V=\operatorname{dim} X+\operatorname{dim} Y$, $\operatorname{dim} Y$ is the difference $\operatorname{dim} V-$ nullity $T$. It suffices to show that $Y$ and range $T$ have the same dimension.

Let $R: Y \rightarrow$ range $T$ be the restriction of $T$ to $Y$, regarded as taking values in range $T$. Thus $R y=T y$ for $y \in Y$, by definition.

The map $R$ is linear; if we can show that it is both $1-1$ and onto, we can then deduce the required equality $\operatorname{dim}$ range $T=\operatorname{dim} Y$.

## Proof of the fundamental theorem

Let $R: Y \rightarrow$ range $T$ be the restriction of $T$ to $Y$, regarded as taking values in range $T$. Thus $R y=T y$ for $y \in Y$, by definition. The map $R$ is linear; it suffices to show that it is both $1-1$ and onto.

Onto: Each element of range $T$ is of the form $T v$ with $v \in V$. Write $v=x+y$. Then

$$
T v=T(x+y)=T x+T y=0+T y=R y .
$$

Thus each element of range $T$ is in the range (= image) of $R$.
1-1: It is equivalent to show that the null space of $R$ is $\{0\}$. Let $y$ be in this null space. Then $0=R y=T y$, so that $y \in \operatorname{null} T=X$. Thus $y \in Y \cap X=\{0\}$, so that $y=0$.

## Formula for $\operatorname{dim}(X+Y)$

We return to the situation of an earlier slide: $X$ and $Y$ are subspaces of a finite-dimensional space $V$, and we wish to deduce the formula

$$
\operatorname{dim}(X+Y)=(\operatorname{dim} X+\operatorname{dim} Y)-\operatorname{dim}(X \cap Y)
$$
from the rank-nullity formula.
Consider as usual the linear map
$$
S: X \times Y \rightarrow V, \quad(x, y) \mapsto x+y .
$$

The range of $S$ is $X+Y$. We need to check two things:

- The dimension of $X \cap Y$ is the dimension of null $S$.
- The dimension of $X \times Y$ is $\operatorname{dim} X+\operatorname{dim} Y$.


## Formula for $\operatorname{dim}(X+Y)$

Consider as usual the linear map

$$
S: X \times Y \rightarrow V, \quad(x, y) \mapsto x+y .
$$

The range of $S$ is $X+Y$. We need to check two things:

- The dimension of $X \cap Y$ is the dimension of null $S$.
- The dimension of $X \times Y$ is $\operatorname{dim} X+\operatorname{dim} Y$.

For the first item, note that null $S=\{(t, t) \mid t \in X \cap Y\}$. Formally, the map

$$
t \in X \cap Y \mapsto(t,-t) \in \operatorname{null} S
$$
is a linear map that is $1-1$ and onto. Hence the two spaces have the same dimension.

The second item requires its own slide. . . .

## Dimension of a cartesian product

## Proposition

Suppose that $X$ and $Y$ are finite-dimensional vector spaces.
Then $\operatorname{dim} X \times Y=\operatorname{dim} X+\operatorname{dim} Y$.
The space $X \times Y$ is the direct sum of its subspaces $X \times\{0\}$ and $\{0\} \times Y$. We will see that the first space has dimension $\operatorname{dim} X$ and that the second has dimension dim $Y$. The formula will then follow from the formula for the dimension of a direct sum.

The point is that $X \times\{0\}=\{(x, 0) \mid x \in X\}$ is the "same thing" as $X$-it's just that we tack on 0 as a second entry when we write vectors of $X$. In particular, if $x_{1}, \ldots, x_{t}$ is a basis of $X$, then $\left(x_{1}, 0\right), \ldots,\left(x_{t}, 0\right)$ is basis of $X \times\{0\}$. Hence $\operatorname{dim}(X \times\{0\})=\operatorname{dim} X$. Similarly $\operatorname{dim}(\{0\} \times Y)=\operatorname{dim} Y$.

## Isomorphisms

A linear map $T: V \rightarrow W$ is an isomorphism if it both 1-1 and onto. Note that

$$
\begin{aligned}
T \text { is } 1-1 & \Longleftrightarrow \text { null } T=\{0\} ; \\
T \text { is onto } & \Longleftrightarrow \text { range } T=W .
\end{aligned}
$$

By Math 55, $T$ is a bijection (i.e., 1-1 and onto) if and only if there is a function $f: W \rightarrow V$ such that $f \circ T=\operatorname{id}_{V}, T \circ f=\operatorname{id}_{W}$. If $f$ exists, it is unique; it's called the "set-theoretic inverse" of $T$ (viewed as a function).

Because $T$ is linear, its set-theoretic inverse is also linear when $T$ is a bijection. How come? See next slide.

## Inverse of Linear Map is Linear

Start with $T: V \rightarrow W$ linear with $T$ both 1-1 and onto. Let $f$ be the inverse function to $T$. Then the claim is:

- $f\left(w_{1}+w_{2}\right)=f\left(w_{1}\right)+f\left(w_{2}\right)$ for all $w_{1}, w_{2} \in W$;
- $f(\lambda w)=\lambda f(w)$ for all $\lambda \in \mathbf{F}, w \in W$.

Let's check the first item:
$f\left(w_{1}\right)$ is the unique $v_{1} \in V$ such that $T v_{1}=w_{1}$, and similarly $f\left(w_{2}\right)$ is the unique $v_{2} \in V$ such that $T v_{2}=w_{2}$. Then $T\left(v_{1}+v_{2}\right)=T v_{1}+T v_{2}=w_{1}+w_{2}$; thus $v_{1}+v_{2}$ maps to $w_{1}+w_{2}$ under $T$. Since $f\left(w_{1}+w_{2}\right)$ is the unique vector of $V$ mapping to $w_{1}+w_{2}$ under $T$, we conclude
$f\left(w_{1}+w_{2}\right)=v_{1}+v_{2}=f\left(w_{1}\right)+f\left(w_{2}\right)$.
I leave the second item to you.

## Inverse of an isomorphism

If $T: V \rightarrow W$ is both 1-1 and onto, we have just seen that its (set-theoretic) inverse is linear. The linear map inverse to $T$ is usually denoted $T^{-1}$. We could describe the inverse of $T$ as the unique linear map $U: W \rightarrow V$ such that

$$
U \circ T=\mathrm{id}_{V}, \quad T \circ U=\mathrm{id}_{W} .
$$

An isomorphism $V \rightarrow W$ and its inverse $W \rightarrow V$ are linear dictionaries that allow us to pass back and forth between the two spaces $V$ and $W$.

