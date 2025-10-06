---
course: CS 189
semester: Fall 2025
type: pdf
title: sep15
source_type: pdf
source_file: sep15.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: sep15-processed
---

# Lists going wild 

## Professor K. A. Ribet

September 15, 2025

## Office Hours

## 885 Evans Hall Mondays, 1:30-3 PM Thursdays, 10:30-noon

Office full $\Longrightarrow$ possible move to a nearby classroom
See you this afternoon in Evans?

I plan to come to the DCs at least once per week. There will be official Residential Life "lunches with Professor Ribet" at noon at Foothill DC on September 18, September 26, October 3 and October 9. There will also be additional lunch gatherings at DCs and the Faculty Club.
Gatherings are optional and not part of Math 110, but I'll continue to list them on slides for those who are interested. Also, you can send me email to subscribe to email announcements.

- Faculty Club lunch today, September 15 at noon
- Crossroads lunch Wednesday, September 17 at 12:30 PM
- First official Lunch with Prof. Ribet on Thursday, September 18 after office hour ends.
Maybe see you at one of these events?


## What we did on Friday

## Lemma (2.19)

Let $v_{1}, \ldots, v_{\ell}$ be a linearly dependent list of vectors of $V$. Then there is some index $k$ such that $v_{k}$ lies in the span of $v_{1}, \ldots, v_{k-1}$. For this $k$, the span of the list with $v_{k}$ deleted is the same as the span of the list $v_{1}, \ldots, v_{\ell}$.

## Theorem (2.22, p. 35)

In a finite-dimensional vector space, the length of every linearly independent list of vectors is less than or equal to the length of every spanning list of vectors.

## What we did on Friday

## Theorem (2.25)

Every subspace of a finite-dimensional vector space is finite-dimensional.

## Proposition

If a vector space has two bases, they are of the same length.

## Waning moments of Friday's class

## Theorem (2.30)

Every spanning list in a vector space can be reduced to a basis of the vector space.

This means that if $v_{1}, \ldots, v_{m}$ spans, we can get a basis of $V$ by ejecting some of the $v_{j}$ from the list.
We can view the proof as an induction on the length of the spanning list $v_{1}, \ldots, v_{m}$. If $m=0, V=\{0\}$ and the spanning list (which is empty) is also a basis. If $m=1$, the list is a single vector, say $v$. Because it spans, $V=\mathbf{F} \cdot v$. If $v$ is nonzero, the list $v$ is linearly independent and represents a basis. If $v=0$, remove it to get the empty list, which is a basis of $V=\{0\}$.

## Refining a spanning list to get a basis

In the induction step, take a spanning list $v_{1}, \ldots, v_{m}$ of length $>1$. If it's linearly independent, it's a basis and we're done.

If it's linearly dependent, some vector in the list is a linear combination of the previous vectors. We can chuck it without disturbing the span of the list-which is all of $V$. The pruned list is still a spanning list, but now it has length $m-1$. Assuming the desired result for lists of that length, we deduce that the pruned list can be pruned further, if necessary, to yield a basis of $V$.

## Finite-dimensional spaces have bases

Corollary (2.31)
If $V$ is finite-dimensional, it has a basis.
To prove it, take a spanning list and prune it if necessary to get a basis.

## Extending linearly independent lists

## Proposition (2.32)

Every linearly independent list of vectors in a finite-dimensional vector space can be extended to a basis of the vector space.

Start with a linearly independent list $v_{1}, \ldots, v_{n}$ in a finite-dimensional vector space $V$. The idea is to extend the list incrementally until it spans (and thus is a basis).
Ask first whether $v_{1}, \ldots, v_{n}$ is already a basis. If not, there is $v_{n+1} \notin \operatorname{span}\left(v_{1}, \ldots, v_{n}\right)$. The key is that $v_{1}, \ldots, v_{n}, v_{n+1}$ is then again linearly independent (to be explained in class). Does it span? If not, it can be grown a second time to a longer linearly independent list $v_{1}, \ldots, v_{n}, v_{n+1}, v_{n+2}$.
list doesn't span? $\Longrightarrow$ it can grow.
However, a linearly independent list can't be longer than $\operatorname{dim} V$. Growth has to stop. When it does, we have a linearly independent spanning list (= a basis).
![The image is a sign affixed to a trash bin, emphasizing proper coat storage etiquette. It features a](https://cdn.mathpix.com/cropped/2025_10_06_011b1bbd0f2788a7ae25g-12.jpg?height=594&width=967&top_left_y=141&top_left_x=147)

**Image description:** The image is a sign affixed to a trash bin, emphasizing proper coat storage etiquette. It features a simple design with bold text stating, "PLEASE HANG UP COATS." The following text reads, "They do not need to be washed after every use." The sign is surrounded by a black border with a logo labeled "California." Its purpose is to encourage individuals to hang coats instead of discarding them, promoting sustainable practices and cleanliness.


## Existence of complements

## Theorem (2.33)

Every subspace of a finite-dimensional vector space has a complement in the larger space.

In symbols: If $V$ is finite-dimensional and $X \subseteq V$ is a subspace of $V$, then there is a subspace $Y$ of $V$ such that $V=X \oplus Y$.

To prove this, note first that $X$ is finite-dimensional by 2.25 . Let $x_{1}, \ldots, x_{t}$ be a basis of $X$. Extend extend this linearly independent list of vectors of $V$ to a basis $x_{1}, \ldots, x_{t} ; y_{1}, \ldots, y_{d}$ of $V$. Let $Y=\operatorname{span}\left(y_{1}, \ldots, y_{d}\right)$.
We will show that $V=X+Y$ and that $X+Y=X \oplus$.

## A very important theorem

We first show that $V=X+Y$. Take a vector $v \in V$; the aim is to write it as a sum of a vector in $X$ and a vector in $Y$.

Write $v$ is a linear combination of the $t+d$ basis vectors of $V$ :

$$
v=\left(\lambda_{1} x_{1}+\cdots+\lambda_{t} x_{t}\right)+\left(\mu_{1} y_{1}+\cdots+\mu_{d} y_{d}\right) .
$$

Observe that the first summand is in $X$, while the second is in $Y$. Hence $v$ belongs to $X+Y$, as desired.
To show that $X+Y$ is a direct sum, it suffices to show that $X \cap Y \stackrel{?}{=}\{0\}$. If $v$ is in the intersection, then there are coefficients $\lambda_{j}$ and $\mu_{k}$ so that

$$
v=\lambda_{1} x_{1}+\cdots+\lambda_{t} x_{t}=\mu_{1} y_{1}+\cdots+\mu_{d} y_{d}
$$

Then

$$
0=\left(\lambda_{1} x_{1}+\cdots+\lambda_{t} x_{t}\right)-\left(\mu_{1} y_{1}+\cdots+\mu_{d} y_{d}\right) .
$$

By the linear independence of the basis of $V$, all the $\lambda_{j}$ and $\mu_{k}$ are 0 . Hence $v=0$, as required.

## Dimensions

The dimension of a finite-dimensional vector space $V$ is the length of a basis of $V$. This makes sense because all bases have the same number of elements (proved on Friday). One writes $\operatorname{dim} V$ for the dimension.

## Proposition

Let $V$ be a finite-dimensional vector space, and let $X$ be a subspace of $Y$. If $Y$ is a subspace of $V$ such that $V=X \oplus Y$, then

$$
\operatorname{dim} V=\operatorname{dim} X+\operatorname{dim} Y .
$$

To prove the proposition, take bases $x_{1}, \ldots, x_{t}$ and $y_{1}, \ldots, y_{d}$ of $X$ and $Y$. The equality of the proposition would follow from the statement that the concatenated list $x_{1}, \ldots, x_{t} ; y_{1}, \ldots, y_{d}$ is a basis of $V$. This means that the big lists spans $V$ and is linearly independent.

Let's prove the first statement, leaving the second as an exercise. If $v$ is an element of $V$, we may write it $x+y$ with $x \in X, y \in Y$. The two summands are linear combinations of the lists $x_{1}, \ldots, x_{t}$ and $y_{1}, \ldots, y_{d}$, respectively. Hence $x+y$ is a linear combination of the big list $x_{1}, \ldots, x_{t} ; y_{1}, \ldots, y_{d}$.

## Dimensions of subspaces

Proposition (2.37)
If $X$ is a subspace of a finite-dimensional vector space $V$, then $\operatorname{dim} X \leq \operatorname{dim} V$.

To prove this, we can for example choose a complement $Y$ to $X$. Then

$$
\operatorname{dim} V=\operatorname{dim} X+\operatorname{dim} Y \geq \operatorname{dim} X .
$$

## Subspace of big dimension

Proposition (2.39)
If $X \subseteq V$ is a subspace of the finite-dimensional vector space $V$, and if $\operatorname{dim} X=\operatorname{dim} V$, then $X=V$.

Form $Y$ so that $V=X \oplus Y$. By the previous proposition and the hypothesis that $\operatorname{dim} X=\operatorname{dim} V, Y$ has dimension 0 . Hence it is the vector space $\{0\}$. Then $V=X \oplus\{0\}=X$.

## Long linearly independent lists

## Proposition (2.38)

If $V$ is a finite-dimensional vector space and $v_{1}, \ldots, v_{\ell}$ is a linearly independent list of length $\ell=\operatorname{dim} V$, then $v_{1}, \ldots, v_{\ell}$ is a basis of $V$.

Let $U=\operatorname{span}\left(v_{1}, \ldots, v_{\ell}\right)$. Then the list is a basis of $U$ (linearly independent and spanning). Hence $\operatorname{dim} U=\ell=\operatorname{dim} V$. Hence $U=V$ by the previous proposition. Thus $v_{1}, \ldots, v_{\ell}$ spans $V$ and is a basis of $V$.

## Short spanning lists

## Proposition

Assume that $v_{1}, \ldots, v_{\ell}$ is a spanning list for $V$ and again that $\ell=\operatorname{dim} V$. Then $v_{1}, \ldots, v_{\ell}$ is a basis of $V$.

We can prune this spanning list if necessary to get a basis of $V$ (Theorem 2.30). The resulting basis has length $\operatorname{dim} V$, which happens also to be the length of the unpruned list. Hence no pruning happens.

## Dimension of a sum

Let $X$ and $Y$ be subspaces of $V$, with $V$ of finite dimension. What is the dimension of $X+Y$ ? If the sum $X+Y$ is direct, then

$$
\operatorname{dim}(X+Y)=\operatorname{dim}(X \oplus Y)=\operatorname{dim} X+\operatorname{dim} Y
$$
by the proposition a few slides ago. If the sum $X+Y$ is not assumed to be direct, then $X \cap Y$ may be different from $\{0\}$. In that case, we get a modified version of the formula above.

## Theorem (2.43)

The dimension of $X+Y$ is $\operatorname{dim} X+\operatorname{dim} Y-\operatorname{dim}(X \cap Y)$.
The proof in LADR (p. 47) is unappetizing. I will explain soon how the formula follows from the Fundamental Theorem for Linear Maps.

## Dimension of a cartesian product

## Proposition

Suppose that $V$ and $W$ are finite-dimensional vector spaces.
Then $\operatorname{dim} V \times W=\operatorname{dim} V+\operatorname{dim} W$.
The space $V \times W$ is the direct sum of its subspaces $V \times\{0\}$ and $\{0\} \times W$. It suffices to show that the first space has dimension $\operatorname{dim} V$ and that the second has dimension $\operatorname{dim} W$.

The point is that $V \times\{0\}=\{(V, 0) \mid V \in V\}$ is the "same thing" as $V$-it's just that we tack on 0 as a second entry when we write vectors of $V$. In particular, if $v_{1}, \ldots, v_{t}$ is a basis of $V$, then $\left(v_{1}, 0\right), \ldots,\left(v_{t}, 0\right)$ is basis of $V \times\{0\}$. Hence $\operatorname{dim}(V \times\{0\})=\operatorname{dim} V$. Similarly $\operatorname{dim}(\{0\} \times W)=\operatorname{dim} W$.

## Lists and linear maps

A linear map (Chapter 3) between $\mathbf{F}$-vector spaces $V$ and $W$ is a function $T: V \rightarrow W$ that takes sums to sums and scalar products to scalar products. (We've seen the formal definition several times.)

- The space $V$ is the domain or source of $T$.
- The null space of $T$, null $T$, is the set of $v \in V$ such that $T V=0$. The null space is a subspace of $V$.
- If null $T$ is finite-dimensional (for example because $V$ is finite-dimensional), the dimension of null $T$ is called the nullity of $T$.
- The range or image of $T$ is the set of all $T v$. The range is a subspace of $W$.
- If range $T$ is finite-dimensional (for example because $W$ is finite-dimensional), the dimension of range $T$ is called the rank of $T$.


## Fundamental theorem

## Theorem (3.21)

If $T: V \rightarrow W$ is a linear map and $V$ is finite-dimensional, then range $T$ and null $T$ are also finite-dimensional, and

$$
\operatorname{dim} V=\operatorname{dim} n u l l T+\operatorname{dim} \operatorname{range} T .
$$

The nullity and rank of a linear map add up to the dimension of the source (or domain) of the linear map.

We can prove this very soon, but first l'll explain how the formula 2.43 of LADR is a special case.

## The summation map

Let $X$ and $Y$ be subspaces of a finite-dimensional vector space $V$. Consider the summation map

$$
S: X \times Y \rightarrow V, \quad(x, y) \mapsto x+y .
$$

The dimension of the domain of this map is $\operatorname{dim} X+\operatorname{dim} Y$, as we have seen. The range of the map is $X+Y$. By the Fundamental Theorem,

$$
\operatorname{dim} X+\operatorname{dim} Y=\operatorname{dim}(X+Y)+\text { nullity } S
$$

The null space of $S$ is the set of pairs $(v,-v)$ with $v \in X \cap Y$. Hence it is the same thing as $X \cap Y$ and has dimension equal to $\operatorname{dim}(X \cap Y)$. Thus

$$
\operatorname{dim} X+\operatorname{dim} Y=\operatorname{dim}(X+Y)+\operatorname{dim}(X \cap Y),
$$
so that 2.43 is a consequence of the Fundamental Theorem.

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

