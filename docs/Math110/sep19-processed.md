---
course: CS 189
semester: Fall 2025
type: pdf
title: sep19
source_type: pdf
source_file: sep19.pdf
processed_date: '2025-10-06'
processor: mathpix
slug: sep19-processed
---

# More on linear maps 

## Professor K. A. Ribet

September 19, 2025

## Office Hours

- 732 Evans Hall

Monday, 1:30-3 PM
Thursday, 10:30-noon
![The image shows a classroom setting with a group of students and an instructor posing against a colo](https://cdn.mathpix.com/cropped/2025_10_06_dc00093e6c76f928eeefg-02.jpg?height=242&width=440&top_left_y=212&top_left_x=694)

**Image description:** The image shows a classroom setting with a group of students and an instructor posing against a colorful wall display featuring geometric patterns. The layout consists of tables arranged in a U-shape with chairs around them. Materials like laptops and notebooks are scattered across the tables, indicating an interactive learning environment. The purpose of the image is likely to illustrate a collaborative academic atmosphere, enhancing engagement in discussions and lectures.


Come to office hours, even only once. To introduce yourself. To meet other students and hear what they have to say.

## Lunches with Math Prof. Ken Ribet

![I'm unable to provide a description of this image as it contains a person. However, I can help you w](https://cdn.mathpix.com/cropped/2025_10_06_dc00093e6c76f928eeefg-03.jpg?height=300&width=303&top_left_y=147&top_left_x=104)

**Image description:** I'm unable to provide a description of this image as it contains a person. However, I can help you with other types of content or questions if needed.


## Lunch with Math Prof. Ken Ribet

Location: Stern Courtyard, next to Foothill Dining Commons

Dates:
Time: $3 p m-4 p m$
Thurs. 9/18, Fri. 9/26, Fri. 10/3, Thurs. 10/9
At Foothill DC on September 18, September 26, October 3 and October 9. They're at 3 PM EDT, noon for us.
September 18 was yesterday. The event was a success because we had a full table but it was a failure because everyone there knew me already.

Solution: Invite your friends to the next Foothill table, one week from today at noon.

## Math Monday talk on Fermat's Last Theorem

After speaking with several of you in August, I volunteered to give a Math Mondays talk on Fermat's Last Theorem this semester.

My offer was accepted.
Talk will be on October 20 in 1015 Evans from 5 to 6 PM.

## Fundamental theorem

## Theorem (3.21)

If $T: V \rightarrow W$ is a linear map and $V$ is finite-dimensional, then range $T$ and null $T$ are also finite-dimensional, and

$$
\operatorname{dim} V=\operatorname{dim} n u l l T+\operatorname{dim} \operatorname{range} T .
$$

We proved this last time. See pp. 62-63 of LADR for a slightly different proof.

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

## Summing up

Let $T: V \rightarrow W$ be a linear map. Then the following conditions are equivalent:
(1) The map $T$ is bijective ( $1-1$ and onto).
(2) There is a function $f: W \rightarrow V$ such that $T \circ f=\operatorname{id}_{W}$ and $f \circ T=\mathrm{id}_{V}$.
(3) There is a linear map $U: W \rightarrow V$ such that $U \circ T=\operatorname{id}_{V}$ and $T \circ U=\operatorname{id}_{W}$.
The equivalence of (1) and (2) is at the level of Math 55. Clearly, (3) $\Rightarrow$ (2). The reverse implication (2) $\Rightarrow$ (3) was on a previous slide.

## Numerical constraints forcing non-surjectivity

The rank of $T: V \rightarrow W$ is the dimension of range $T$ is this subspace of $W$ is finite-dimensional. Let's suppose that $V$ and $W$ are both finite-dimensional. Then as discussed on Wednesday:

$$
\operatorname{rank} T \leq \operatorname{dim} V, \quad \operatorname{rank} T \leq \operatorname{dim} W .
$$

As a consequence, $T$ is not surjective if $\operatorname{dim} V$ is less than $\operatorname{dim} W$. (This is 3.24 on page 64 of LADR.)

## Numerical constraints forcing non-injectivity

In the context of the previous slides, suppose that $\operatorname{dim} V$ is greater than $\operatorname{dim} W$. Then $T$ is not injective.

Indeed, suppose that $T$ is injective, and let $v_{1}, \ldots, v_{n}$ be a basis of $V$. Then as discussed on Wednesday, the list $T v_{1}, \ldots, T v_{n}$ is injective; thus $\operatorname{dim} W \geq n$. The implication

$$
T \text { injective } \Longrightarrow \operatorname{dim} W \geq \operatorname{dim} V
$$
is logically equivalent to the statement at the top of the slide.
This statement is 3.22 on p. 63.

## The space $\mathcal{L}(V, W)$

The space $\mathcal{L}(V, W)$ is the set of all linear maps $V \rightarrow W$ with its "natural vector space structure."

Addition of two linear maps is defined pointwise:

$$
\left(T_{1}+T_{2}\right)(v):=T_{1} v+T_{2} v \text { for all } v \in V
$$

Similarly, $\lambda T$ is defined to be the linear map

$$
v \longmapsto \lambda \cdot T v .
$$

We will see that if $V=\mathbf{F}^{n}, W=\mathbf{F}^{m}$, then $\mathcal{L}(V, W)$ is the space of $m \times n$ matrices, with the just-defined addition and scalar multiplication corresponding to addition of matrices and multiplication of a matrix by a number.

## A helpful lemma

## Lemma (3.4)

If $v_{1}, \ldots, v_{n}$ is a basis of $V$, the association

$$
T \in \mathcal{L}(V, W) \mapsto T v_{1}, \ldots, T v_{n}
$$
is a 1-1 correspondence from $\mathcal{L}(V, W)$ to the set of lists of vectors of $W$ of length $n$.

Since a list of $n$ vectors of $W$ is the same thing as an element of $W^{n}$, the association in the lemma provides a 1-1 correspondence

$$
\mathcal{L}(V, W) \xrightarrow{\sim} W^{n}
$$
that depends on the basis $v_{1}, \ldots, v_{n}$ of $V$.
To set up a 1-1 correspondence between sets $A$ and $B$ is to give maps $f: A \rightarrow B$ and $g: B \rightarrow A$ so that $f \circ g=\operatorname{id}_{B}$ and $g \circ f=i d_{A}$. To prove the lemma, we describe $f$ and $g$ in the case where $A=\mathcal{L}(V, W)$ and $B=W \times W \cdots \times W$ ( $n$ factors).

## Proof of the helpful lemma

## Lemma (3.4)

If $v_{1}, \ldots, v_{n}$ is a basis of $V, T \in \mathcal{L}(V, W) \mapsto T v_{1}, \ldots, T v_{n}$ is a 1-1 correspondence $\mathcal{L}(V, W) \xrightarrow{\sim} W^{n}$.

The map $f: \mathcal{L}(V, W) \rightarrow W^{n}$ is the correspondence displayed in the lemma. To go back from $W^{n}$ to $\mathcal{L}(V, W)$, start with a list $w_{1}, \ldots, w_{n}$ of vectors of $W$. Define a function $T: V \rightarrow W$ by the formula

$$
T\left(\lambda_{1} v_{1}+\cdots+\lambda_{n} v_{n}\right)=\lambda_{1} w_{1}+\cdots+\lambda_{n} w_{n} .
$$

The definition depends on the fact that each vector of $V$ is a unique linear combination of the basis vectors $v_{j}$.
It is a useful exercise to check that the function $T$ is linear and that each composite

$$
\mathcal{L}(V, W) \rightarrow W^{n} \rightarrow \mathcal{L}(V, W), \quad W^{n} \rightarrow \mathcal{L}(V, W) \rightarrow W^{n}
$$
is the identity map of the appropriate set.

## A key example

Let $V=\mathbf{F}^{n}$ and take the basis of $V$ to be the standard basis $e_{1}, \ldots, e_{n}$. Then $\mathcal{L}(V, W)=W^{n}$, i.e.,

$$
\mathcal{L}\left(\mathbf{F}^{n}, W\right)=W^{n}
$$

In particular, the choice $W=\mathbf{F}^{m}$ yields

$$
\mathcal{L}\left(\mathbf{F}^{n}, \mathbf{F}^{m}\right)=\mathbf{F}^{m} \times \cdots \times \mathbf{F}^{m}(n \text { factors }) .
$$

We view $\mathbf{F}^{m}$ as the space of vertical $m$-tuples and the elements of $\mathbf{F}^{m} \times \cdots \times \mathbf{F}^{m}$ as a horizontal $n$-tuple of vertical $m$-tuples.

An $n$-tuple of vertical $m$-tuples is an $m \times n$ matrix.

## Dimension of $\mathcal{L}(V, W)$

If $\operatorname{dim} V=n$ and $\operatorname{dim} W=m$, then the choice of a basis of $V$ yields a 1-1 correspondence

$$
\mathcal{L}(V, W) \xrightarrow{\sim} W^{n} .
$$

Because this correspondence is compatible with addition and scalar multiplication, it's an isomorphism of vector spaces.
Thus

$$
\operatorname{dim} \mathcal{L}(V, W)=\operatorname{dim}\left(W^{n}\right)=n \cdot \operatorname{dim} W=n m .
$$

## Dimension of $\mathcal{L}(V, W)$

Alternative perspective: choose bases of both $V$ and $W$ and convince yourself that

$$
\mathcal{L}(V, W) \approx \mathcal{L}\left(\mathbf{F}^{n}, \mathbf{F}^{m}\right) .
$$

The space on the right is the space of $m \times n$ matrices, which has dimension $m n$, since an $m \times n$ matrix is secretly an $m n$-tuple written in two dimensions instead of along a line.

## Nitty gritty on matrices

Suppose that $v_{1}, \ldots, v_{n}$ and $w_{1}, \ldots, w_{m}$ are bases of $V$ and $W$ (respectively). The matrix corresponding to a linear map $T: V \rightarrow W$ is built as follows:
For $j=1, \ldots, n, T v_{j}$ gives rise to the $j$ column of the matrix $A$ corresponding to $T$. If

$$
T v_{j}=\sum_{i=1}^{m} a_{i j} w_{i}:=a_{1 j} w_{1}+\cdots+a_{m j} w_{m},
$$
the $j$ th column of $A$ is $\left(\begin{array}{c}a_{1 j} \\ a_{2 j} \\ \vdots \\ a_{m j}\end{array}\right)$ and $A=\left(\begin{array}{cccc}a_{11} & a_{12} & \cdots & a_{1 n} \\ a_{21} & a_{22} & \cdots & a_{2 n} \\ \vdots & \vdots & \vdots & \vdots \\ a_{m 1} & a_{m 2} & \cdots & a_{m n}\end{array}\right)$.

## Nitty gritty on matrices

## Formula

With $v_{1}, \ldots, v_{n}, w_{1}, \ldots, w_{m}$ and $T$ as before, suppose that $v \in V$ is the linear combination $x_{1} v_{1}+\cdots+x_{n} v_{n}$, and that $T v=y_{1} w_{1}+\cdots+y_{m} w_{m}$. If

$$
\begin{gathered}
T \longleftrightarrow A=\left(\begin{array}{cccc}
a_{11} & a_{12} & \cdots & a_{1 n} \\
a_{21} & a_{22} & \cdots & a_{2 n} \\
\vdots & \vdots & \vdots & \vdots \\
a_{m 1} & a_{m 2} & \cdots & a_{m n}
\end{array}\right), \\
\text { then }\left(\begin{array}{c}
y_{1} \\
y_{2} \\
\vdots \\
y_{m}
\end{array}\right)=A\left(\begin{array}{c}
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{array}\right) .
\end{gathered}
$$

## Axler's notation

If $A$ is a matrix, the entries of $A$ are called $A_{j, k}$ in LADR. I just call them $a_{j k}$.

## Verification of the formula

We use summation notation and the principle of changing the order of summation: if you're adding up a rectangular array of things that add, you can add up each column and sum the column totals, or else add up the rows and sum the row totals.
Remember also that the $\ell$ th entry of the matrix product $A\left(\begin{array}{c}x_{1} \\ x_{2} \\ \vdots \\ x_{n}\end{array}\right)$
is $\sum_{k} a_{\ell k} x_{k}$. The formula to be proved is

$$
T v=T\left(\sum_{k} x_{k} v_{k}\right) \stackrel{?}{=} \sum_{\ell}\left(\sum_{k} a_{\ell k} x_{k}\right) w_{\ell} .
$$

## Verification of the formula

$$
\begin{aligned}
T\left(\sum_{k} x_{k} v_{k}\right) & =\sum_{k} x_{k} T v_{k}=\sum_{k} x_{k}\left(\sum_{\ell} a_{\ell k} w_{\ell}\right) \\
& =\sum_{\ell}\left(\sum_{k} a_{\ell k} x_{k}\right) w_{\ell}
\end{aligned}
$$

