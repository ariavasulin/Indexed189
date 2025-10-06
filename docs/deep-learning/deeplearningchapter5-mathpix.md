---
course: CS 189
semester: Fall 2025
type: textbook
title: DeepLearningChapter5
source_type: pdf
source_file: DeepLearningChapter5.pdf
processed_date: '2025-10-04'
processor: mathpix
---

### 4.1. Linear Regression

The goal of regression is to predict the value of one or more continuous target variables $t$ given the value of a $D$-dimensional vector $\mathbf{x}$ of input variables. Typically we are given a training data set comprising $N$ observations $\left\{\mathbf{x}_{n}\right\}$, where $n=1, \ldots, N$, together with corresponding target values $\left\{t_{n}\right\}$, and the goal is to predict the value of $t$ for a new value of $\mathbf{x}$. To do this, we formulate a function $y(\mathbf{x}, \mathbf{w})$ whose values for new inputs $\mathbf{x}$ constitute the predictions for the corresponding values of $t$, and where $\mathbf{w}$ represents a vector of parameters that can be learned from the training data.

The simplest model for regression is one that involves a linear combination of the input variables:

$$
y(\mathbf{x}, \mathbf{w})=w_{0}+w_{1} x_{1}+\ldots+w_{D} x_{D}
$$
where $\mathbf{x}=\left(x_{1}, \ldots, x_{D}\right)^{\mathrm{T}}$. The term linear regression sometimes refers specifically to this form of model. The key property of this model is that it is a linear function of the parameters $w_{0}, \ldots, w_{D}$. It is also, however, a linear function of the input variables $x_{i}$, and this imposes significant limitations on the model.

### 4.1.1 Basis functions

We can extend the class of models defined by (4.1) by considering linear combinations of fixed nonlinear functions of the input variables, of the form

$$
y(\mathbf{x}, \mathbf{w})=w_{0}+\sum_{j=1}^{M-1} w_{j} \phi_{j}(\mathbf{x})
$$

Section 4.3

Section 6.1
where $\phi_{j}(\mathbf{x})$ are known as basis functions. By denoting the maximum value of the index $j$ by $M-1$, the total number of parameters in this model will be $M$.

The parameter $w_{0}$ allows for any fixed offset in the data and is sometimes called a bias parameter (not to be confused with bias in a statistical sense). It is often convenient to define an additional dummy basis function $\phi_{0}(x)$ whose value is fixed at $\phi_{0}(\mathrm{x})=1$ so that (4.2) becomes

$$
y(\mathbf{x}, \mathbf{w})=\sum_{j=0}^{M-1} w_{j} \phi_{j}(\mathbf{x})=\mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}(\mathbf{x})
$$
where $\mathbf{w}=\left(w_{0}, \ldots, w_{M-1}\right)^{\mathrm{T}}$ and $\boldsymbol{\phi}=\left(\phi_{0}, \ldots, \phi_{M-1}\right)^{\mathrm{T}}$. We can represent the model (4.3) using a neural network diagram, as shown in Figure 4.1.

By using nonlinear basis functions, we allow the function $y(\mathbf{x}, \mathbf{w})$ to be a nonlinear function of the input vector $\mathbf{x}$. Functions of the form (4.2) are called linear models, however, because they are linear in $\mathbf{w}$. It is this linearity in the parameters that will greatly simplify the analysis of this class of models. However, it also leads to some significant limitations.

Figure 4.1 The linear regression model (4.3) can be expressed as a simple neural network diagram involving a single layer of parameters. Here each basis function $\phi_{j}(\mathrm{x})$ is represented by an input node, with the solid node representing the 'bias' basis function $\phi_{0}$, and the function $y(\mathbf{x}, \mathbf{w})$ is represented by an output node. Each of the parameters $w_{j}$ is shown by a line connecting the corresponding basis function to the output.
![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-02.jpg?height=348&width=554&top_left_y=276&top_left_x=1325)

Before the advent of deep learning it was common practice in machine learning to use some form of fixed pre-processing of the input variables $\mathbf{x}$, also known as feature extraction, expressed in terms of a set of basis functions $\left\{\phi_{j}(\mathbf{x})\right\}$. The goal was to choose a sufficiently powerful set of basis functions that the resulting learning task could be solved using a simple network model. Unfortunately, it is very difficult to hand-craft suitable basis functions for anything but the simplest applications. Deep learning avoids this problem by learning the required nonlinear transformations of the data from the data set itself.

We have already encountered an example of a regression problem when we dis- cussed curve fitting using polynomials. The polynomial function (1.1) can be expressed in the form (4.3) if we consider a single input variable $x$ and if we choose basis functions defined by $\phi_{j}(x)=x^{j}$. There are many other possible choices for the basis functions, for example

$$
\phi_{j}(x)=\exp \left\{-\frac{\left(x-\mu_{j}\right)^{2}}{2 s^{2}}\right\}
$$
where the $\mu_{j}$ govern the locations of the basis functions in input space, and the parameter $s$ governs their spatial scale. These are usually referred to as 'Gaussian' basis functions, although it should be noted that they are not required to have a probabilistic interpretation. In particular the normalization coefficient is unimportant because these basis functions will be multiplied by learnable parameters $w_{j}$.

Another possibility is the sigmoidal basis function of the form

$$
\phi_{j}(x)=\sigma\left(\frac{x-\mu_{j}}{s}\right)
$$
where $\sigma(a)$ is the logistic sigmoid function defined by
$$
\sigma(a)=\frac{1}{1+\exp (-a)} .
$$

Equivalently, we can use the tanh function because this is related to the logistic sigmoid by $\tanh (a)=2 \sigma(2 a)-1$, and so a general linear combination of logistic sigmoid functions is equivalent to a general linear combination of tanh functions in the sense that they can represent the same class of input-output functions. These various choices of basis function are illustrated in Figure 4.2.

![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-03.jpg?height=451&width=1562&top_left_y=306&top_left_x=287)

**Image Description:** The image contains three mathematical plots arranged horizontally. Each plot has the same Cartesian coordinate system, with the x-axis ranging from -1 to 1 and the y-axis ranging from -1 to 1. 

1. The left plot displays a cubic function with various curves, including green, blue, red, and purple lines. 
2. The middle plot shows oscillatory behavior, with multiple sine-like curves layered over each other. 
3. The right plot features closely spaced, wavy lines indicating higher frequency oscillations. 

All three plots represent different mathematical relationships and their behaviors in a coordinate plane.

Figure 4.2 Examples of basis functions, showing polynomials on the left, Gaussians of the form (4.4) in the centre, and sigmoidal basis functions of the form (4.5) on the right.

Yet another possible choice of basis function is the Fourier basis, which leads to an expansion in sinusoidal functions. Each basis function represents a specific frequency and has infinite spatial extent. By contrast, basis functions that are localized to finite regions of input space necessarily comprise a spectrum of different spatial frequencies. In signal processing applications, it is often of interest to consider basis functions that are localized in both space and frequency, leading to a class of functions known as wavelets (Ogden, 1997; Mallat, 1999; Vidakovic, 1999). These are also defined to be mutually orthogonal, to simplify their application. Wavelets are most applicable when the input values live on a regular lattice, such as the successive time points in a temporal sequence or the pixels in an image.

Most of the discussion in this chapter, however, is independent of the choice of basis function set, and so we will not specify the particular form of the basis functions, except for numerical illustration. Furthermore, to keep the notation simple, we will focus on the case of a single target variable $t$, although we will briefly outline

Section 4.1.7

## Section 1.2

the modifications needed to deal with multiple target variables.

### 4.1.2 Likelihood function

We solved the problem of fitting a polynomial function to data by minimizing a sum-of-squares error function, and we also showed that this error function could be motivated as the maximum likelihood solution under an assumed Gaussian noise model. We now return to this discussion and consider the least-squares approach, and its relation to maximum likelihood, in more detail.

As before, we assume that the target variable $t$ is given by a deterministic function $y(\mathbf{x}, \mathbf{w})$ with additive Gaussian noise so that

$$
t=y(\mathbf{x}, \mathbf{w})+\epsilon
$$
where $\epsilon$ is a zero-mean Gaussian random variable with variance $\sigma^{2}$. Thus, we can write
$$
p\left(t \mid \mathbf{x}, \mathbf{w}, \sigma^{2}\right)=\mathcal{N}\left(t \mid y(\mathbf{x}, \mathbf{w}), \sigma^{2}\right) .
$$

Now consider a data set of inputs $\mathbf{X}=\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}\right\}$ with corresponding target values $t_{1}, \ldots, t_{N}$. We group the target variables \{ $t_{n}$ \} into a column vector that we denote by $\mathbf{t}$ where the typeface is chosen to distinguish it from a single observation of a multivariate target, which would be denoted $\mathbf{t}$. Making the assumption that these data points are drawn independently from the distribution (4.8), we obtain an expression for the likelihood function, which is a function of the adjustable parameters $\mathbf{w}$ and $\sigma^{2}$ :

$$
p\left(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \sigma^{2}\right)=\prod_{n=1}^{N} \mathcal{N}\left(t_{n} \mid \mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right), \sigma^{2}\right)
$$
where we have used (4.3). Taking the logarithm of the likelihood function and making use of the standard form (2.49) for the univariate Gaussian, we have
\$\$\begin{aligned}

\ln p\left(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \sigma^{2}\right) \& =\sum_{n=1}^{N} \ln \mathcal{N}\left(t_{n} \mid \mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right), \sigma^{2}\right) <br>
\& =-\frac{N}{2} \ln \sigma^{2}-\frac{N}{2} \ln (2 \pi)-\frac{1}{\sigma^{2}} E_{D}(\mathbf{w})

\end{aligned}$$
where the sum-of-squares error function is defined by
$$E_{D}(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N}\left\{t_{n}-\mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\}^{2} .\$\$

Section 2.3.4
The first two terms in (4.10) can be treated as constants when determining $\mathbf{w}$ because they are independent of $\mathbf{w}$. Therefore, as we saw previously, maximizing the likelihood function under a Gaussian noise distribution is equivalent to minimizing the sum-of-squares error function (4.11).

### 4.1.3 Maximum likelihood

Having written down the likelihood function, we can use maximum likelihood to determine $\mathbf{w}$ and $\sigma^{2}$. Consider first the maximization with respect to $\mathbf{w}$. The gradient of the log likelihood function (4.10) with respect to $\mathbf{w}$ takes the form

$$
\nabla_{\mathbf{w}} \ln p\left(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \sigma^{2}\right)=\frac{1}{\sigma^{2}} \sum_{n=1}^{N}\left\{t_{n}-\mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)^{\mathrm{T}} .
$$

Setting this gradient to zero gives

$$
0=\sum_{n=1}^{N} t_{n} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)^{\mathrm{T}}-\mathbf{w}^{\mathrm{T}}\left(\sum_{n=1}^{N} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right) \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)^{\mathrm{T}}\right) .
$$

Solving for $\mathbf{w}$ we obtain

$$
\mathbf{w}_{\mathrm{ML}}=\left(\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{\mathrm{T}} \mathbf{t},
$$
which are known as the normal equations for the least-squares problem. Here $\boldsymbol{\Phi}$ is an $N \times M$ matrix, called the design matrix, whose elements are given by $\Phi_{n j}=\phi_{j}\left(\mathbf{x}_{n}\right)$, so that
\$\$\mathbf{\Phi}=\left(\begin{array}{cccc}

\phi_{0}\left(\mathbf{x}_{1}\right) \& \phi_{1}\left(\mathbf{x}_{1}\right) \& \cdots \& \phi_{M-1}\left(\mathbf{x}_{1}\right) <br>
\phi_{0}\left(\mathbf{x}_{2}\right) \& \phi_{1}\left(\mathbf{x}_{2}\right) \& \cdots \& \phi_{M-1}\left(\mathbf{x}_{2}\right) <br>
\vdots \& \vdots \& \ddots \& \vdots <br>
\phi_{0}\left(\mathbf{x}_{N}\right) \& \phi_{1}\left(\mathbf{x}_{N}\right) \& \cdots \& \phi_{M-1}\left(\mathbf{x}_{N}\right)

\end{array}\right)\$\$

The quantity

$$
\boldsymbol{\Phi}^{\dagger} \equiv\left(\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{\mathrm{T}}
$$
is known as the Moore-Penrose pseudo-inverse of the matrix $\boldsymbol{\Phi}$ (Rao and Mitra, 1971; Golub and Van Loan, 1996). It can be regarded as a generalization of the notion of a matrix inverse to non-square matrices. Indeed, if $\boldsymbol{\Phi}$ is square and invertible, then using the property $(\mathbf{A B})^{-1}=\mathbf{B}^{-1} \mathbf{A}^{-1}$ we see that $\boldsymbol{\Phi}^{\dagger} \equiv \boldsymbol{\Phi}^{-1}$.

At this point, we can gain some insight into the role of the bias parameter $w_{0}$. If we make the bias parameter explicit, then the error function (4.11) becomes

$$
E_{D}(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N}\left\{t_{n}-w_{0}-\sum_{j=1}^{M-1} w_{j} \phi_{j}\left(\mathbf{x}_{n}\right)\right\}^{2}
$$

Setting the derivative with respect to $w_{0}$ equal to zero and solving for $w_{0}$, we obtain

$$
w_{0}=\bar{t}-\sum_{j=1}^{M-1} w_{j} \overline{\phi_{j}}
$$
where we have defined
$$
\bar{t}=\frac{1}{N} \sum_{n=1}^{N} t_{n}, \quad \overline{\phi_{j}}=\frac{1}{N} \sum_{n=1}^{N} \phi_{j}\left(\mathbf{x}_{n}\right)
$$

Thus, the bias $w_{0}$ compensates for the difference between the averages (over the training set) of the target values and the weighted sum of the averages of the basis function values.

We can also maximize the log likelihood function (4.10) with respect to the variance $\sigma^{2}$, giving

$$
\sigma_{\mathrm{ML}}^{2}=\frac{1}{N} \sum_{n=1}^{N}\left\{t_{n}-\mathbf{w}_{\mathrm{ML}}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\}^{2}
$$
and so we see that the maximum likelihood value of the variance parameter is given by the residual variance of the target values around the regression function.

Figure 4.3 Geometrical interpretation of the leastsquares solution in an $N$-dimensional space whose axes are the values of $t_{1}, \ldots, t_{N}$. The least-squares regression function is obtained by finding the orthogonal projection of the data vector $\mathbf{t}$ onto the subspace spanned by the basis functions $\phi_{j}(x)$ in which each basis function is viewed as a vector $\varphi_{j}$ of length $N$ with elements $\phi_{j}\left(\mathrm{x}_{n}\right)$.
![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-06.jpg?height=389&width=589&top_left_y=276&top_left_x=1306)

**Image Description:** The diagram depicts a geometric representation in a three-dimensional space. It shows a parallelogram, labeled \( S \), with two vectors, \( y \) and \( t \), originating from a point. The vector \( y \) is in black, while \( t \) is presented in blue, green, and red colors, indicating different orientations. The angles \( \varphi_1 \) and \( \varphi_2 \) are marked, suggesting relationships between the vectors and the plane \( S \). The axes are not explicitly shown, but the context implies a focus on vector orientation and projection onto the vector space defined by \( S \).


### 4.1.4 Geometry of least squares

At this point, it is instructive to consider the geometrical interpretation of the least-squares solution. To do this, we consider an $N$-dimensional space whose axes are given by the $t_{n}$, so that $\mathbf{t}=\left(t_{1}, \ldots, t_{N}\right)^{\mathrm{T}}$ is a vector in this space. Each basis function $\phi_{j}\left(\mathbf{x}_{n}\right)$, evaluated at the $N$ data points, can also be represented as a vector in the same space, denoted by $\varphi_{j}$, as illustrated in Figure 4.3. Note that $\varphi_{j}$ corresponds to the $j$ th column of $\boldsymbol{\Phi}$, whereas $\phi\left(\mathrm{x}_{n}\right)$ corresponds to the transpose of the $n$th row of $\boldsymbol{\Phi}$. If the number $M$ of basis functions is smaller than the number $N$ of data points, then the $M$ vectors $\phi_{j}\left(\mathbf{x}_{n}\right)$ will span a linear subspace $\mathcal{S}$ of dimensionality $M$. We define $\mathbf{y}$ to be an $N$-dimensional vector whose $n$th element is given by $y\left(\mathbf{x}_{n}, \mathbf{w}\right)$, where $n=1, \ldots, N$. Because $\mathbf{y}$ is an arbitrary linear combination of the vectors $\varphi_{j}$, it can live anywhere in the $M$-dimensional subspace. The sum-of-squares error (4.11) is then equal (up to a factor of $1 / 2$ ) to the squared Euclidean distance between $\mathbf{y}$ and $\mathbf{t}$. Thus, the least-squares solution for $\mathbf{w}$ corresponds to that choice of $\mathbf{y}$ that lies in subspace $\mathcal{S}$ and is closest to $\mathbf{t}$. Intuitively, from Figure 4.3, we anticipate that this solution corresponds to the orthogonal projection of $\mathbf{t}$ onto the subspace $\mathcal{S}$. This is indeed the case, as can easily be verified by noting that the solution for $\mathbf{y}$ is given by $\boldsymbol{\Phi} \mathbf{w}_{\text {ML }}$ and then confirming that this takes the form of an orthogonal projection.

In practice, a direct solution of the normal equations can lead to numerical difficulties when $\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}$ is close to singular. In particular, when two or more of the basis vectors $\varphi_{j}$ are co-linear, or nearly so, the resulting parameter values can have large magnitudes. Such near degeneracies will not be uncommon when dealing with real data sets. The resulting numerical difficulties can be addressed using the technique of singular value decomposition, or SVD (Deisenroth, Faisal, and Ong, 2020). Note that the addition of a regularization term ensures that the matrix is non-singular, even in the presence of degeneracies.

### 4.1.5 Sequential learning

The maximum likelihood solution (4.14) involves processing the entire training set in one go and is known as a batch method. This can become computationally costly for large data sets. If the data set is sufficiently large, it may be worthwhile to use sequential algorithms, also known as online algorithms, in which the data points are considered one at a time and the model parameters updated after each such presentation. Sequential learning is also appropriate for real-time applications in which the data observations arrive in a continuous stream and predictions must be
made before all the data points are seen.
We can obtain a sequential learning algorithm by applying the technique of stochastic gradient descent, also known as sequential gradient descent, as follows. If the error function comprises a sum over data points $E=\sum_{n} E_{n}$, then after presentation of data point $n$, the stochastic gradient descent algorithm updates the parameter vector w using

$$
\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}-\eta \nabla E_{n}
$$
where $\tau$ denotes the iteration number, and $\eta$ is a suitably chosen learning rate parameter. The value of $\mathbf{w}$ is initialized to some starting vector $\mathbf{w}^{(0)}$. For the sum-ofsquares error function (4.11), this gives
$$
\mathbf{w}^{(\tau+1)}=\mathbf{w}^{(\tau)}+\eta\left(t_{n}-\mathbf{w}^{(\tau) \mathrm{T}} \boldsymbol{\phi}_{n}\right) \boldsymbol{\phi}_{n}
$$

where $\phi_{n}=\phi\left(\mathrm{x}_{n}\right)$. This is known as the least-mean-squares or the LMS algorithm.

### 4.1.6 Regularized least squares

We have previously introduced the idea of adding a regularization term to an error function to control over-fitting, so that the total error function to be minimized takes the form

$$
E_{D}(\mathbf{w})+\lambda E_{W}(\mathbf{w})
$$
where $\lambda$ is the regularization coefficient that controls the relative importance of the data-dependent error $E_{D}(\mathbf{w})$ and the regularization term $E_{W}(\mathbf{w})$. One of the simplest forms of regularizer is given by the sum of the squares of the weight vector elements:
$$
E_{W}(\mathbf{w})=\frac{1}{2} \sum_{j} w_{j}^{2}=\frac{1}{2} \mathbf{w}^{\mathrm{T}} \mathbf{w} .
$$

If we also consider the sum-of-squares error function given by

$$
E_{D}(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N}\left\{t_{n}-\mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\}^{2},
$$
then the total error function becomes
$$
\frac{1}{2} \sum_{n=1}^{N}\left\{t_{n}-\mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\}^{2}+\frac{\lambda}{2} \mathbf{w}^{\mathrm{T}} \mathbf{w} .
$$

In statistics, this regularizer provides an example of a parameter shrinkage method because it shrinks parameter values towards zero. It has the advantage that the error function remains a quadratic function of $\mathbf{w}$, and so its exact minimizer can be found in closed form. Specifically, setting the gradient of (4.26) with respect to $\mathbf{w}$ to zero and solving for $\mathbf{w}$ as before, we obtain

$$
\mathbf{w}=\left(\lambda \mathbf{I}+\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{\mathrm{T}} \mathbf{t} .
$$

This represents a simple extension of the least-squares solution (4.14).

Figure 4.4 Representation of a linear regression model as a neural network having a single layer of connections. Each basis function is represented by a node, with the solid node representing the 'bias' basis function $\phi_{0}$. Likewise each output $y_{1}, \ldots, y_{K}$ is represented by a node. The links between the nodes represent the corresponding weight and bias parameters.
![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-08.jpg?height=348&width=684&top_left_y=276&top_left_x=1195)

**Image Description:** The image depicts a directed acyclic graph representing a multi-layer neural network structure. It consists of multiple layers denoted by circles labeled $\phi_0(x)$, $\phi_1(x)$, ..., $\phi_{M-1}(x)$, where each $\phi_i(x)$ indicates the output of layer $i$. Arrows indicate the flow of information between layers, with inputs $y_1(x, w)$ to $y_K(x, w)$ showcased. The horizontal axis represents the layers, while the vertical axis signifies the outputs akin to a classification or regression task. The diagram illustrates the connections and transformations in the neural network architecture.


### 4.1.7 Multiple outputs

So far, we have considered situations with a single target variable $t$. In some applications, we may wish to predict $K>1$ target variables, which we denote collectively by the target vector $\mathbf{t}=\left(t_{1}, \ldots, t_{K}\right)^{\mathrm{T}}$. This could be done by introducing a different set of basis functions for each component of $\mathbf{t}$, leading to multiple, independent regression problems. However, a more common approach is to use the same set of basis functions to model all of the components the target vector so that

$$
\mathbf{y}(\mathbf{x}, \mathbf{w})=\mathbf{W}^{\mathrm{T}} \boldsymbol{\phi}(\mathbf{x})
$$
where $\mathbf{y}$ is a $K$-dimensional column vector, $\mathbf{W}$ is an $M \times K$ matrix of parameters, and $\phi(\mathrm{x})$ is an $M$-dimensional column vector with elements $\phi_{j}(\mathrm{x})$ with $\phi_{0}(\mathrm{x})=1$ as before. Again, this can be represented as a neural network having a single layer of parameters, as shown in Figure 4.4.

Suppose we take the conditional distribution of the target vector to be an isotropic Gaussian of the form

$$
p\left(\mathbf{t} \mid \mathbf{x}, \mathbf{W}, \sigma^{2}\right)=\mathcal{N}\left(\mathbf{t} \mid \mathbf{W}^{\mathrm{T}} \boldsymbol{\phi}(\mathbf{x}), \sigma^{2} \mathbf{I}\right) .
$$

If we have a set of observations $\mathbf{t}_{1}, \ldots, \mathbf{t}_{N}$, we can combine these into a matrix $\mathbf{T}$ of size $N \times K$ such that the $n$th row is given by $\mathbf{t}_{n}^{\mathrm{T}}$. Similarly, we can combine the input vectors $\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}$ into a matrix $\mathbf{X}$. The log likelihood function is then given by

$$
\begin{aligned}
\ln p\left(\mathbf{T} \mid \mathbf{X}, \mathbf{W}, \sigma^{2}\right) & =\sum_{n=1}^{N} \ln \mathcal{N}\left(\mathbf{t}_{n} \mid \mathbf{W}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right), \sigma^{2} \mathbf{I}\right) \\
& =-\frac{N K}{2} \ln \left(2 \pi \sigma^{2}\right)-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left\|\mathbf{t}_{n}-\mathbf{W}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\|^{2} .
\end{aligned}
$$

As before, we can maximize this function with respect to $\mathbf{W}$, giving

$$
\mathbf{W}_{\mathrm{ML}}=\left(\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{\mathrm{T}} \mathbf{T}
$$
where we have combined the input feature vectors $\phi\left(\mathrm{x}_{1}\right), \ldots, \phi\left(\mathrm{x}_{N}\right)$ into a matrix $\boldsymbol{\Phi}$. If we examine this result for each target variable $t_{k}$, we have
$$
\mathbf{w}_{k}=\left(\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{\mathrm{T}} \mathbf{t}_{k}=\boldsymbol{\Phi}^{\dagger} \mathbf{t}_{k}
$$

where $\mathbf{t}_{k}$ is an $N$-dimensional column vector with components $t_{n k}$ for $n=1, \ldots N$. Thus, the solution to the regression problem decouples between the different target variables, and we need compute only a single pseudo-inverse matrix $\boldsymbol{\Phi}^{\dagger}$, which is shared by all the vectors $\mathbf{w}_{k}$.

The extension to general Gaussian noise distributions having arbitrary covari-

Exercise 4.7

Section 3.2.7 ance matrices is straightforward. Again, this leads to a decoupling into $K$ independent regression problems. This result is unsurprising because the parameters $\mathbf{W}$ define only the mean of the Gaussian noise distribution, and we know that the maximum likelihood solution for the mean of a multivariate Gaussian is independent of the covariance. From now on, we will therefore consider a single target variable $t$ for simplicity.

### 4.2. Decision theory

We have formulated the regression task as one of modelling a conditional probability distribution $p(t \mid \mathbf{x})$, and we have chosen a specific form for the conditional probability, namely a Gaussian (4.8) with an $\mathbf{x}$-dependent mean $y(\mathbf{x}, \mathbf{w})$ governed by parameters $\mathbf{w}$ and with variance given by the parameter $\sigma^{2}$. Both $\mathbf{w}$ and $\sigma^{2}$ can be learned from data using maximum likelihood. The result is a predictive distribution given by

$$
p\left(t \mid \mathbf{x}, \mathbf{w}_{\mathrm{ML}}, \sigma_{\mathrm{ML}}^{2}\right)=\mathcal{N}\left(t \mid y\left(\mathbf{x}, \mathbf{w}_{\mathrm{ML}}\right), \sigma_{\mathrm{ML}}^{2}\right) .
$$

The predictive distribution expresses our uncertainty over the value of $t$ for some new input $\mathbf{x}$. However, for many practical applications we need to predict a specific value for $t$ rather than returning an entire distribution, particularly where we must take a specific action. For example, if our goal is to determine the optimal level of radiation to use for treating a tumour and our model predicts a probability distribution over radiation dose, then we must use that distribution to decide the specific dose to be administered. Our task therefore breaks down into two stages. In the first stage, called the inference stage, we use the training data to determine a predictive distribution $p(t \mid \mathbf{x})$. In the second stage, known as the decision stage, we use this predictive distribution to determine a specific value $f(\mathbf{x})$, which will be dependent on the input vector $\mathbf{x}$, that is optimal according to some criterion. We can do this by minimizing a loss function that depends on both the predictive distribution $p(t \mid \mathbf{x})$ and on $f$.

Intuitively we might choose the mean of the conditional distribution, so that we would use $f(\mathbf{x})=y\left(\mathbf{x}, \mathbf{w}_{\text {ML }}\right)$. In some cases this intuition will be correct, but in other situations it can give very poor results. It is therefore useful to formalize this so that we can understand when it applies and under what assumptions, and the framework for doing this is called decision theory.

Suppose that we choose a value $f(\mathrm{x})$ for our prediction when the true value is $t$. In doing so, we incur some form of penalty or cost. This is determined by a loss, which we denote $L(t, f(\mathbf{x}))$. Of course, we do not know the true value of $t$, so instead of minimizing $L$ itself, we minimize the average, or expected, loss which is

Figure 4.5 The regression function $f^{\star}(x)$, which minimizes the expected squared loss, is given by the mean of the conditional distribution $p(t \mid x)$.
![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-10.jpg?height=568&width=741&top_left_y=281&top_left_x=1149)

**Image Description:** The diagram is a graph representing a function \( f^*(x) \) plotted against the variable \( x \) on the horizontal axis. The vertical axis shows the variable \( t \). The graph features a red curve indicating \( f^*(x) \) which exhibits a sigmoidal shape, and a blue curve representing \( p(t | x_0, w, \sigma^2) \), intersecting the vertical axis at a defined point. A vertical line on the graph signifies a threshold or division between two regions. The intersection of the curves suggests a transition point relevant to probabilistic modeling.


given by

$$
\mathbb{E}[L]=\iint L(t, f(\mathbf{x})) p(\mathbf{x}, t) \mathrm{d} \mathbf{x} \mathrm{~d} t
$$
where we are averaging over the distribution of both input and target variables, weighted by their joint distribution $p(\mathbf{x}, t)$. A common choice of loss function in regression problems is the squared loss given by $L(t, f(\mathbf{x}))=\{f(\mathbf{x})-t\}^{2}$. In this case, the expected loss can be written
$$
\mathbb{E}[L]=\iint\{f(\mathbf{x})-t\}^{2} p(\mathbf{x}, t) \mathrm{d} \mathbf{x} \mathrm{~d} t
$$

It is important not to confuse the squared-loss function with the sum-of-squares error function introduced earlier. The error function is used to set the parameters during training in order to determine the conditional probability distribution $p(t \mid \mathbf{x})$, whereas the loss function governs how the conditional distribution is used to arrive at a predictive function $f(x)$ specifying a prediction for each value of $\mathbf{x}$.

Our goal is to choose $f(\mathbf{x})$ so as to minimize $\mathbb{E}[L]$. If we assume a completely flexible function $f(\mathrm{x})$, we can do this formally using the calculus of variations to give

$$
\frac{\delta \mathbb{E}[L]}{\delta f(\mathbf{x})}=2 \int\{f(\mathbf{x})-t\} p(\mathbf{x}, t) \mathrm{d} t=0 .
$$

Solving for $f(\mathbf{x})$ and using the sum and product rules of probability, we obtain

$$
f^{\star}(\mathbf{x})=\frac{1}{p(\mathbf{x})} \int t p(\mathbf{x}, t) \mathrm{d} t=\int t p(t \mid \mathbf{x}) \mathrm{d} t=\mathbb{E}_{t}[t \mid \mathbf{x}]
$$
which is the conditional average of $t$ conditioned on $\mathbf{x}$ and is known as the regression function. This result is illustrated in Figure 4.5. It can readily be extended to multiple target variables represented by the vector $\mathbf{t}$, in which case the optimal solution is the

## Exercise 4.8

conditional average $\mathbf{f}^{\star}(\mathbf{x})=\mathbb{E}_{t}[\mathbf{t} \mid \mathbf{x}]$. For a Gaussian conditional distribution of the
form (4.8), the conditional mean will be simply

$$
\mathbb{E}[t \mid \mathbf{x}]=\int t p(t \mid \mathbf{x}) \mathrm{d} t=y(\mathbf{x}, \mathbf{w}) .
$$

The use of calculus of variations to derive (4.37) implies that we are optimizing over all possible functions $f(\mathbf{x})$. Although any parametric model that we can implement in practice is limited in the range of functions that it can represent, the framework of deep neural networks, discussed extensively in later chapters, provides a highly flexible class of functions that, for many practical purposes, can approximate any desired function to high accuracy.

We can derive this result in a slightly different way, which will also shed light on the nature of the regression problem. Armed with the knowledge that the optimal solution is the conditional expectation, we can expand the square term as follows

$$
\begin{aligned}
& \{f(\mathbf{x})-t\}^{2}=\{f(\mathbf{x})-\mathbb{E}[t \mid \mathbf{x}]+\mathbb{E}[t \mid \mathbf{x}]-t\}^{2} \\
& \quad=\{f(\mathbf{x})-\mathbb{E}[t \mid \mathbf{x}]\}^{2}+2\{f(\mathbf{x})-\mathbb{E}[t \mid \mathbf{x}]\}\{\mathbb{E}[t \mid \mathbf{x}]-t\}+\{\mathbb{E}[t \mid \mathbf{x}]-t\}^{2}
\end{aligned}
$$

where, to keep the notation uncluttered, we use $\mathbb{E}[t \mid \mathbf{x}]$ to denote $\mathbb{E}_{t}[t \mid \mathbf{x}]$. Substituting into the loss function (4.35) and performing the integral over $t$, we see that the crossterm vanishes and we obtain an expression for the loss function in the form

$$
\mathbb{E}[L]=\int\{f(\mathbf{x})-\mathbb{E}[t \mid \mathbf{x}]\}^{2} p(\mathbf{x}) \mathrm{d} \mathbf{x}+\int \operatorname{var}[t \mid \mathbf{x}] p(\mathbf{x}) \mathrm{d} \mathbf{x}
$$

The function $f(\mathrm{x})$ we seek to determine appears only in the first term, which will be minimized when $f(\mathbf{x})$ is equal to $\mathbb{E}[t \mid \mathbf{x}]$, in which case this term will vanish. This is simply the result that we derived previously, and shows that the optimal least-squares predictor is given by the conditional mean. The second term is the variance of the distribution of $t$, averaged over $\mathbf{x}$, and represents the intrinsic variability of the target data and can be regarded as noise. Because it is independent of $f(\mathbf{x})$, it represents the irreducible minimum value of the loss function.

The squared loss is not the only possible choice of loss function for regression. Here we consider briefly one simple generalization of the squared loss, called the Minkowski loss, whose expectation is given by

$$
\mathbb{E}\left[L_{q}\right]=\iint|f(\mathbf{x})-t|^{q} p(\mathbf{x}, t) \mathrm{d} \mathbf{x} \mathrm{~d} t,
$$
which reduces to the expected squared loss for $q=2$. The function $|f-t|^{q}$ is plotted against $f-t$ for various values of $q$ in Figure 4.6. The minimum of $\mathbb{E}\left[L_{q}\right]$ is given by the conditional mean for $q=2$, the conditional median for $q=1$, and the conditional mode for $q \rightarrow 0$.

Note that the Gaussian noise assumption implies that the conditional distribution of $t$ given $\mathbf{x}$ is unimodal, which may be inappropriate for some applications. In this case a squared loss can lead to very poor results and we need to develop more sophisticated approaches. For example, we can extend this model by using mixtures

![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-12.jpg?height=1153&width=1608&top_left_y=273&top_left_x=287)

**Image Description:** The image consists of four graphs arranged in a 2x2 grid. Each graph depicts the relationship between the variables \( |f - t|^q \) on the vertical axis and \( f - t \) on the horizontal axis. The parameter \( q \) varies across the graphs: \( q = 0.3 \) (top left), \( q = 1 \) (top right), \( q = 2 \) (bottom left), and \( q = 10 \) (bottom right). Each curve illustrates how the shape of the function changes with different values of \( q \), showing varying degrees of curvature and steepness.

Figure 4.6 Plots of the quantity $L_{q}=|f-t|^{q}$ for various values of $q$.

Section 6.5 of Gaussians to give multimodal conditional distributions, which often arise in the solution of inverse problems. Our focus in this section has been on decision theory for regression problems, and in the next chapter we shall develop analogous concepts for classification tasks.
Section 5.2

### 4.3. The Bias-Variance Trade-off

## Section 1.2

So far in our discussion of linear models for regression, we have assumed that the form and number of basis functions are both given. We have also seen that the use of maximum likelihood can lead to severe over-fitting if complex models are trained using data sets of limited size. However, limiting the number of basis functions to avoid over-fitting has the side effect of limiting the flexibility of the model to capture interesting and important trends in the data. Although a regularization term can control over-fitting for models with many parameters, this raises the question of how to determine a suitable value for the regularization coefficient $\lambda$. Seeking the
solution that minimizes the regularized error function with respect to both the weight vector $\mathbf{w}$ and the regularization coefficient $\lambda$ is clearly not the right approach, since this leads to the unregularized solution with $\lambda=0$.

It is instructive to consider a frequentist viewpoint of the model complexity issue, known as the bias-variance trade-off. Although we will introduce this concept in the context of linear basis function models, where it is easy to illustrate the ideas using simple examples, the discussion has very general applicability. Note, however, that over-fitting is really an unfortunate property of maximum likelihood and does not arise when we marginalize over parameters in a Bayesian setting (Bishop, 2006).

When we discussed decision theory for regression problems, we considered various loss functions, each of which leads to a corresponding optimal prediction once we are given the conditional distribution $p(t \mid \mathbf{x})$. A popular choice is the squared-loss function, for which the optimal prediction is given by the conditional expectation, which we denote by $h(\mathbf{x})$ and is given by

$$
h(\mathbf{x})=\mathbb{E}[t \mid \mathbf{x}]=\int t p(t \mid \mathbf{x}) \mathrm{d} t
$$

We have also seen that the expected squared loss can be written in the form

$$
\mathbb{E}[L]=\int\{f(\mathbf{x})-h(\mathbf{x})\}^{2} p(\mathbf{x}) \mathrm{d} \mathbf{x}+\iint\{h(\mathbf{x})-t\}^{2} p(\mathbf{x}, t) \mathrm{d} \mathbf{x} \mathrm{~d} t
$$

Recall that the second term, which is independent of $f(\mathrm{x})$, arises from the intrinsic noise on the data and represents the minimum achievable value of the expected loss. The first term depends on our choice for the function $f(\mathbf{x})$, and we will seek a solution for $f(\mathbf{x})$ that makes this term a minimum. Because it is non-negative, the smallest value that we can hope to achieve for this term is zero. If we had an unlimited supply of data (and unlimited computational resources), we could in principle find the regression function $h(\mathbf{x})$ to any desired degree of accuracy, and this would represent the optimal choice for $f(\mathbf{x})$. However, in practice we have a data set $\mathcal{D}$ containing only a finite number $N$ of data points, and consequently, we cannot know the regression function $h(\mathbf{x})$ exactly.

If we were to model $h(\mathbf{x})$ using a function governed by a parameter vector $\mathbf{w}$, then from a Bayesian perspective, the uncertainty in our model would be expressed through a posterior distribution over $\mathbf{w}$. A frequentist treatment, however, involves making a point estimate of $\mathbf{w}$ based on the data set $\mathcal{D}$ and tries instead to interpret the uncertainty of this estimate through the following thought experiment. Suppose we had a large number of data sets each of size $N$ and each drawn independently from the distribution $p(t, \mathbf{x})$. For any given data set $\mathcal{D}$, we can run our learning algorithm and obtain a prediction function $f(\mathbf{x} ; \mathcal{D})$. Different data sets from the ensemble will give different functions and consequently different values of the squared loss. The performance of a particular learning algorithm is then assessed by taking the average over this ensemble of data sets.

Consider the integrand of the first term in (4.42), which for a particular data set $\mathcal{D}$ takes the form

$$
\{f(\mathbf{x} ; \mathcal{D})-h(\mathbf{x})\}^{2} .
$$

Because this quantity will be dependent on the particular data set $\mathcal{D}$, we take its average over the ensemble of data sets. If we add and subtract the quantity $\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]$ inside the braces, and then expand, we obtain

$$
\begin{aligned}
& \left\{f(\mathbf{x} ; \mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]+\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]-h(\mathbf{x})\right\}^{2} \\
& =\left\{f(\mathbf{x} ; \mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]\right\}^{2}+\left\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]-h(\mathbf{x})\right\}^{2} \\
& +2\left\{f(\mathbf{x} ; \mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]\right\}\left\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]-h(\mathbf{x})\right\}
\end{aligned}
$$

We now take the expectation of this expression with respect to $\mathcal{D}$ and note that the final term will vanish, giving

$$
\begin{aligned}
& \mathbb{E}_{\mathcal{D}}\left[\{f(\mathbf{x} ; \mathcal{D})-h(\mathbf{x})\}^{2}\right] \\
& =\underbrace{\left\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]-h(\mathbf{x})\right\}^{2}}_{(\text {bias })^{2}}+\underbrace{\mathbb{E}_{\mathcal{D}}\left[\left\{f(\mathbf{x} ; \mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]\right\}^{2}\right]}_{\text {variance }} .
\end{aligned}
$$

We see that the expected squared difference between $f(\mathbf{x} ; \mathcal{D})$ and the regression function $h(\mathrm{x})$ can be expressed as the sum of two terms. The first term, called the squared bias, represents the extent to which the average prediction over all data sets differs from the desired regression function. The second term, called the variance, measures the extent to which the solutions for individual data sets vary around their average, and hence, this measures the extent to which the function $f(\mathbf{x} ; \mathcal{D})$ is sensitive to the particular choice of data set. We will provide some intuition to support these definitions shortly when we consider a simple example.

So far, we have considered a single input value $\mathbf{x}$. If we substitute this expansion back into (4.42), we obtain the following decomposition of the expected squared loss:

$$
\text { expected loss }=(\text { bias })^{2}+\text { variance }+ \text { noise }
$$
where
\$\$\begin{aligned}

(bias)^{2} \& =\int\left\{\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]-h(\mathbf{x})\right\}^{2} p(\mathbf{x}) \mathrm{d} \mathbf{x} <br>
variance \& =\int \mathbb{E}_{\mathcal{D}}\left[\left\{f(\mathbf{x} ; \mathcal{D})-\mathbb{E}_{\mathcal{D}}[f(\mathbf{x} ; \mathcal{D})]\right\}^{2}\right] p(\mathbf{x}) \mathrm{d} \mathbf{x} <br>
noise \& =\iint\{h(\mathbf{x})-t\}^{2} p(\mathbf{x}, t) \mathrm{d} \mathbf{x} \mathrm{~d} t

\end{aligned}\$\$
and the bias and variance terms now refer to integrated quantities.
Our goal is to minimize the expected loss, which we have decomposed into the sum of a (squared) bias, a variance, and a constant noise term. As we will see, there is a trade-off between bias and variance, with very flexible models having low bias and high variance, and relatively rigid models having high bias and low variance. The model with the optimal predictive capability is the one that leads to the best balance between bias and variance. This is illustrated by considering the sinusoidal data set introduced earlier. Here we independently generate 100 data sets, each containing
$N=25$ data points, from the sinusoidal curve $h(x)=\sin (2 \pi x)$. The data sets are indexed by $l=1, \ldots, L$, where $L=100$. For each data set $\mathcal{D}^{(l)}$, we fit a model with $M=24$ Gaussian basis functions along with a constant 'bias' basis function to give a total of 25 parameters. By minimizing the regularized error function (4.26), we obtain a prediction function $f^{(l)}(x)$, as shown in Figure 4.7.

The top row corresponds to a large value of the regularization coefficient $\lambda$ that gives low variance (because the red curves in the left plot look similar) but high bias (because the two curves in the right plot are very different). Conversely on the bottom row, for which $\lambda$ is small, there is large variance (shown by the high variability between the red curves in the left plot) but low bias (shown by the good fit between the average model fit and the original sinusoidal function). Note that the result of averaging many solutions for the complex model with $M=25$ is a very good fit to the regression function, which suggests that averaging may be a beneficial procedure. Indeed, a weighted averaging of multiple solutions lies at the heart of a Bayesian approach, although the averaging is with respect to the posterior distribution of parameters, not with respect to multiple data sets.

We can also examine the bias-variance trade-off quantitatively for this example. The average prediction is estimated from

$$
\bar{f}(x)=\frac{1}{L} \sum_{l=1}^{L} f^{(l)}(x)
$$
and the integrated squared bias and integrated variance are then given by
\$\$\begin{aligned}

(bias)^{2} \& =\frac{1}{N} \sum_{n=1}^{N}\left\{\bar{f}\left(x_{n}\right)-h\left(x_{n}\right)\right\}^{2} <br>
variance \& =\frac{1}{N} \sum_{n=1}^{N} \frac{1}{L} \sum_{l=1}^{L}\left\{f^{(l)}\left(x_{n}\right)-\bar{f}\left(x_{n}\right)\right\}^{2}

\end{aligned}\$\$
where the integral over $x$, weighted by the distribution $p(x)$, is approximated by a finite sum over data points drawn from that distribution. These quantities, along with their sum, are plotted as a function of $\ln \lambda$ in Figure 4.8. We see that small values of $\lambda$ allow the model to become finely tuned to the noise on each individual data set leading to large variance. Conversely, a large value of $\lambda$ pulls the weight parameters towards zero leading to large bias.

Note that the bias-variance decomposition is of limited practical value because it is based on averages with respect to ensembles of data sets, whereas in practice we have only the single observed data set. If we had a large number of independent training sets of a given size, we would be better off combining them into a single larger training set, which of course would reduce the level of over-fitting for a given model complexity. Nevertheless, the bias-variance decomposition often provides useful insights into the model complexity issue, and although we have introduced it in this chapter from the perspective of regression problems, the underlying intuition has broad applicability.

![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-16.jpg?height=1568&width=1605&top_left_y=379&top_left_x=290)

**Image Description:** The image consists of a 2x3 grid of plots illustrating the behavior of a system parameterized by the natural logarithm of lambda, \(\ln \lambda\). Each subplot has \(t\) on the vertical axis and \(x\) on the horizontal axis. 

1. The top row shows \(\ln \lambda = 3\) (smooth curves), the middle row shows \(\ln \lambda = 1\) (more oscillatory), and the bottom row shows \(\ln \lambda = -3\) (high-frequency oscillations). 
2. Each subplot features multiple curves, with variations in amplitude and frequency based on the value of \(\ln \lambda\).

Figure 4.7 Illustration of the dependence of bias and variance on model complexity governed by a regularization parameter $\lambda$, using the sinusoidal data from Chapter 1. There are $L=100$ data sets, each having $N=25$ data points, and there are 24 Gaussian basis functions in the model so that the total number of parameters is $M=25$ including the bias parameter. The left column shows the result of fitting the model to the data sets for various values of $\ln \lambda$ (for clarity, only 20 of the 100 fits are shown). The right column shows the corresponding average of the 100 fits (red) along with the sinusoidal function from which the data sets were generated (green).

Figure 4.8 Plot of squared bias and variance, together with their sum, corresponding to the results shown in Figure 4.7. Also shown is the average test set error for a test data set size of 1,000 points. The minimum value of (bias) ${ }^{2}+$ variance occurs around $\ln \lambda=0.43$, which is close to the value that gives the minimum error on the test data.
![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-17.jpg?height=573&width=960&top_left_y=276&top_left_x=935)

**Image Description:** The image is a diagram depicting a bias-variance tradeoff in machine learning. It features a Cartesian coordinate system with the x-axis labeled as "ln λ" ranging from -3 to 3, and the y-axis ranging from 0 to 0.25. There are four curves plotted: the red curve represents "(bias)²," the blue curve indicates "variance," the green curve shows "(bias)² + variance," and the magenta curve illustrates "test error." Each curve has a corresponding legend in the upper right quadrant identifying their colors and meanings.


## Exercises

4.1 ( $\star$ ) Consider the sum-of-squares error function given by (1.2) in which the function $y(x, \mathbf{w})$ is given by the polynomial (1.1). Show that the coefficients $\mathbf{w}=\left\{w_{i}\right\}$ that minimize this error function are given by the solution to the following set of linear equations:

$$
\sum_{j=0}^{M} A_{i j} w_{j}=T_{i}
$$
where
$$
A_{i j}=\sum_{n=1}^{N}\left(x_{n}\right)^{i+j}, \quad T_{i}=\sum_{n=1}^{N}\left(x_{n}\right)^{i} t_{n}
$$

Here a suffix $i$ or $j$ denotes the index of a component, whereas $(x)^{i}$ denotes $x$ raised to the power of $i$.
$4.2(\star)$ Write down the set of coupled linear equations, analogous to (4.53), satisfied by the coefficients $w_{i}$ that minimize the regularized sum-of-squares error function given by (1.4).
4.3 ( $\star$ ) Show that the tanh function defined by

$$
\tanh (a)=\frac{e^{a}-e^{-a}}{e^{a}+e^{-a}}
$$
and the logistic sigmoid function defined by (4.6) are related by
$$
\tanh (a)=2 \sigma(2 a)-1
$$

Hence, show that a general linear combination of logistic sigmoid functions of the form

$$
y(x, \mathbf{w})=w_{0}+\sum_{j=1}^{M} w_{j} \sigma\left(\frac{x-\mu_{j}}{s}\right)
$$
is equivalent to a linear combination of tanh functions of the form
$$
y(x, \mathbf{u})=u_{0}+\sum_{j=1}^{M} u_{j} \tanh \left(\frac{x-\mu_{j}}{2 s}\right)
$$

and find expressions to relate the new parameters $\left\{u_{1}, \ldots, u_{M}\right\}$ to the original parameters $\left\{w_{1}, \ldots, w_{M}\right\}$.
$4.4(\star \star \star)$ Show that the matrix

$$
\boldsymbol{\Phi}\left(\boldsymbol{\Phi}^{\mathrm{T}} \boldsymbol{\Phi}\right)^{-1} \boldsymbol{\Phi}^{\mathrm{T}}
$$
takes any vector $\mathbf{v}$ and projects it onto the space spanned by the columns of $\boldsymbol{\Phi}$. Use this result to show that the least-squares solution (4.14) corresponds to an orthogonal projection of the vector $\mathbf{t}$ onto the manifold $\mathcal{S}$, as shown in Figure 4.3.
$4.5(\star)$ Consider a data set in which each data point $t_{n}$ is associated with a weighting factor $r_{n}>0$, so that the sum-of-squares error function becomes
$$
E_{D}(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N} r_{n}\left\{t_{n}-\mathbf{w}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right\}^{2} .
$$

Find an expression for the solution $\mathbf{w}^{\star}$ that minimizes this error function. Give two alternative interpretations of the weighted sum-of-squares error function in terms of (i) data-dependent noise variance and (ii) replicated data points.
4.6 ( $\star$ ) By setting the gradient of (4.26) with respect to $\mathbf{w}$ to zero, show that the exact minimum of the regularized sum-of-squares error function for linear regression is given by (4.27).
$4.7(\star \star)$ Consider a linear basis function regression model for a multivariate target variable $\mathbf{t}$ having a Gaussian distribution of the form

$$
p(\mathbf{t} \mid \mathbf{W}, \boldsymbol{\Sigma})=\mathcal{N}(\mathbf{t} \mid \mathbf{y}(\mathbf{x}, \mathbf{W}), \boldsymbol{\Sigma})
$$
where
$$
\mathbf{y}(\mathbf{x}, \mathbf{W})=\mathbf{W}^{\mathrm{T}} \boldsymbol{\phi}(\mathbf{x})
$$

together with a training data set comprising input basis vectors $\phi\left(\mathrm{x}_{n}\right)$ and corresponding target vectors $\mathbf{t}_{n}$, with $n=1, \ldots, N$. Show that the maximum likelihood solution $\mathbf{W}_{\text {ML }}$ for the parameter matrix $\mathbf{W}$ has the property that each column is given by an expression of the form (4.14), which was the solution for an isotropic noise distribution. Note that this is independent of the covariance matrix $\boldsymbol{\Sigma}$. Show that the maximum likelihood solution for $\boldsymbol{\Sigma}$ is given by

$$
\boldsymbol{\Sigma}=\frac{1}{N} \sum_{n=1}^{N}\left(\mathbf{t}_{n}-\mathbf{W}_{\mathrm{ML}}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right)\left(\mathbf{t}_{n}-\mathbf{W}_{\mathrm{ML}}^{\mathrm{T}} \boldsymbol{\phi}\left(\mathbf{x}_{n}\right)\right)^{\mathrm{T}}
$$
$4.8(\star)$ Consider the generalization of the squared-loss function (4.35) for a single target variable $t$ to multiple target variables described by the vector $\mathbf{t}$ given by
$$
\mathbb{E}[L(\mathbf{t}, \mathbf{f}(\mathbf{x}))]=\iint\|\mathbf{f}(\mathbf{x})-\mathbf{t}\|^{2} p(\mathbf{x}, \mathbf{t}) \mathrm{d} \mathbf{x} \mathrm{~d} \mathbf{t}
$$

Using the calculus of variations, show that the function $\mathbf{f}(\mathbf{x})$ for which this expected loss is minimized is given by

$$
\mathbf{f}(\mathbf{x})=\mathbb{E}_{t}[\mathbf{t} \mid \mathbf{x}]
$$
4.9 ( $\star$ ) By expansion of the square in (4.64), derive a result analogous to (4.39) and, hence, show that the function $\mathbf{f}(\mathbf{x})$ that minimizes the expected squared loss for a vector $\mathbf{t}$ of target variables is again given by the conditional expectation of $\mathbf{t}$ in the form (4.65).
$4.10(\star \star)$ Rederive the result (4.65) by first expanding (4.64) analogous to (4.39).
$4.11(\star \star)$ The following distribution
$$
p\left(x \mid \sigma^{2}, q\right)=\frac{q}{2\left(2 \sigma^{2}\right)^{1 / q} \Gamma(1 / q)} \exp \left(-\frac{|x|^{q}}{2 \sigma^{2}}\right)
$$

is a generalization of the univariate Gaussian distribution. Here $\Gamma(x)$ is the gamma function defined by

$$
\Gamma(x)=\int_{-\infty}^{\infty} u^{x-1} e^{-u} \mathrm{~d} u
$$

Show that this distribution is normalized so that

$$
\int_{-\infty}^{\infty} p\left(x \mid \sigma^{2}, q\right) \mathrm{d} x=1
$$
and that it reduces to the Gaussian when $q=2$. Consider a regression model in which the target variable is given by $t=y(\mathbf{x}, \mathbf{w})+\epsilon$ and $\epsilon$ is a random noise variable drawn from the distribution (4.66). Show that the log likelihood function over $\mathbf{w}$ and $\sigma^{2}$, for an observed data set of input vectors $\mathbf{X}=\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{N}\right\}$ and corresponding target variables $\mathbf{t}=\left(t_{1}, \ldots, t_{N}\right)^{\mathrm{T}}$, is given by
$$
\ln p\left(\mathbf{t} \mid \mathbf{X}, \mathbf{w}, \sigma^{2}\right)=-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left|y\left(\mathbf{x}_{n}, \mathbf{w}\right)-t_{n}\right|^{q}-\frac{N}{q} \ln \left(2 \sigma^{2}\right)+\text { const }
$$

where 'const' denotes terms independent of both $\mathbf{w}$ and $\sigma^{2}$. Note that, as a function of $\mathbf{w}$, this is the $L_{q}$ error function considered in Section 4.2.
$4.12(\star \star)$ Consider the expected loss for regression problems under the $L_{q}$ loss function given by (4.40). Write down the condition that $y(\mathbf{x})$ must satisfy to minimize $\mathbb{E}\left[L_{q}\right]$. Show that, for $q=1$, this solution represents the conditional median, i.e., the function $y(\mathbf{x})$ such that the probability mass for $t<y(\mathbf{x})$ is the same as for $t \geqslant y(\mathbf{x})$. Also show that the minimum expected $L_{q}$ loss for $q \rightarrow 0$ is given by the conditional mode, i.e., by the function $y(\mathbf{x})$ being equal to the value of $t$ that maximizes $p(t \mid \mathbf{x})$ for each x .
![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-20.jpg?height=1330&width=1310&top_left_y=281&top_left_x=580)

**Image Description:** The image is a title slide from an academic lecture, featuring the number "5" prominently at the top in red font. Below it, the title reads "Single-layer Networks: Classification" in black text against a colorful abstract background. The background consists of light and dark swirls of color, resembling a network of interconnected fibers or nodes. The design is non-technical and serves as a visual element rather than a diagram or formula, focusing on the theme of the lecture topic.


In the previous chapter, we explored a class of regression models in which the output variables were linear functions of the model parameters and which can therefore be expressed as simple neural networks having a single layer of weight and bias parameters. We turn now to a discussion of classification problems, and in this chapter, we will focus on an analogous class of models that again can be expressed as single-layer neural networks. These will allow us to introduce many of the key concepts of classification before dealing with more general deep neural networks in later chapters.

The goal in classification is to take an input vector $\mathbf{x} \in \mathbb{R}^{D}$ and assign it to one of $K$ discrete classes $\mathcal{C}_{k}$ where $k=1, \ldots, K$. In the most common scenario, the classes are taken to be disjoint, so that each input is assigned to one and only one class. The input space is thereby divided into decision regions whose boundaries are called decision boundaries or decision surfaces. In this chapter, we consider linear
models for classification, by which we mean that the decision surfaces are linear functions of the input vector $\mathbf{x}$ and, hence, are defined by ( $D-1$ )-dimensional hyperplanes within the $D$-dimensional input space. Data sets whose classes can be separated exactly by linear decision surfaces are said to be linearly separable. Linear classification models can be applied to data sets that are not linearly separable, although not all inputs will be correctly classified.

We can broadly identify three distinct approaches to solving classification problems. The simplest involves constructing a discriminant function that directly assigns each vector $\mathbf{x}$ to a specific class. A more powerful approach, however, models the conditional probability distributions $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ in an inference stage and subsequently uses these distributions to make optimal decisions. Separating inference and deci-
Section 5.2.4 sion brings numerous benefits. There are two different approaches to determining the conditional probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$. One technique is to model them directly, for example by representing them as parametric models and then optimizing the parameters using a training set. This will be called a discriminative probabilistic model. Alternatively, we can model the class-conditional densities $p\left(\mathbf{x} \mid \mathcal{C}_{k}\right)$, together with the prior probabilities $p\left(\mathcal{C}_{k}\right)$ for the classes, and then compute the required posterior probabilities using Bayes' theorem:

$$
p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)=\frac{p\left(\mathbf{x} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right)}{p(\mathbf{x})} .
$$

This will be called a generative probabilistic model because it offers the opportunity to generate samples from each of the class-conditional densities $p\left(\mathbf{x} \mid \mathcal{C}_{k}\right)$. In this chapter, we will discuss examples of all three approaches: discriminant functions, generative probabilistic models, and discriminative probabilistic models.

### 5.1. Discriminant Functions

A discriminant is a function that takes an input vector $\mathbf{x}$ and assigns it to one of $K$ classes, denoted $\mathcal{C}_{k}$. In this chapter, we will restrict attention to linear discriminants, namely those for which the decision surfaces are hyperplanes. To simplify the discussion, we consider first two classes and then investigate the extension to $K>2$ classes.

### 5.1.1 Two classes

The simplest representation of a linear discriminant function is obtained by taking a linear function of the input vector so that

$$
y(\mathbf{x})=\mathbf{w}^{\mathrm{T}} \mathbf{x}+w_{0}
$$
where $\mathbf{w}$ is called a weight vector, and $w_{0}$ is a bias (not to be confused with bias in the statistical sense). An input vector $\mathbf{x}$ is assigned to class $\mathcal{C}_{1}$ if $y(\mathbf{x}) \geqslant 0$ and to class $\mathcal{C}_{2}$ otherwise. The corresponding decision boundary is therefore defined by the relation $y(\mathrm{x})=0$, which corresponds to a ( $D-1$ )-dimensional hyperplane within

Figure 5.1 Illustration of the geometry of a linear discriminant function in two dimensions. The decision surface, shown in red, is perpendicular to w , and its displacement from the origin is controlled by the bias parameter $w_{0}$. Also, the signed orthogonal distance of a general point $x$ from the decision surface is given by $y(\mathbf{x}) /\|\mathbf{w}\|$.
![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-22.jpg?height=755&width=955&top_left_y=273&top_left_x=943)

**Image Description:** The image presents a two-dimensional coordinate system with axes labeled \(x_1\) and \(x_2\). It features two different regions, R1 (above the red line) and R2 (below the red line), demarcated by the red line representing a linear inequality \(y > 0\). The green vector \(w\) indicates a weight direction, while the dashed blue line represents a function \(y(x)\). Additional elements include arrows illustrating vector projections on the axes. The visual emphasizes geometric interpretations of inequalities and vector relationships in a two-dimensional space.


the $D$-dimensional input space. Consider two points $\mathbf{x}_{\mathrm{A}}$ and $\mathbf{x}_{\mathrm{B}}$ both of which lie on the decision surface. Because $y\left(\mathbf{x}_{\mathrm{A}}\right)=y\left(\mathbf{x}_{\mathrm{B}}\right)=0$, we have $\mathbf{w}^{\mathrm{T}}\left(\mathbf{x}_{\mathrm{A}}-\mathbf{x}_{\mathrm{B}}\right)=0$ and hence the vector $\mathbf{w}$ is orthogonal to every vector lying within the decision surface, and so $\mathbf{w}$ determines the orientation of the decision surface. Similarly, if $\mathbf{x}$ is a point on the decision surface, then $y(x)=0$, and so the normal distance from the origin to the decision surface is given by

$$
\frac{\mathbf{w}^{\mathrm{T}} \mathbf{x}}{\|\mathbf{w}\|}=-\frac{w_{0}}{\|\mathbf{w}\|} .
$$

We therefore see that the bias parameter $w_{0}$ determines the location of the decision surface. These properties are illustrated for the case of $D=2$ in Figure 5.1.

Furthermore, note that the value of $y(x)$ gives a signed measure of the perpendicular distance $r$ of the point $\mathbf{x}$ from the decision surface. To see this, consider an arbitrary point $\mathbf{x}$ and let $\mathbf{x}_{\perp}$ be its orthogonal projection onto the decision surface, so that

$$
\mathbf{x}=\mathbf{x}_{\perp}+r \frac{\mathbf{w}}{\|\mathbf{w}\|} .
$$

Multiplying both sides of this result by $\mathbf{w}^{\mathrm{T}}$ and adding $w_{0}$, and making use of $y(\mathbf{x})= \mathbf{w}^{\mathrm{T}} \mathbf{x}+w_{0}$ and $y\left(\mathbf{x}_{\perp}\right)=\mathbf{w}^{\mathrm{T}} \mathbf{x}_{\perp}+w_{0}=0$, we have

$$
r=\frac{y(\mathbf{x})}{\|\mathbf{w}\|} .
$$

This result is illustrated in Figure 5.1.
Section 4.1.1
As with linear regression models, it is sometimes convenient to use a more compact notation in which we introduce an additional dummy 'input' value $x_{0}=1$ and then define $\widetilde{\mathbf{w}}=\left(w_{0}, \mathbf{w}\right)$ and $\widetilde{\mathbf{x}}=\left(x_{0}, \mathbf{x}\right)$ so that

$$
y(\mathbf{x})=\widetilde{\mathbf{w}}^{\mathrm{T}} \widetilde{\mathbf{x}}
$$

![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-23.jpg?height=706&width=1541&top_left_y=273&top_left_x=292)

**Image Description:** The image consists of two diagrams depicting geometric configurations involving lines and regions in a plane. 

1. The left diagram shows three red lines \( R_1, R_2, R_3 \) intersecting, forming a green triangular region labeled with a question mark. The lines \( C_1 \) and \( C_2 \) are indicated with arrows, suggesting they might interact with the triangular area.
   
2. The right diagram features a similar layout, with lines \( R_1, R_2, R_3 \) and additional line \( C_3 \) shown. The question mark indicates an unspecified area or relationship in that configuration.

Both diagrams explore relationships in planar geometry involving intersections and areas.

Figure 5.2 Attempting to construct a $K$-class discriminant from a set of two-class discriminant functions leads to ambiguous regions, as shown in green. On the left is an example with two discriminant functions designed to distinguish points in class $\mathcal{C}_{k}$ from points not in class $\mathcal{C}_{k}$. On the right is an example involving three discriminant functions each of which is used to separate a pair of classes $\mathcal{C}_{k}$ and $\mathcal{C}_{j}$.

In this case, the decision surfaces are $D$-dimensional hyperplanes passing through the origin of the ( $D+1$ )-dimensional expanded input space.

### 5.1.2 Multiple classes

Now consider the extension of linear discriminant functions to $K>2$ classes. We might be tempted to build a $K$-class discriminant by combining a number of two-class discriminant functions. However, this leads to some serious difficulties (Duda and Hart, 1973), as we now show.

Consider a model with $K-1$ classifiers, each of which solves a two-class problem of separating points in a particular class $\mathcal{C}_{k}$ from points not in that class. This is known as a one-versus-the-rest classifier. The left-hand example in Figure 5.2 shows an example involving three classes where this approach leads to regions of input space that are ambiguously classified.

An alternative is to introduce $K(K-1) / 2$ binary discriminant functions, one for every possible pair of classes. This is known as a one-versus-one classifier. Each point is then classified according to a majority vote amongst the discriminant functions. However, this too runs into the problem of ambiguous regions, as illustrated in the right-hand diagram of Figure 5.2.

We can avoid these difficulties by considering a single $K$-class discriminant comprising $K$ linear functions of the form

$$
y_{k}(\mathbf{x})=\mathbf{w}_{k}^{\mathrm{T}} \mathbf{x}+w_{k 0}
$$
and then assigning a point $\mathbf{x}$ to class $\mathcal{C}_{k}$ if $y_{k}(\mathbf{x})>y_{j}(\mathbf{x})$ for all $j \neq k$. The decision boundary between class $\mathcal{C}_{k}$ and class $\mathcal{C}_{j}$ is therefore given by $y_{k}(\mathbf{x})=y_{j}(\mathbf{x})$ and

Figure 5.3 Illustration of the decision regions for a multi-class linear discriminant, with the decision boundaries shown in red. If two points $\mathrm{x}_{\mathrm{A}}$ and $\mathrm{x}_{\mathrm{B}}$ both lie inside the same decision region $\mathcal{R}_{k}$, then any point $\widehat{\mathbf{x}}$ that lies on the line connecting these two points must also lie in $\mathcal{R}_{k}$, and hence, the decision region must be singly connected and convex.
![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-24.jpg?height=443&width=629&top_left_y=276&top_left_x=1263)

**Image Description:** The image depicts a geometric diagram illustrating a three-dimensional spatial relationship. It features three red lines labeled \( R_i \), \( R_j \), and \( R_k \), converging at a point, representing different vector directions. A blue line runs horizontally, labeled with points \( X_A \), \( \hat{X} \), and \( X_B \), indicating a position on the line. The diagram emphasizes the relationship between the vectors and their respective spatial coordinates in a defined system.


hence corresponds to a ( $D-1$ )-dimensional hyperplane defined by

$$
\left(\mathbf{w}_{k}-\mathbf{w}_{j}\right)^{\mathrm{T}} \mathbf{x}+\left(w_{k 0}-w_{j 0}\right)=0 .
$$

This has the same form as the decision boundary for the two-class case discussed in Section 5.1.1, and so analogous geometrical properties apply.

The decision regions of such a discriminant are always singly connected and convex. To see this, consider two points $\mathrm{x}_{\mathrm{A}}$ and $\mathrm{x}_{\mathrm{B}}$ both of which lie inside decision region $\mathcal{R}_{k}$, as illustrated in Figure 5.3. Any point $\widehat{\mathbf{x}}$ that lies on the line connecting $\mathrm{x}_{\mathrm{A}}$ and $\mathrm{x}_{\mathrm{B}}$ can be expressed in the form

$$
\widehat{\mathbf{x}}=\lambda \mathbf{x}_{\mathrm{A}}+(1-\lambda) \mathbf{x}_{\mathrm{B}}
$$
where $0 \leqslant \lambda \leqslant 1$. From the linearity of the discriminant functions, it follows that
$$
y_{k}(\widehat{\mathbf{x}})=\lambda y_{k}\left(\mathbf{x}_{\mathrm{A}}\right)+(1-\lambda) y_{k}\left(\mathbf{x}_{\mathrm{B}}\right) .
$$

Because both $\mathrm{x}_{\mathrm{A}}$ and $\mathrm{x}_{\mathrm{B}}$ lie inside $\mathcal{R}_{k}$, it follows that $y_{k}\left(\mathrm{x}_{\mathrm{A}}\right)>y_{j}\left(\mathrm{x}_{\mathrm{A}}\right)$ and that $y_{k}\left(\mathbf{x}_{\mathrm{B}}\right)>y_{j}\left(\mathbf{x}_{\mathrm{B}}\right)$, for all $j \neq k$, and hence $y_{k}(\widehat{\mathbf{x}})>y_{j}(\widehat{\mathbf{x}})$, and so $\widehat{\mathbf{x}}$ also lies inside $\mathcal{R}_{k}$. Thus, $\mathcal{R}_{k}$ is singly connected and convex.

Note that for two classes, we can either employ the formalism discussed here, based on two discriminant functions $y_{1}(\mathbf{x})$ and $y_{2}(\mathbf{x})$, or else use the simpler but essentially equivalent formulation based on a single discriminant function $y(\mathbf{x})$.

### 5.1.3 1-of- $K$ coding

For regression problems, the target variable $\mathbf{t}$ was simply the vector of real numbers whose values we wish to predict. In classification, there are various ways of using target values to represent class labels. For two-class problems, the most convenient is the binary representation in which there is a single target variable $t \in\{0,1\}$ such that $t=1$ represents class $\mathcal{C}_{1}$ and $t=0$ represents class $\mathcal{C}_{2}$. We can interpret the value of $t$ as the probability that the class is $\mathcal{C}_{1}$, with the probability values taking only the extreme values of 0 and 1 . For $K>2$ classes, it is convenient to use a 1 -of- $K$ coding scheme, also known as the one-hot encoding scheme, in which $\mathbf{t}$ is a vector of length $K$ such that if the class is $\mathcal{C}_{j}$, then all elements $t_{k}$ of $\mathbf{t}$ are zero
except element $t_{j}$, which takes the value 1 . For instance, if we have $K=5$ classes, then a data point from class 2 would be given the target vector

$$
\mathbf{t}=(0,1,0,0,0)^{\mathrm{T}} .
$$

Again, we can interpret the value of $t_{k}$ as the probability that the class is $\mathcal{C}_{k}$ in which the probabilities take only the values 0 and 1 .

### 5.1.4 Least squares for classification

With linear regression models, the minimization of a sum-of-squares error func- tion leads to a simple closed-form solution for the parameter values. It is therefore tempting to see if we can apply the same least-squares formalism to classification problems. Consider a general classification problem with $K$ classes and a 1 -of- $K$ binary coding scheme for the target vector $\mathbf{t}$. One justification for using least squares in such a context is that it approximates the conditional expectation $\mathbb{E}[\mathbf{t} \mid \mathbf{x}]$ of the target values given the input vector. For a binary coding scheme, this conditional ex-
Exercise 5.1 pectation is given by the vector of posterior class probabilities. Unfortunately, these probabilities are typically approximated rather poorly, and indeed the approximations can have values outside the range $(0,1)$. However, it is instructional to explore these simple models and to understand how these limitations arise.

Each class $\mathcal{C}_{k}$ is described by its own linear model so that

$$
y_{k}(\mathbf{x})=\mathbf{w}_{k}^{\mathrm{T}} \mathbf{x}+w_{k 0}
$$
where $k=1, \ldots, K$. We can conveniently group these together using vector notation so that
$$
\mathbf{y}(\mathbf{x})=\widetilde{\mathbf{W}}^{\mathrm{T}} \widetilde{\mathbf{x}}
$$

where $\widetilde{\mathbf{W}}$ is a matrix whose $k$ th column comprises the ( $D+1$ )-dimensional vector $\widetilde{\mathbf{w}}_{k}=\left(w_{k 0}, \mathbf{w}_{k}^{\mathrm{T}}\right)^{\mathrm{T}}$ and $\widetilde{\mathbf{x}}$ is the corresponding augmented input vector $\left(1, \mathbf{x}^{\mathrm{T}}\right)^{\mathrm{T}}$ with a dummy input $x_{0}=1$. A new input $\mathbf{x}$ is then assigned to the class for which the output $y_{k}=\widetilde{\mathbf{w}}_{k}^{\mathrm{T}} \widetilde{\mathbf{x}}$ is largest.

We now determine the parameter matrix $\widetilde{\mathbf{W}}$ by minimizing a sum-of-squares error function. Consider a training data set $\left\{\mathbf{x}_{n}, \mathbf{t}_{n}\right\}$ where $n=1, \ldots, N$, and define a matrix $\mathbf{T}$ whose $n$th row is the vector $\mathbf{t}_{n}^{\mathrm{T}}$, together with a matrix $\widetilde{\mathbf{X}}$ whose $n$th row is $\widetilde{\mathbf{x}}_{n}^{\mathrm{T}}$. The sum-of-squares error function can then be written as

$$
E_{D}(\widetilde{\mathbf{W}})=\frac{1}{2} \operatorname{Tr}\left\{(\widetilde{\mathbf{X}} \widetilde{\mathbf{W}}-\mathbf{T})^{\mathrm{T}}(\widetilde{\mathbf{X}} \widetilde{\mathbf{W}}-\mathbf{T})\right\} .
$$

Setting the derivative with respect to $\widetilde{\mathbf{W}}$ to zero and rearranging, we obtain the solution for $\widetilde{\mathbf{W}}$ in the form

$$
\widetilde{\mathbf{W}}=\left(\widetilde{\mathbf{X}}^{\mathrm{T}} \widetilde{\mathbf{X}}\right)^{-1} \widetilde{\mathbf{X}}^{\mathrm{T}} \mathbf{T}=\widetilde{\mathbf{X}}^{\dagger} \mathbf{T}
$$

Section 4.1.3
where $\widetilde{\mathbf{X}}^{\dagger}$ is the pseudo-inverse of the matrix $\widetilde{\mathbf{X}}$. We then obtain the discriminant
function in the form

$$
\mathbf{y}(\mathbf{x})=\widetilde{\mathbf{W}}^{\mathrm{T}} \widetilde{\mathbf{x}}=\mathbf{T}^{\mathrm{T}}\left(\widetilde{\mathbf{X}}^{\dagger}\right)^{\mathrm{T}} \widetilde{\mathbf{x}}
$$

An interesting property of least-squares solutions with multiple target variables is that if every target vector in the training set satisfies some linear constraint

$$
\mathbf{a}^{\mathrm{T}} \mathbf{t}_{n}+b=0
$$
for some constants $\mathbf{a}$ and $b$, then the model prediction for any value of $\mathbf{x}$ will satisfy

## Exercise 5.3

Section 2.3.4

Section 5.4.3 the same constraint so that

$$
\mathbf{a}^{\mathrm{T}} \mathbf{y}(\mathbf{x})+b=0
$$

Thus, if we use a 1 -of- $K$ coding scheme for $K$ classes, then the predictions made by the model will have the property that the elements of $\mathbf{y}(\mathbf{x})$ will sum to 1 for any value of $\mathbf{x}$. However, this summation constraint alone is not sufficient to allow the model outputs to be interpreted as probabilities because they are not constrained to lie within the interval $(0,1)$.

The least-squares approach gives an exact closed-form solution for the discriminant function parameters. However, even as a discriminant function (where we use it to make decisions directly and dispense with any probabilistic interpretation), it suffers from some severe problems. We have seen that the sum-of-squares error function can be viewed as the negative log likelihood under the assumption of a Gaussian noise distribution. If the true distribution of the data is markedly different from being Gaussian, then least squares can give poor results. In particular, least squares is very sensitive to the presence of outliers, which are data points located a long way from the bulk of the data. This is illustrated in Figure 5.4. Here we see that the additional data points in the right-hand figure produce a significant change in the location of the decision boundary, even though these points would be correctly classified by the original decision boundary in the left-hand figure. The sum-of-squares error function gives too much weight to data points that are a long way from the decision boundary, even though they are correctly classified. Outliers can arise due to rare events or may simply be due to mistakes in the data set. Techniques that are sensitive to a very few data points are said to lack robustness. For comparison, Figure 5.4 also shows results from a technique called logistic regression, which is more robust to outliers.

The failure of least squares should not surprise us when we recall that it corresponds to maximum likelihood under the assumption of a Gaussian conditional distribution, whereas binary target vectors clearly have a distribution that is far from Gaussian. By adopting more appropriate probabilistic models, we can obtain classification techniques with much better properties than least squares, and which can also be generalized to give flexible nonlinear neural network models, as we will see in later chapters.

![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-27.jpg?height=743&width=1560&top_left_y=306&top_left_x=292)

**Image Description:** The image consists of two scatter plot diagrams side by side. Both plots display data points in a two-dimensional space. The left plot contains red 'X' markers representing one class and blue circular markers representing another class. A green line distinguishes the two classes, indicating a decision boundary. The right plot features a similar arrangement, with an altered decision boundary represented by a magenta line. The x-axis and y-axis scales appear consistent across both plots, likely ranging from -4 to 8 and -8 to 4, respectively.

Figure 5.4 The left-hand plot shows data from two classes, denoted by red crosses and blue circles, together with the decision boundaries found by least squares (magenta curve) and by a logistic regression model (green curve). The right-hand plot shows the corresponding results obtained when extra data points are added at the bottom right of the diagram, showing that least squares is highly sensitive to outliers, unlike logistic regression.

### 5.2. Decision Theory

When we discussed linear regression we saw how the process of making predictions in machine learning can be broken down into the two stages of inference and decision. We now explore this perspective in much greater depth specifically in the context of classifiers.

Suppose we have an input vector $\mathbf{x}$ together with a corresponding vector $\mathbf{t}$ of target variables, and our goal is to predict $\mathbf{t}$ given a new value for $\mathbf{x}$. For regression problems, $\mathbf{t}$ will comprise continuous variables and in general will be a vector as we may wish to predict several related quantities. For classification problems, $\mathbf{t}$ will represent class labels. Again, $\mathbf{t}$ will in general be a vector if we have more than two classes. The joint probability distribution $p(\mathbf{x}, \mathbf{t})$ provides a complete summary of the uncertainty associated with these variables. Determining $p(\mathbf{x}, \mathbf{t})$ from a set of training data is an example of inference and is typically a very difficult problem whose solution forms the subject of much of this book. In a practical application, however, we must often make a specific prediction for the value of $\mathbf{t}$ or more generally take a specific action based on our understanding of the values $\mathbf{t}$ is likely to take, and this aspect is the subject of decision theory.

Consider, for example, our earlier medical diagnosis problem in which we have taken an image of a skin lesion on a patient, and we wish to determine whether the patient has cancer. In this case, the input vector $\mathbf{x}$ is the set of pixel intensities in
the image, and the output variable $t$ will represent the absence of cancer, which we denote by the class $\mathcal{C}_{1}$, or the presence of cancer, which we denote by the class $\mathcal{C}_{2}$. We might, for instance, choose $t$ to be a binary variable such that $t=0$ corresponds to class $\mathcal{C}_{1}$ and $t=1$ corresponds to class $\mathcal{C}_{2}$. We will see later that this choice of label values is particularly convenient when working with probabilities. The general inference problem then involves determining the joint distribution $p\left(\mathbf{x}, \mathcal{C}_{k}\right)$, or equivalently $p(\mathbf{x}, t)$, which gives us the most complete probabilistic description of the variables. Although this can be a very useful and informative quantity, ultimately, we must decide either to give treatment to the patient or not, and we would like this choice to be optimal according to some appropriate criterion (Duda and Hart, 1973). This is the decision step, and the aim of decision theory is that it should tell us how to make optimal decisions given the appropriate probabilities. We will see that the decision stage is generally very simple, even trivial, once we have solved the inference problem. Here we give an introduction to the key ideas of decision theory as required for the rest of the book. Further background, as well as more detailed accounts, can be found in Berger (1985) and Bather (2000).

Before giving a more detailed analysis, let us first consider informally how we might expect probabilities to play a role in making decisions. When we obtain the skin image $\mathbf{x}$ for a new patient, our goal is to decide which of the two classes to assign the image to. We are therefore interested in the probabilities of the two classes, given the image, which are given by $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$. Using Bayes' theorem, these probabilities can be expressed in the form

$$
p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)=\frac{p\left(\mathbf{x} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right)}{p(\mathbf{x})}
$$

Note that any of the quantities appearing in Bayes' theorem can be obtained from the joint distribution $p\left(\mathbf{x}, \mathcal{C}_{k}\right)$ by either marginalizing or conditioning with respect to the appropriate variables. We can now interpret $p\left(\mathcal{C}_{k}\right)$ as the prior probability for the class $\mathcal{C}_{k}$ and $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ as the corresponding posterior probability. Thus, $p\left(\mathcal{C}_{1}\right)$ represents the probability that a person has cancer, before the image is taken. Similarly, $p\left(\mathcal{C}_{1} \mid \mathbf{x}\right)$ is the posterior probability, revised using Bayes' theorem in light of the information contained in the image. If our aim is to minimize the chance of assigning $\mathbf{x}$ to the wrong class, then intuitively we would choose the class having the higher posterior probability. We now show that this intuition is correct, and we also discuss more general criteria for making decisions.

### 5.2.1 Misclassification rate

Suppose that our goal is simply to make as few misclassifications as possible. We need a rule that assigns each value of $\mathbf{x}$ to one of the available classes. Such a rule will divide the input space into regions $\mathcal{R}_{k}$ called decision regions, one for each class, such that all points in $\mathcal{R}_{k}$ are assigned to class $\mathcal{C}_{k}$. The boundaries between decision regions are called decision boundaries or decision surfaces. Note that each decision region need not be contiguous but could comprise some number of disjoint regions. To find the optimal decision rule, consider first the case of two classes, as in the cancer problem, for instance. A mistake occurs when an input vector belonging
to class $\mathcal{C}_{1}$ is assigned to class $\mathcal{C}_{2}$ or vice versa. The probability of this occurring is given by

$$
\begin{aligned}
p(\text { mistake }) & =p\left(\mathbf{x} \in \mathcal{R}_{1}, \mathcal{C}_{2}\right)+p\left(\mathbf{x} \in \mathcal{R}_{2}, \mathcal{C}_{1}\right) \\
& =\int_{\mathcal{R}_{1}} p\left(\mathbf{x}, \mathcal{C}_{2}\right) \mathrm{d} \mathbf{x}+\int_{\mathcal{R}_{2}} p\left(\mathbf{x}, \mathcal{C}_{1}\right) \mathrm{d} \mathbf{x}
\end{aligned}
$$

We are free to choose the decision rule that assigns each point $\mathbf{x}$ to one of the two classes. Clearly, to minimize $p$ (mistake) we should arrange that each $\mathbf{x}$ is assigned to whichever class has the smaller value of the integrand in (5.20). Thus, if $p\left(\mathbf{x}, \mathcal{C}_{1}\right)>p\left(\mathbf{x}, \mathcal{C}_{2}\right)$ for a given value of $\mathbf{x}$, then we should assign that $\mathbf{x}$ to class $\mathcal{C}_{1}$. From the product rule of probability, we have $p\left(\mathbf{x}, \mathcal{C}_{k}\right)=p\left(\mathcal{C}_{k} \mid \mathbf{x}\right) p(\mathbf{x})$. Because the factor $p(\mathbf{x})$ is common to both terms, we can restate this result as saying that the minimum probability of making a mistake is obtained if each value of $\mathbf{x}$ is assigned to the class for which the posterior probability $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ is largest. This result is illustrated for two classes and a single input variable $x$ in Figure 5.5.

For the more general case of $K$ classes, it is slightly easier to maximize the probability of being correct, which is given by

$$
\begin{aligned}
p(\text { correct }) & =\sum_{k=1}^{K} p\left(\mathbf{x} \in \mathcal{R}_{k}, \mathcal{C}_{k}\right) \\
& =\sum_{k=1}^{K} \int_{\mathcal{R}_{k}} p\left(\mathbf{x}, \mathcal{C}_{k}\right) \mathrm{d} \mathbf{x}
\end{aligned}
$$

which is maximized when the regions $\mathcal{R}_{k}$ are chosen such that each $\mathbf{x}$ is assigned to the class for which $p\left(\mathbf{x}, \mathcal{C}_{k}\right)$ is largest. Again, using the product rule $p\left(\mathbf{x}, \mathcal{C}_{k}\right)= p\left(\mathcal{C}_{k} \mid \mathbf{x}\right) p(\mathbf{x})$, and noting that the factor of $p(\mathbf{x})$ is common to all terms, we see that each $\mathbf{x}$ should be assigned to the class having the largest posterior probability $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$.

### 5.2.2 Expected loss

For many applications, our objective will be more complex than simply minimizing the number of misclassifications. Let us consider again the medical diagnosis problem. We note that, if a patient who does not have cancer is incorrectly diagnosed as having cancer, the consequences may be that they experience some distress plus there is the need for further investigations. Conversely, if a patient with cancer is diagnosed as healthy, the result may be premature death due to lack of treatment. Thus, the consequences of these two types of mistake can be dramatically different. It would clearly be better to make fewer mistakes of the second kind, even if this was at the expense of making more mistakes of the first kind.

We can formalize such issues through the introduction of a loss function, also called a cost function, which is a single, overall measure of loss incurred in taking any of the available decisions or actions. Our goal is then to minimize the total loss incurred. Note that some authors consider instead a utility function, whose value

![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-30.jpg?height=1538&width=1348&top_left_y=287&top_left_x=420)

**Image Description:** The image consists of two diagrams (labeled (a) and (b)) illustrating probability distributions. 

- In diagram (a), the x-axis represents regions \(R_1\) and \(R_2\), while the y-axis shows probability density, \(p(x, C_1)\) and \(p(x, C_2)\). It features a curve peaking in region \(R_1\) (green) and \(R_2\) (red), with a vertical line at \(x_0\) indicating a decision boundary.

- Diagram (b) illustrates a similar distribution but shifts the areas: the previously green area is now predominantly blue, suggesting a change in classification regions, with \(x\) marked by \( \hat{x} \).

Figure 5.5 Schematic illustration of the joint probabilities $p\left(x, \mathcal{C}_{k}\right)$ for each of two classes plotted against $x$, together with the decision boundary $x=\widehat{x}$. Values of $x \geqslant \widehat{x}$ are classified as class $\mathcal{C}_{2}$ and hence belong to decision region $\mathcal{R}_{2}$, whereas points $x<\widehat{x}$ are classified as $\mathcal{C}_{1}$ and belong to $\mathcal{R}_{1}$. Errors arise from the blue, green, and red regions, so that for $x<\widehat{x}$, the errors are due to points from class $\mathcal{C}_{2}$ being misclassified as $\mathcal{C}_{1}$ (represented by the sum of the red and green regions). Conversely for points in the region $x \geqslant \widehat{x}$, the errors are due to points from class $\mathcal{C}_{1}$ being misclassified as $\mathcal{C}_{2}$ (represented by the blue region). By varying the location $\widehat{x}$ of the decision boundary, as indicated by the red double-headed arrow in (a), the combined areas of the blue and green regions remains constant, whereas the size of the red region varies. The optimal choice for $\widehat{x}$ is where the curves for $p\left(x, \mathcal{C}_{1}\right)$ and $p\left(x, \mathcal{C}_{2}\right)$ cross, as shown in (b) and corresponding to $\widehat{x}=x_{0}$, because in this case the red region disappears. This is equivalent to the minimum misclassification rate decision rule, which assigns each value of $x$ to the class having the higher posterior probability $p\left(\mathcal{C}_{k} \mid x\right)$.

Figure 5.6 An example of a loss matrix with elements $L_{k j}$ for the cancer treatment problem. The rows correspond to the true class, whereas the columns correspond to the assignment of class made by our decision criterion.

$$
\begin{aligned}
& \text { normal } \\
& \text { normal } \\
& \text { cancer }
\end{aligned}\left(\begin{array}{cc}
0 & \text { cancer } \\
100 & 0
\end{array}\right)
$$

they aim to maximize. These are equivalent concepts if we take the utility to be simply the negative of the loss. Throughout this text we will use the loss function convention. Suppose that, for a new value of $\mathbf{x}$, the true class is $\mathcal{C}_{k}$ and that we assign $\mathbf{x}$ to class $\mathcal{C}_{j}$ (where $j$ may or may not be equal to $k$ ). In so doing, we incur some level of loss that we denote by $L_{k j}$, which we can view as the $k, j$ element of a loss matrix. For instance, in our cancer example, we might have a loss matrix of the form shown in Figure 5.6. This particular loss matrix says that there is no loss incurred if the correct decision is made, there is a loss of 1 if a healthy patient is diagnosed as having cancer, whereas there is a loss of 100 if a patient having cancer is diagnosed as healthy.

The optimal solution is the one that minimizes the loss function. However, the loss function depends on the true class, which is unknown. For a given input vector $\mathbf{x}$, our uncertainty in the true class is expressed through the joint probability distribution $p\left(\mathrm{x}, \mathcal{C}_{k}\right)$, and so we seek instead to minimize the average loss, where the average is computed with respect to this distribution and is given by

$$
\mathbb{E}[L]=\sum_{k} \sum_{j} \int_{\mathcal{R}_{j}} L_{k j} p\left(\mathbf{x}, \mathcal{C}_{k}\right) \mathrm{d} \mathbf{x}
$$

Each x can be assigned independently to one of the decision regions $\mathcal{R}_{j}$. Our goal is to choose the regions $\mathcal{R}_{j}$ to minimize the expected loss (5.22), which implies that for each $\mathbf{x}$, we should minimize $\sum_{k} L_{k j} p\left(\mathbf{x}, \mathcal{C}_{k}\right)$. As before, we can use the product rule $p\left(\mathbf{x}, \mathcal{C}_{k}\right)=p\left(\mathcal{C}_{k} \mid \mathbf{x}\right) p(\mathbf{x})$ to eliminate the common factor of $p(\mathbf{x})$. Thus, the decision rule that minimizes the expected loss assigns each new $\mathbf{x}$ to the class $j$ for which the quantity

$$
\sum_{k} L_{k j} p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)
$$
is a minimum. Once we have chosen values for the loss matrix elements $L_{k j}$, this is clearly trivial to do.

### 5.2.3 The reject option

We have seen that classification errors arise from the regions of input space where the largest of the posterior probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ is significantly less than unity or equivalently where the joint distributions $p\left(\mathbf{x}, \mathcal{C}_{k}\right)$ have comparable values. These are the regions where we are relatively uncertain about class membership. In some applications, it will be appropriate to avoid making decisions on the difficult cases in anticipation of obtaining a lower error rate on those examples for which a classification decision is made. This is known as the reject option. For example, in our hypothetical cancer screening example, it may be appropriate to use an automatic

Figure 5.7 Illustration of the reject option. Inputs $x$ such that the larger of the two posterior probabilities is less than or equal to some threshold $\theta$ will be rejected.
![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-32.jpg?height=551&width=714&top_left_y=276&top_left_x=1170)

**Image Description:** The diagram is a probability density function showing the conditional probabilities \( p(C_1 | x) \) and \( p(C_2 | x) \) along the x-axis. The x-axis is labeled "reject region," indicating areas of classification. The y-axis ranges from 0 to 1, representing probability values. The blue curve represents \( p(C_1 | x) \), while the red curve represents \( p(C_2 | x) \). Green vertical lines demarcate the boundaries of the reject region, and a horizontal dashed green line at \( \theta \) indicates a decision threshold where classification changes occur.


system to classify those images for which there is little doubt as to the correct class, while requesting a biopsy to classify the more ambiguous cases. We can achieve this by introducing a threshold $\theta$ and rejecting those inputs $\mathbf{x}$ for which the largest of the posterior probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ is less than or equal to $\theta$. This is illustrated for two classes and a single continuous input variable $x$ in Figure 5.7. Note that setting $\theta=1$ will ensure that all examples are rejected, whereas if there are $K$ classes, then setting $\theta<1 / K$ will ensure that no examples are rejected. Thus, the fraction of examples that are rejected is controlled by the value of $\theta$.

We can easily extend the reject criterion to minimize the expected loss, when a loss matrix is given, by taking account of the loss incurred when a reject decision is made.

### 5.2.4 Inference and decision

We have broken the classification problem down into two separate stages, the inference stage in which we use training data to learn a model for $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ and the subsequent decision stage in which we use these posterior probabilities to make optimal class assignments. An alternative possibility would be to solve both problems together and simply learn a function that maps inputs $\mathbf{x}$ directly into decisions. Such a function is called a discriminant function.

In fact, we can identify three distinct approaches to solving decision problems, all of which have been used in practical applications. These are, in decreasing order of complexity, as follows:
(a) First, solve the inference problem of determining the class-conditional densities $p\left(\mathbf{x} \mid \mathcal{C}_{k}\right)$ for each class $\mathcal{C}_{k}$ individually. Separately infer the prior class probabilities $p\left(\mathcal{C}_{k}\right)$. Then use Bayes' theorem in the form

$$
p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)=\frac{p\left(\mathbf{x} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right)}{p(\mathbf{x})}
$$
to find the posterior class probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$. As usual, the denominator in

Bayes' theorem can be found in terms of the quantities in the numerator, using

$$
p(\mathbf{x})=\sum_{k} p\left(\mathbf{x} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right)
$$

Equivalently, we can model the joint distribution $p\left(\mathbf{x}, \mathcal{C}_{k}\right)$ directly and then normalize to obtain the posterior probabilities. Having found the posterior probabilities, we use decision theory to determine the class membership for each new input x . Approaches that explicitly or implicitly model the distribution of inputs as well as outputs are known as generative models, because by sampling from them, it is possible to generate synthetic data points in the input space.
(b) First, solve the inference problem of determining the posterior class probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$, and then subsequently use decision theory to assign each new $\mathbf{x}$ to one of the classes. Approaches that model the posterior probabilities directly are called discriminative models.
(c) Find a function $f(\mathbf{x})$, called a discriminant function, that maps each input $\mathbf{x}$ directly onto a class label. For instance, for two-class problems, $f(\cdot)$ might be binary valued and such that $f=0$ represents class $\mathcal{C}_{1}$ and $f=1$ represents class $\mathcal{C}_{2}$. In this case, probabilities play no role.

Let us consider the relative merits of these three alternatives. Approach (a) is the most demanding because it involves finding the joint distribution over both $\mathbf{x}$ and $\mathcal{C}_{k}$. For many applications, $\mathbf{x}$ will have high dimensionality, and consequently, we may need a large training set to be able to determine the class-conditional densities to reasonable accuracy. Note that the class priors $p\left(\mathcal{C}_{k}\right)$ can often be estimated simply from the fractions of the training set data points in each of the classes. One advantage of approach (a), however, is that it also allows the marginal density of data $p(\mathbf{x})$ to be determined from (5.25). This can be useful for detecting new data points that have low probability under the model and for which the predictions may be of low accuracy, which is known as outlier detection or novelty detection (Bishop, 1994; Tarassenko, 1995).

However, if we wish only to make classification decisions, then it can be wasteful of computational resources and excessively demanding of data to find the joint distribution $p\left(\mathbf{x}, \mathcal{C}_{k}\right)$ when in fact we really need only the posterior probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$, which can be obtained directly through approach (b). Indeed, the classconditional densities may contain a significant amount of structure that has little effect on the posterior probabilities, as illustrated in Figure 5.8. There has been much interest in exploring the relative merits of generative and discriminative approaches to machine learning and in finding ways to combine them (Jebara, 2004; Lasserre, Bishop, and Minka, 2006).

An even simpler approach is (c) in which we use the training data to find a discriminant function $f(\mathrm{x})$ that maps each x directly onto a class label, thereby combining the inference and decision stages into a single learning problem. In the example of Figure 5.8, this would correspond to finding the value of $x$ shown by

![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-34.jpg?height=741&width=1584&top_left_y=295&top_left_x=284)

**Image Description:** The image consists of two subplots. The left plot is a probability density function showing class densities \( p(x|C_1) \) (blue curve) and \( p(x|C_2) \) (red curve) against the variable \( x \), ranging from 0 to 1 on the x-axis and showing density values on the y-axis, with peaks indicating the most likely values for each class. The right plot illustrates the posterior probabilities \( p(C_1|x) \) and \( p(C_2|x) \) against \( x \), featuring a threshold at \( x = 0.5 \) where the two curves intersect.

Figure 5.8 Example of the class-conditional densities for two classes having a single input variable $x$ (left plot) together with the corresponding posterior probabilities (right plot). Note that the left-hand mode of the class-conditional density $p\left(\mathbf{x} \mid \mathcal{C}_{1}\right)$, shown in blue on the left plot, has no effect on the posterior probabilities. The vertical green line in the right plot shows the decision boundary in $x$ that gives the minimum misclassification rate, assuming the prior class probabilities, $p\left(\mathcal{C}_{1}\right)$ and $p\left(\mathcal{C}_{2}\right)$, are equal.

the vertical green line, because this is the decision boundary giving the minimum probability of misclassification.

With option (c), however, we no longer have access to the posterior probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$. There are many powerful reasons for wanting to compute the posterior probabilities, even if we subsequently use them to make decisions. These include:

Minimizing risk. Consider a problem in which the elements of the loss matrix are subjected to revision from time to time (such as might occur in a financial application). If we know the posterior probabilities, we can trivially revise the minimum risk decision criterion by modifying (5.23) appropriately. If we have only a discriminant function, then any change to the loss matrix would require that we return to the training data and solve the inference problem afresh.

Reject option. Posterior probabilities allow us to determine a rejection criterion that will minimize the misclassification rate, or more generally the expected loss, for a given fraction of rejected data points.

Section 2.1.1
Compensating for class priors. Consider our cancer screening example again, and suppose that we have collected a large number of images from the general population for use as training data, which we use to build an automated screening system. Because cancer is rare amongst the general population, we might find that, say, only 1 in every 1,000 examples corresponds to the presence of cancer.

If we used such a data set to train an adaptive model, we could run into severe difficulties due to the small proportion of those in the cancer class. For instance, a classifier that assigned every point to the normal class would achieve $99.9 \%$ accuracy, and it may be difficult to avoid this trivial solution. Also, even a large data set will contain very few examples of skin images corresponding to cancer, and so the learning algorithm will not be exposed to a broad range of examples of such images and hence is not likely to generalize well. A balanced data set with equal numbers of examples from each of the classes would allow us to find a more accurate model. However, we then have to compensate for the effects of our modifications to the training data. Suppose we have used such a modified data set and found models for the posterior probabilities. From Bayes' theorem (5.24), we see that the posterior probabilities are proportional to the prior probabilities, which we can interpret as the fractions of points in each class. We can therefore simply take the posterior probabilities obtained from our artificially balanced data set, divide by the class fractions in that data set, and then multiply by the class fractions in the population to which we wish to apply the model. Finally, we need to normalize to ensure that the new posterior probabilities sum to one. Note that this procedure cannot be applied if we have learned a discriminant function directly instead of determining posterior probabilities.

Combining models. For complex applications, we may wish to break the problem into a number of smaller sub-problems each of which can be tackled by a separate module. For example, in our hypothetical medical diagnosis problem, we may have information available from, say, blood tests as well as skin images. Rather than combine all of this heterogeneous information into one huge input space, it may be more effective to build one system to interpret the images and a different one to interpret the blood data. If each of the two models gives posterior probabilities for the classes, then we can combine the outputs systematically using the rules of probability. One simple way to do this is to assume that, for each class separately, the distributions of inputs for the images, denoted by $\mathbf{x}_{\mathrm{I}}$, and the blood data, denoted by $\mathbf{x}_{\mathrm{B}}$, are independent, so that

$$
p\left(\mathbf{x}_{\mathrm{I}}, \mathbf{x}_{\mathrm{B}} \mid \mathcal{C}_{k}\right)=p\left(\mathbf{x}_{\mathrm{I}} \mid \mathcal{C}_{k}\right) p\left(\mathbf{x}_{\mathrm{B}} \mid \mathcal{C}_{k}\right) .
$$

This is an example of a conditional independence property, because the independence holds when the distribution is conditioned on the class $\mathcal{C}_{k}$. The posterior probability, given both the image and blood data, is then given by

$$
\begin{aligned}
p\left(\mathcal{C}_{k} \mid \mathbf{x}_{\mathrm{I}}, \mathbf{x}_{\mathrm{B}}\right) & \propto p\left(\mathbf{x}_{\mathrm{I}}, \mathbf{x}_{\mathrm{B}} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right) \\
& \propto p\left(\mathbf{x}_{\mathrm{I}} \mid \mathcal{C}_{k}\right) p\left(\mathbf{x}_{\mathrm{B}} \mid \mathcal{C}_{k}\right) p\left(\mathcal{C}_{k}\right) \\
& \propto \frac{p\left(\mathcal{C}_{k} \mid \mathbf{x}_{\mathrm{I}}\right) p\left(\mathcal{C}_{k} \mid \mathbf{x}_{\mathrm{B}}\right)}{p\left(\mathcal{C}_{k}\right)}
\end{aligned}
$$

Thus, we need the class prior probabilities $p\left(\mathcal{C}_{k}\right)$, which we can easily estimate from the fractions of data points in each class, and then we need to normalize

Figure 5.9 The confusion matrix for the cancer treatment problem, in which the rows correspond to the true class and the columns correspond to the assignment of class made by our decision criterion. The elements of the matrix show the numbers of true negatives, false positives, false negatives, and true positives.

$$
\begin{aligned}
& \text { normal } \\
& \text { norcer }
\end{aligned}\left(\begin{array}{cc}
\text { normal } & \text { cancer } \\
N_{\mathrm{TN}} & N_{\mathrm{FP}} \\
N_{\mathrm{FN}} & N_{\mathrm{TP}}
\end{array}\right)
$$

## Section 11.2.3

## Chapter 7

the resulting posterior probabilities so they sum to one. The particular conditional independence assumption (5.26) is an example of a naive Bayes model. Note that the joint marginal distribution $p\left(\mathbf{x}_{\mathrm{I}}, \mathbf{x}_{\mathrm{B}}\right)$ will typically not factorize under this model. We will see in later chapters how to construct models for combining data that do not require the conditional independence assumption (5.26). A further advantage of using models that output probabilities rather than decisions is that they can easily be made differentiable with respect to any adjustable parameters (such as the weight coefficients in the polynomial regression example), which allows them to be composed and trained jointly using gradient-based optimization methods.

### 5.2.5 Classifier accuracy

The simplest measure of performance for a classifier is the fraction of test set points that are correctly classified. However, we have seen that different types of error can have different consequences, as expressed through the loss matrix, and often we therefore do not simply wish to minimize the number of misclassifications. By changing the location of the decision boundary, we can make trade-offs between different kinds of error, for example with the goal of minimizing an expected loss. Because this is such an important concept, we will introduce some definitions and terminology so that the performance of a classifier can be better characterized.
Section 2.1.1
We will consider again our cancer screening example. For each person tested, there is a 'true label' of whether they have cancer or not, and there is also the prediction made by the classifier. If, for a particular person, the classifier predicts cancer and this is in fact the true label, then the prediction is called a true positive. However, if the person does not have cancer it is a false positive. Likewise, if the classifier predicts that a person does not have cancer and this is correct, then the prediction is called a true negative, otherwise it is a false negative. The false positives are also known as type 1 errors whereas the false negatives are called type 2 errors. If $N$ is the total number of people taking the test, then $N_{\mathrm{TP}}$ is the number of true positives, $N_{\mathrm{FP}}$ is the number of false positives, $N_{\mathrm{TN}}$ is the number of true negatives, and $N_{\mathrm{FN}}$ is the number of false negatives, where

$$
N=N_{\mathrm{TP}}+N_{\mathrm{FP}}+N_{\mathrm{TN}}+N_{\mathrm{FN}} .
$$

This can be represented as a confusion matrix as shown in Figure 5.9. Accuracy, measured by the fraction of correct classifications, is then given by

$$
\text { Accuracy }=\frac{N_{\mathrm{TP}}+N_{\mathrm{TN}}}{N_{\mathrm{TP}}+N_{\mathrm{FP}}+N_{\mathrm{TN}}+N_{\mathrm{FN}}} .
$$

We can see that accuracy can be misleading if there are strongly imbalanced classes. In our cancer screening example, for instance, where only 1 person in 1,000 has cancer, a naive classifier that simply decides that nobody has cancer will achieve $99.9 \%$ accuracy and yet is completely useless.

Several other quantities can be defined in terms of these numbers, of which the most commonly encountered are

$$
\begin{aligned}
\text { Precision } & =\frac{N_{\mathrm{TP}}}{N_{\mathrm{TP}}+N_{\mathrm{FP}}} \\
\text { Recall } & =\frac{N_{\mathrm{TP}}}{N_{\mathrm{TP}}+N_{\mathrm{FN}}} \\
\text { False positive rate } & =\frac{N_{\mathrm{FP}}}{N_{\mathrm{FP}}+N_{\mathrm{TN}}} \\
\text { False discovery rate } & =\frac{N_{\mathrm{FP}}}{N_{\mathrm{FP}}+N_{\mathrm{TP}}}
\end{aligned}
$$

In our cancer screening example, precision represents an estimate of the probability that a person who has a positive test does indeed have cancer, whereas recall is an estimate of the probability that a person who has cancer is correctly detected by the test. The false positive rate is an estimate of the probability that a person who is normal will be classified as having cancer, whereas the false discovery rate represents the fraction of those testing positive who do not in fact have cancer.

By altering the location of the decision boundary, we can change the trade-offs between the two kinds of errors. To understand this trade-off, we revisit Figure 5.5, but now we label the various regions as shown in Figure 5.10. We can relate the labelled regions to the various true and false rates as follows:

$$
\begin{aligned}
& N_{\mathrm{FP}} / N=E \\
& N_{\mathrm{TP}} / N=D+E \\
& N_{\mathrm{FN}} / N=B+C \\
& N_{\mathrm{TN}} / N=A+C
\end{aligned}
$$

where we are implicitly considering the limit $N \rightarrow \infty$ so that we can relate number of observations to probabilities.

### 5.2.6 ROC curve

A probabilistic classifier will output a posterior probability, which can be converted to a decision by setting a threshold. As the value of the threshold is varied, we can reduce type 1 errors at the expense of increasing type 2 errors, or vice versa. To better understand this trade-off, it is useful to plot the receiver operating characteristic or ROC curve (Fawcett, 2006), a name that originates from procedures to measure the performance of radar receivers. This is a graph of true positive rate versus false positive rate, as shown in Figure 5.11.

As the decision boundary in Figure 5.10 is moved from $-\infty$ to $\infty$, the ROC curve is traced out and can then be generated by plotting the cumulative fraction of

![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-38.jpg?height=695&width=1340&top_left_y=276&top_left_x=425)

**Image Description:** The diagram depicts a probability density function \( p(x, C) \) represented by a black curve spanning between regions \( R_1 \) and \( R_2 \). The x-axis denotes the variable \( x \), with \( x_0 \) marking a vertical dashed line dividing regions \( A \), \( B \), \( C \), \( D \), and \( E \) under the curve. The areas \( B \) (red) and \( C \) (green) highlight two distinct probabilities associated with different conditions \( C_1 \) and \( C_2 \), respectively. The horizontal double-headed arrow indicates a comparison between two regions, enhancing its interpretative clarity.

Figure 5.10 As in Figure 5.5, with the various regions labelled. In the cancer classification problem, region $\mathcal{R}_{1}$ is assigned to the normal class whereas region $\mathcal{R}_{2}$ is assigned to the cancer class.

correct detection of cancer on the $y$-axis versus the cumulative fraction of incorrect detection on the $x$-axis. Note that a specific confusion matrix represents one point along the ROC curve. The best possible classifier would be represented by a point at the top left corner of the ROC diagram. The bottom left corner represents a simple classifier that assigns every point to the normal class and therefore has no true positives but also no false positives. Similarly, the top right corner represents a classifier that assigns everything to the cancer class and therefore has no false negatives but also no true negatives. In Figure 5.11, the classifiers represented by the blue curve are better than those of the red curve for any choice of, say, false positive rate. It is also possible, however, for such curves to cross over, in which case the choice of which is better will depend on the choice of operating point.

As a baseline, we can consider a random classifier that simply assigns each data point to cancer with probability $\rho$ and to normal with probability $1-\rho$. As we vary the value of $\rho$ it will trace out an ROC curve given by a diagonal straight line, as shown in Figure 5.11. Any classifier below the diagonal line performs worse than random guessing.

Sometimes it is useful to have a single number that characterises the whole ROC curve. One approach is to measure the area under the curve (AUC). A value of 0.5 for the AUC represents random guessing whereas a value of 1.0 represents a perfect classifier.

Another measure is the $F$-score, which is the geometric mean of precision and

Figure 5.11 The receiver operator characteristic (ROC) curve is a plot of true positive rate against false positive rate, and it characterizes the trade-off between type 1 and type 2 errors in a classification problem. The upper blue curve represents a better classifier than the lower red curve. Here the dashed curve represents the performance of a simple random classifier.
![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-39.jpg?height=738&width=735&top_left_y=276&top_left_x=1149)

**Image Description:** The image is a Receiver Operating Characteristic (ROC) curve diagram. It displays the True Positive Rate (TPR) on the vertical axis and the False Positive Rate (FPR) on the horizontal axis, both ranging from 0 to 1. Two curves are plotted: one in blue and another in red, indicating the performance of two different classification models. A diagonal gray line represents a random classifier, serving as a baseline for comparison. The curves illustrate trade-offs between sensitivity and specificity for varying threshold settings.


recall, and is therefore defined by

$$
\begin{aligned}
F & =\frac{2 \times \text { precision } \times \text { recall }}{\text { precision }+ \text { recall }} \\
& =\frac{2 N_{\mathrm{TP}}}{2 N_{\mathrm{TP}}+N_{\mathrm{FP}}+N_{\mathrm{FN}}} .
\end{aligned}
$$

Of course, we can also combine the confusion matrix in Figure 5.9 with the loss matrix in Figure 5.6 to compute the expected loss by multiplying the elements pointwise and summing the resulting products.

Although the ROC curve can be extended to more than two classes, it rapidly becomes cumbersome as the number of classes increases.

### 5.3. Generative Classifiers

We turn next to a probabilistic view of classification and show how models with linear decision boundaries arise from simple assumptions about the distribution of the data. We have already discussed the distinction between the discriminative and

## Section 5.2.4

the generative approaches to classification. Here we will adopt a generative approach in which we model the class-conditional densities $p\left(\mathbf{x} \mid \mathcal{C}_{k}\right)$ as well as the class priors $p\left(\mathcal{C}_{k}\right)$ and then use these to compute posterior probabilities $p\left(\mathcal{C}_{k} \mid \mathbf{x}\right)$ through Bayes' theorem.

First, consider problems having two classes. The posterior probability for class

Figure 5.12 Plot of the logistic sigmoid function $\sigma(a)$ defined by (5.42), shown in red, together with the scaled probit function $\Phi(\lambda a)$, for $\lambda^{2}=\pi / 8$, shown in dashed blue, where $\Phi(a)$ is defined by (5.86). The scaling factor $\pi / 8$ is chosen so that the derivatives of the two curves are equal for $a=0$.
![](https://cdn.mathpix.com/cropped/2025_10_04_d6841743a465c395e721g-40.jpg?height=518&width=766&top_left_y=298&top_left_x=1113)

**Image Description:** The image depicts a graph plotting a sigmoid function. The x-axis ranges from approximately -5 to 5, while the y-axis goes from 0 to 1. Two curves are presented: a solid red line and a dashed blue line, both representing the function's behavior. The graph illustrates the sigmoid function's gradual increase and its asymptotic approach to the values at the extremes of the x-axis. The intersection point near x=0 shows where the function transitions from lower to higher values, highlighting its characteristic 'S' shape.


$\mathcal{C}_{1}$ can be written as

$$
\begin{aligned}
p\left(\mathcal{C}_{1} \mid \mathbf{x}\right) & =\frac{p\left(\mathbf{x} \mid \mathcal{C}_{1}\right) p\left(\mathcal{C}_{1}\right)}{p\left(\mathbf{x} \mid \mathcal{C}_{1}\right) p\left(\mathcal{C}_{1}\right)+p\left(\mathbf{x} \mid \mathcal{C}_{2}\right) p\left(\mathcal{C}_{2}\right)} \\
& =\frac{1}{1+\exp (-a)}=\sigma(a)
\end{aligned}
$$

where we have defined

$$
a=\ln \frac{p\left(\mathbf{x} \mid \mathcal{C}_{1}\right) p\left(\mathcal{C}_{1}\right)}{p\left(\mathbf{x} \mid \mathcal{C}_{2}\right) p\left(\mathcal{C}_{2}\right)}
$$
and $\sigma(a)$ is the logistic sigmoid function defined by
$$
\sigma(a)=\frac{1}{1+\exp (-a)},
$$

which is plotted in Figure 5.12. The term 'sigmoid' means S-shaped. This type of function is sometimes also called a 'squashing function' because it maps the whole real axis into a finite interval. The logistic sigmoid has been encountered already in earlier chapters and plays an important role in many classification algorithms. It satisfies the following symmetry property:

$$
\sigma(-a)=1-\sigma(a)
$$
as is easily verified. The inverse of the logistic sigmoid is given by
$$
a=\ln \left(\frac{\sigma}{1-\sigma}\right)
$$

and is known as the logit function. It represents the log of the ratio of probabilities $\ln \left[p\left(\mathcal{C}_{1} \mid \mathbf{x}\right) / p\left(\mathcal{C}_{2} \mid \mathbf{x}\right)\right]$ for the two classes, also known as the log odds.

Note that in (5.40), we have simply rewritten the posterior probabilities in an equivalent form, and so the appearance of the logistic sigmoid may seem artificial.

