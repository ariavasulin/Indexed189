---
course: CS 189
semester: Fall 2025
type: lecture
title: Logistic Regression (2)
source_type: slides
source_file: Lecture 11 -- Logistic Regression (2).pptx
processed_date: '2025-10-04'
processor: mathpix
---

## Lecture 11

## Logistic Regression (2)

Logistic Regression Optimization, Decisions, and Evaluation

## EECS 189/289, Fall 2025 @ UC Berkeley

Joseph E. Gonzalez and Narges Norouzi

# III Join at slido.com <br> '1َيْL \#1643395 

## Roadmap

- Regularization with Logistic Regression
- Logistic Regression Optimization
- Making Decisions
- Multi-Class Classification
- Model Evaluation


## The Sigmoid ( $\sigma$ ) Function

The S-shaped curve is formally known as the sigmoid function

$$
\sigma(a)=\frac{1}{1+e^{-a}}
$$
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-04.jpg?height=583&width=1026&top_left_y=544&top_left_x=1892)

**Image Description:** The image displays a sigmoid function diagram. The x-axis represents the variable \( a \), ranging from approximately -5 to 5, while the y-axis represents \( \sigma(a) \), which values between 0 and 1. The curve gradually rises, showing a characteristic "S" shape, illustrating the smooth transition from low to high values of \( \sigma(a) \) as \( a \) increases. The function approaches 0 as \( a \) becomes negative and approaches 1 as \( a \) becomes positive, with a midpoint around \( a = 0 \).

$1-\sigma(a)=\frac{e^{-a}}{1+e^{-a}}=\sigma(-a)$
Domain
Range
Reflectio
n/
$-\infty<a<+\infty$
$0<\sigma(a)<1$
Symmetr
Derivative
$\mathrm{y}_{\text {Inverse }}$
$a=\sigma^{-1}(p)=\log \left(\frac{p}{1-p}\right)$
$\frac{d}{d a} \sigma(a)=\sigma(a)(1-\sigma(a))=\sigma(a) \sigma(-a)$

## The Logistic Regression Objective (MLEq)

$$
p(t \mid w, \mathcal{D})=\prod_{n=1}^{N} y_{n}^{t_{n}}\left(1-y_{n}\right)^{1-t_{n}}
$$

We can define an error function by taking the negative logarithm of the likelihood, which gives the cross-entropy error function.

$$
E(w)=-\ln p(t \mid w, \mathcal{D})=-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right]
$$

# Regularization with Logistic Regression 

## What Is the Value of $w$ ?

A. $w=-1$
$w=\underset{w}{\operatorname{argmin}}-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right]$
B. $w=1$

Assume $\phi(\mathrm{x})=\mathrm{x}$
C. $\quad w=-\infty$
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-07.jpg?height=1068&width=2102&top_left_y=807&top_left_x=72)

**Image Description:** The image depicts a two-dimensional Cartesian coordinate system. The horizontal axis (x-axis) ranges from -1 to 1, while the vertical axis (y-axis) ranges from -1 to 1. Two points are plotted: one at coordinates (-1, 1) marked with a blue dot and another at (1, 0), also marked with a blue dot. The title "The Data" is positioned above the vertical axis, indicating that the plotted points represent specific data values in the coordinate plane.

D. $w=+\infty$
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-07.jpg?height=1045&width=981&top_left_y=824&top_left_x=2227)

**Image Description:** The image shows two plots of logistic functions. The top graph depicts the sigmoid curve defined on the range of x from -4 to 4, with the y-axis representing the output of the logistic function ranging from 0 to 1, illustrating the S-shaped growth. The vertical line at x = 0 indicates the midpoint of the curve. The bottom graph shows a similar curve, possibly representing a modified logistic function or its inverse, with an asymptotic approach towards 0 and 1 as x approaches negative and positive infinity, respectively. Both plots provide insights into logistic growth dynamics.

-
-
0 What is the best value for w ?

## What Is the Value of $w$ ?

A. $w=-1$
$w=\underset{w}{\operatorname{argmin}}-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right]$
B. $w=1$

Assume $\phi(\mathrm{x})=\mathrm{x}$
C. $\quad w=-\infty$
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-09.jpg?height=1068&width=2102&top_left_y=807&top_left_x=72)

**Image Description:** The diagram is a Cartesian coordinate system displaying two data points. The x-axis ranges from -1 to 1, while the y-axis ranges from -1 to 1. The point (-1, 1) is located in the upper left quadrant, and the point (1, 0) is positioned in the lower right quadrant. Both points are marked with blue dots, indicating their specific coordinates. The diagram visually represents the relationship between the two data points in a two-dimensional space.

D. $w=+\infty$
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-09.jpg?height=1045&width=981&top_left_y=824&top_left_x=2227)

**Image Description:** The image contains two graphs stacked vertically. Each graph represents a nonlinear function plotted on Cartesian coordinates. The x-axis ranges from -4 to 4, while the y-axis ranges from 0 to 1. The top graph shows a sigmoid-like curve that increases asymptotically towards 1 as x approaches infinity and approaches 0 as x approaches negative infinity. The bottom graph depicts a logistic function that decreases asymptotically towards 0 as x approaches infinity and approaches 1 as x approaches negative infinity. Both graphs provide visual representations of the behavior of these functions across the specified range.


## What Is the Value of $w$ ?

$$
w=\underset{w}{\operatorname{argmin}}-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right]
$$

For the point ( $-1,1$ ):

## Objective:

$$
-\ln \sigma\left(w^{T} \mathrm{x}\right)=-\ln \sigma(-w) \quad \Rightarrow w \rightarrow-\infty
$$
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-10.jpg?height=1090&width=1102&top_left_y=310&top_left_x=2228)

## What Is the Value of $w$ ?

$$
w=\underset{w}{\operatorname{argmin}}-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right]
$$

For the point $(1,0)$ :
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-11.jpg?height=945&width=1999&top_left_y=922&top_left_x=0)

**Image Description:** The image contains a diagram and an equation. The diagram is a 2D Cartesian coordinate system with the x-axis labeled from 0 to 1 and a blue point positioned at (1, 0). The equation displayed is the objective function, given as $$ -\ln(1 - \sigma(w^T x)) = -\ln(1 - \sigma(w)) \Rightarrow w \rightarrow -\infty $$, where $\sigma$ denotes the sigmoid function. The diagram illustrates a specific case related to the objective function's behavior as $w$ approaches negative infinity.


## What Is the Value of $w$ ?

For the point ( $-1,1$ ):

$$
-\ln \sigma\left(w^{T} \mathrm{x}\right) \quad \Rightarrow w \rightarrow-\infty
$$

For the point $(1,0)$ :

$$
-\ln \left(1-\sigma\left(w^{T} \mathrm{x}\right)\right) \quad \Rightarrow w \rightarrow-\infty
$$
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-12.jpg?height=711&width=1281&top_left_y=1139&top_left_x=2015)

**Image Description:** The image depicts a two-dimensional Cartesian coordinate system with x and y-axes. The x-axis ranges from -1 to 1, while the y-axis ranges from -1 to 1. Two points are highlighted: (-1,1) is marked with a blue dot and labeled "The Data," and (1,0) is marked with another blue dot. An orange horizontal line extends from (-1,1) to (1,1), and a vertical line drops from (1,0) to intersect the x-axis. The phrase "Overly confident!" is prominently displayed in bold, yellow text in the upper right corner.


## Adding Regularization to Logistic Regression

$$
w=\underset{w}{\operatorname{argmin}}-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right]+\frac{\lambda}{2} \sum_{d=1}^{D} w^{2}
$$

Prevents weights from diverging on linearly separable data.
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-13.jpg?height=694&width=1234&top_left_y=1063&top_left_x=293)

**Image Description:** The image is a diagram illustrating a curve labeled "Without Regularization." It features a grid with the horizontal axis (θ) representing a parameter range, likely from -5 to 5, and the vertical axis indicating the output value, reaching up to approximately 5. The curve starts at the origin (0,0) and rises steeply, demonstrating a non-linear relationship. An earlier example is referenced on the left. The curve is depicted in blue, emphasizing the lack of regularization in the model's behavior.

![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-13.jpg?height=694&width=1060&top_left_y=1063&top_left_x=1573)

**Image Description:** The slide features a diagram displaying a convex curve representing a cost function in a regularization context. The x-axis, labeled \(\theta\), ranges from approximately -4 to 4, while the y-axis, representing the cost value, ranges from 0 to 7. A blue curve illustrates the relationship between \(\theta\) and the cost, indicating the optimal point marked by a blue dot at \(\theta \approx -2\). The title notes regularization with \(\lambda = 0.1\), emphasizing the impact of this parameter on the cost function's shape.


# Logistic Regression Optimization 

## Logistic Regression Optimization

$$
\begin{aligned}
& E(w)=-\ln p(\mathrm{t} \mid w, \mathcal{D})=-\sum_{n=1}^{N}[\underbrace{\left[t_{n} \ln y_{n}\right.}_{\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}}+\underbrace{\left(1-t_{n}\right) \ln \left(1-y_{n}\right)}_{\begin{array}{l}
\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w} \\
\downarrow
\end{array}}] \\
& \begin{array}{c}
\text { Derivative wrt } w \\
\text { using chain rule: }
\end{array} \\
& \xrightarrow{\partial E(w)}=-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}+\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w}\right]
\end{aligned}
$$

## Logistic Regression Optimization

$$
\begin{aligned}
E(w)=-\ln p(\mathrm{t} \mid w, \mathcal{D}) & =-\sum_{n=1}^{N}[\underbrace{t_{n} \ln y_{n}}_{\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}}+\underbrace{\left(1-t_{n}\right) \ln \left(1-y_{n}\right)}_{\downarrow}] \\
\begin{array}{l}
\text { Derivative wrt } w \\
\text { using chain rule: }
\end{array} & ] \\
\xrightarrow{\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w}} & =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}+\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w}\right] \\
& =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}-\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial y_{n}}{\partial w}\right]
\end{aligned}
$$

$$
\frac{\partial\left(1-y_{n}\right)}{\partial w}=-\frac{\partial y_{n}}{\partial w}
$$

$$
\begin{aligned}
E(w)=-\ln p(\mathrm{t} \mid w, \mathcal{D}) & =-\sum_{n=1}^{N}[\underbrace{\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}}+\underbrace{\left(1-t_{n}\right) \ln \left(1-y_{n}\right)}_{\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w}}] \\
\begin{array}{l}
\text { Derivative wrt } w \\
\text { using chain rule: }
\end{array} & \frac{\partial E(w)}{\partial w} \\
& =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}+\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w}\right] \\
& =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}-\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial y_{n}}{\partial w}\right] \\
& =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}}-\frac{\left(1-t_{n}\right)}{1-y_{n}}\right] \frac{\partial y_{n}}{\partial w}
\end{aligned}
$$

$$
\begin{aligned}
E(w)=-\ln p(\mathrm{t} \mid w, \mathcal{D}) & =-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right] \\
\frac{\partial E(w)}{\partial w} & =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}+\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w}\right] \\
& =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}-\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial y_{n}}{\partial w}\right] \\
& =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}}-\frac{\left(1-t_{n}\right)}{1-y_{n}}\right]\left[\frac{\partial y_{n}}{\partial w} \quad \begin{array}{c}
\frac{\partial\left(1-y_{n}\right)}{\partial w}=-\frac{\partial y_{n}}{\partial w} \\
\text { Need to apply chain rule }
\end{array}\right. \\
\frac{\partial E(w)}{\partial w} & =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}}-\frac{\left(1-t_{n}\right)}{1-y_{n}}\right] y_{n}\left(1-y_{n}\right) \phi\left(\mathrm{x}_{n}\right) \\
\frac{\partial y_{n}}{\partial w}=\frac{\partial \sigma\left(w^{T} \phi\left(\mathrm{x}_{n}\right)\right)}{\partial w} & =\sigma\left(w^{T} \phi\left(\mathrm{x}_{n}\right)\right)\left(1-\sigma\left(w^{T} \phi\left(\mathrm{x}_{n}\right)\right)\right) \phi\left(\mathrm{x}_{n}\right)=y_{n}\left(1-y_{n}\right) \phi\left(\mathrm{x}_{n}\right)
\end{aligned}
$$

$$
\begin{aligned}
E(w)=-\ln p(\mathrm{t} \mid w, \mathcal{D}) & =-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right] \\
\frac{\partial E(w)}{\partial w} & =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}}-\frac{\left(1-t_{n}\right)}{1-y_{n}}\right] y_{n}\left(1-y_{n}\right) \phi\left(\mathrm{x}_{n}\right) \quad \text { Separating the terms } \\
& =-\sum_{n=1}^{N}\left[t_{n}\left(1-y_{n}\right) \phi\left(\mathrm{x}_{n}\right)-\left(1-t_{n}\right) y_{n} \phi\left(\mathrm{x}_{n}\right)\right] \\
& =-\sum_{n=1}^{N}\left[t_{n} \phi\left(\mathrm{x}_{n}\right)-t_{n} y_{n} \phi\left(\mathrm{x}_{n}\right)-y_{n} \phi\left(\mathrm{x}_{n}\right)+t_{n} y_{n} \phi\left(\mathrm{x}_{n}\right)\right] \\
& =-\sum_{n=1}^{N}\left[t_{n} \phi\left(\mathrm{x}_{n}\right)-y_{n} \phi\left(\mathrm{x}_{n}\right)\right]=\sum_{n=1}^{N}\left[y_{n}-t_{n}\right] \phi\left(\mathrm{x}_{n}\right)
\end{aligned}
$$

## We Cannot Solve Directly

$$
\begin{aligned}
E(w)=-\ln p(\mathrm{t} \mid w, \mathcal{D}) & =-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right] \\
\frac{\partial E(w)}{\partial w} & =\sum_{n=1}^{N}\left[y_{n}-t_{n}\right] \phi\left(\mathrm{x}_{n}\right) \\
& =\sum_{n=1}^{N}\left[\sigma\left(w^{T} \phi\left(\mathrm{x}_{n}\right)\right)-t_{n}\right] \phi\left(\mathrm{x}_{n}\right)
\end{aligned}
$$

No Closed Form Solution
$w$ is inside the $\sigma$ for every term so we cannot pull out $w$.

We cannot apply logit to inverse $\sigma$ : logit $(a+b) \neq \operatorname{logit}(a)+\operatorname{logit}(b)$

More on this in lectures 12 and 13 with Gradient Descent.

## Decisions = Posteriors + Loss

- We separate inference from decision.

Inference gives posteriors $p\left(C_{k} \mid \mathrm{x}\right)$.
The task gives a loss matrix $L_{k j}$.

The right action is the one with minimum expected loss at this x .

$$
\begin{aligned}
& \text { Expected loss at } \mathrm{x} \text { if } \\
& \text { we choose class } j
\end{aligned} \quad=\sum_{k} L_{k j} p\left(C_{k} \mid \mathrm{x}\right)
$$

## Decisions = Posteriors + Loss

The right action is the one with minimum expected loss at this x .
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-23.jpg?height=703&width=3169&top_left_y=650&top_left_x=85)

**Image Description:** The image is a flowchart illustrating a process in machine learning. It depicts three main components: 

1. An input labeled \( x \) on the left.
2. A central box labeled "Model outputs" which is affected by the input.
3. A right section labeled "Decision computes," leading to a final box labeled "Pick." 

Arrows indicate the flow from input to decision. Additionally, there is an equation for expected loss represented as $$E[\text{Loss} | x, j] = \sum_k L_{kj} P(c_k | x)$$, integrating probability and loss function considerations.


Change the costs $L \rightarrow$ decisions change immediately;
No retraining. That's why we prefer probabilities over hard labels.

## Expected Loss Example

In the example of cancer prediction, let's assume for a data sample we have: $p\left(C_{0} \mid \mathrm{x}\right)=0.7$ and $p\left(C_{1} \mid \mathrm{x}\right)=0.3$.

Assume also that the Loss matrix is given as:

$$
\left.\begin{array}{l} 
\\
\text { normal } \\
\text { cancer }
\end{array} \begin{array}{cc}
C_{0} & C_{1} \\
\text { normal } & \text { cancer } \\
0 & 1 \\
100 & 0
\end{array}\right)
$$

$\left.\begin{array}{cll}\begin{array}{c}\text { If we predict normal } \\ j=C_{1}\end{array} & \sum_{k=C_{0}} L(j, k) p\left(C_{1} \mid \mathrm{x}\right) & =100 \times 0.3=30 \\ \begin{array}{c}\text { If we predict cancer } \\ j=C_{0}\end{array} & \sum_{k=C_{1}} L(j, k) p\left(C_{0} \mid \mathrm{x}\right) & =1 \times 0.7=0.7\end{array}\right\} \begin{aligned} & 0.7<30 \text { and predicting cancer gives } \\ & \text { smaller loss so we predict the class of } \\ & \text { cancer. }\end{aligned}$

## The Rejection Option

Reject decisions when max posterior probability $\leq$ threshold $\theta$.

- Control error rate vs. rejection fraction.

Multi-Class Classification

## Multiple Classes

## One-versus-the-rest classifier

- A model with $K-1$ classifiers, each of which solves a two-class problem of separating points in a particular class $C_{k}$ from points not in that class.


## One-versus-one classifier

- Introduce $\frac{K(K-1)}{2}$ binary discriminant functions, one for every possible pair of classes.


## Better Option：Multi－class Discriminant Solutio䣼慮能

－Single $K$－class discriminant with linear functions：

$$
y_{k}(\mathrm{x})=w_{k}^{T} \mathrm{x}+w_{k 0}
$$
－Classify x into class $C_{k}$ if：
$$
y_{k}(\mathrm{x})>y_{j}(\mathrm{x}) \quad \forall j \neq k
$$

－The decision boundary between the two classes are always singly connected and convex，defined as：

$$
\left(w_{k}-w_{j}\right)^{T} \mathrm{x}+\left(w_{k 0}-w_{j 0}\right)=0 .
$$

## K > 2 Classes: Softmax Function

Let the model output scores (logits) $a_{1}, \ldots, a_{K} \in R$.

- We want to turn these scores into $p\left(C_{k} \mid \mathrm{x}\right)$.
-We generalize binary logit:
For every two class $i, j: \ln \frac{p_{i}}{p_{j}}=a_{i}-a_{j}$
- Now we solve to find $p_{k}^{\prime} s$, knowing that $\sum_{k=1}^{K} p_{k}=1$.

$$
1=\sum_{i=1}^{K} p_{i}=p_{j} \sum_{k=1}^{K} e^{a_{i}-a_{j}} \rightarrow p_{j}=\frac{1}{\sum_{k=1}^{K} e^{a_{i}-a_{j}}}=\frac{e^{a_{j}}}{\sum_{k=1}^{K} e^{a_{i}}} \quad\left(\begin{array}{c}
\text { Softmax Function } \\
p_{j}=\frac{e^{a_{j}}}{\sum_{k=1}^{K} e^{a_{i}}}
\end{array}\right.
$$

Writing the equation based on one class $j$

How? $\quad \ln \frac{p_{i}}{p_{j}}=a_{i}-a_{j} \longrightarrow \quad p_{i}=e^{a_{i}-a_{j}} p_{j}$

Which of the following are true about the softmax function

Model Evaluation

## Classifier Performance

"Positive" means a prediction of 1.
"Negative" means a prediction of $\mathbf{0}$.
"True" means correct prediction. "False" means incorrect prediction.

Predictio
|  | n |  |  |
| :--- | :--- | :--- | :--- |
|  |  | 0 | 1 |
| Actual | 0 | True Negative (TN) | False Positive (FP) |
|  | 1 | False Negative (FN) | True Positive (TP) |


## Classifier Performance

A confusion matrix plots TP, TN, FP, FN for a particular classifier, dataset, and threshold (!).

Our $\mathbf{0} / \mathbf{1}$ predictions depend on our choice of probability threshold, so the confusion matrix can also change with a new threshold!

Predicted
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-33.jpg?height=758&width=1179&top_left_y=956&top_left_x=2049)

**Image Description:** The image is a confusion matrix represented as a 2x2 grid, commonly used in classification tasks. The axes are labeled "Actual" on the vertical side and "Predicted" on the horizontal side. The cells display counts: top-left (True Negatives) = 266, top-right (False Positives) = 20, bottom-left (False Negatives) = 39, bottom-right (True Positives) = 130. A color gradient on the right indicates the frequency of values, transitioning from dark blue to light blue, with darker shades indicating higher counts. This matrix visually summarizes the performance of a classification model.


## Classifier Accuracy

$$
\begin{aligned}
& \text { accuracy }=\frac{\# \text { of points classified correctly }}{\# \text { Total points }} \\
& \text { accuracy }=\frac{T P+T N}{T P+T N+F P+F N}=\frac{T P+T N}{N}
\end{aligned}
$$

The most basic evaluation metric for a classifier is accuracy.

## Pitfalls of Accuracy: A Case Study

Suppose we're trying to build a classifier to filter spam emails.
Each email is spam (1) or ham (0).
We have 100 emails: 5 of them are spam, and the remaining 95 are real (i.e., ham).

## Classifier 1:

Classify every email as ham
(0).

What is the accuracy of Classifier 1?

## Pitfalls of Accuracy: A Case Study

Suppose we're trying to build a classifier to filter spam emails.
Each email is spam (1) or ham (0).
We have 100 emails: 5 of them are spam, and the remaining 95 are real (i.e., ham).

## Classifier 1:

Classify every email as ham (0).

$$
\operatorname{accuracy}_{1}=\frac{95}{100}=0.95
$$

High accuracy...
...but we detected none $\triangle$ of the spam!!!

## Classifier 2:

Classify every email as spam (1).
$\operatorname{accuracy}_{2}=\frac{5}{100}=0.05$ ...but we detected all of the spam!!!

While widely used, the accuracy metric is is problematic if there is class @dian balance.

Suppose I increase the probability threshold converting my estimated probabilities from logistic regression into $\mathbf{0 / 1}$ predictions.

I do not change my fitted model or test data.
What can happen to the number of true positives (TP) in the test data? Select all that apply.

Effect of changing the probability threshold on T．P

I increase the probability threshold．What happens to the count of true positives（TP）

Threshold $=0$ ．

|  | Thresholc 3 |  |  |
| :--- | :--- | :--- | :--- |
| Estimated probabilities | ヒ．く＊ | 6 0.63 | Threshold＝0． 8 |
| Prediction（ $\mathrm{T}=0.3$ ） | 0 （TN） | 1 （TP） | 1 （TP） |
| Prediction（ $\mathrm{T}=0.6$ ） | 0 （TN） | 1 （TP） | 1 （TP） |
| Prediction（ $\mathrm{T}=0.8$ ） | 0 （TN） | 0 （FN） | 1 （TP） |

$$
\text { TP = } 2
$$

No change！

$$
\text { TP = } 1
$$

If we increase the threshold，it becomes harder to make a positive prediction．
The number of positives（ $\mathbf{P}$ ）will stay the same or decrease．

So．TP and FP will either ao down or stav the same．too．

## Accuracy, Precision, and Recall

Predictio
| Actual | n |  |  |
| :--- | :--- | :--- | :--- |
|  | 0 | TN | FP |
|  | 1 | FN | TP |


Predictio
|  |  | 0 | 1 |
| :--- | :--- | :--- | :--- |
| Actual | 0 | TN | FP |
|  | 1 | FN | TP |


$$
\text { accuracy }=\frac{T P+T N}{N}
$$

What proportion of all points were correctly classified?
"What \% of emails did I classify correctly?"
precision $=\frac{T P}{P}=\frac{T P}{T P+F P}$
Of all positives ( $\mathbf{P}$ ), what proportion were correct (TP)?
"What \% of spam predictions were correct?"

1643395
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-39.jpg?height=613&width=609&top_left_y=276&top_left_x=2398)

**Image Description:** The image presents a confusion matrix, a common tool in classification problems. It consists of a 2x2 grid with the axes labeled "Actual" and "Predicted." The cells represent:

- True Negative (TN) in the top left.
- False Positive (FP) in the top right.
- False Negative (FN) in the bottom left.
- True Positive (TP) in the bottom right.

The labels indicate the predicted and actual classes, allowing for assessment of classification performance through metrics such as accuracy, precision, and recall.

recall $=\frac{T P}{\text { Actual } 1 s}=\frac{T P}{T P+F N}$
Of all actual 1s, what proportion did our model detect?
What \% of spam emails were detected?"

Recall is also known as sensitivity.

## Back to the Spam

Suppose we're trying to build a class

|  | $\mathbf{0}$ | $\mathbf{1}$ |
| :---: | :---: | :---: |
| 0 | TN: 0 | FP: <br> 95 |
| 1 | FN: 0 | TP: 5 |

$$
\begin{gathered}
\text { accuracy }=\frac{T P+T N}{N} \\
\text { precision }=\frac{T P}{T P+F P} \\
\text { recall }=\frac{T P}{T P+F N} \\
\hline
\end{gathered}
$$

Each email is spam (1) or ham (0).
Let's say we have 100 emails, of which only 5 are truly spam, and the remaining 95 are ham.

## Classifier 1:

Classify every email as ham
(0).
accuracy $_{1}=\frac{95}{100}=0.95$
precision $_{1}=\frac{0}{0+0}=$ undefined
recall $_{1}=\frac{0}{0+5}=0$

## Classifier 2:

Classify every email as spam (1).

$$
\left.\left.\begin{array}{l}
\operatorname{accuracy}_{2}=\frac{5}{100}=0.05 \\
\operatorname{precision}_{2}=\frac{5}{5+95}=0.05 \\
\operatorname{recall}_{2}=\frac{5}{5+0}=1
\end{array}\right\} \begin{array}{l}
\text { Many false } \\
\text { positives! }
\end{array}\right\} \begin{aligned}
& \text { No false } \\
& \text { negatives! }
\end{aligned}
$$

## Precision vs. Recall

$$
\text { precision }=\frac{T P}{T P+\boxed{F P}}
$$

Precision penalizes FPs.

$$
\text { recall }=\frac{T P}{T P+E N}
$$

Recall penalizes FNs.

There is a tradeoff between precision and recall; they are often inversely related.

FPs and FNs can have different costs based on the Loss matrix.

- You should adjust your threshold to optimize for the expected Loss.


## "Sweeping Through" Thresholds

In the following slides, we talk about computing metrics over different thresholds (T).
Here's a diagram of this process for a general train-validate split over 99 thresholds:
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-42.jpg?height=1192&width=3322&top_left_y=671&top_left_x=0)

**Image Description:** The image is a flow diagram illustrating the process of estimating probabilities for validation data in a model fitting context. It consists of multiple horizontal sections, each detailing steps to convert probabilities to binary predictions using different threshold values (T), ranging from 0.01 to 0.99. Arrows connect these sections to a "Plot metrics as a function of T" box, indicating that the resulting accuracy metrics are computed for each threshold. The diagram emphasizes the iterative evaluation process for determining model performance based on varying probability thresholds.


## Choosing an Accuracy Threshold

The choice of threshold T impacts classification performance.

- High T: Most predictions are 0 . Lots of FNs.
- Low T: Most predictions are 1. Lots of FPs.

Do we get max accuracy when $T=0.5$ ? Not always the case...

Accuracy vs. Threshold
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-43.jpg?height=1464&width=1213&top_left_y=293&top_left_x=1297)

1643395
Accuracy Max 0.87912

## PrecisionRecall Curves

To construct a precision-recall curve, we:
1)Generate predictions for many different ${ }_{1643395}$ thresholds.
2)For each threshold, compute precision and recall.

Performance Metrics vs. Threshold
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-44.jpg?height=720&width=877&top_left_y=773&top_left_x=1326)

**Image Description:** The image is a line graph displaying the relationship between various performance metrics (Accuracy, Precision, Recall) against a threshold value. The x-axis represents the threshold ranging from 0 to 1, while the y-axis shows the corresponding value of the metrics, also ranging from 0 to 1. Three lines are depicted: the red line for Accuracy, the blue line for Precision, and the green line for Recall, each showing distinct trends as the threshold varies. The graph aids in visualizing the trade-offs between these performance metrics as the threshold is adjusted.


Precision vs. Recall
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-44.jpg?height=754&width=788&top_left_y=769&top_left_x=2364)

**Image Description:** The image is a precision-recall curve, a graphical representation used in machine learning to evaluate the performance of classification models. The x-axis represents recall (sensitivity), ranging from 0 to 1, while the y-axis represents precision, also ranging from 0 to 1. The curve shows the trade-off between precision and recall at various threshold settings. A steep curve indicates a good balance between precision and recall, with areas closer to the top left corner of the plot being more desirable for model performance. The curve appears to plateau at higher recall values.


We often choose a threshold that keeps both precision and recall high. But, if the cost of FPs and TPs differs a lot, we may not.

## Computing an F1 Score

One way to balance precision and recall is to maximize the $\boldsymbol{F}_{\mathbf{1}}$ Score:

$$
\mathrm{F} 1 \text { Score }=\frac{2}{\frac{1}{\text { Precision }}+\frac{1}{\text { Recall }}}=\frac{2 \times \text { Precision } \times \text { Recall }}{\text { Precision }+ \text { Recall }}
$$
- The harmonic mean of precision and recall
- Often used when there is a large class imbalance

How do you use it?

- Pick the threshold that maximizes the $\boldsymbol{F}_{\boldsymbol{1}}$ score

Note: If FPs and FNs have different costs, we may not want to balance precision + recall.

## Maximizing F1 Score

We can maximize the F1 score by evaluating the F1 score
for many threshold values.

Finding F1 Score Maximum
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-46.jpg?height=821&width=991&top_left_y=765&top_left_x=1264)

**Image Description:** The image is a line graph depicting the relationship between the threshold and the F1 score. The x-axis represents the threshold, ranging from 0 to 1, while the y-axis shows the F1 score, which varies between 0 and 1. A blue line illustrates the F1 score's dependency on the threshold with a peak at approximately 0.6, highlighted by a red dot indicating the maximum F1 score of 0.54855. The graph serves to analyze the performance of a classification model as threshold values change.


Precision vs. Recall
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-46.jpg?height=788&width=843&top_left_y=782&top_left_x=2330)

**Image Description:** The image is a precision-recall curve diagram. The x-axis represents Recall, ranging from 0 to 1, while the y-axis represents Precision, also ranging from 0 to 1. The curve illustrates the trade-off between precision and recall, indicating model performance across different threshold settings. A red dot marks the point of maximum F1 score, which is 0.54855, highlighting the optimal balance between precision and recall. The curve generally slopes downward, depicting decreasing precision as recall increases. The background is light blue, and gridlines are present for clearer interpretation.


## True and False Positive Rates

Two more performance metrics worth knowing!
$F P R=\frac{F P}{T N+F P}$
False Positive Rate (FPR): Out of all actual 0s, how many did we classify incorrectly?
$\mathrm{T} P R=\frac{T P}{T P+F N}$
True Positive Rate (TPR): Out of all actual 1s, how many did we classify
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-47.jpg?height=622&width=618&top_left_y=527&top_left_x=2517)

**Image Description:** The image is a confusion matrix, used in machine learning to evaluate the performance of classification models. It has two rows and two columns, representing predicted classes (Predicto) vs. actual classes (Actual). The axes are labeled as follows: the vertical axis indicates actual values (0 and 1), while the horizontal axis indicates predicted values (0 and 1). The cells contain values: True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP), providing a summary of correct and incorrect classifications.

correctly? Same as recall.

Lots of classification metrics out there! All based on TP, TN, FP, FN.

## ROC Curves

We can perform a similar process with FPR and TPR.

1) Try many thresholds
2) Compute the FPR and TPR for each threshold
3) Choose a threshold that keeps FPR low and TPR high

ROC = "receiver operating characteristic" (comes from radar in WWII)
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-48.jpg?height=992&width=1035&top_left_y=340&top_left_x=1632)

**Image Description:** The image is a diagram representing a Receiver Operating Characteristic (ROC) curve. The x-axis is labeled "FPR" (False Positive Rate), ranging from 0 to 1. The y-axis is labeled "TPR" (True Positive Rate), also ranging from 0 to 1. The curve is plotted in blue, showing a gradual increase in TPR as FPR increases, indicating the trade-off between sensitivity and specificity at various threshold settings for a binary classifier. The plot typically assesses model performance across different classification thresholds.


Threshold is high:
FPR low, TPR low (few positive predictions)

Threshold is low: FPR high, TPR high (many positive predictions)

## ROC Curves

A perfect predictor has $\mathrm{TPR}=1$ and $\mathrm{FPR}=0$

## ROC Curve

![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-49.jpg?height=984&width=1043&top_left_y=348&top_left_x=1632)

**Image Description:** The image is a Receiver Operating Characteristic (ROC) curve diagram, which visualizes the performance of a binary classifier. The x-axis represents the False Positive Rate (FPR) ranging from 0 to 1, while the y-axis denotes the True Positive Rate (TPR) also from 0 to 1. The curve is plotted in blue, illustrating the trade-off between sensitivity and specificity at various threshold settings. The area under the curve (AUC) can be used to assess the classifier's accuracy. The grid lines and axes enhance readability and interpretation of the model’s performance.


The best Area Under the ROC Curve (AUC$\mathbf{R O C}$ ) is 1 .

Because we want our classifier to be as close as possible to the perfect predictor, we aim to maximize the AUC.

## ROC Curves

A predictor that predicts randomly has an AUC-ROC of 0.5

ROC Curve
![](https://cdn.mathpix.com/cropped/2025_10_04_5981aa95f0ce529ee756g-50.jpg?height=1039&width=1043&top_left_y=293&top_left_x=1632)

**Image Description:** The image is a Receiver Operating Characteristic (ROC) curve diagram. It has True Positive Rate (TPR) on the y-axis and False Positive Rate (FPR) on the x-axis, both ranging from 0 to 1. The blue line represents the performance of a classifier, demonstrating its sensitivity versus the fall-out at various threshold settings. The purple diagonal line indicates a random classifier, serving as a baseline for comparison. The area under the curve (AUC) quantifies the quality of the classifier, with a value closer to 1 indicating better performance.


Real-world classifiers have an AUC between 0.5 and 1 .

## Summary: How do I pick my threshold

It comes down to the cost of FPs and FNs. Choose threshold that minimizes expected loss.
General guidelines when FP and FN costs are similar:
When classes are balanced and you care equally about good negative+positive predictions:

- Pick threshold to maximize accuracy (often close to 0.5 )

When classes are imbalanced or you care more about good positive predictions:

- Pick threshold to maximize $F_{1}$ score

Accuracy and $\mathbf{F}_{\mathbf{1}}$-score are threshold-dependent while AUC metrics are not.

## Logistic Regression (2)

Credit: Joseph E. Gonzalez and Narges Norouzi
Reference Book Chapters: Chapter 5 Sections 5.1.2, 5.1.3, 5.2

