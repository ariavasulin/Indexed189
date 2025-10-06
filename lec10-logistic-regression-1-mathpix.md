---
course: CS 189
semester: Fall 2025
type: lecture
title: Logistic Regression (1)
source_type: slides
source_file: Lecture 10 -- Logistic Regression (1).pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Lecture 10

## Logistic Regression (1)

## Introduction to Logistic Regression

## EECS 189/289, Fall 2025 @ UC Berkeley

Joseph E. Gonzalez and Narges Norouzi

# III Join at slido.com <br> '1َيْL \#1965715 

## \section*{\section*{Roadmap <br> <br> <br> Roadmap <br> <br> <br> Roadmap}}

- Classification Task
- Defining a New Model for Classification
- Linear Discriminant Functions
- Discriminative Probabilistic Models
- Sigmoid for Classification
- Logistic Regression Objective
- Regularization with Logistic Regression
- Logistic Regression Optimization
- Making Decisions


## Classification Task

- Classification Task
- Defining a New Model for Classification
- Linear Discriminant Functions
- Discriminative Probabilistic Models
- Sigmoid for Classification
- Logistic Regression Objective
- Regularization with Logistic Regression
- Logistic Regression Optimization
- Making Decisions


## So Far ...

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-05.jpg?height=1387&width=2991&top_left_y=480&top_left_x=178)

**Image Description:** The slide features a diagram illustrating a mathematical optimization problem within the context of machine learning. It includes a multidimensional feature space denoted as "Domain" with axes labeled as \( x \) and \( w \). Arrows indicate mapping from domain features \( \phi(x_n) \) to target values \( y(x, w) \). Additionally, an equation for optimization is presented: 

$$
\arg\min_{w} \frac{1}{2} \sum_{n=1}^{N} (t_n - w^T \phi(x_n))^2 + \frac{\lambda}{2} w^T w
$$ 

This equation combines squared error and regularization terms.


## Classification Task Shift

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-06.jpg?height=652&width=3059&top_left_y=484&top_left_x=178)

**Image Description:** The image depicts a conceptual diagram illustrating a function mapping from a domain to binary classification. The left side features a shaded area labeled "Domain," indicating the input space. The central area is a Cartesian coordinate system with x- and y-axes extending to negative and positive infinity. An arrow labeled \(y(x, w)\) points towards a coordinate in the system, represented as a black dot. To the right, the expression “isBenign?” signifies the output, which is classified within the set \(\{0, 1\}\), indicating a binary outcome of benign or not.


## Kinds of Classification

e want to predict some categorical variable, or response, $y$.

- Binary classification
- Two classes
- Responses ( $y$ ) are either 0 or 1
- Multiclass classification
- Many classes
- Image labeling and sentiment analysis (positive, negative, or neutral).
Structured prediction tasks
- Predicting a structured object instead of a discrete class
- Examples: Translation and ChatGPT.
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-07.jpg?height=230&width=1204&top_left_y=1021&top_left_x=1448)

**Image Description:** The image consists of two photographs. The left image shows a penguin resting on a rocky surface, characterized by its black and white plumage, with a blurred natural background that suggests a coastal environment. The right image features a black and white cat lying on its side, partially obscured by its paws. Its fur is fluffy, and it appears relaxed in a domestic setting, with a soft gray couch visible. Both images contrast the behavioral postures of a bird and a mammal.

win or lose
disease or
no disease
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-07.jpg?height=345&width=558&top_left_y=1029&top_left_x=2730)

**Image Description:** This is a miscellaneous image depicting a cat resting on a bed. The cat features a black and white coat, with its front paws covering its face, suggesting a relaxed pose. The background includes white bedding and soft natural light filtering through a window. The setting appears calm and cozy, providing a serene atmosphere.



## Classification

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-08.jpg?height=652&width=3050&top_left_y=484&top_left_x=178)

**Image Description:** The image features a diagram illustrating a function mapping from a "Domain" (represented as a blue shape) to an output indicating whether an input is benign (denoted as $\{0,1\}$). 

The x-axis and y-axis are depicted as two intersecting lines, suggesting a coordinate system. Arrows indicate a transformation represented by the function $\phi$. The output, $y(x, w)$, is connected to the benign classification question. The image includes references to negative and positive infinity along the axes, emphasizing the comprehensive range of inputs.


## Can we just use least squares?

$$
\underset{w}{\operatorname{argmin}} \frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-w^{T} \phi\left(\mathrm{x}_{n}\right)\right)^{2}+\frac{\lambda}{2} w^{T} w
$$

## Demo

We see the distribution of the two classes in the figure. Y axis shows the class of the image (benign or malignant) and the X axis shows Mean Radius.
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-09.jpg?height=1030&width=1804&top_left_y=450&top_left_x=1407)

**Image Description:** The image is a scatter plot representing a binary classification of tumors based on their mean radius. The x-axis is labeled "Mean Radius," indicating the size measurement of the tumors. The y-axis denotes tumor classification, with "0" for benign (blue dots) and "1" for malignant (red dots). The plot illustrates a clear separation between benign and malignant tumors, with benign tumors clustering at lower mean radius values and malignant tumors at higher values. The distinct color coding aids in visual differentiation, emphasizing the relationship between tumor radius and malignancy.


## Demo

We usually add jittering to be able to see all data points clearly.
def jitter(data, amt=0.1):
return data +
amt*np. random. rand(len(data)

- amt/2.0
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-10.jpg?height=1026&width=1800&top_left_y=450&top_left_x=1411)

**Image Description:** The diagram is a scatter plot illustrating the relationship between "Mean Radius" on the x-axis and a binary classification of "Malignant" on the y-axis, with values encoded as 0 for Benign (blue dots) and 1 for Malignant (red dots). The x-axis displays the mean radius measurements, ranging approximately from 10 to 25, while the y-axis is a categorical indication, showing a clear separation of benign and malignant cases based on radius measurements.



## Demo

Here is the probability density of the data for both classes.

The Probability Density of the Data
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-11.jpg?height=1035&width=1851&top_left_y=450&top_left_x=1360)

**Image Description:** The image is a density plot comparing two distributions of "Mean Radius" for malignant (red) and benign (blue) tumors. The x-axis represents "Mean Radius" with values ranging from 0 to 30, while the y-axis indicates density. Histograms in the background show the frequency of occurrences for each category, with overlaying smoothed density curves for visual comparison. Additional tick marks at the bottom display individual data points for malignant and benign tumors, color-coded accordingly. The plot enables the analysis of the differences in radius distribution between the two tumor types.


## Demo

What if we test linear regression model for this task?

Here, the orange line is a regression line fit to the Mean Radius feature, mapping it to the scalar target values (0s and 1s).

Linear Regression Fitted to the Dataset
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-12.jpg?height=1026&width=1821&top_left_y=450&top_left_x=1386)

**Image Description:** The image is a scatter plot illustrating the relationship between "Mean Radius" (x-axis) and "Malignant" status (y-axis), where "Malignant" is coded as 0 for benign (blue) and 1 for malignant (red). The plot features a least squares regression line (in orange) that indicates a positive correlation between mean radius and the likelihood of malignancy. The x-axis ranges from approximately 10 to 25, while the y-axis ranges from 0 to 1. Data points cluster around two distinct categories, highlighting the separation between benign and malignant cases.


## Demo

We needed a decision function (e.g., $f(x)>0.5$ ) ...

If we use the threshold of 0.5 between the two target values 0 and 1 and highlight all data points that are labeled as benign with blue and all malignant predictions with red, you will see that a decision boundary is being formed between the two classes with the black

Use of 0.5 as the Decision Boundary Between the Two Classes
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-13.jpg?height=1030&width=1851&top_left_y=450&top_left_x=1360)

**Image Description:** The image is a scatter plot illustrating the classification of tumors as benign (0) and malignant (1) based on two variables. The x-axis represents the "Mean Radius" of tumors, while the y-axis indicates the binary classification (0 for benign, 1 for malignant). Red points depict malignant tumors, and blue points indicate benign ones. The plot includes a least squares decision boundary represented by a solid yellow line, separating the two classes. A vertical dotted line is also present, marking a specific threshold for classification.


## Demo

It is difficult to interpret a model that outputs values below 0 and above 1 so we capped the output between 0 and 1.

Breast Cancer Prediction using Linear Regression
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-14.jpg?height=1060&width=1914&top_left_y=565&top_left_x=1297)

**Image Description:** The image is a scatter plot diagram illustrating the probability of malignancy against mean radius in a dataset. The x-axis represents the mean radius, ranging approximately from 1 to 25, while the y-axis displays the probability of malignancy, scaled between -0.5 and 1. Points are color-coded: red indicates classified malignant cases, and blue indicates benign cases. A green curve labeled "Least Squares Decision Boundary" depicts the decision threshold, while a yellow line indicates the least squares fit. A vertical black dotted line is present, aligning with a particular mean radius value for reference.


## Demo

This model is also sensitive to outliers.

## Breast Cancer Prediction with an Outlier

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-15.jpg?height=1030&width=1859&top_left_y=450&top_left_x=1352)

**Image Description:** The image is a scatter plot showing the classification of points as malignant or benign. The x-axis represents "Mean Radius" while the y-axis indicates the binary classification of tumors (0 for benign, 1 for malignant). Red points indicate malignant tumors and blue points represent benign tumors. A solid black line denotes the least squares regression line, whereas the dashed lines indicate the decision boundary. An orange line represents the new least squares decision boundary. An extreme point is highlighted in green. This diagram illustrates a classification problem in a medical context.


## Classification

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-16.jpg?height=1395&width=3127&top_left_y=480&top_left_x=110)

**Image Description:** The slide features a combination of a diagram and an equation. 

1. The diagram includes a Cartesian coordinate system with labeled axes, showing a function \(y(x, w)\) which indicates prediction outcomes classified as benign (0) or malignant (1). A question on classification is posed ("Is Benign?"). The domain is represented by a blue shape labeled "Domain" with a summation symbol prominently displaying the function’s output. 

2. The equation presented is:

$$
\arg \min_w \sum_{n=1}^{N} (t_n - w^T \phi(x_n))^2 + \lambda \frac{1}{2} w^T w
$$ 

3. A large orange and white label prominently states "Don't use Least Squares for Classification".


## Least squares assumes Gaussian prior, so it fails with binary data.

## Takeaway!!!

## Defining a New Model for Classification

- Classification Task
- Defining a New Model for Classification
- Linear Discriminant Functions
- Discriminative Probabilistic Models
- Sigmoid for Classification
- Logistic Regression Objective
- Regularization with Logistic Regression
- Logistic Regression Optimization
- Making Decisions


## Classification Setup

- Goal: Map an input vector $\mathrm{x} \in \mathbb{R}^{D}$ to one of the $K$ discrete classes $C_{k}$ where $k=1,2, \ldots, K$.
- $K=2$ : Binary Classification
- $K>2$ : Multi-class Classification
- Input space is divided into decision regions. Boundaries are decision boundaries or decision surfaces.
- ( $D-1$ )-dimensional hyperplanes within the D -dimensional input space.


## Classification Setup

- Input space is divided into decision regions. Boundaries are decision boundaries or decision surfaces.
- ( $D-1$ )-dimensional hyperplanes within the D -dimensional input space.
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-20.jpg?height=609&width=974&top_left_y=748&top_left_x=136)

**Image Description:** The image is a scatter plot illustrating a decision boundary in a binary classification context. The x-axis represents "mean radius," while the y-axis is unlabelled. The plot contains two clusters of points: blue (representing one class) and red (representing another class). The vertical dashed line indicates the decision boundary at a specific mean radius, demarcating the regions where the classification changes from one class to the other. The density of points suggests the distribution of data relative to the boundary.

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-20.jpg?height=563&width=975&top_left_y=752&top_left_x=1122)

**Image Description:** The image is a scatter plot showing a two-dimensional feature space with data points colored in blue and red. The x-axis represents "mean texture" and the y-axis likely represents another feature (not labeled). The points are divided into two groups, with a dashed diagonal line indicating the decision boundary separating the blue points from the red points. The decision boundary is stated to transform the two-dimensional space into a one-dimensional line, illustrating the classification technique being discussed.

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-20.jpg?height=618&width=724&top_left_y=748&top_left_x=2113)

**Image Description:** The image is a three-dimensional scatter plot showing data points classified into two categories, represented by red and blue colors. The axes are labeled "mean texture," "mean radius," and "mean perimeter." The plot illustrates a decision boundary in the form of a plane, indicating the separation between the two classes based on the three feature dimensions. Data points cluster according to these features, with the boundary suggesting regions of classification for each category.

- Dataset with classes that are separable by linear decision surfaces are said to be linearly separable.


## 3 Approaches in Solving Classification

1. Discriminant functions (direct mapping): Direct assignment to a class.
Ex: SVM (margin-based view)
2. Generative probabilistic models (model conditional probability $p\left(\mathrm{x} \mid C_{k}\right)$ and prior $p\left(C_{k}\right)$ then apply Bayes).

Ex: Naïve Bayes, ChatGPT

$$
p\left(C_{k} \mid \mathrm{x}\right)=\frac{p\left(\mathrm{x} \mid C_{k}\right) p\left(C_{k}\right)}{p(\mathrm{x})}
$$
3. Discriminative probabilistic models (model conditional probability $p\left(C_{k} \mid \mathrm{x}\right)$ directly.

Ex: Logistic Regression

## Linear Discriminant Functions

- Classification Task
- Defining a New Model for Classification
- Linear Discriminant Functions
- Discriminative Probabilistic Models
- Sigmoid for Classification
- Logistic Regression Objective
- Regularization with Logistic Regression
- Logistic Regression Optimization
- Making Decisions


## Discriminant Classification with Two Classes

- Basic representation:

A linear function of the input vector so that $y(x)=w^{T} x+w_{0}$.
-Decision rule:

$$
\text { Class Assignment }=\left\{\begin{array}{lr}
C_{1} & \text { if } y(\mathrm{x}) \geq 0.5 \\
C_{2} & \text { otherwise }
\end{array}\right.
$$

- Decision boundary:

$$
y(x)=0.5
$$

Which is a ( $D-1$ )-dimensional surface.

$$
\begin{aligned}
y(\mathrm{x}) & =0.5 \\
w^{T} \mathrm{x}+w_{0} & =0.5 \\
\mathrm{x}_{\text {Mean_radius }} \times w_{1}+\mathrm{x}_{\text {Mean_texture }} \times w_{2}+w_{0} & =0.5
\end{aligned}
$$

$D=2 \rightarrow$ decision boundary is 1-D (a line)
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-23.jpg?height=515&width=1133&top_left_y=1233&top_left_x=1951)

**Image Description:** The diagram illustrates a scatter plot depicting a decision boundary for a binary classification problem. The x-axis represents an input variable, while the y-axis represents the predicted output, labeled as \(y(x)\). Data points are shown in two colors: red and blue, indicating different classes. The dashed line marks the decision boundary where \(y(x) = 0.5\), with areas above the boundary corresponding to \(y(x) > 0.5\) and areas below to \(y(x) < 0.5\). An arrow indicates the direction relative to the decision boundary, likely highlighting the classification distinction.


## Geometry of Linear Discriminant Functions

For any two points on the line $\mathrm{x}_{\mathrm{A}}$ and $\mathrm{x}_{\mathrm{B}}$ :

$$
\begin{gathered}
y\left(\mathrm{x}_{A}\right)=y\left(\mathrm{x}_{B}\right)=0 \\
w^{T}\left(\mathrm{x}_{A}-\mathrm{x}_{B}\right)=0
\end{gathered}
$$

- $w$ is orthogonal to every vector lying within the decision surface/boundary.


## Geometry of Linear Discriminant Functions

- For every point on the decision surface, $y(\mathrm{x})=w^{T} \mathrm{x}+w_{0}=0$.
- The normal distance of the surface to the origin is:

$$
\frac{w^{T} \mathrm{X}}{\|w\|}=-\frac{w_{0}}{\|w\|}
$$

## $w_{0}$ gives the location of the surface

If our model does not have an intercept term, the decision boundary must pass through the origin.

## Discriminative Probabilistic Models

- Classification Task
- Defining a New Model for Classification
- Linear Discriminant Functions
- Discriminative Probabilistic Models
- Sigmoid for Classification
- Logistic Regression Objective
- Regularization with Logistic Regression
- Logistic Regression Optimization
- Making Decisions


## Discriminative Probabilistic Models

'_ogistic regression is the most widely used discriminative probabilistic model for classification.

Logistic regression models the log-odds of class $C_{1}$ versus $C_{2}$ as a linear model.

$$
\begin{aligned}
& \text { Odds Ratio }=\frac{p\left(C_{1} \mid \mathrm{x}\right)}{p\left(C_{2} \mid \mathrm{x}\right)}=\frac{p\left(C_{1} \mid \mathrm{x}\right)}{1-p\left(C_{1} \mid \mathrm{x}\right)} \\
& \text { Log Odds }=\ln \left(\frac{p\left(C_{1} \mid \mathrm{x}\right)}{p\left(C_{2} \mid \mathrm{x}\right)}\right)=\ln \left(\frac{p\left(C_{1} \mid \mathrm{x}\right)}{1-p\left(C_{1} \mid \mathrm{x}\right)}\right)=\mathrm{w}^{\top} \mathrm{x}
\end{aligned}
$$

We can solve for $p\left(C_{1} \mid \mathrm{x}\right)$ to obtain the form of our model.

## Deriving the Logistic Sigmoid Function

We can solve for $p\left(C_{1} \mid \mathrm{x}\right)$ to obtain the form of our model.

$$
\text { Log Odds }=\ln \left(\frac{p\left(C_{1} \mid \mathrm{X}\right)}{p\left(C_{2} \mid \mathrm{X}\right)}\right)=\ln \left(\frac{p\left(C_{1} \mid \mathrm{X}\right)}{1-p\left(C_{1} \mid \mathrm{X}\right)}\right)=\mathrm{w}^{\top} \mathrm{X}
$$

$$
\begin{aligned}
& \frac{p\left(C_{1} \mid \mathrm{x}\right)}{1-p\left(C_{1} \mid \mathrm{x}\right)}=\exp \left(\mathrm{w}^{\top} \mathrm{x}\right) \\
& p\left(C_{1} \mid \mathrm{x}\right)=\exp \left(\mathrm{w}^{\top} \mathrm{x}\right)\left(1-p\left(C_{1} \mid \mathrm{x}\right)\right) \\
& p\left(C_{1} \mid \mathrm{x}\right)=\exp \left(\mathrm{w}^{\top} \mathrm{x}\right)-\exp \left(\mathrm{w}^{\top} \mathrm{x}\right) p\left(C_{1} \mid \mathrm{x}\right) \\
& p\left(C_{1} \mid \mathrm{x}\right)+\exp \left(\mathrm{w}^{\top} \mathrm{x}\right) p\left(C_{1} \mid \mathrm{x}\right)=\exp \left(\mathrm{w}^{\top} \mathrm{x}\right) \\
& p\left(C_{1} \mid \mathrm{x}\right)=\frac{\exp \left(\mathrm{w}^{\top} \mathrm{x}\right)}{1+\exp \left(\mathrm{w}^{\top} \mathrm{x}\right)}=\frac{1}{1+\exp \left(-\mathrm{w}^{\top} \mathrm{x}\right)}
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-29.jpg?height=537&width=890&top_left_y=1309&top_left_x=2321)

**Image Description:** The image is a diagram representing a sigmoid function. The x-axis ranges from -5 to 5, and the y-axis measures output values from 0 to 1. The curve starts near 0, transitions smoothly through the inflection point at the origin (0, 0.5), and asymptotically approaches 1 as x increases. The graph is characterized by an S-shaped curve, indicating the function's gradual increase before stabilizing.


## The Sigmoid ( $\sigma$ ) Function

Also known as the logistic function.
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-30.jpg?height=652&width=1099&top_left_y=612&top_left_x=1041)

**Image Description:** The image is a graph depicting a sigmoid function. The x-axis ranges from -5 to 5, while the y-axis ranges from 0 to 1. The curve starts near 0, steeply increases around the origin (x=0), and asymptotically approaches 1 as x increases. The curve is smooth and S-shaped, indicating a characteristic growth pattern often used in statistics and neural networks. The graph is plotted in a thick red line against a white background with labeled axes.


$$
\sigma(a)=\frac{1}{1+e^{-a}}
$$
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-30.jpg?height=294&width=831&top_left_y=1581&top_left_x=1445)

**Image Description:** The image depicts the sigmoid function, commonly used in statistics and machine learning. It features a smooth S-shaped curve, representing a mathematical model where the x-axis represents the input values and the y-axis represents the output values ranging between 0 and 1. The curve asymptotically approaches 0 as x decreases and approaches 1 as x increases. The function is characterized by its midpoint at (0, 0.5). The label "sigmoid function" is presented below the diagram, indicating the subject of the image.


## The Sigmoid ( $\sigma$ ) Function

Also known as the logistic function.
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-31.jpg?height=652&width=1090&top_left_y=612&top_left_x=365)

**Image Description:** The image is a graph depicting a sigmoid function. The x-axis ranges from -5 to 5, while the y-axis spans from 0 to 1. The curve starts near 0 for negative x-values, gradually increasing in the vicinity of x = 0, and approaches 1 for positive x-values. The graph is characterized by a smooth, S-shaped curve, demonstrating the transition of values in a nonlinear fashion. It effectively illustrates the behavior of the sigmoid function in various mathematical and applied contexts.


$$
\sigma(a)=\frac{1}{1+e^{-a}}
$$

The term Sigmoid means Sshaped.

A 'squashing function' since it maps the real axis into a finite interval.

## Transforming the Sigmoid ( $\sigma$ ) Function

We can shift and scale the sigmoid function just like any other function:
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-32.jpg?height=656&width=1089&top_left_y=625&top_left_x=1046)

**Image Description:** The image is a graph displaying a sigmoid function. The x-axis ranges from -5 to 5, while the y-axis extends from 0 to 1. The curve starts near 0 for negative x values, gradually increases around x = 0, and approaches 1 for positive x values. The graph is colored in red, highlighting the S-shaped curve characteristic of sigmoid functions, which are commonly used in statistics and machine learning to represent probabilities.


$$
\sigma(A a+B)=\frac{1}{1+e^{-(A a+B)}}
$$

Bigger A $\rightarrow$ Horizontal shrink.
Steeper!

## The Sigmoid ( $\sigma$ ) Function

The S-shaped curve is formally known as the sigmoid function

$$
\sigma(a)=\frac{1}{1+e^{-a}}
$$
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-33.jpg?height=579&width=1026&top_left_y=548&top_left_x=1892)

**Image Description:** The image is a graph representing a sigmoid function, often used in statistics and machine learning. The x-axis is labeled \( a \) and spans from -5 to 5, while the y-axis is labeled \( \sigma(a) \) and ranges from 0 to 1. The curve is S-shaped, starting near 0 for negative values of \( a \), steeply increasing around \( a = 0 \), and approaching 1 for positive values of \( a \). The line is depicted in a thick, red style, indicating its significance in the context of probability or transformations.

$\begin{array}{ll}\text { Reflectio } & 1-\sigma(a)=\frac{e^{-a}}{1+e^{-a}}=\sigma(-a) \\ \mathrm{n} / & \end{array}$
Domain
$-\infty<a<+\infty$
Range
$0<\sigma(a)<1$
Symmetr
$\mathrm{y}_{\text {Inverse }}$
$a=\sigma^{-1}(p)=\log \left(\frac{p}{1-p}\right)$
Derivative
$\frac{d}{d a} \sigma(a)=\sigma(a)(1-\sigma(a))=\sigma(a) \sigma(-a)$
-
-
0
Which of the following are properties of the sigmoid function, $\sigma(x)$ ?

## Sigmoid for Classification

- Classification Task
- Defining a New Model for Classification
- Linear Discriminant Functions
- Discriminative Probabilistic Models
- Sigmoid for Classification
- Logistic Regression Objective
- Regularization with Logistic Regression
- Logistic Regression Optimization
- Making Decisions


## Logistic Regression

Widely used models for binary classification:
$x=$ "Get a FREE sample ..."

$$
\phi(\mathrm{x})=[2.0,0, \ldots, 1.0,0.5] \quad \Rightarrow \quad y=1 \quad \begin{aligned}
& 1=\text { "Spam" } \\
& 0=\text { "Ham" }
\end{aligned}
$$

- Models the probability $\mathrm{p}\left(\mathrm{C}_{1} \mid \phi(\mathrm{x})\right)$.

Why is ham good and spam bad? ...
(https://www.youtube.com/ watch?v=anwy2MPT5RE )

$$
P\left(C_{1} \mid \phi(\mathrm{x})\right)=\sigma\left(w^{T} \phi(\mathrm{x})\right)=\frac{1}{1+\exp \left(-w^{T} \phi(\mathrm{x})\right)}
$$

## Logistic Regression

## Linear Model

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-37.jpg?height=1383&width=3139&top_left_y=484&top_left_x=0)

**Image Description:** The slide features a diagram illustrating a generalized linear model. It includes a logistic function curve showcasing the relationship between the input \( \phi(x) \) and the output probability \( P(C_1 | \phi(x)) \). The x-axis represents the linear transformation \( w^T \phi(x) \), while the y-axis denotes the probability ranging from 0 to 1. Accompanying this is the equation for the model, presented as $$ P(C_1 | \phi(x)) = \sigma(w^T \phi(x)) = \frac{1}{1 + \exp(-w^T \phi(x))} $$


## Logistic Regression

- Widely used models for binary classification:
$x=$ "Get a FREE sample ..."

$$
\phi(\mathrm{x})=[2.0,0, \ldots, 1.0,0.5] \quad \Rightarrow \quad y=1 \quad \begin{aligned}
& 1=\text { "Spam" } \\
& 0=\text { "Ham" }
\end{aligned}
$$

- Models the probability $\mathrm{p}\left(\mathrm{C}_{1} \mid \phi(\mathrm{x})\right)$.

Why is ham good and spam bad? ...
(https://www.youtube.com/ watch?v=anwy2MPT5RE )

$$
\begin{aligned}
& P\left(C_{1} \mid \phi(\mathrm{x})\right)=\sigma\left(w^{T} \phi(\mathrm{x})\right)=\frac{1}{1+\exp \left(-w^{T} \phi(\mathrm{x})\right)} \\
& P\left(C_{2} \mid \phi(\mathrm{x})\right)=1-P\left(C_{1} \mid \phi(\mathrm{x})\right)
\end{aligned}
$$

## Our Cancer Classification Example

We've discovered the (simple) logistic regression model! It just fits a sigmoid to the data.

Logistic Regression Fitted to the Dataset
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-39.jpg?height=567&width=1031&top_left_y=782&top_left_x=1160)

**Image Description:** The image is a scatter plot illustrating the relationship between "Mean Radius" (x-axis) and the probability of a tumor being malignant (y-axis). Data points are colored: blue indicates benign tumors, while red indicates malignant tumors. The x-axis ranges from 0 to 25, and the y-axis represents probability values from 0 to 1. A logistic regression curve, plotted as a black line, models the probability of malignancy based on mean radius, showing an S-shaped curve transitioning from benign to malignant classifications.


$$
p\left(C_{1} \mid \phi(\mathrm{x})\right)=\sigma\left(w_{0}+w_{1} \mathrm{x}\right)=\frac{1}{1+e^{-\left(w_{0}+w_{1} \mathrm{x}\right)}}
$$

How do we identify the optimal $w$ 's? More shortly!

## Logistic Regression and Decision Boundary警等

1965715

Logistic Regression Fitted to the Dataset
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-40.jpg?height=771&width=1264&top_left_y=705&top_left_x=289)

**Image Description:** The diagram is a scatter plot illustrating the probability of a tumor being malignant as a function of the mean radius. The x-axis represents the "Mean Radius," while the y-axis indicates "Prob(Malignant = 1 | x)." Data points are displayed in blue for benign tumors and red for malignant tumors. A black curve, representing logistic regression, shows the sigmoid function approximating the probabilities, indicating a transition from lower probabilities for benign tumors to higher probabilities as mean radius increases. The plot effectively visualizes the relationship between tumor size and malignancy.


![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-40.jpg?height=762&width=987&top_left_y=272&top_left_x=1807)

**Image Description:** The diagram presents a logistic regression decision boundary plotted with jittered points. The x-axis represents "Mean Radius," while the y-axis indicates "Prob(Malignant = 1 | x)." Data points are color-coded: red for malignant (above the decision boundary) and blue for benign (below). Circular markers depict the training data, with a solid black curve illustrating the probability of malignancy, a dashed vertical line indicating a threshold at 15, and shaded regions for classified outcomes (malignant and benign). The graph effectively visualizes the model’s classification efficacy in a binary outcome context.

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-40.jpg?height=767&width=971&top_left_y=1058&top_left_x=1811)

**Image Description:** The image is a diagram depicting a logistic regression decision boundary. The x-axis represents "Mean Radius," while the y-axis denotes "Probability of Malignant." It features a red curve indicating the probability curve, with shaded areas for classified data points: red for malignant and blue for benign. Jittered data points are scattered, with malignant points marked with red "X" and benign points with blue "X." A dashed horizontal line at a probability of 0.5 indicates the threshold for classification, and a vertical dashed line illustrates a specific mean radius.


## Basis Functions and Decision Boundary 1965715

- We can still apply basis functions $\phi(\mathrm{x})$ to the classification inputs. The resulting decision boundaries will be linear in the feature space $\phi$, and these correspond to nonlinear decision boundaries in the original x .

Non-linear decision boundary in input space.

Linear decision boundary in the Gaussian space.

Centers of the Gaussian basis functions
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-41.jpg?height=482&width=409&top_left_y=1105&top_left_x=820)

**Image Description:** The image depicts a curved arrow diagram, illustrating a potential energy curve in a physics or chemistry context. The x-axis represents the reaction coordinate or distance, while the y-axis denotes potential energy. The curve initially rises sharply, indicating an increase in energy, followed by a level section, and then a gradual descent, showing energy release. The arrows indicate the direction of a reaction or process, suggesting an initial input of energy followed by a favorable transition to a lower energy state. This visual representation is relevant in discussions of reaction mechanisms and thermodynamic stability.


## Logistic Regression Objective

- Classification Task
- Defining a New Model for Classification
- Linear Discriminant Functions
- Discriminative Probabilistic Models
- Sigmoid for Classification
- Logistic Regression Objective
- Regularization with Logistic Regression
- Logistic Regression Optimization
- Making Decisions


## Can We Use Squared Error As Objective?

## Toy Dataset: Squared Error

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-44.jpg?height=600&width=1128&top_left_y=51&top_left_x=1292)

**Image Description:** The slide features a logistic regression model with the equation:

$$
p(C_1 | \phi(x)) = \sigma(w^T \phi(x))
$$

Accompanying the equation is a table with two columns labeled "x" and "y." The "x" values range from -4.0 to 5.0, including -2.0, -0.5, 1.0, 3.0, and 5.0. The corresponding "y" values indicate binary outcomes, with a '1' appearing for values of "x" at -2.0, 1.0, 3.0, and 5.0, and a '0' for -4.0 and -0.5. The slide notes the assumption of no intercept.


So x and $w$ are scalars.

## Squared Error:

$E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-\sigma\left(w^{T} \phi(\mathrm{x})\right)\right)^{2}$
The squared error surface for logistic regression has many issues!

Error on Toy Classification Data
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-44.jpg?height=585&width=933&top_left_y=1149&top_left_x=2313)

**Image Description:** The image is a graph showing the relationship between the variable \( w \) (x-axis) and the corresponding "Error" (y-axis). The x-axis ranges approximately from -10 to 10, while the y-axis displays values of Error, starting from around 2 and decreasing sharply to approximately 0.4 as \( w \) approaches 0, then stabilizing at values close to 0.8 for larger \( w \). The curve is smooth, indicating a continuous function, primarily displaying a sharp drop in error at \( w = 0 \). The line is colored blue.


## Pitfalls of Squared Error

1. Non-convex. Gets stuck in local minima.

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-\sigma\left(w^{T} \phi(\mathrm{x})\right)\right)^{2}
$$

Secant line crosses function, ${ }_{\text {Error on Toy Classfication Dota }}$ so $E^{\prime \prime}(w)$ is not greater than 0 for all $w$.
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-45.jpg?height=554&width=920&top_left_y=518&top_left_x=2270)

**Image Description:** The image is a graph depicting the relationship between the variable \( w \) on the x-axis (ranging from -10 to 10) and the corresponding "Error" values on the y-axis (with a visible range around 0 to 2). It features a blue curve illustrating the error's variation as \( w \) changes, with two notable red points indicating specific error values at certain \( w \) positions. This graph likely represents a loss function in optimization contexts, showing how error decreases and plateaus as \( w \) approaches certain values.


## Pitfalls of Squared Error

1. Non-convex. Gets stuck in local minima.

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-\sigma\left(w^{T} \phi(\mathrm{x})\right)\right)^{2}
$$

Secant line crosses function, so $E^{\prime \prime}(w)$ is not greater than 0 for all $w$.

Error on Toy Classification Data
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-46.jpg?height=562&width=920&top_left_y=514&top_left_x=2270)

**Image Description:** The image is a graph depicting the relationship between variable \( w \) and the corresponding error value. The x-axis ranges from -10 to 10, labeled as \( w \), while the y-axis represents error values ranging approximately from 0 to 2. The graph features a curve that sharply decreases around \( w = 0 \), indicating a minimum error, with points marked at \( w = -5 \) (purple) and \( w = 5 \) (green). The behavior suggests a quadratic or similar function, emphasizing the sensitivity of error to changes in \( w \).


from scipy.optimize import minimize
minimize(ss_error_on_toy_data, $x 0=\underline{\mathbf{0}}$ )[" $x$ "] $[0]$
minimize(ss_error_on_toy_data, $x 0=-5)[$ " $x$ " $][0]$
-10.858380927026204

## Pitfalls of Squared Error

1. Non-convex. Gets stuck in local minima.

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-\sigma\left(w^{T} \phi(\mathrm{x})\right)\right)^{2}
$$

Secant line crosses function, so
Error on Toy Classification Data $E^{\prime \prime}(w)$ is not greater than 0 for all $w$.
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-47.jpg?height=669&width=924&top_left_y=1126&top_left_x=1241)

**Image Description:** The image is a graphical representation of a logistic regression model. It features a 2D scatter plot with the x-axis labeled as \( x \) and the y-axis labeled as \( y \). Two types of data points are plotted: blue circles represent a binary outcome of 0, while red circles indicate a binary outcome of 1. A black curve illustrates the logistic regression model, with the equation for the model indicated as \( w = 0.54 \). The curve shows a steep transition between the two classes as \( x \) approaches 0.

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-47.jpg?height=660&width=916&top_left_y=1135&top_left_x=2236)

**Image Description:** The image is a graphical representation of a logistic regression model. It features a scatter plot with the x-axis labeled as "x" and the y-axis labeled as "str_y," indicating a dependent variable. Red circles represent data points with a value of 1, while blue circles represent data points with a value of 0. A black curve illustrates the logistic regression model, showing the relationship between the independent variable x and the predicted probability on the y-axis, with a weight (w) of 0.54. The curve indicates a sigmoid shape, typical of logistic functions.


## Squared Error Gives Bounded Penalty

- Best case scenario:

Estimate matches outcome.
Squared error $=(0-0)^{2}=(1-1)^{2}=0$. Perfect prediction!

- Worst case scenarios:

Estimated probability $=0$, Actual outcome $=1$.
Squared error $=(0-1)^{2}=1$
Estimated probability $=1$, Actual outcome $=0$.
Squared error $=(1-0)^{2}=1$

- The largest possible squared error is 1 . Not a huge penalty for a big mistake!


## Pitfalls of Squared Error

1. Non-convex. Gets stuck in local minima.
2. Bounded. Not a good measure of model error.

We'd like error functions to penalize "off" predictions. SE never gets very large, because both response and estimated probability are bounded

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-\sigma\left(w^{T} \phi(\mathrm{x})\right)\right)^{2}
$$

Squared Loss for One Individual when $y=1$
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-49.jpg?height=652&width=1060&top_left_y=1071&top_left_x=1309)

**Image Description:** The image is a graph depicting a curve that represents the relationship between the variable \( p \) on the x-axis and the "Squared Error" on the y-axis. The x-axis ranges from 0 to 1, while the y-axis ranges from 0 to 0.5. The curve is smooth and decreases monotonically, starting near 0.5 at \( p = 0 \) and approaching 0 as \( p \) approaches 1. The line has a blue hue and does not display any grid lines or data points. This representation likely illustrates the performance of a model with respect to the probability parameter \( p \).


If true $t=1$ but predicted nrohahilitı- $n$.
$(t-p)^{2}=(1-0)^{2}=1$

A New Error Function for Logistic Regression

## The Logistic Regression Objective (MLENTBASIN

In logistic regression, we can adopt maximum likelihood estimation.

- Define target variable $t \in\{0,1\}$, with $t=1$ representing class $C_{1}$ and $t=0$ for class $C_{2}$.
- The likelihood of $t$ given x under our model $p\left(C_{1} \mid \mathrm{x}\right)=\sigma\left(w^{\top} \mathrm{x}\right)$ :

$$
p(t \mid w)=\underbrace{\sigma\left(w^{\top} \mathbf{x}\right)^{t}}_{\substack{\text { for } t_{n}=\mathbf{1}, \text { only } \\ \text { this term stays }}} \underbrace{\left[1-\sigma\left(w^{\top} \mathbf{x}\right)\right]^{1-t}}_{\substack{\text { for } t_{n}=\mathbf{0}, \text { only } \\ \text { this term stays }}}
$$

## The Logistic Regression Objective (MLE)

In logistic regression, we can adopt maximum likelihood estimation.

- Define target variable $t \in\{0,1\}$, with $t=1$ representing class $C_{1}$ and $t=0$ for class $C_{2}$.
- The likelihood of $t$ given x under our model $p\left(C_{1} \mid \mathrm{x}\right)=\sigma\left(w^{\top} \mathrm{x}\right)$ :

$$
p(t \mid w)=\sigma\left(w^{\top} \mathrm{x}\right)^{t}\left[1-\sigma\left(w^{\top} \mathrm{x}\right)\right]^{1-t}
$$
- For a dataset $\mathcal{D}=\left\{\mathrm{x}_{n}, t_{n}\right\}$, where $y_{n}=\sigma\left(w^{\top} \mathrm{x}\right)$ the likelihood function can be written as:
$$
p(\mathcal{D} \mid w)=\prod_{n=1}^{N} y_{n}^{t_{n}}\left(1-y_{n}\right)^{1-t_{n}}
$$

## The Logistic Regression Objective (MLE춨ss is

$$
p(\mathcal{D} \mid w)=\prod_{n=1}^{N} y_{n}^{t_{n}}\left(1-y_{n}\right)^{1-t_{n}}
$$

We can define an error function by taking the negative logarithm of the likelihood, which gives the cross-entropy error function.

$$
E(w)=-\ln p(\mathcal{D} \mid w)=-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right]
$$

## Cross-Entropy: Two Error Functions In One!

The Cross-Entropy (CE) function is often written more compactly as the product of the label and the log probability, summed over both
CE error $= \begin{cases}-\ln y_{n}, & \text { if } t_{n}=1 \Longleftarrow-[\underbrace{t_{n} \ln y_{n}}_{\text {for } t_{n}=1, \text { only this term stays }}+\underbrace{\left.1-t_{n}\right) \ln \left(1-y_{n}\right)}_{\text {for } t_{n}=0, \text { only this term stays }}]\end{cases}$
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-54.jpg?height=763&width=1056&top_left_y=990&top_left_x=327)

**Image Description:** The image is a comparative graph illustrating error metrics for a single observation with \( y = 1 \). The x-axis represents the probability \( p \), ranging from 0 to 1. The y-axis denotes the error values. Two curves are plotted: the blue line represents the Squared Error, which decreases steadily, while the red line depicts the Negative Log Error, which approaches infinity as \( p \) approaches 0. The graph effectively visualizes the differences in error behavior for the two metrics under fixed conditions.

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-54.jpg?height=767&width=1064&top_left_y=990&top_left_x=1854)

**Image Description:** The image is a diagram comparing two types of error: Squared Error (shown in blue) and Negative Log Error (shown in red) as functions of a variable \( p \) ranging from 0 to 1. The x-axis represents the variable \( p \), while the y-axis depicts the error values. The Squared Error curve increases gradually, while the Negative Log Error curve rises sharply, indicating larger errors as \( p \) approaches 1. The legend identifies the error types.


Unbounded for wrong-way certainty
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-55.jpg?height=452&width=541&top_left_y=561&top_left_x=178)

**Image Description:** This image is a stylized representation of clouds, featuring two distinct cloud shapes. The larger cloud is situated at the top, with a rounded, puffed appearance and a light purple hue. Below it, a smaller cloud mirrors the style, also in light purple. The design emphasizes a simple, illustrative aesthetic, suitable for visual communication rather than detailed scientific representation. It is likely used in a context related to weather, climate, or metaphorical representation of concepts.


What w value will achieve 0 cross-entropy error if we train on the datapoint $x=1, t=1$ ?

## Convexity By Picture

$$
w=\underset{w}{\operatorname{argmin}}-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right]
$$
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-56.jpg?height=1077&width=779&top_left_y=714&top_left_x=102)

**Image Description:** The image is a scatter plot displaying points in a two-dimensional Cartesian coordinate system. The x-axis ranges from -5 to 5, while the y-axis ranges from 0 to 1. Points are color-coded: red for the category "1" of the variable "str_y" and blue for category "0." The corresponding data table below lists five observations with values for both x and y, showing pairs: (-4, 0), (-2, 0), (-0.5, 1), (3, 1), and (5, 1).


## Squared Error Surface

Error on Toy Classification Data
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-56.jpg?height=566&width=950&top_left_y=1046&top_left_x=1139)

**Image Description:** The image is a plot depicting the relationship between a variable \( w \) on the x-axis and an "Error" metric on the y-axis. The graph features a blue line that demonstrates a non-linear response, exhibiting a sharp drop in error values around \( w = -5 \). Two red points highlight key values on the curve, with one at approximately \( w = -5 \) corresponding to a local minimum of the error, while the other point indicates a higher error value at a different \( w \).


## A straight line crosses the curve Non-convex

## Cross-Entropy Error Surface

Cross-Entropy on Toy Classification Data
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-56.jpg?height=490&width=839&top_left_y=1063&top_left_x=2253)

**Image Description:** The image is a graph depicting two curves representing cross-entropy loss against an unspecified variable, likely a model parameter or iteration count. The x-axis is labeled with values ranging from approximately -10 to 4, while the y-axis indicates cross-entropy values ranging from 0 to 20. The blue curve shows a decreasing trend, indicating a reduction in loss, while the red curve appears to be a linear approximation that more sharply decreases, intersecting key points at the left and right extrema. Relevant points on the curves are marked with solid red dots, emphasizing specific loss values.


Convex!

## Regularization with Logistic Regression

- Classification Task
- Defining a New Model for Classification
- Linear Discriminant Functions
- Discriminative Probabilistic Models
- Sigmoid for Classification
- Logistic Regression Objective
- Regularization with Logistic Regression
- Logistic Regression Optimization
- Making Decisions


## What Is the Value of $w$ ?

A. $w=-1$
$w=\underset{w}{\operatorname{argmin}}-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right]$
B. $w=1$

Assume $\phi(\mathrm{x})=\mathrm{x}$
C. $w=-\infty$

The Data
D. $w=+\infty$
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-58.jpg?height=948&width=2051&top_left_y=927&top_left_x=123)

**Image Description:** The image is a two-dimensional Cartesian coordinate system. The x-axis ranges from -1 to 1, and the y-axis ranges from -1 to 1, with both axes labeled. Two points are marked: one at (-1, 1) and the other at (1, 0). The point (-1, 1) is situated in the second quadrant, indicated by a blue dot, while the point (1, 0) lies on the positive x-axis in the first quadrant, also marked with a blue dot. The grid is clean and focused on these points and their locations within the coordinate system.

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-58.jpg?height=1045&width=961&top_left_y=823&top_left_x=2247)

**Image Description:** The image contains two graphs arranged vertically. The top graph depicts a sigmoid function, characterized by an S-shaped curve that asymptotes at y = 0 and y = 1, with x-axis ranging approximately from -4 to 4. The y-axis shows values between 0 and 1. The bottom graph presents a complementary sigmoid shape, resembling an inverted S-curve, transitioning from 1 to 0. The x-axis maintains the same range, while the y-axis similarly represents values from 0 to 1. Both graphs illustrate the relationship between the input (x) and output (y) values of the sigmoid function and its complement.


## What is the best value of W ?

## What Is the Value of $w$ ?

A. $w=-1$
$w=\underset{w}{\operatorname{argmin}}-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right]$
B. $w=1$

Assume $\phi(\mathrm{x})=\mathrm{x}$
C. $w=-\infty$

The Data
D. $w=+\infty$
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-60.jpg?height=948&width=2051&top_left_y=927&top_left_x=123)

**Image Description:** The image is a two-dimensional Cartesian coordinate plane. The horizontal axis (x-axis) ranges from -1 to 1, while the vertical axis (y-axis) ranges from -1 to 1. Two points are plotted: one at (-1, 1) represented by a blue dot in the upper left quadrant, and another at (1, 0) indicated by a blue dot on the horizontal axis to the right. The diagram illustrates the representation of specific coordinate values within the Cartesian system.

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-60.jpg?height=1045&width=961&top_left_y=823&top_left_x=2247)

**Image Description:** The image contains two separate line graphs arranged vertically. 

1. **Top Graph**: Depicts a sigmoid-like curve, starting near 0 for negative x-values and asymptotically approaching 1 as x increases. The x-axis ranges from approximately -4 to 4, with a vertical line at x = 0.
2. **Bottom Graph**: Displays a corresponding inverse relationship, starting near 1 for negative x-values and approaching 0 as x increases. 

Both graphs illustrate smooth transitions, likely representing a logistic function or similar mathematical model. The y-axis is normalized between 0 and 1.


## What Is the Value of $w$ ?

$$
w=\underset{w}{\operatorname{argmin}}-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right]
$$

For the point (-1,1):
Objective:

$$
-\ln \sigma\left(w^{T} \mathrm{x}\right)=-\ln \sigma(-w) \quad \Rightarrow w \rightarrow-\infty
$$

$$
\begin{array}{cc|}
(-1,1) & \\
\bullet \text { The Data } & 1 \\
& \\
\hline-1 & 0
\end{array}
$$

## What Is the Value of $w$ ?

$$
w=\underset{w}{\operatorname{argmin}}-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right]
$$

For the point $(1,0)$ :
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-62.jpg?height=953&width=1999&top_left_y=922&top_left_x=0)

**Image Description:** The image contains an equation and a diagram. The equation displayed is $$ -\ln(1 - \sigma(w^T x)) = -\ln(1 - \sigma(w)) \Rightarrow w \rightarrow -\infty $$, where $\sigma$ denotes the sigmoid function. Below the equation, there is a diagram illustrating a point on a coordinate plane: the x-axis ranges from 0 to 1, while the y-axis is marked from 0 to 1, and a blue dot is positioned at coordinates (1, 0). The diagram appears to visualize the behavior or output of the function as $w$ changes.


## What Is the Value of $w$ ?

For the point ( $-1,1$ ):

$$
-\ln \sigma\left(w^{T} \mathrm{x}\right) \quad \Rightarrow w \rightarrow-\infty
$$

For the point $(1,0)$ :

$$
-\ln \left(1-\sigma\left(w^{T} \mathrm{x}\right)\right) \quad \Rightarrow w \rightarrow-\infty
$$
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-63.jpg?height=715&width=1282&top_left_y=1136&top_left_x=2015)

**Image Description:** The image presents a two-dimensional Cartesian coordinate system. The x-axis ranges from -1 to 1 and the y-axis ranges from -1 to 1. Two points are plotted: (-1, 1) and (1, 0), represented by blue dots. A horizontal line connects these points, labeled "The Data." Above the diagram, the text "Overly confident!" is displayed in a yellow box, suggesting a thematic focus on the confidence levels related to the data points. The diagram effectively illustrates the relationship between the coordinates and presents a visual interpretation of confidence.


## Adding Regularization to Logistic Regression

$$
w=\underset{w}{\operatorname{argmin}}-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right]+\frac{\lambda}{2} \sum_{d=1}^{D} w^{2}
$$

Prevents weights from diverging on linearly separable data.
![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-64.jpg?height=694&width=1234&top_left_y=1063&top_left_x=293)

**Image Description:** The image is a graph depicting a function without regularization. The x-axis is labeled \( \theta \), ranging approximately from -4 to 4. The y-axis shows function values ranging from 0 to about 5. A blue curve illustrates the relationship between \( \theta \) and the function, starting at the origin (0,0) and increasing steeply as \( \theta \) approaches 4. An annotation on the left states "Earlier Example," indicating it may reference a previous concept discussed in the lecture. The graph employs a simplistic design with a clear line, enhancing interpretability.

![](https://cdn.mathpix.com/cropped/2025_10_01_c9a6cc269659d8d570a4g-64.jpg?height=698&width=1060&top_left_y=1063&top_left_x=1573)

**Image Description:** The image features a graph illustrating a cost function in the context of regularization. The x-axis represents the parameter \( \theta \), ranging from approximately -4 to 4, while the y-axis represents the cost value, ranging from 0 to 7. A blue curve depicts the cost function, which has a minimum point marked by a solid blue dot at \( \theta \approx -2 \) and a cost value of 1. The title indicates that the graph incorporates regularization with \( \lambda = 0.1 \). An arrow is directed from the left toward the graph.


## Logistic Regression Optimization

- Classification Task
- Defining a New Model for Classification
- Linear Discriminant Functions
- Discriminative Probabilistic Models
- Sigmoid for Classification
- Logistic Regression Objective
- Regularization with Logistic Regression
- Logistic Regression Optimization
- Making Decisions


## Logistic Regression Optimization

$$
\begin{gathered}
{\left[\begin{array}{c}
E(w)=-\ln p(\mathrm{t} \mid w)=-\sum_{n=1}^{N}[\underbrace{\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}} \underbrace{\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w}}+\underbrace{\left(1-t_{n}\right) \ln \left(1-y_{n}\right)}_{\downarrow} \\
\downarrow \\
\begin{array}{c}
\text { Derivative wrt } w \\
\text { using chain rule: }
\end{array}
\end{array}\right]} \\
\Longrightarrow \frac{\partial E(w)}{\partial w}=-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}+\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w}\right]
\end{gathered}
$$

## Logistic Regression Optimization

$$
\left[\begin{array}{rl}
E(w)=-\ln p(\mathrm{t} \mid w) & =-\sum_{n=1}^{N}[\underbrace{t_{n} \ln y_{n}}_{\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}}+\underbrace{\left(1-t_{n}\right) \ln \left(1-y_{n}\right)}_{\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w}}] \\
\begin{array}{c}
\text { Derivative wrt } w \\
\text { using chain rule: }
\end{array} & \\
\frac{\partial E(w)}{\partial w} & =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}+\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w}\right] \overbrace{\frac{\partial\left(1-y_{n}\right)}{\partial w}=-\frac{\partial y_{n}}{\partial w}} \\
& =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}-\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial y_{n}}{\partial w}\right]
\end{array}\right.
$$

$$
\begin{aligned}
E(w)=-\ln p(\mathrm{t} \mid w) & =-\sum_{n=1}^{N}[\underbrace{\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}}+\underbrace{\left(1-t_{n}\right) \ln \left(1-y_{n}\right)}_{\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w}}] \\
\begin{array}{l}
\text { Derivative wrt } w \\
\text { using chain rule: }
\end{array} & \frac{\partial E(w)}{\partial w} \\
& =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}+\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w}\right] \xrightarrow{\frac{\partial\left(1-y_{n}\right)}{\partial w}=-\frac{\partial y_{n}}{\partial w}} \\
& =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}-\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial y_{n}}{\partial w}\right] \\
& =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}}-\frac{\left(1-t_{n}\right)}{1-y_{n}}\right] \frac{\partial y_{n}}{\partial w}
\end{aligned}
$$

$$
\begin{aligned}
E(w)=-\ln p(\mathrm{t} \mid w) & =-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right] \\
\frac{\partial E(w)}{\partial w} & =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}+\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial\left(1-y_{n}\right)}{\partial w}\right] \\
& =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}} \frac{\partial y_{n}}{\partial w}-\frac{\left(1-t_{n}\right)}{1-y_{n}} \frac{\partial y_{n}}{\partial w}\right] \stackrel{\frac{\partial\left(1-y_{n}\right)}{\partial w}=-\frac{\partial y_{n}}{\partial w}}{\underset{n=1}{N}}\left[\frac{t_{n}}{y_{n}}-\frac{\left(1-t_{n}\right)}{1-y_{n}}\right] \frac{\partial y_{n}}{\partial w} \quad \begin{array}{c}
y_{n}=\sigma\left(w^{T} \phi\left(\mathrm{x}_{n}\right)\right) \\
\text { Need to apply chain rule }
\end{array} \\
& =-\sum_{n} \\
\frac{\partial E(w)}{\partial w} & =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}}-\frac{\left(1-t_{n}\right)}{1-y_{n}}\right] y_{n}\left(1-y_{n}\right) \phi\left(\mathrm{x}_{n}\right) \\
\frac{\partial y_{n}}{\partial w}=\frac{\partial \sigma\left(w^{T} \phi\left(\mathrm{x}_{n}\right)\right)}{\partial w} & =\sigma\left(w^{T} \phi\left(\mathrm{x}_{n}\right)\right)\left(1-\sigma\left(w^{T} \phi\left(\mathrm{x}_{n}\right)\right)\right) \phi\left(\mathrm{x}_{n}\right)=y_{n}\left(1-y_{n}\right) \phi\left(\mathrm{x}_{n}\right)
\end{aligned}
$$

$$
\begin{aligned}
E(w)=-\ln p(\mathrm{t} \mid w) & =-\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right] \\
\frac{\partial E(w)}{\partial w} & =-\sum_{n=1}^{N}\left[\frac{t_{n}}{y_{n}}-\frac{\left(1-t_{n}\right)}{1-y_{n}}\right] y_{n}\left(1-y_{n}\right) \phi\left(\mathrm{x}_{n}\right) \quad \text { Separating the terms } \\
& =-\sum_{n=1}^{N}\left[t_{n}\left(1-y_{n}\right) \phi\left(\mathrm{x}_{n}\right)-\left(1-t_{n}\right) y_{n} \phi\left(\mathrm{x}_{n}\right)\right] \\
& =-\sum_{n=1}^{N}\left[t_{n} \phi\left(\mathrm{x}_{n}\right)-t_{n} y_{n} \phi\left(\mathrm{x}_{n}\right)-y_{n} \phi\left(\mathrm{x}_{n}\right)+t_{n} y_{n} \phi\left(\mathrm{x}_{n}\right)\right] \\
& =-\sum_{n=1}^{N}\left[t_{n} \phi\left(\mathrm{x}_{n}\right)-y_{n} \phi\left(\mathrm{x}_{n}\right)\right]=\sum_{n=1}^{N}\left[y_{n}-t_{n}\right] \phi\left(\mathrm{x}_{n}\right)
\end{aligned}
$$

## We Cannot Solve Directly

$$
\begin{aligned}
E(w)=-\ln p(\mathrm{t} \mid w)= & -\sum_{n=1}^{N}\left[t_{n} \ln y_{n}+\left(1-t_{n}\right) \ln \left(1-y_{n}\right)\right] \\
\frac{\partial E(w)}{\partial w} & =\sum_{n=1}^{N}\left[y_{n}-t_{n}\right] \phi\left(\mathrm{x}_{n}\right) \\
& =\sum_{n=1}^{N}\left[\sigma\left(w^{T} \phi\left(\mathrm{x}_{n}\right)\right)-t_{n}\right] \phi\left(\mathrm{x}_{n}\right)
\end{aligned}
$$

No Closed Form Solution
$w$ is inside the $\sigma$ for every term so we cannot pull out $w$.

We cannot apply logit to inverse $\sigma$ : logit $(a+b) \neq \operatorname{logit}(a)+\operatorname{logit}(b)$

More on this in lectures 12 and 13 with Gradient Descent.

## Making Decisions

- Classification Task
- Defining a New Model for Classification
- Linear Discriminant Functions
- Discriminative Probabilistic Models
- Sigmoid for Classification
- Logistic Regression Objective
- Regularization with Logistic Regression
- Logistic Regression Optimization
- Making Decisions


## Decisions = Posteriors + Loss

- We separate inference from decision.

Inference gives posteriors $p\left(C_{k} \mid \mathrm{x}\right)$.
The task gives a loss matrix $L_{k j}$.
$\boldsymbol{L}_{k j}:$ Loss if we predict $j$
when the true class is $k$.

The right action is the one with minimum expected loss at this x .

$$
\begin{aligned}
& \text { Expected loss at x if } \\
& \text { we choose class } j
\end{aligned}=\sum_{k} L_{k j} p\left(C_{k} \mid \mathrm{x}\right)
$$

## Decisions = Posteriors + Loss

The right action is the one with minimum expected loss at this x .

$$
\underset{\text { we choose class } j}{\text { Expected loss at } \mathrm{x} \text { if }}=\sum_{k} L_{k j} p\left(C_{k} \mid \mathrm{x}\right)
$$

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


## Lecture 10

## Logistic Regression (1)

Credit: Joseph E. Gonzalez and Narges Norouzi
Reference Book Chapters: Chapter 5 Sections 5.1, 5.2(up to 5.2.5), 5.4 (up to 5.4.4)

