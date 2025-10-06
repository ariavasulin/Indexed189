---
course: CS 189
semester: Fall 2025
type: lecture
title: Linear Regression (2)
source_type: slides
source_file: Lecture 07 -- Linear Regression (2).pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Lecture 7

## Linear Regression (2)

Geometric interpretation of least squares and probabilistic view of linear regression

## EECS 189/289, Fall 2025 @ UC Berkeley

Joseph E. Gonzalez and Narges Norouzi

# III Join at slido.com \#6101346 

## Roadmap

- Error Minimization
- Geometric Interpretation
- Evaluation
- Regularized Least Squares
-When Normal Equation Gets Tricky


## Error Minimization

- Error Minimization
- Geometric Interpretation
- Evaluation
- Regularized Least Squares
-When Normal Equation Gets Tricky


## Optimization

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-05.jpg?height=1490&width=3024&top_left_y=331&top_left_x=187)

**Image Description:** The image presents a flowchart representing a machine learning process. It is divided into three sections: "Learning Problem (L)," "Model Design (M)," and "Optimization (N)." Arrows connect these components, illustrating the workflow. The "Learning Problem" section describes supervised learning of scalar values. The "Model Design" highlights linear regression using basis functions, $y(\mathbf{x}) = \mathbf{x}^T \mathbf{w}$. The "Optimization" section features an equation: $$E(w) = \frac{1}{N} \sum_{n=1}^{N} (y_n - y(\mathbf{x}_n, \mathbf{w}))^2$$ and indicates deriving a direct solution to minimize error.


## Error Function Minimization

- $E(w)$ is a quadratic function.

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}
$$

Therefore, $\frac{\partial E(w)}{\partial \mathrm{w}}$ is a linear function of $w$ and hence solving $\frac{\partial E(w)}{\partial \mathrm{w}}=0$ has one and only one solution. Let's now derive this solution:

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-w_{0}-\sum_{j=1}^{D} w_{j} x_{n j}\right)_{\substack{\text { Finding the } \\ \text { optimum solution }}}^{2} \frac{\partial E(w)}{\partial \mathrm{w}}=0
$$

## Error Function Minimization

$$
\left[\begin{array}{l}
\frac{\partial E(w)}{\partial \mathrm{w}_{0}}=0 \longrightarrow \frac{\partial\left(\frac{1}{2} \sum_{n=1}^{N}\left(\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)\right)^{2}\right.}{\partial \mathrm{w}_{0}}=0 \\
\quad \text { Sum rule } \\
\quad \text { a } \sum f=\sum \partial f
\end{array}\right.
$$

Chain rule
$\partial f^{2}=2 f \partial f$

Separating
the terms

$$
\frac{\partial E(w)}{\partial \mathrm{w}_{0}}=0 \longrightarrow
$$

## Error Function Minimization

$$
\left[\begin{array}{rl}
\frac{\partial E(w)}{\partial \mathrm{w}_{0}}=0 & \longrightarrow \frac{\partial\left(\frac{1}{2} \sum_{n=1}^{N}\left(\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)\right)^{2}\right.}{\partial \mathrm{w}_{0}}=0 \\
\begin{array}{l}
\text { Sum rule } \\
\partial \sum^{f} f=\sum \partial f
\end{array} & =\frac{\frac{1}{2} \sum_{n=1}^{N} \partial\left(\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)^{2}\right)}{\partial w_{0}} \\
\begin{array}{l}
\text { Chain rule } \\
\partial f^{2}=2 f \partial f
\end{array} & =\frac{1}{2} \sum_{n=1}^{N} 2\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)\left(\frac{\partial\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)}{\partial w_{0}}\right) \\
\begin{array}{l}
\text { Separating } \\
\text { the terms }
\end{array} & =\sum_{n=1}^{N}\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)(-1)=-\left(\sum_{n=1}^{N} t_{n}-n w_{0}-\sum_{n=1}^{N}\left(\sum_{d=1}^{D} w_{d} x_{n d}\right)\right) \\
\frac{\partial E(w)}{\partial \mathrm{w}_{0}}=0 \longrightarrow w_{0}=\frac{1}{n}\left(\sum_{n=1}^{N} t_{n}-\sum_{n=1}^{N}\left(\sum_{d=1}^{D} w_{d} x_{n d}\right)\right)
\end{array}\right.
$$

## Error Function Minimization

$$
\begin{aligned}
\frac{\partial E(w)}{\partial \mathrm{w}_{\mathrm{k}}}=0 \longrightarrow & \frac{\partial\left(\frac{1}{2} \sum_{n=1}^{N}\left(\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)\right)^{2}\right.}{\partial \mathrm{w}_{\mathrm{k}}}=0 \\
\begin{aligned}
\text { Sum rule } & \\
\partial \sum^{2} f=\sum \partial f & =
\end{aligned} & \frac{1}{2} \sum_{n=1}^{N} \partial\left(\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)^{2}\right) \\
\begin{array}{l}
\text { Chain rule } \\
\partial f^{2}=2 f \partial f
\end{array} & =\frac{1}{2} \sum_{n=1}^{N} 2\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)\left(\frac{\partial\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)}{\partial w_{k}}\right) \\
\text { Reordering } & =\sum_{n=1}^{N}\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)\left(-x_{n k}\right) \\
\frac{\partial E(w)}{\partial \mathrm{w}_{\mathrm{k}}} & =-\sum_{n=1}^{N} x_{n k}\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right)
\end{aligned}
$$

## Error Function Minimization

$$
\begin{aligned}
& \frac{\partial E(w)}{\partial \mathrm{w}_{\mathrm{k}}}=-\sum_{n=1}^{N} x_{n k}\left(t_{n}-w_{0}-\sum_{d=1}^{D} w_{d} x_{n d}\right) \\
& \sum_{n=1}^{N} x_{n k} t_{n}=\sum_{n=1}^{N} x_{n k}\left(w_{0}+\sum_{d=1}^{D} w_{d} x_{n d}\right) \longrightarrow \underbrace{\sum_{n=1}^{N} x_{n k} t_{n}}_{\mathbb{X}^{T} T}=\underbrace{w_{0} \sum_{n=1}^{N} x_{n k}+\sum_{d=1}^{D} w_{d} \sum_{n=1}^{N} x_{n k} x_{n d}}_{\mathbb{X}^{T} \mathbb{X} u} \\
& \begin{array}{l}
\text { Normal equations for } \\
\text { the least squares } \\
\text { problem } \mathbb{X}^{T} \mathbb{X} w
\end{array} \\
& \left(\mathbb{X}^{T} \mathbb{X}\right)^{-1} \mathbb{X}^{T} T=w \quad \mathbb{W}^{*}=\left(\mathbb{X}^{T} \mathbb{X}\right)^{-}
\end{aligned}
$$

## Geometric Interpretation

- Error Minimization
- Geometric Interpretation
- Evaluation
- Regularized Least Squares
-When Normal Equation Gets Tricky


## [Linear Algebra] Span

- The set of all possible linear combinations of the columns of $\mathbb{X}$ is called the span of $\mathbb{X}$ (denoted span( $\mathbb{X}$ )), also called the column space.
- Intuitively, this is all vectors you can "reach" using the columns of $\mathbb{X}$.
- If each column of $\mathbb{X}$ has length $D, \operatorname{span}(\mathbb{X})$ is a subspace of $\mathbb{R}^{D}$.
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-12.jpg?height=1103&width=1481&top_left_y=620&top_left_x=1849)

**Image Description:** The image features a 3D geometric representation of a subspace in \(\mathbb{R}^D\) spanned by a set \(X\). The diagram is a green, translucent vector space indicated by a parallelogram-like shape, with arrows extending in two directions labeled \(X_{.,2}\) and \(X_{.,1}\). The axis orientations denote dimensions corresponding to the vectors in the space, illustrating the linear combination of vectors forming the subspace. Text annotations highlight that this structure is a subspace, affirming the conceptual understanding of vector spaces in linear algebra.



## [Linear Algebra] Matrix-Vector Multiplication

Approach 1: So far, we've thought of our model as horizontally stacked predictions per datapoint:

$$
\left[\begin{array}{l}
\mid \\
\mathbb{Y} \\
\mid
\end{array}\right]=\left[\begin{array}{lcl}
- & \mathbf{x}_{1}^{T} & - \\
- & \mathbf{x}_{2}^{T} & - \\
& \vdots & \\
- & \mathbf{x}_{N}^{T} & -
\end{array}\right]\left[\begin{array}{l}
W \\
\end{array}\right] \quad \begin{aligned}
& \text { Matrix sizes } \\
& N \times 1=(N \times(D+1))((D+1) \times 1)
\end{aligned}
$$

- Approach 2: However, it is helpful sometimes to think of matrix-vector multiplication as performed by columns. We can also think of $\mathbb{Y}$ as a linear combination of feature vectors, scaled by parameters.
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-13.jpg?height=477&width=2391&top_left_y=1386&top_left_x=854)

**Image Description:** The image presents a mathematical equation formatted in LaTeX. The equation is:

$$
Y = \begin{bmatrix} X_{:,0} & \cdots & X_{:,D} \end{bmatrix} W = w_0 X_{:,0} + w_1 X_{:,1} + \cdots + w_D X_{:,D}
$$

This equation represents a linear combination of vectors \(X\) weighted by coefficients \(W\), suggesting a relationship in a predictive modeling or regression context. The brackets indicate matrix/vector structures, with \(Y\) as the resultant output variable.



## Prediction Is a Linear Combination of Columns

Our prediction of $\mathbb{Y}(\mathbb{X}, \vec{w})=\mathbb{X} \vec{w}$ is a linear combination of columns of $\mathbb{X}$.

Interpret: Our linear prediction $\mathbb{Y}$ will be in $\operatorname{span}(\mathbb{X})$, even if ground-truth values $t$ are not.

Goal: Find vector of $\mathbb{Y}$ in $\operatorname{span}(\mathbb{X})$ that is closest to t .
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-14.jpg?height=1096&width=1489&top_left_y=602&top_left_x=1845)

**Image Description:** The image displays a three-dimensional diagram representing a subspace spanned by vectors in \( \mathbb{R}^D \). The diagram includes a green plane, labeled "Subspace of \( \mathbb{R}^D \) spanned by \( X \)," showing vectors \( X_{.,1} \) and \( Y \), represented by arrows pointing downward and diagonally, respectively. An arrow labeled \( t \) points upward, indicating a specific direction within the subspace. The diagram visually illustrates the concept of span in linear algebra, emphasizing the relation between the vectors and their spatial representation.

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-15.jpg?height=452&width=541&top_left_y=561&top_left_x=178)

**Image Description:** The image is a stylized graphic representation of clouds, featuring a combination of two cloud shapes. The larger cloud is purple with a subtle gradient, suggesting a sense of depth, while the smaller cloud is a lighter shade of purple. This graphic likely symbolizes concepts related to weather, data storage (cloud computing), or environmental themes in an academic context. The design is simple and modern, focusing on rounded contours without any intricate details.


# What's the geometry word for 'closest point in a subspace'? 

## Finding Optimum Predictions ( $\mathbb{Y}$ )

To minimize distance between vector of $\mathbb{Y}$ and t , we should minimize the length of the residual vector.
$\vec{e}$ is minimized if it is the orthogonal projection of t on the $\operatorname{span}(\mathbb{X})$.
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-16.jpg?height=451&width=1043&top_left_y=1399&top_left_x=2245)

**Image Description:** The image contains a mathematical equation related to the \( L_2 \) norm of a residual vector. It is presented in LaTeX format, showing the formula:

$$
L_2 - \text{norm} (e) = \| e \|_2 = \sqrt{\sum_{d=0}^{D} e_d^2}
$$

This equation quantifies the length of a vector \( e \) by calculating the square root of the sum of the squares of its components \( e_d \).


## Geometry of Least Squares in Plotly

Interactive link
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-17.jpg?height=1247&width=1570&top_left_y=446&top_left_x=1403)

**Image Description:** The image is a 3D vector diagram illustrating the transformation of vectors in a coordinate system. The axes are labeled \( x \), \( y \), and \( z \). Two vectors, \( \mathbf{X}_{1} \) (black) and \( \mathbf{X}_{2} \) (green), originate from the origin and extend into the positive \( z \)-direction. A third vector, \( \mathbf{Y} \) (blue), is shown as the result of a transformation \( Y = Xw \), where \( w \) likely represents a coefficient or weight. Vectors and transformation labels are positioned near their respective arrows for clarity.


## [Linear Algebra] Orthogonality

- A vector $\vec{v}$ is orthogonal to the $\operatorname{span}(\mathbb{X})$ if and only if it is orthogonal to all columns of $\mathbb{X}$.

Assuming $\vec{v} \in \mathbb{R}^{N}$ and $\mathbb{X}=\left[\begin{array}{ccc}\mid & & \mid \\ \mathbb{X}_{i, 0} & \cdots & \mathbb{X}_{i, D} \\ \mid & & \mid\end{array}\right] \in \mathbb{R}^{N \times D}$
$\vec{v}$ orthogonal to each column means:

We will use this shortly

$$
\begin{gathered}
\mathbb{X}_{:, 0}^{T} \vec{v}=0 \\
\mathbb{X}_{:, 1}^{T} \vec{v}=0 \\
\vdots \\
\mathbb{X}_{:, D}^{T} \vec{v}=0
\end{gathered} \longrightarrow\left[\begin{array}{c}
\mathbb{X}_{:, 0}^{T} \vec{v} \\
\mathbb{X}_{:, 1}^{T} \vec{v} \\
\vdots \\
\mathbb{X}_{:, D}^{T} \vec{v}
\end{array}\right]=\left[\begin{array}{c}
0 \\
0 \\
\vdots \\
0
\end{array}\right] \longmapsto \mathbb{X}^{T} \vec{v}=0
$$

## Going Back to Our Error Function

$$
E(w)=\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}=\frac{1}{2}\left(\|\vec{e}\|_{2}\right)^{2}
$$

Equivalently, vector $\vec{e}$ is orthogonal to the span( $\mathbb{X}$ ), meaning that:
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-19.jpg?height=966&width=3075&top_left_y=901&top_left_x=0)

**Image Description:** The image displays a series of equations related to the derivation of the Normal Equation in linear regression. It includes the following components:

1. The first equation shows the residual definition: $X' e = 0$.
2. The second equation introduces the residual in terms of predicted values: $X' (t - Y(w^*)) = 0$.
3. The third rearranges terms: $X' t - X' X w^* = 0$.
4. Finally, the Normal Equation is presented: $$ w^* = (X'X)^{-1}X' t $$, with a note that it holds if $X'X$ is invertible.


## Evaluation

- Error Minimization
- Geometric Interpretation
- Evaluation
- Regularized Least Squares
-When Normal Equation Gets Tricky


## Predict and Evaluate

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-21.jpg?height=1490&width=3026&top_left_y=324&top_left_x=188)

**Image Description:** The image illustrates a flowchart titled "Machine Learning Process" divided into four quadrants: Learning Problem (L), Model Design (M), Optimization (O), and Predict & Evaluate (P). Each quadrant is represented by arrows connecting to adjacent quadrants. The Learning Problem quadrant emphasizes supervised learning for scalar target values. Model Design features linear regression. The Optimization quadrant includes the equation $$ E(w) = \frac{1}{N} \sum_{n=1}^{N} (y_{test} - w^T x_{n})^2 $$ for deriving a direct solution. Predict & Evaluate highlights model evaluation and test error metrics.


## Evaluation Visualization

Residual plot shows the trend of the residuals $\mathrm{e}_{\mathrm{n}}=\left(t_{n}-y\left(x_{n}, w\right)\right)$ with respect to predictions $y\left(x_{n}, w\right)$.
What a good residual plot looks like:
Points scattered randomly around 0 (no pattern).
Roughly constant vertical spread (homoscedasticity).
No obvious trend with fitted values or with predictors.
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-22.jpg?height=783&width=1000&top_left_y=561&top_left_x=1267)

**Image Description:** The image is a scatter plot depicting a simple linear regression analysis. The x-axis represents "Median Income (MedInc)" with values ranging from 0 to 9, while the y-axis shows "Median House Value (Y)" with values extending from 0 to approximately 5. The plot features numerous red data points indicating individual observations of house values against income levels, with a blue fitted line illustrating the positive correlation. The title of the plot is "Simple Linear Fit: y ~ MedInc."

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-22.jpg?height=822&width=1009&top_left_y=527&top_left_x=2279)

**Image Description:** The image is a scatter plot titled "Residuals vs Fitted." The x-axis represents "Fitted values," while the y-axis shows "Residuals." The plot contains numerous red points indicating residuals for different fitted values. A dashed horizontal line at zero represents the "Zero Residual Line." The distribution of points forms a fan shape, suggesting heteroscedasticity, where the variance of residuals changes with fitted values. The pattern indicates that the residuals are not randomly dispersed, highlighting potential issues with model assumptions.


# When you see a fan shape in the residual plot, what comes to mind? 

## Evaluation - Metrics

## Evaluation - Metrics

## Mean Squared Error (MSE)

## Evaluation - Metrics

## Mean Squared Error (MSE)

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
$\stackrel{i}{\sim}\left(\sum_{i=1}^{N}\left(c_{n}-x\left(c_{n}-c_{n}\right) s^{2}\right)\right.$
$\sqrt{\frac{\lambda}{\lambda}\left(\sum_{i=1}^{N}(c+n-x(\alpha \ldots n))^{\nu}\right)}$

Root Mean Squared Error
Mean Squared Error (MSE)
(Namoves, the metric
Root Mean Squared Error (RMSE)
$\sqrt{\frac{1}{N}\left(\sum_{\text {mack }} \text { bacto }\right.}$ (RMSE) original unit of the data compared to MSE

## Evaluation - Metrics

## Mean Squared Error (MSE)

## Root Mean Squared Error (RMSE)

## R-Squared ( $\mathrm{R}^{2}$ ) Score

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-27.jpg?height=184&width=1195&top_left_y=399&top_left_x=1667)

**Image Description:** The image presents three statistical formulas related to model evaluation:

1. Mean Squared Error (MSE):
   $$ MSE = \frac{1}{N} \sum_{n=1}^{N} (t_n - y(x_n, w))^2 $$

2. Root Mean Squared Error (RMSE):
   $$ RMSE = \sqrt{\frac{1}{N} \sum_{n=1}^{N} (t_n - y(x_n, w))^2} $$

3. R-Squared (R²) Score:
   $$ R^2 = 1 - \frac{\sum_{n=1}^{N} (t_n - \bar{t})^2}{\sum_{n=1}^{N} (t_n - y(x_n, w))^2} $$

These formulas evaluate the performance of predictive models.


## Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)
R-Squared ( $R^{2}$ ) Score

## Mean Squared Error (MSE) <br> Root Mean Squared Error (RMSE) <br> R-Squared ( $\mathbf{R}^{2}$ ) Score

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-27.jpg?height=188&width=397&top_left_y=684&top_left_x=2457)

**Image Description:** The image contains two mathematical equations related to error calculation in statistical models. The first equation presents the Mean Squared Error (MSE) as follows:

$$
\frac{1}{N} \sum_{n=1}^{N} (t_n - y(x_n, w))^2 
$$

The second equation represents the Calculation of Coefficient of Determination (R²):

$$
1 - \frac{\sum_{n=1}^{N} (t_n - y(x_n, w))^2}{\sum_{n=1}^{N} (t_n - \bar{t})^2}
$$

These equations are typically used in regression analysis to assess model performance.

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-27.jpg?height=171&width=397&top_left_y=999&top_left_x=2457)

**Image Description:** The image contains three mathematical expressions related to error calculation in a statistical or machine learning context. 

1. The first formula is the Mean Squared Error (MSE), expressed as:
   $$ 
   \frac{1}{N} \sum_{n=1}^{N} (t_n - y(x_n, w))^2 
   $$
   
2. The second formula appears to represent the root mean squared error (RMSE):
   $$ 
   \sqrt{\frac{1}{N} \left( \sum_{n=1}^{N} (t_n - y(x_n, w))^2 \right)} 
   $$

3. The third expression indicates the ratio of two sums, likely related to variance:
   $$
   \frac{\sum_{n=1}^{N} (t_n - y(x_n, w))^2}{\sum_{n=1}^{N} (t_n - \overline{t})^


## Visualizing the Sum of Squared Error of Regression Model

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-28.jpg?height=1213&width=2455&top_left_y=374&top_left_x=382)

**Image Description:** The slide features a graphical representation of a regression problem. The x-axis represents a vector \(\mathbf{x}\), while the y-axis shows the dependent variable \(y\). Several data points are indicated as black circles, with dashed boxes around them symbolizing residuals—errors between observed and predicted values. A solid line denotes the regression line, which aims to minimize the total area of the boxes (residuals) as stated in the caption. The goal of the regression is clearly articulated.


## Visualizing the Sum of Squared Error of Intercept Model

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-29.jpg?height=1234&width=2463&top_left_y=357&top_left_x=378)

**Image Description:** The image depicts a graph illustrating a piecewise function with a horizontal axis representing the variable \( x \) and a vertical axis representing the output \( y \). The diagram includes data points represented by black circles and segments depicted with red lines connecting the points to the baseline. Vertical dotted lines indicate the intervals of the function's definition. Additionally, the equation \( y(\mathbf{x}, \mathbf{w}) = \bar{t}_n \) is displayed, suggesting a relationship between inputs \( \mathbf{x} \) and weights \( \mathbf{w} \).


## R²: Quality of the Fit Relative to Intercept Model

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-30.jpg?height=775&width=1323&top_left_y=352&top_left_x=51)

**Image Description:** The image features a function representation in a linear model context. It shows the equation \( y(\mathbf{x}, \mathbf{w}) = \mathbf{x}^T \mathbf{w} \) alongside a graphical representation. The diagram includes a dashed line (representing the decision boundary) bisecting the space, with two black circles indicating data points. Additionally, there are blue dotted rectangles surrounding the points, suggesting boundaries or margins. The axis is not labeled in the image but illustrates the relationship between input features \( \mathbf{x} \) and weights \( \mathbf{w} \) in a classification setting.

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-30.jpg?height=516&width=1442&top_left_y=352&top_left_x=1671)

**Image Description:** The image depicts a mathematical function represented as \( y(\mathbf{\bar{x}}, \mathbf{\bar{w}}) = \bar{t}_n \). It features several components: a horizontal axis representing a baseline, with blue dotted vertical lines extending from it to black circles indicating points of interest. Each point has a solid red vertical line connecting it to the baseline, illustrating the relationship between \( y \) values and their corresponding \( t_n \) values. The diagram effectively visualizes how specific inputs correspond to outputs along a quantitative axis.


$$
R^{2}=\frac{\Delta \text { in area }}{\text { Constant Model Area }}=1-\frac{\sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}}{\sum_{n=1}^{N}\left(t_{n}-\overline{t_{n}}\right)^{2}}
$$
unitless and only compares performance relative to mean baseline

## Evaluation - Metrics

## Mean Squared Error (MSE)

## Root Mean Squared Error (RMSE)

## R-Squared ( $\mathrm{R}^{2}$ ) Score

## Mean Absolute Error (MAE)

| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{m=1}^{N}\left(t_{m}-\nu\left(x_{m} \ldots n\right)\right)^{2}\right)$ |
| :--- | :--- |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{m=1}^{N}\left(t_{m}-\nu\left(x_{m}, \ldots p\right)^{2}\right)\right.}$ |
| R-Squared ( $\mathbf{R}^{\mathbf{2}}$ ) Score | $1-\frac{\sum_{n=1}^{N}-1\left(E_{n}-x\left(x_{n}, \ldots\right)\right)^{2}}{\left.\sum_{n=1}^{N}\left(E_{n}\right)-E_{n}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{m=1}^{N} 1 \epsilon_{m}-\nu\left(x_{m} \ldots m\right) 1\right)$ |
| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{n=1}^{N}\left(c_{n}-y\left(c_{n}, w\right)\right)^{2}\right)$ |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{n=1}^{N}\left(\epsilon_{n}-y\left(x_{n}, \ldots p\right)\right)^{2}\right)}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{n=1}^{N}\left\|t_{n}-\nu\left(x_{n}, \ldots\right)\right\|\right)$ |
| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{n=1}^{N}\left(c_{n}-N\left(x_{n}, n\right)\right)^{2}\right)$ |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{m=1}^{N}\left(t_{m}-\nu\left(x_{m}, \ldots\right)\right)^{2}\right)}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N^{N}}\left(\sum_{m=1}^{N} 1 \epsilon_{m}-\nu\left(x_{m} \ldots n\right) 1\right)$ |
| Mean Squared Error (MSE) | In the same unit as |
| R-Squared ( $\mathbf{R}^{\mathbf{2}}$ ) Score | MSE but differs in how |

## Evaluation - Metrics

## Mean Squared Error (MSE)

## Root Mean Squared Error (RMSE)

R-Squared ( $\mathrm{R}^{2}$ ) Score

Mean Absolute Error (MAE)

Mean Absolute Percentage Error (MAPE)

| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{m=1}^{N}\left(x_{m}-y\left(x_{m}, \ldots p\right)^{2}\right)\right.$ |
| :--- | :--- |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{m=1}^{N}\left(c_{m}-\nu\left(x_{m} \ldots m\right)\right)^{2}\right)}$ |
| R-Squared ( $\mathbf{R}^{\mathbf{2}}$ ) Score | $1-\frac{\sum m_{-1}\left(t_{m}-y \mid\left(x_{m}, m\right)\right)^{2}}{\left.\sum m_{-1}\left(t_{m}\right)-\frac{t_{m}}{t_{m}}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \mid x_{m}-\nu\left(x_{m}, \ldots, D\right)\right)$ |
| Mean Absolute Percentage Error (MAPE) | $\frac{1}{N}\left(\sum_{n=1}^{N} \frac{x_{n}-x_{n}\left(x_{n}, w\right)}{x_{n}}\right) \times 100$ |
| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{n=1}^{N}\left(\varepsilon_{n}-y\left(x_{n}, w\right)\right)^{2}\right)$ |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{m=1}^{N}\left(t_{m}-\nu\left(x_{m}+w\right)\right)^{2}\right)}$ |
| R-Squared ( $\mathbf{R}^{\mathbf{2}}$ ) Score | $1-\frac{\sum_{m=1}^{M}\left(t_{m}-y\left(x_{m}, m\right)\right)^{2}}{\left.\sum_{m=1}^{N}\left(t_{m}\right)-E_{m}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{n=1}^{N}\left\|t_{n}-\nu\right\|\left(x_{n}, w\right) \mid\right)$ |
| Mean Absolute Percentage Error (MAPE) | $\frac{1}{N}\left(\sum_{n=1}^{N} \frac{t_{m}-y\left(x_{m}, w\right)}{t_{m}}\right) \times 100$ |
| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{m=1}^{N}\left(x_{m}-\nu\left(x_{m}, \ldots p\right)^{2}\right)\right.$ |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{m=1}^{N}\left(c_{m}-\nu\left(x_{m} \ldots m\right)\right)^{2}\right)}$ |
| R-Squared ( $\mathbf{R}^{\mathbf{2}}$ ) Score | $1-\frac{\sum m_{-1}\left(t_{m}-3\right)\left(x_{m} m p\right)^{2}}{\sum \sum_{m-1}\left(t_{m}\right)-\left(t_{m}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \mid x_{m}-\nu\left(x_{m}, \ldots, D\right)\right)$ |
| Mean Absolute Percentage Error (MAPE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \frac{x_{m}-N\left(x_{m}, W\right)}{x_{m}}\right) \times 100$ |
| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{m=1}^{m}\left(m_{m}-\nu\left(x_{m}, \ldots p\right)^{\bar{x}}\right)\right.$ |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{n=1}^{N}\left(t_{n}-\nu\left(x_{n} \ldots m\right)\right)^{2}\right)}$ |
| R-Squared ( $\mathbf{R}^{\mathbf{2}}$ ) score | $1-\frac{\sum m_{-1}\left(t_{m}-3 \cdot\left(x_{m}, \ldots\right)\right)^{2}}{\left.\sum x_{m}^{m}-1\left(t_{m}\right)-\frac{\left(x_{m}\right)}{x_{m}}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \mid x_{m=}-\nu\left(x_{m . . m p}\right)\right.$ |
| Mean Absolute Percentage Error (MAPE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \frac{c_{m n}-N\left(x_{m n}, W\right)}{N_{m n}}\right) \times 100$ |
| Mean Squared Error (MSE) | $\frac{1}{N}\left(\sum_{m=1}^{m}\left(x_{m}-\nu\left(x_{m}, \ldots p\right)^{z}\right)\right.$ |
| Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N}\left(\sum_{n=1}^{N}\left(c_{n m}-\nu\left(x_{n \ldots} \omega\right)\right)^{2}\right)}$ |
| R-Squared ( $R^{2}$ ) Score | $1-\frac{\sum m_{-1}\left(t_{m}-y \mid\left(x_{m} m p\right)^{2}\right.}{\left.\sum m_{-1}^{m}\left(t_{m}\right)-\varepsilon_{m}\right)^{2}}$ |
| Mean Absolute Error (MAE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \mid x_{m}-\nu\left(x_{m}, \ldots, 1\right)\right.$ |
| Mean Absolute Percentage Error (MAPE) | $\frac{1}{N}\left(\sum_{m=1}^{N} \frac{E_{m}-s\left(x_{m}, W\right)}{x_{m}}\right) \times 100$ |

## Regularized Least Squares

- Error Minimization
- Geometric Interpretation
- Evaluation
- Regularized Least Squares
-When Normal Equation Gets Tricky


## Complexity and Overfitting

Raw data

## More complex isn't always better.

## Fit models to samples of data

## \}

## Regularization

Regularization is the process of adding constraints or penalties to the learning process to improve generalization.

Many models and learning algorithms have methods to tune the regularization during the training process.

## Regularization

Regularization restricts complexity by limiting the magnitudes of the model parameters $w$.

We capture this restriction through the function $\operatorname{Reg}[w]$ and restricting its value to be less than $\boldsymbol{C}$.

Our Error function becomes a constrained function, so we minimize:

$$
\text { Error }\left[h_{w} ; \mathfrak{D}_{\text {training }}\right] \quad \text { such that } \operatorname{Reg}[w]<\boldsymbol{C}
$$

## Regularization - Lagrangian Duality

Our Error function becomes a constrained function, so we minimize:

$$
\text { Error }\left[h_{w} ; \mathfrak{D}_{\text {training }}\right] \quad \text { such that } \quad \operatorname{Reg}[w]<\boldsymbol{C}
$$

This constraint complicates our optimization.
What if we assign a penalty for each violation of the constraint?

Idea of Lagrangian
Introduce a penalty for breaking the constraint.
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-37.jpg?height=393&width=758&top_left_y=973&top_left_x=1173)

**Image Description:** This image presents a formula related to regularization in machine learning. It is structured to illustrate the relationship between an imposed penalty and the violation magnitude. The equation features a lambda symbol ($\lambda$) representing the regularization parameter, followed by the expression $(Reg[w] - C)$, where $Reg[w]$ signifies the regularization term dependent on weights $w$, and $C$ represents a constant threshold. The diagram visually conveys the trade-off between regularization strength and model complexity.


The updated Error function

$$
E(w)=\operatorname{Error}\left[h_{w} ; \mathfrak{D}_{\text {training }}\right]+\lambda \operatorname{Reg}[w]
$$

## From Reg $[w] \rightarrow$ Norms

One simple example of $\boldsymbol{\operatorname { R e g }}[w]$ is the sum of the squares of the weight vector:

$$
\frac{1}{2} \sum w_{d}^{2}=\frac{1}{2} w^{T} w
$$
- Generally, norms are the standard choices for $\operatorname{Reg}[w]$.
- What is a norm?
- Nonnegativity \& definiteness: $\|v\| \geq 0,\|v\|=0 \Leftrightarrow \vec{v}=\overrightarrow{0}$
- Positive homogeneity: $\|\propto v\|=|\propto|\|v\|$
- Triangle inequality: $\|u+v\| \leq\|u\|+\|v\|$
- Norm definition: $L_{p}(w):\|w\|_{p}=\left(\sum_{d}\left|w_{d}\right|^{p}\right)^{\frac{1}{p}}$

## From Reg $[w] \rightarrow$ Norms

- Norm definition: $L_{p}(w)$ : $\|w\|_{p}=\left(\sum_{d}\left|w_{d}\right|^{p}\right)^{\frac{1}{p}}$
- Example of norms:
- $L_{1}(w):\|w\|_{1}=\sum_{d}\left|w_{d}\right|$
- $L_{2}(w):\|w\|_{2}=\sqrt{\sum_{d} w_{d}{ }^{2}}$
- $L_{\infty}(w):\|w\|_{\infty}=\max _{d}\left|w_{d}\right|$


## $\boldsymbol{\operatorname { R e g }}[w] \rightarrow$ Norms $\rightarrow L_{1}$ Regularization

- Norm definition: $L_{p}(w):\|w\|_{p}=\left(\sum_{d}\left|w_{d}\right|^{p}\right)^{\frac{1}{p}}$
- Example of norms:
- $L_{1}(w):\|w\|_{1}=\sum_{d}\left|w_{d}\right|$
- $L_{2}(w):\|w\|_{2}=\sqrt{\Sigma_{d} w_{d}{ }^{2}}$
- $L_{\infty}(w):\|w\|_{\infty}=\max _{d}\left|w_{d}\right|$
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-40.jpg?height=1013&width=1423&top_left_y=531&top_left_x=1911)

**Image Description:** The image is a diagram representing a geometric depiction of the $\ell_1$ norm constraint in two dimensions. It features a diamond shape centered at the origin, with vertices at $(\pm C, 0)$ and $(0, \pm C)$, where $C$ is a positive constant. The x-axis is labeled as $w_1$ and the y-axis is implicitly understood to represent another variable $w_2$. An arrow points from the center towards the diamond, highlighting the constraint $\|w\|_1 = C$. This visually indicates the feasible region for the weights under the $\ell_1$ norm constraint.



## $\boldsymbol{\operatorname { R e g }}[w] \rightarrow$ Norms $\rightarrow L_{2}$ Regularization

- Norm definition: $L_{p}(w):\|w\|_{p}=\left(\sum_{d}\left|w_{d}\right|^{p}\right)^{\frac{1}{p}} \quad W_{2}$
- Example of norms:
- $L_{1}(w):\|w\|_{1}=\sum_{d}\left|w_{d}\right|$
- $L_{2}(w):\|w\|_{2}=\sqrt{\sum_{d} w_{d}{ }^{2}}$
- $L_{\infty}(w):\|w\|_{\infty}=\max _{d}\left|w_{d}\right|$
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-41.jpg?height=1008&width=1417&top_left_y=540&top_left_x=1913)

**Image Description:** The image illustrates a two-dimensional diagram representing the $L_2$ norm of a vector \( \mathbf{w} = [w_1, w_2] \). The axes are labeled \( w_1 \) (horizontal) and \( w_2 \) (vertical), each ranging from \( -C \) to \( C \). The circular contour, centered at the origin, indicates the locus of points where the $L_2$ norm \( ||\mathbf{w}||_2 = C \) holds. The circle's radius is \( C \), visually demonstrating the relationship between the norm and the vector components. An arrow points from the center to the circumference, enhancing interpretation.



## Regularization

We typically use iterative optimization algorithms to train (fit) the model by solving the following problem:
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-42.jpg?height=835&width=2570&top_left_y=650&top_left_x=352)

**Image Description:** The image presents an equation that defines the optimal weight \( w^* \) in a machine learning context. The equation is structured as follows:

$$
w^* = \arg \min_w \left( E_{W}^{training} + \lambda \, \text{Reg}[W] \right)
$$

It illustrates a minimization problem, with \( E \) representing data-dependent error, \( \lambda \) as a regularization parameter, and \( \text{Reg}[W] \) indicating a regularization term. Below the equation, the phrase "Data-dependent Error" is clearly labeled, emphasizing the goal of minimizing both error and regularization.


## Regularization

We typically use iterative optimization algorithms to train (fit) the model by solving the following problem:
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-43.jpg?height=826&width=2016&top_left_y=654&top_left_x=897)

**Image Description:** The image presents an equation representing an optimization problem. The formula is expressed as:

$$
w^* = \arg \min_{w} \left( E_D(w) + \lambda \, Reg[w] \right)
$$

where \( w^* \) is the optimal solution, \( E_D(w) \) denotes the data-dependent error, \( \lambda \) is a regularization parameter, and \( Reg[w] \) signifies the regularization term. An arrow emphasizes the relationship between the minimization expression and the concept of data-dependent error.


## Regularization

We typically use iterative optimization algorithms to train (fit) the model by solving the following problem:
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-44.jpg?height=831&width=2012&top_left_y=654&top_left_x=867)

**Image Description:** The image depicts an equation detailing an optimization problem. It presents the formula for a weight vector \( w^* \) as the argument that minimizes the sum of two components: a data-dependent error \( E_D(w) \) and a regularization function \( E_W[w] \), scaled by a regularization parameter \( \lambda \). The equation is presented as \( w^* = \arg \min (E_D(w) + \lambda E_W[w]) \). Arrows connect each component to their respective labels, indicating their roles in the minimization process.


## Regularization

We typically use iterative optimization algorithms to train (fit) the model by solving the following problem:
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-45.jpg?height=817&width=2421&top_left_y=663&top_left_x=854)

**Image Description:** The image features an equation presented in LaTeX format. It is:

$$
w^* = \arg \min (E_D(w) + \lambda E_W[|w|])
$$

The equation indicates the optimization problem of finding \( w^* \), which minimizes the sum of a data-dependent error \( E_D(w) \) and a regularization term involving a hyperparameter \( \lambda \) and a regularization function \( E_W[|w|] \). The diagram includes directional arrows that connect to definitions of "Data-dependent Error," "Regularizer Hyperparameter," and "Regularization Function," clarifying the components of the equation.


## Linear Regression with L2 Regularization (Ridge)

$$
\begin{aligned}
& E(w)=\mathbf{E}_{\mathrm{D}}[w]+\lambda \mathbf{E}_{\mathrm{w}}[w] \\
& \mathrm{w}^{*}=\underset{w}{\arg \min } \mathbf{E}_{\mathrm{D}}[w]+\lambda \mathbf{E}_{\mathrm{w}}[w]
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-46.jpg?height=1240&width=1913&top_left_y=564&top_left_x=1419)

**Image Description:** The image is a diagram depicting a contour plot in a two-dimensional space representing the functions \( E_W[w] \) and \( E_D[w] \). The horizontal axis (W1) represents the values of a variable \( w_1 \), while the vertical axis represents \( E_W[w] \). The contour lines illustrate levels of energy or error, with darker shades indicating lower values. The curves intersect at points where \( E_W[w] = E_D[w] \), with circles and ellipses displaying varying levels of the dimensions parameterized by \( c \) and \( -c \).


L2 Regularization (Ridge)
$\boldsymbol{L}_{\mathbf{2}}[w]=\sqrt{\sum_{\boldsymbol{d}} w_{\boldsymbol{d}}^{2}} \mathrm{w}^{*}=\underset{w}{\arg \min } \underbrace{\mathbf{E}_{\mathrm{D}}[w]}+\lambda \mathbf{E}_{\mathrm{w}}[w]$
$\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}$
$\frac{1}{2}\left(\left\|t_{n}-y\left(\mathrm{x}_{n}, w\right)\right\|_{2}\right)^{2}$
$\frac{1}{2}\left(\left\|e_{n}\right\|_{2}\right)^{2}$
length of the residual vector
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-47.jpg?height=1009&width=1034&top_left_y=459&top_left_x=2228)

**Image Description:** The image is a contour plot representing a function \( E_D[w] \) in a two-dimensional space defined by the axes \( w_1 \) and \( w_2 \). The plot features elliptical contour lines, indicating levels of the function values, with a red star marking the minimum point. The color gradient transitions from dark blue (indicating lower values) to lighter blue, illustrating the function's optimization landscape. This visual representation aids in understanding the optimization process in the given parameter space.


L2 Regularization (Ridge) $\quad \boldsymbol{L}_{\mathbf{2}}[w]=\sqrt{\sum_{d} w_{d}^{2}}$ $\mathrm{w}^{*}=\arg \min \mathrm{E}_{\mathrm{D}}[w]+\lambda \mathrm{E}_{\mathrm{w}}[w]$

$$
w_{2} \quad\|w\|_{2}=c
$$
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-48.jpg?height=73&width=413&top_left_y=603&top_left_x=918)
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-48.jpg?height=125&width=414&top_left_y=586&top_left_x=1581)

**Image Description:** The image depicts a simple line diagram illustrating a concave shape with two upward arcs meeting at a central downward point, resembling a parabolic curve. The left and right ends are extended horizontally, while the central point forms a sharp dip, possibly indicating a mathematical or physical concept related to potential energy minima or stability in systems. No numerical axes are present, indicating that the diagram is likely conceptual rather than quantitative.

$\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}$
$\frac{1}{2} \boldsymbol{L}_{\mathbf{2}}[w]^{\mathbf{2}}=\frac{1}{2} \sum_{\boldsymbol{d}} w_{\boldsymbol{d}}^{\mathbf{2}}$
$$
\frac{1}{2}\left(\left\|t_{n}-y\left(\mathrm{x}_{n}, w\right)\right\|_{2}\right)^{2}
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-48.jpg?height=882&width=1102&top_left_y=586&top_left_x=2228)

**Image Description:** The image illustrates a contour plot showing a two-dimensional optimization landscape defined by the variables \( w_1 \) and \( w_2 \) on the horizontal and vertical axes, respectively. The contours represent levels of performance or loss associated with different parameter values. A circular path indicates a specific trajectory of adjustment in the parameter space, with a point marked to signify an optimal or target value, referred to as \( E_D[w] \). The color gradient transitions from dark blue to light blue, indicating variations in intensity or value.


$$
\frac{1}{2}\left(\left\|e_{n}\right\|_{2}\right)^{2}
$$
length of the residual vector
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-49.jpg?height=736&width=822&top_left_y=510&top_left_x=1394)

**Image Description:** The image depicts a contour plot representing a two-dimensional function, with the x-axis ranging from -5 to 10 and the y-axis from -5 to 10. The contours illustrate levels of function values, indicated by varying shades of blue. A magenta circle is drawn around a point marked with a green cross, indicating a specific location of interest, likely a minimum or critical point. The buttons at the bottom suggest an interactive element, possibly to animate or manipulate the diagram.

$\lambda=0.00$
$\lambda=\begin{aligned} & \text { । } \\ & \lambda=0.00\end{aligned} \quad \begin{gathered}\text { । } \\ \lambda=2.00\end{gathered} \quad \begin{gathered}\text { । } \\ \lambda=4.00\end{gathered} \quad \begin{gathered}\text { । } \\ \lambda=6.00\end{gathered}$
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-49.jpg?height=960&width=938&top_left_y=470&top_left_x=2287)

**Image Description:** The image is a multi-colored graph depicting several functions related to a variable, \(\lambda\). The x-axis represents \(\lambda\), ranging from 0 to 15, while the y-axis represents the values of different functions, with a scale from 0 to 30. The curves include a star marking at \((0, E_D(0))\) with no regularization, an increasing purple curve labeled \(E(w(\lambda))\), an orange curve showing \(E(w(\lambda))\) with a regularization term, and a light green curve indicating \(|w(\lambda)|\).


L2 Regularization (Lasso) $\quad L_{1}[w]=\sum_{j}\left|w_{j}\right|$ $\mathrm{w}^{*}=\arg \min \mathrm{E}_{\mathrm{D}}[w]+\lambda \mathrm{E}_{\mathrm{w}}[w] \quad w_{2} \quad\|w\|_{2}=c$
$\frac{1}{2} \sum_{n=1}^{N}\left(t_{n}-y\left(\mathrm{x}_{n}, w\right)\right)^{2}$
$\frac{1}{2} \boldsymbol{L}_{\mathbf{1}}[w]=\frac{1}{2} \sum_{\boldsymbol{d}}\left|w_{\boldsymbol{d}}\right|$
$\frac{1}{2}\left(\left\|t_{n}-y\left(\mathrm{x}_{n}, w\right)\right\|_{2}\right)^{2}$
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-50.jpg?height=886&width=1102&top_left_y=582&top_left_x=2228)

**Image Description:** The diagram is a 2D contour plot illustrating a function \( E_D[w] \) on a Cartesian coordinate system. The x-axis represents \( w_1 \) and the y-axis is not labeled but likely represents another variable, possibly \( w_2 \). The contours depict regions of equal function value, with a gradient color scale indicating function values. A magenta diamond shape, centered within the contours, highlights a specific region of interest. An arrow points to this diamond, suggesting it is significant in the context of the function being analyzed.

$\frac{1}{2}\left(\left\|e_{n}\right\|_{2}\right)^{2}$
length of the residual vector
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-51.jpg?height=945&width=1902&top_left_y=476&top_left_x=1335)

**Image Description:** The image consists of two sections: a contour plot and a function graph. 

1. **Contour Plot (Left)**: Displays the function $k(w_1, w_2)$. The axes represent $w_1$ (horizontal) and $w_2$ (vertical), with contours indicating levels of the function's value. A star denotes a specific point $(w_1^*, w_2^*)$. A rhombus outlines a feasible region.

2. **Function Graph (Right)**: Plots multiple curves corresponding to different values of a parameter $\lambda$. The x-axis represents $\lambda$, and the y-axis shows function values. Curves indicate relationships between variables, with intersections at marked points.

Overall, the image illustrates optimization or decision-making concepts in a multi-dimensional space.


## Impact of Regularization - Ridge

$\lambda$ controls complexity and helps avoid overfitting.
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-52.jpg?height=1205&width=2293&top_left_y=603&top_left_x=127)

**Image Description:** The image consists of two diagrams. 

1. The left graph displays a scatter plot with training data (red squares) and testing data (blue circles) against the x-axis (independent variable) and the y-axis (dependent variable) showcasing a fitted curve (green line) representing the relationship. A horizontal red line indicates a baseline.

2. The right graph presents a line plot depicting "Loss vs. $\lambda$" with a logarithmic x-axis. Two lines represent training loss (blue) and test loss (orange), showing how loss changes with regularization parameter $\lambda$. The vertical dashed line marks a specific $\lambda$ value.


## Impact of Regularization - Ridge

## $\lambda$ controls complexity and helps avoid overfitting.

| lambda $=1.00000 \mathrm{e}+02$ |  | lambda=3.50000e+01 | lambda=2.00000e+01 | lambda $=1.00000 \mathrm{e}+00$ | lambda=2.00000e-01 | lambda $=5.00000 \mathrm{e}-02$ | lambda=1.00000e-02 | lambda=3.00000e-03 | lambda=9.00000e-04 | lambda=2.50000e-04 | lambda=1.30000e-04 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Polynomial Degree |  |  |  |  |  |  |  |  |  |  |  |
| 0 | -0.018966 | -0.052435 | -0.088490 | -0.750335 | -1.178313 | -0.318366 | 4.162396 | 8.485826 | 11.662871 | 14.992023 | 17.289582 |
| 1 | -0.019036 | -0.052632 | -0.088828 | -0.777657 | -1.686689 | -3.508704 | -8.271798 | -12.817502 | -17.319566 | -25.023898 | -31.248832 |
| 2 | -0.016455 | -0.045379 | -0.076367 | -0.611629 | -1.230065 | -2.784778 | -7.316610 | -11.216560 | -13.384204 | -13.996051 | -13.938533 |
| 3 | -0.013521 | -0.037154 | -0.062269 | -0.426514 | -0.623082 | -1.219114 | -3.232832 | -4.667422 | -4.177037 | 0.170150 | 4.339540 |
| 4 | -0.010943 | -0.029948 | -0.049953 | -0.273166 | -0.113433 | 0.200421 | 0.657859 | 1.322626 | 2.988265 | 7.711474 | 11.909930 |
| 5 | -0.008863 | -0.024148 | -0.040075 | -0.159327 | 0.251802 | 1.238814 | 3.527585 | 5.538996 | 7.166978 | 9.322972 | 10.980703 |
| 6 | -0.007234 | -0.019625 | -0.032402 | -0.079540 | 0.488914 | 1.910190 | 5.373464 | 8.096688 | 9.045151 | 7.495621 | 5.653301 |
| 7 | -0.005969 | -0.016127 | -0.026494 | -0.025775 | 0.628286 | 2.293662 | 6.412034 | 9.413875 | 9.471402 | 4.241110 | -1.029883 |

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-53.jpg?height=813&width=1379&top_left_y=1054&top_left_x=922)

**Image Description:** The image is a line plot titled "Coefficient Magnitudes vs. Lambda." The x-axis represents "Lambda" on a logarithmic scale, with values ranging from \(10^{-12}\) to \(10^{0}\). The y-axis indicates "Coefficient Magnitudes," displaying values from approximately \(-30\) to \(10\). Multiple colored lines (representing coefficients 0 to 7) illustrate how the magnitudes of the coefficients change with varying lambda values. Each line is distinctively colored, providing visual differentiation for analysis of the relationships.


## Impact of Regularization - Lasso

$\lambda$ controls complexity and helps avoid overfitting.
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-54.jpg?height=1200&width=2284&top_left_y=608&top_left_x=123)

**Image Description:** The image consists of two panels: 

1. The left panel shows a scatter plot titled "Fitted Function with Train/Test." It features red squares for training data, blue circles for testing data, a fitted curve (dashed green line), and a horizontal line (red) indicating a baseline. The x-axis represents the independent variable, while the y-axis denotes the dependent variable.

2. The right panel presents a line graph titled "Loss vs A." It depicts three curves: train mean squared error (MSE) (orange), test MSE (blue), and fitted line (red). The x-axis is in logarithmic scale, showing parameter \(A\), while the y-axis indicates loss values.


## Impact of Regularization - Lasso

## $\lambda$ controls complexity and helps avoid overfitting.

| lambda=1.00000e+02 |  | lambda=3.50000e+01 | lambda=2.00000e+01 | lambda $=1.00000 \mathrm{e}+00$ | lambda=2.00000e-01 | lambda=5.00000e-02 | lambda=1.00000e-02 | lambda=3.00000e-03 | lambda=9.00000e-04 | lambda=2.50000e-04 | lambda=1.30000e-04 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Polynomial Degree |  |  |  |  |  |  |  |  |  |  |  |
| 0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | 0.000000 | 4.986034 | 8.175296 | 11.697458 | 13.179180 |
| 1 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -2.095965 | -1.975668 | -10.457935 | -16.722823 | -26.709688 | -31.561898 |
| 2 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | -0.767903 | -0.000000 | -0.000000 | 0.000000 | 1.888739 |
| 3 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | -0.000000 | 0.000000 | 0.000000 | 9.784207 | 12.196142 |
| 4 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | -0.000000 | 0.000000 | 4.504546 | 9.382764 | 10.631082 |
| 5 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | 0.000000 | 0.000000 | 3.684894 | 1.077110 | 2.682036 |
| 6 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | 0.000000 | 4.632838 | 0.000000 | -0.000000 | -0.000000 |
| 7 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | 0.000000 | 0.000000 | 0.000000 | -6.925711 | -11.495779 |

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-55.jpg?height=813&width=1213&top_left_y=1054&top_left_x=927)

**Image Description:** The diagram is a line graph titled "Coefficient Magnitudes vs Lambda." The x-axis represents "Lambda" on a logarithmic scale, with values ranging from \(10^{-2}\) to \(10^{-6}\). The y-axis displays "Coefficient Magnitudes" with values from approximately -30 to 10. Different colored lines represent distinct coefficients, showing how their magnitudes change as lambda varies. The plot features markers at specific lambda values, indicating data points. The graph visualizes the relationship between lambda and the magnitudes of several coefficients, highlighting trends and stability across the range.

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-55.jpg?height=277&width=141&top_left_y=1097&top_left_x=2160)

**Image Description:** The image is a color-coded legend or key illustrating coefficients numbered from 0 to 7. Each coefficient is associated with a specific color and represented as a circular marker on a horizontal line. The markers progress vertically, with colors transitioning for each increasing number: from light blue for 0 to gray for 7. This categorization visually represents a quantitative range or classification system, potentially for data visualization in a scientific or statistical context.


## Impact of Regularization - Lasso

## $\lambda$ controls complexity and helps avoid overfitting.

| lambda=1.00000e +02 |  | lambda=3.50000e+01 | lambda=2.00000e+01 | lambda=1.00000e+00 | lambda=2.00000e-01 | lambda=5.00000e-02 | lambda=1.00000e-02 | lambda=3.00000e-03 | lambda=9.00000e-04 | lambda=2.50000e-04 | lambda=1.30000e-04 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Polynomial Degree |  |  |  |  |  |  |  |  |  |  |  |
| 0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | 0.000000 | 4.986034 | 8.175296 | 11.697458 | 13.179180 |
| 1 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -2.095965 | -1.975668 | -10.457935 | -16.722823 | -26.709688 | -31.561898 |
| 2 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | -0.767903 | -0.000000 | -0.000000 | 0.000000 | 1.888739 |
| 3 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | -0.000000 | 0.000000 | 0.000000 | 9.784207 | 12.196142 |
| 4 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | -0.000000 | 0.000000 | 4.504546 | 9.382764 | 10.631082 |
| 5 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | 0.000000 | 0.000000 | 3.684894 | 1.077110 | 2.682036 |
| 6 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | 0.000000 | 4.632838 | 0.000000 | -0.000000 | -0.000000 |
| 7 | -0.0 | -0.0 | -0.0 | -0.0 | -0.0 | -0.000000 | 0.000000 | 0.000000 | 0.000000 | -6.925711 | -11.495779 |

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-56.jpg?height=817&width=1217&top_left_y=1012&top_left_x=93)

**Image Description:** The image is a scatter plot titled "Coefficient Magnitudes vs Lambda." The x-axis is labeled "Lambda (log scale)" with specific discrete values marked along it, while the y-axis shows "Coefficient Magnitudes." The plot features multiple colored lines connecting points that represent the coefficients' magnitudes at different lambda values. Notably, the y-axis ranges from -30 to 10, displaying both positive and negative values. Each point is color-coded, providing insights into the relationship between lambda and the respective coefficient magnitudes. The use of a logarithmic scale on the x-axis suggests a focus on exponential decay or growth characteristics.

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-56.jpg?height=277&width=142&top_left_y=1063&top_left_x=1326)

**Image Description:** The image is a legend indicating the color-coded representation of coefficients for different variables in a graphical context. It consists of circular markers in various colors paired with numerical labels ranging from 0 to 7. Each color corresponds to a specific coefficient value, aiding in visual data interpretation. The arrangement suggests an increasing order from top to bottom, with no additional axis or diagram provided in the image itself.

![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-56.jpg?height=473&width=528&top_left_y=1101&top_left_x=1522)

**Image Description:** The image illustrates a contour plot with concentric circles indicating levels of a function's value, likely representing a gradient or potential field. The background is a gradient from dark blue to lighter blue. In the foreground, a magenta diamond shape is overlaid on the plot, with two distinct points marked by a red star and a teal asterisk. The plot indicates the relationship between the position of these points and the changing values of the function throughout the domain.


Lasso
Least Absolute Shrinkage and Selection Operator

- Error Minimization


## When Normal Equation Gets Tricky

- Geometric Interpretation
- Evaluation
- Regularized Least Squares
- When Normal Equation

Gets Tricky

## When Normal Equation Gets Tricky

-When is $\mathbb{X}^{T} \mathbb{X}$ invertible?

- Full column rank: $\operatorname{rank}(\mathbb{X})=D+1$
- Requires $\mathrm{N} \geq \mathrm{D}+1$ (at least as many samples as features)
- No perfect collinearity: $\operatorname{det}\left(\mathbb{X}^{T} \mathbb{X}\right) \neq 0$
- All singular values be non-zero: $\sigma_{1}, \sigma_{2}, \ldots, \sigma_{D+1} \geq 0$
- Illconditioning (numerical unstability)
- Even if $\operatorname{det}\left(\mathbb{X}^{T} \mathbb{X}\right) \neq 0$
- Condition number $\kappa=\frac{\sigma_{\max }}{\sigma_{\min }} \gg 10^{8}$ small noise in the data creates huge weight swing
- Why?
- Rounding error will be amplified when calculating $\kappa \varepsilon_{\text {machine }}$.

If we plug in the SVD of $X$, what is the simplified expression for the solution to the normal equation?

## Fix \#1: Ridge Trick

Stabilize with $\left(\mathbb{X}^{\mathrm{T}} \mathbb{X}+\lambda I\right) w=\mathbb{X}^{\mathrm{T}} t$

- $\lambda>0$ boosts all singular values $\rightarrow$ lower $\kappa$.
- Closed-form solution: $w^{*}=\left(\mathbb{X}^{\mathrm{T}} \mathbb{X}+\lambda I\right)^{-1} \mathbb{X}^{\mathrm{T}} t$


## Illconditionin g Example

With the condition factor that is large, due to very small eigenvalues resulting from very similar features, the weights become unstable.

Try changing the noise and you will see substantial swings in weights.

```
N, D = 500, 2 # one bias + 2 features
X = np.ones((N, D + 1)) # first column = bias
x1 = np.random.randn(N)
x2 = x1 + 1e-4 * np.random.randn(N) # almost identical feature
X[:, 1] = x1 Condition ( (X'X) =
X[:, 2] = x2 420140223.27925146
t = 3 + 2*x1 - 1*x2 + 0.1*np.random.randn(N) # targets
XtX = X.T @ X
Xty = X.T @ t
w_ne = np.linalg.solve(XtX, Xty)
lam = 1e-2
w_ridge = np.linalg.solve(XtX + lam*np.eye(D+1), Xty)
w_neq: [ 3.00273423 12.87110771
-11.87562795]
w_ridge : [3.00275573 0.50070697
0.49481053]
```


## Fix \#2: Moore-Penrose Pseudo-Inverse

- Singular Value Decomposition (SVD) recipe:
- $X=U \Sigma V^{T}$ where $\Sigma=\operatorname{diag}\left(\sigma_{1}, \ldots, \sigma_{r}\right)$ with $r=\operatorname{rank}(X)$.
- Every real matrix can be factorized this way; $U$ and $V$ have orthogonal columns ( $U^{T} U=V^{T} V=I$ ), $\Sigma$ holds the "stretch factors".
- Moore-Penrose Pseudo-inverse:

$$
\Sigma_{i i}^{+}= \begin{cases}\frac{1}{\sigma_{i}} & \text { if } \sigma_{i}>\varepsilon \sigma_{\max } \\ 0 & \text { otherwise }\end{cases}
$$
- We safely "flip" only the non-tiny singular values.
$$
X^{+}=V \Sigma^{+} U^{T}
$$

- Penrose conditions: $X X^{+} X=X, X^{+} X X^{+}=X^{+},\left(X X^{+}\right)^{T}=X X^{+},\left(X^{+} X\right)^{T}= X^{+} X$. These make $X^{+}$unique.


## Fix \#2: Moore-Penrose Pseudo-Inverse

## How can Penrose Pseudo-Inverse help us with solving normal equation?

$$
\begin{aligned}
& \text { o-inverse: } \\
& \Sigma_{i i}^{+}= \begin{cases}\frac{1}{\sigma_{i}} & \text { if } \sigma_{i}>\varepsilon \sigma_{\max } \\
0 & \text { otherwise }\end{cases}
\end{aligned}
$$

- We safely "flip" only the non-tiny singular values.
$X^{+}=V \Sigma^{+} U^{T}$ Penrose conditions: $X X^{+} X=X, X^{+} X X^{+}=X^{+},\left(X^{+}\right)^{T}=X X^{+},\left(X^{+} X\right)^{T}= X^{+} X$. These make $X^{+}$unique.

Closed-form solution for least squares:

$$
w=\mathbb{X}^{+} t
$$

This works no matter whether $\mathbb{X}$ is tall ( $N>D$ ), square, or wide ( $N<D$ ). Also regardless of singular values.

## Fix 2: Why It Helps?

| Issue | How pseudo-inverse fixes it? |  |
| :--- | :--- | :--- |
| Rank deficiency (perfect multicollinearity) | Issue Rank deficiency (perfect | How pseudo-inverse fixes it? Sets 0 for $\sigma=0 \Rightarrow$ removes redundant directions, |

## Fix 2: Why It Helps?

| Issue | How pseudo-inverse fixes it? |  |
| :--- | :--- | :--- |
| Rank deficiency (perfect multicollinearity) | Issue Rank deficioncy (perfect multicollinearity) Ml-conditioning | How pseudo-inverse fixes it? Sets o for $\sigma=0 \rightarrow$ removes redundant directions, gives the minimum-norm weight vector. Truncation threshold discards numerically meaningless directions, implicitly adds ridge-like regularization. |
| III-conditioning |  | Truncation threshold discards numerically meaningless directions, implicitly adds ridgelike regularization. |

## Fix 2: Why It Helps?

| Issue | How pseudo-inverse fixes it? |  |
| :--- | :--- | :--- |
| Rank deficiency (perfect multicollinearity) | Ramk aefinioney perfect Rank define iency multiodlinearity Ill-conditioning <br> Undordotexmined |  |
| III-conditioning | Truncation threshold discards numerically meaningless directions, implicitly adds ridgelike regularization. |  |
| ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-66.jpg?height=50&width=881&top_left_y=1149&top_left_x=135) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-66.jpg?height=48&width=1032&top_left_y=1171&top_left_x=140) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-66.jpg?height=60&width=1038&top_left_y=1225&top_left_x=135) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-66.jpg?height=66&width=1038&top_left_y=1273&top_left_x=135) | reaming amonemor coerect <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-66.jpg?height=60&width=460&top_left_y=1203&top_left_x=1220) <br> प्रत्ये विश्ववर्त्या | E0. N..................................... <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-66.jpg?height=87&width=1238&top_left_y=1171&top_left_x=1922) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-66.jpg?height=80&width=1249&top_left_y=1221&top_left_x=1922) поштык. |

## Fix 2：Why It Helps？

| Issue | How pseudo－inverse fixes it？ |  |
| :--- | :--- | :--- |
| Rank deficiency （perfect multicollinearity） | ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=87&width=676&top_left_y=519&top_left_x=1220) multuodusamus <br> Undoxdotormined Sextems（N＜D） tropurismantation | ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=81&width=1232&top_left_y=519&top_left_x=1922) elives the minimum－nor－m weight vector． <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=60&width=1249&top_left_y=616&top_left_x=1922) Resularization ique solution with the smallest La－ norm among infinitely many exact fits． <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=83&width=1303&top_left_y=744&top_left_x=1922) |
| III－conditioning | Truncation threshold discards numerically meaningless directions，implicitly adds ridge－ like regularization． |  |
|  |  | ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=108&width=1296&top_left_y=1145&top_left_x=1922)

**Image Description:** The slide discusses the concept of the pseudo-inverse in relation to finding a minimum-norm weight vector. It includes a text box with key points: setting σ to zero removes redundant directions and identifies the minimum-norm weight vector, while a truncation threshold eliminates numerically insignificant directions. This highlights the efficacy of the pseudo-inverse in optimizing weight vectors in mathematical modeling or machine learning contexts. The content is primarily textual, focused on explaining theoretical concepts rather than presenting a diagram or mathematical equation.
 <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=86&width=1287&top_left_y=1197&top_left_x=1922) |
| ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=97&width=1037&top_left_y=1145&top_left_x=140) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=59&width=1032&top_left_y=1221&top_left_x=140) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=55&width=1043&top_left_y=1269&top_left_x=140) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=66&width=260&top_left_y=1307&top_left_x=140) | 茶茶。由 문……．．．．．．．．．．．． ＝＝＝＝＝＝＝＝＝＝ |  |
|  |  |  |
| Implementation convenience | ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=109&width=465&top_left_y=1377&top_left_x=1215)

**Image Description:** The image is a structured text slide from an academic lecture. It presents a list of issues related to statistical modeling or regression analysis. The issues are: "Rank deficiency (multicollinearity)" and "Ill-conditioning." Each issue is highlighted prominently, indicating potential problems in model estimation or interpretation. The layout suggests clarity and organization, reinforcing the importance of recognizing these issues in the context of the lecture topic.
 －－－ 준주느…⿴囗十 | ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=82&width=1292&top_left_y=1372&top_left_x=1922) <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=102&width=1249&top_left_y=1406&top_left_x=1922)

**Image Description:** $$ \sigma = 0 \Rightarrow \text{removes redundant directions, gives the minimum-norm weight vector.} $$

The text describes a truncation threshold that eliminates numerically insignificant directions in a mathematical model. This process implicitly introduces ridge-like regularization, which is used to improve model stability by mitigating overfitting.
 <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=103&width=1275&top_left_y=1447&top_left_x=1922)

**Image Description:** The image contains a textual description regarding the concept of truncation threshold in a mathematical context. It explains that truncation threshold discards numerically meaningless directions, adding ridge-like regularization. Additionally, it states that this method yields the unique solution with the smallest \(L_2\)-norm among infinitely many exact fits. The text is informative, discussing both the implications of the threshold and the resulting solution characteristics.
 <br> ![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-67.jpg?height=77&width=1319&top_left_y=1523&top_left_x=1922) |

## Fix 3: Sequential Learning

- In practice, we use sequential algorithms, also known as online algorithms, in which the data points are considered one at a time and the model parameters updated after each such presentation.
- Algorithm: Stochastic gradient descent, also known as sequential gradient descent:
- If we assume that the error function is calculated based on all samples, then $E= \sum_{n} E_{n}$.
- After presentation of datapoint $n$, the gradient descent algorithm updates parameters using $w^{(\tau+1)}=w^{(\tau)}-\eta \nabla E_{n}$
![](https://cdn.mathpix.com/cropped/2025_10_01_dc95ce31a76cb18463adg-68.jpg?height=477&width=2447&top_left_y=1377&top_left_x=118)

**Image Description:** The image contains a formula related to the Least Mean Squares (LMS) algorithm. It is presented in a rectangular layout with an arrow pointing to the term "Learning rate," emphasizing its significance in the equation. The formula is structured as follows:

$$
w^{(\tau + 1)} = w^{\tau} - \eta \left( t_n - w^{(\tau)T} \phi(x_n) \right) \phi(x_n)
$$

This equation updates the weight vector \( w \) based on the learning rate \( \eta \), the target value \( t_n \), and a feature transformation \( \phi(x_n) \).



## Lecture 7

## Linear Regression (2)

Credit: Joseph E. Gonzalez and Narges Norouzi
Reference Book Chapters: Chapter 1.2, Chapter 4.[1.4-1.6]

