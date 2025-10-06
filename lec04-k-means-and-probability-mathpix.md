---
course: CS 189
semester: Fall 2025
type: lecture
title: K-means and Probability
source_type: slides
source_file: Lecture 04 -- K-means and Probability.pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Lecture 4

## K-Means and Probability

An introduction to unsupervised learning and a review of core concepts in probability

## EECS 189/289, Fall 2025 @ UC Berkeley

Joseph E. Gonzalez and Narges Norouzi

# III Join at slido.com \#1041260 

-K-means Clustering

- Scikit-Learn
- Lloyd's Algorithm
- Pixel K-Means
- Probability (Review)

Roadmap

- Joint Distributions
-Wake Word Example
Questions
-K-means Clustering
- Scikit-Learn
- Lloyd's Algorithm
- Pixel K-Means
- Probability (Review)

K-means Clustering

- Joint Distributions
- Wake Word Example

Questions

## Prof. Gonzalez's Messy Biking Records

Professor Gonzalez has too many bikes
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-05.jpg?height=337&width=550&top_left_y=518&top_left_x=246)

**Image Description:** This is a miscellaneous image of a bicycle, specifically a road bike model. The bike features a sleek, aerodynamic design with a vibrant yellow and black color scheme. It has drop handlebars, a lightweight frame, and disc brakes, indicating performance suitability for road cycling. The wheels are equipped with thin tires designed for speed on pavement. The overall design showcases modern cycling technology, emphasizing efficiency and aerodynamics.


![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-05.jpg?height=354&width=592&top_left_y=501&top_left_x=978)

**Image Description:** This is a misc. image of a mountain bike, specifically a Giant brand model. The bike features a black frame with blue accents, highlighting the brand name "Giant." It has a rigid fork and wide tires suitable for off-road terrain. The bike is equipped with disc brakes, enhancing stopping power, and has a simple gear setup typical for mountain biking. The composition showcases the bike in a side view, allowing for clear visibility of its design and components.


![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-05.jpg?height=350&width=592&top_left_y=514&top_left_x=1709)

**Image Description:** The image depicts a yellow cargo bike designed for transporting children. It features two child seats mounted at the back, with safety harnesses visible. A small blue bicycle is also secured on the bike's frame. The cargo area is enclosed with a sturdy black storage bag, enhancing the bike's utility. The design emphasizes stability and safety, suitable for family transport. The overall configuration showcases an innovative approach to urban mobility.


![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-05.jpg?height=328&width=559&top_left_y=510&top_left_x=2440)

**Image Description:** This image is of a mountain bike, specifically a Canyon model. It features a lightweight frame, disc brakes, and wide tires designed for off-road performance. The bike has a modern geometric design, with a straight handlebar and suspension fork. The color is primarily light yellow, with branding prominently displayed on the frame. The components include a gear system visible on the rear wheel hub and front derailleur. Its rugged tires are equipped for traction on uneven terrain, indicative of a focus on durability and performance in outdoor cycling environments.



He has been recording the speed and length of his bike rides but not which bike he used.

He would like to answer questions like:

1. What is the average ride time for each bike?
2. Are there any rides that are abnormally short or long?
3. What is the most likely bike for each ride?

## Learning Problem

We have unlabeled data and we would like to divide the records into $\mathbf{4}$ groups (clusters) corresponding to the four bikes.

- Unsupervised Learning: we don't have the labels.
- Clustering:
we are trying to infer the unobserved (latent) bike choice.
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-06.jpg?height=873&width=1889&top_left_y=714&top_left_x=1365)

**Image Description:** The image is a diagram showing a flowchart categorized into "Supervised Learning" and "Unsupervised Learning." 

- **Supervised Learning**: It branches to "Quantitative Label" and "Categorical Label," which further leads to "Regression" (illustrated with a stock prediction graph) and "Classification" (showing images of labeled objects).
- **Unsupervised Learning**: It is divided into "Dimensionality Reduction" and "Clustering," depicted with scatter plots illustrating different clustering results. 

Arrows indicate the relationships between these concepts, enhancing comprehension of machine learning methodologies.


**Image Description:** The diagram presents a conceptual framework for machine learning, dividing it into three main categories: Supervised Learning, Unsupervised Learning, and Reinforcement Learning. 

1. **Supervised Learning** is further divided into two branches: Regression (depicted with a stock prediction curve) and Classification (illustrated with labeled images). 
2. **Reinforcement Learning** leads to an example involving Alpha Go, showing a game board image.
3. **Unsupervised Learning** branches into Dimensionality Reduction and Clustering, illustrated with scatter plot images depicting data clusters and dimensionality transformations. 

Axes are not explicitly labeled, focusing instead on categorical relationships.


Let's try
k-means clustering

## SkLearn K-means Clustering

K-Means Clustering of Bike Segments

## Demo

Using sklearn k-means clustering
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-07.jpg?height=1349&width=1532&top_left_y=425&top_left_x=1462)

**Image Description:** The image is a scatter plot representing the outcome of k-means clustering using the Scikit-learn library. The x-axis denotes "Speed" and the y-axis represents "Length." Data points are color-coded by cluster: red, blue, green, and purple, with a legend indicating cluster labels (0, 1, 2, 3). Centroids of these clusters are indicated by black markers. The plot visually differentiates the distribution and grouping of data points based on their features, illustrating the effectiveness of the k-means algorithm in clustering.



## Clustering With K-Means in Scikit-Learn ${ }_{1041200}^{\text {BSA }}$

In the demo we wrote a few lines of code and obtained a reasonable clustering of the ride-times.

```
from sklearn.cluster import KMeans
# Create a KMeans model with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
# Fit the model to the data
kmeans.fit(bikes[['Speed', 'Length']])
# Predict the cluster assignments
bikes['c'] = kmeans.predict(bikes[['Speed', 'Length']])
Today we will learn how this algorithm
We will review concepts in probability and explore more general density estimation techniques.
```

-K-means Clustering

- Scikit-Learn
- Lloyd's Algorithm
- Pixel K-Means
- Probability (Review)

Lloyd's Algorithm

- Joint Distributions
- Wake Word Example

Questions

## K-Means Clustering

Input: A collection of data points $\mathcal{D}=\left\{x_{1}, \ldots, x_{N}\right\}$ where $x_{n} \in \mathbb{R}^{D}$.
Output: $K$ cluster centers $\mu_{k} \in \mathbb{R}^{D}$ and an assignment $z_{n} \in\{1, \ldots, K\}$ of each data point to one of the cluster centers.

- The means $\mu_{k}$ are model parameters of the model ( $z_{n}$ can be computed from $\mu)$ that are fit to the data.
- $K$ is a hyperparameter which we need to select.

Objective: Each data point should be close to its assigned center.

$$
\arg \min _{\mu, z} \sum_{n=1}^{N}\left\|x_{n}-\mu_{z_{n}}\right\|_{2}^{2}
$$
$\left(\mathrm{L}_{2}-\text { Norm }\right)^{2}$
Euclidean Distance Squared
$$
\|a\|_{2}^{2}=\sum_{d=1}^{D} a_{d}^{2}
$$

## K-Means Cluster (Lloyd's) Algorithm

'nitialization: Choose $K$ points at random to be the initial cluster centers $\mu$.

## Iterate until convergence:

![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-11.jpg?height=280&width=799&top_left_y=516&top_left_x=2286)

**Image Description:** The image consists of three groups of stars and circles arranged in a horizontal layout. Each group features a differently colored star (orange, green, and purple) surrounded by several smaller blue circles. The arrangement may represent clusters or categories in a scatter plot format, indicating distinct groups or classifications in a spatial framework. The stars symbolize centroids or distinct points of interest, while the circles likely represent data points or features associated with each centroid. The visual suggests a conceptual illustration of clustering or classification in a dataset.



1. Update Assignments: Assign each point to its nearest cluster center.

$$
z=\arg \min _{z} \sum_{n=1}^{N}\left\|x_{n}-\mu_{z_{n}}\right\|_{2}^{2}
$$
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-11.jpg?height=188&width=171&top_left_y=1007&top_left_x=2296)

**Image Description:** This image depicts a simple illustration featuring a prominent yellow star at the center, surrounded by smaller yellow circles. The star is five-pointed with a dark outline. The circles appear to represent celestial or significant points of interest around the star, creating a visual hierarchy where the star is the focal point. There are no axes or quantitative measures, and the image does not convey a specific scientific diagram or equation, making it a decorative or conceptual graphic rather than a technical one.


![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-11.jpg?height=166&width=150&top_left_y=1080&top_left_x=2585)

**Image Description:** The image features a green star at the center, surrounded by several green circular dots. The star appears to represent a focal point or significant data point, while the circles may denote related concepts or entities. The overall layout suggests an illustrative relation, possibly indicating a network or relationship diagram. However, no axes or quantitative data are present, thus it functions more as a conceptual representation rather than a detailed analytical diagram.


2. Update Centers: Recompute centers by averaging assigned points.

Why is this the mean of each cluster?

$$
\mu=\arg \min _{\mu} \sum_{n=1}^{N}\left\|x_{n}-\mu_{z_{n}}\right\|_{2}^{2}
$$
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-11.jpg?height=158&width=158&top_left_y=1607&top_left_x=2577)

**Image Description:** The image features a star-shaped symbol in green, prominently placed in the center. Behind this star, there are several lighter, translucent star shapes arranged in various sizes, creating a layered effect. Surrounding the central star, there are several circular shapes with a green outline and a lighter fill, suggesting a dynamic context. An orange arrow points towards the central star, indicating emphasis or direction. This composition may be used to illustrate concepts such as focus, importance, or centrality in a given topic.


-

## Updating the Cluster Centers

'What is the value that minimizes?

$$
\arg \min _{\mu} \sum_{n=1}^{N}\left\|x_{n}-\mu_{z_{n}}\right\|_{2}^{2}
$$

We can re-arrange the objective to optimize with respect to $\mu$ :

$$
\sum_{n=1}^{N}\left\|x_{n}-\mu_{z_{n}}\right\|_{2}^{2}=\sum_{n=1}^{N} \sum_{d=1}^{D}\left(x_{n d}-\mu_{z_{n} d}\right)^{2}=\sum_{k=1}^{K} \sum_{i=1}^{N_{k}} \sum_{d=1}^{D}\left(x_{k_{i} d}-\mu_{k d}\right)^{2}
$$

## Minimizing the Transformed Objective

'Ne take the derivative with respect to $\mu_{k d}$ :

$$
\frac{\partial}{\partial \mu_{k d}} \sum_{k=1}^{K} \sum_{i=1}^{N_{k}} \sum_{d=1}^{D}\left(x_{k_{i} d}-\mu_{k d}\right)^{2}=\sum_{i=1}^{N_{k}}-2\left(x_{k_{i} d}-\mu_{k d}\right)
$$

Setting the derivative equal to zero and solving for $\mu_{k d}$ :

$$
\sum_{i=1}^{N_{k}}-2\left(x_{k_{i} d}-\mu_{k d}\right)=-2 \sum_{i=1}^{N_{k}} x_{k_{i} d}+2 N_{k} \mu_{k d}=0
$$

Is Lloyd's algorithm guaranteed to converge? Does it always produce an optimal clustering?

## Convergence of K-Means

's the k-means (Lloyd's) algorithm guaranteed to converge?

- Yes -)! Why?

Alternating minimization always decreases the objective

$$
z=\arg \min _{z} \sum_{n=1}^{N}\left\|x_{n}-\mu_{z_{n}}\right\|_{2}^{2} \text { and } \mu=\arg \min _{\mu} \sum_{n=1}^{N}\left\|x_{n}-\mu_{z_{n}}\right\|_{2}^{2}
$$

How do we know when it has converged?

- The cluster assignments ( $z$ ) stop changing.

Do the final $z^{*}$ and $\mu^{*}$ minimize the objective?

- No :)! Can be local minima.


## See live animation in the demove

1041260

## Demo

Animated K-Means Clustering
\&
K-means for pixels

K-Means Clustering
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-16.jpg?height=983&width=1256&top_left_y=446&top_left_x=1598)

**Image Description:** The image is a scatter plot displaying data points categorized by class, indicated by different colors (green, red, blue, and purple). The x-axis represents "Speed," ranging approximately from 0 to 18, while the y-axis represents "Length," ranging from 5 to 30. Each point corresponds to a data observation, and the black crosses denote the centroids of the different classes. The scatter of points suggests clustering based on the class variable, illustrating relationships between speed and length effectively.



![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-16.jpg?height=137&width=924&top_left_y=1450&top_left_x=1777)

**Image Description:** The image is a linear diagram representing a discrete variable for "Iteration" with a scale ranging from 0 to 10. It features a slider positioned at the value 2, illustrating the current iteration state. The x-axis is labeled with integer values from 0 to 10, divided by vertical ticks, while the slider itself is circular, indicating a specific point along the continuum. This diagram visually demonstrates the concept of iteration in a process, allowing for clear interpretation of progress or stages within an iterative framework.



## Choosing the Number of Clusters

There are several ways to select the hyperparameter $K$.

- Use domain understanding (e.g., 4 bikes)
- Use the "elbow method" to select the value of $K$ with diminishing improvements in the objective:

$$
\sum_{n=1}^{N}\left\|x_{n}-\mu_{z_{n}}\right\|_{2}^{2}
$$

## Interpreting the Clusters

We used k -means to compute a cluster assignment for each ride.
Which bike is each cluster?

- We don't know. Could ask for a few labels.
Does each cluster represent a bike?
- Maybe? But it could also correspond to other factors (e.g., group ride, traffic).
Use caution when interpreting clusters!
-K-means Clustering
- Scikit-Learn
- Lloyd's Algorithm
- Pixel K-Means
- Probability (Review)


## Pixel K-Means

- Joint Distributions
- Wake Word Example

Questions

## Pixel K-Means

The pixels in an image can be treated of as vectors in an RGB vector space.

- We can use k -means to compute the clusters of colors and render an image using just a few colors.
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-20.jpg?height=762&width=579&top_left_y=1029&top_left_x=276)

**Image Description:** The image shows a child seated on a bicycle with safety gear, including a helmet and sunglasses. The background features a paved path alongside a residential area with greenery and flowers. There is a bike-mounted storage setup visible, suggesting an emphasis on safety and utility. The image is labeled as "Original Image" at the top. The dimensions indicate a resolution of 800 pixels wide by 600 pixels tall, which captures the child's posture and the surrounding environment effectively.


![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-20.jpg?height=767&width=575&top_left_y=1024&top_left_x=910)

**Image Description:** The image is a K-means compressed representation of an image featuring a child seated on a bicycle. The diagram displays pixel intensity values, where the x-axis represents pixel width (0 to 800) and the y-axis represents pixel height (0 to 600). The color scheme indicates regions of similar pixel values after K-means clustering, demonstrating the segmentation of the image based on pixel color similarities. The image appears in grayscale with reduced colors, emphasizing the clustering effect.


![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-20.jpg?height=741&width=1196&top_left_y=514&top_left_x=1947)

**Image Description:** The image depicts a diagram illustrating the process of flattening a multi-dimensional array, commonly used in image processing or neural networks. It features a 3D grid labeled with three axes: Height, Width, and Color Channel, representing an RGB image. The grid consists of colored squares, each indicating a pixel's color values. To the right, an arrow points to a 1D array labeled "Flatten," which shows the resulting linear structure derived from the grid, indicating a sequential arrangement of pixel values in a single vector format.



Pixels plotted in
RGB vector space.

## Visualize Pixel Clustering

## Demo

K-means on Pixels
1041260

## Hard Cluster Assignments

K-means assigns each data point to exactly one cluster.

Do all the points belong in exactly one cluster?

- Maybe? Each point represents one bike ride and therefore one bike.
- How could we at least measure the uncertainty in the predictions.


## Uncertainty

ML is ultimately about inference making predictions.

- Predictions are inherently uncertain.
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-23.jpg?height=554&width=732&top_left_y=178&top_left_x=2143)

**Image Description:** The image consists of two sections: on the left, a pixelated representation of a pullover garment; on the right, a bar chart displaying the predicted probabilities of various clothing categories. The y-axis lists clothing types (e.g., Trouser, T-shirt/top, Sneaker, Pullover, etc.), while the x-axis ranges from 0 to 0.6, indicating probability values. The bar for "Pullover" is highlighted in blue, showing a probability close to 0.6, indicating it is the most likely classification compared to other categories.




## Source of Uncertainty:

- Epistemic Uncertainty (Reducible) - the systematic uncertainty that arises from a finite training dataset and our modeling process.
- Aleatoric Uncertainty (Irreducible) - the uncertainty that arises from observational noise in our training data.

Need a framework for uncertainty - Probability!
-K-means Clustering

- Scikit-Learn
- Lloyd's Algorithm
- Pixel K-Means
- Probability (Review)

Probability (Review)

- Joint Distributions
- Wake Word Example

Questions

## Probability

Probability provides a framework for quantifying and manipulating uncertainty.
The probability of an event is:
-Frequentist View: The long-run relative frequency of an event in identical repeated trials.

- Bayesian View: The degree of belief (or plausibility) assigned to the event given the available information.

Which one is correct?
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-26.jpg?height=477&width=426&top_left_y=561&top_left_x=229)

**Image Description:** The image is a miscellaneous graphic, depicting a checklist or form. It features a rectangular frame with a rounded corner design, containing a checkmark icon and two lines representing fields or items. There is also a circular button at the bottom left, suggesting interaction or selection. The overall design is simple and modern, using a blue color scheme, making it visually clear for task completion or assessment purposes.


## How do you interpret probabilities?

A brief review of the Basics of Probability

## The Joint Probability Distribution

Let $X$ and $Y$ be discrete random variables that take values:

$$
X \in\left\{x_{1}, \ldots, x_{L}\right\} \text { and } Y \in\left\{y_{1}, \ldots, y_{M}\right\}
$$

The joint probability function $p\left(X=x_{i}, Y=y_{j}\right)$ is the probability of the event $X=x_{i}$ and $Y=y_{j}$.
The joint probability satisfies the following two properties:

$$
\begin{aligned}
& \forall x_{i}, y_{j}: p\left(X=x_{i}, Y=y_{j}\right) \geq 0 \text { (Non-negativity) } \\
& \sum_{i=1}^{L} \sum_{j=1}^{M} p\left(X=x_{i}, Y=y_{j}\right)=1 \text { (Normalization) }
\end{aligned}
$$

![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-28.jpg?height=741&width=1387&top_left_y=918&top_left_x=1892)

**Image Description:** The image depicts a joint probability distribution table \( p(X = x_i, Y = y_j) \) for discrete random variables \( X \) and \( Y \). The rows correspond to values of \( X \) (denoted as \( x_1, x_2, x_3 \)), while the columns represent values of \( Y \) (denoted as \( y_1, y_2 \)). Each cell contains the probability associated with the pairing of \( x_i \) and \( y_j \). The marginal probabilities for \( X \) and \( Y \) are indicated below the table, alongside a total sum of 1 in the bottom right corner, indicating the entire probability space.



## The Joint Probability Distribution

Let $X$ and $Y$ be discrete random variables that take values

$$
X \in\left\{x_{1}, \ldots, x_{L}\right\} \text { and } Y \in\left\{y_{1}, \ldots, y_{M}\right\}
$$

The joint probability function $p\left(X=x_{i}, Y=y_{j}\right)$ is the probability of the event $X=x_{i}$ and $Y=y_{j}$.
The Sum Rule (Marginalization): defines the distribution over a subset of the random variables.

$$
p\left(X=x_{i}\right)=\sum_{j=1}^{M} p\left(X=x_{i}, Y=y_{j}\right)
$$
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-29.jpg?height=584&width=1013&top_left_y=918&top_left_x=1888)

**Image Description:** The image presents a joint probability distribution table. The rows represent the variable \(X\) with values \(x_1, x_2, x_3\), while the columns represent the variable \(Y\) with values \(y_1, y_2\). Each cell contains the probability \(p(X = x_i, Y = y_j)\). The first row showcases the values of \(Y\) corresponding to \(x_1\) with probabilities 0.2 and 0.1, and subsequent rows show similar distributions for \(x_2\) and \(x_3\). The probabilities sum to 1 across all cells, indicating a valid joint probability distribution. 

The equation shown is: $$ p(X = x_i, Y = y_j) $$


**Image Description:** The image depicts a probability table illustrating the joint distribution of two discrete random variables, \(X\) and \(Y\). The rows represent outcomes of \(X\) (\(x_1, x_2, x_3\)), while the columns represent outcomes of \(Y\) (\(y_1, y_2\)). The table contains numerical values indicating the probabilities \(P(X = x_i, Y = y_j)\) for each combination of \(x_i\) and \(y_j\). The values are arranged with probability outcomes between 0 and 1, showcasing their distribution across combinations. The equation shown is as follows: 

$$ p(X = x_i, Y = y_j) $$

$$
p\left(Y=y_{i}\right)
$$

$$
p\left(X=x_{i}\right)
$$
0.3
0.2
0.5

1

## Conditional Probability

We define the conditional probability as the chance of observing $Y=y_{j}$ given we observed $X=x_{i}$

$$
p\left(Y=y_{j} \mid X=x_{i}\right)=\frac{p\left(X=x_{i}, Y=y_{j}\right)}{p\left(X=x_{i}\right)}
$$

## Example:

What is the $p\left(Y=y_{1} \mid X=x_{3}\right)$ ?
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-30.jpg?height=397&width=1587&top_left_y=1326&top_left_x=144)

**Image Description:** The image features a diagram and an equation. The left side shows a segmented bar chart with three regions labeled \( x_3 \), \( 0.15 \), and \( 0.35 \). A horizontal line indicates a normalization point, which sums to \( 0.5 \). The right side presents an equation formatted as \( \frac{p(Y = y_i | X = x_3)}{0.5} = 0.3 + 0.7 \), indicating conditional probabilities associated with variable \( Y \) given \( X \). The components are highlighted with blue and green boxes for emphasis.


1. The left section shows a segmented bar representing the values of $x_3$, with segments labeled 0.15 (orange) and 0.35 (gray), summing to 0.50 (light pink) at the bottom as the total probability. 

2. The equation on the right presents conditional probabilities: 
$$ p(Y = y_i | X = x_3) = 0.3, 0.7 $$ 
indicating the outcomes corresponding to event $Y$ given $X$.

![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-30.jpg?height=754&width=1392&top_left_y=1058&top_left_x=1879)

**Image Description:** The image contains a joint probability distribution table for two random variables \( X \) and \( Y \). The rows represent values \( x_1, x_2 \) of variable \( X \), while the columns represent values \( y_1, y_2 \) of variable \( Y \). Each cell displays the probability \( p(X = x_i, Y = y_j) \). Marginal probabilities for \( X \) and \( Y \) are indicated at the bottom and right margins, respectively. The highlighted central row indicates the marginal distribution of \( Y \).



## Empirical Probability Distributions

Consider $N$ trials $\left\{\left(X_{1}, Y_{1}\right), \ldots,\left(X_{N}, Y_{N}\right)\right\}$ in which we sample ( $X, Y$ ).
Let $n_{i j}$ be the number (count) of trials where ( $X=x_{i}, Y=y_{j}$ ), we can construct a discrete empirical probability distribution:

$$
\hat{p}\left(X=x_{i}, Y=y_{j}\right)=\frac{n_{i j}}{N}
$$

Frequentist View: In the limit as $N \rightarrow \infty$, the empirical probability distribution converges to the true probability distribution:

$$
p\left(X=x_{i}, Y=y_{j}\right)=\lim _{N \rightarrow \infty} \hat{p}\left(X=x_{i}, Y=y_{j}\right)
$$

## The Product Rule of Probability

Recall the definition of the conditional:

$$
p\left(Y=y_{j} \mid X=x_{i}\right)=\frac{p\left(X=x_{i}, Y=y_{j}\right)}{p\left(X=x_{i}\right)}
$$

Multiply both sides by $p\left(X=x_{i}\right)$ you obtain the product rule:

$$
p\left(X=x_{i}, Y=y_{j}\right)=p\left(Y=y_{j} \mid X=x_{i}\right) p\left(X=x_{i}\right)
$$

The product rule is like a chain rule of probability - it allows us to factorize joint probability distributions into conditionals:

$$
p(X, Y, Z)=p(Z \mid X, Y) p(X \mid Y) p(Y)
$$
- Notation: we often drop $x_{i}, y_{j}$, and $z_{k}$ for convenience.

## Independent Random Variables

iwo variables $X$ and $Y$ are independent if the joint probability factorizes:

$$
X \perp Y \Rightarrow \boldsymbol{p}(\boldsymbol{X}, \boldsymbol{Y})=\boldsymbol{p}(\boldsymbol{X}) \boldsymbol{p}(\boldsymbol{Y})
$$

Using the definition of conditionals:

$$
X \perp Y \Rightarrow \boldsymbol{p}(\boldsymbol{Y} \mid \boldsymbol{X})=\frac{p(Y, X)}{p(X)}=\frac{p(Y) p(X)}{p(X)}=\boldsymbol{p}(\boldsymbol{Y})
$$

Independent and Identically Distributed (IID) data ( $X_{i}, Y_{i}$ ) is:

$$
p\left(X_{1}, Y_{1}, \ldots, X_{N}, Y_{N}\right)=\prod_{n=1}^{N} p\left(X_{n}, Y_{n}\right)
$$
- We will often assume IID data (though it is often not true in practice).
-K-means Clustering
- Scikit-Learn
- Lloyd's Algorithm
- Pixel K-Means
- Probability (Review)

Wake Word Example

- Joint Distributions
- Wake Word Example

Questions

## Example: Wake Words

A wake word is a verbal cue that triggers voice assistants to start actively listening.

Example: "Alexa, set an alarm for 6:00AM"
Most voice assistants continuously run a wake word detector model on every sound they hear.
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-35.jpg?height=689&width=2514&top_left_y=1080&top_left_x=464)

**Image Description:** The image is a waveform diagram illustrating audio data over the span of one day, indicated by the horizontal axis labeled "1 day." The vertical amplitude of the waveform represents sound intensity, showing fluctuations in sound levels. Key segments are highlighted with orange arrows, pointing to areas identified as "Rare wake word events," denoting specific moments of heightened audio activity. The overall layout suggests a focus on analyzing occurrences of wake words within a continuous audio recording.



## Example: Wake Words

A wake word is a verbal cue that triggers voice assistants to start actively listening.

Example: "Alexa, set an alarm for 6:00AM"
Most voice assistants continuously run a wake word detector model on every sound they hear.
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-36.jpg?height=537&width=800&top_left_y=1143&top_left_x=0)

**Image Description:** The image is a waveform diagram depicting audio signal patterns over time. The horizontal axis represents time, while the vertical axis indicates amplitude levels of the sound signal. The peaks and troughs illustrate variations in sound intensity, with denser sections indicating louder sound and sparser sections indicating quieter sound. The overall shape of the waveform provides insight into the characteristics of the audio content, such as rhythm and volume dynamics.



Streamed to the cloud for processing.
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-36.jpg?height=736&width=704&top_left_y=0&top_left_x=2346)

**Image Description:** The image features a compact, spherical smart speaker designed for voice-activated functionality. It has a textured upper body made of a gray mesh material, housing microphones and speakers, while a smooth blue band wraps around the base. A blue light ring is present at the bottom, indicating active status. This device is likely associated with smart home integration and audio playback capabilities.



## Example: Wake Words

## Streamed to the cloud for processing.

:'low good does the wake word detector model need to be?

- To answer this question, we can model the joint probability of the wake word detector on an audio segment.

Before we listen to the audio segment, there are two random variables:

1. Was a wake word said: $X \in\{0,1\}$ ( 0 : no, 1 : yes)
2. Was a wake word detected: $Y \in\{0,1\}$ ( 0 : negative, 1 : positive)

## Analyzing the Wake Word Detector

iet's assume wake words are rare occurring less than $0.01 \%$ of segments:

$$
p(X=1)=0.0001 \text { and } p(X=0)=1-p(X=1)=0.9999
$$

Suppose the wake word model detects $\mathbf{9 9 \%}$ of the wake words:

$$
p(Y=1 \mid X=1)=0.99
$$

And falsely detects (false positive) a wake word just $0.1 \%$ of the time:

$$
p(Y=1 \mid X=0)=0.001
$$

If the detector is positive, what is the chance that it was a wake word?

$$
p(X=1 \mid Y=1)=\frac{p(Y=1, X=1)}{p(Y=1)}=\frac{p(Y=1 \mid X=1) p(X=1)}{p(Y=1)}
$$

## Bayes' Theorem

Named after Thomas Bayes who was an $18^{\text {th }}$ century statistician, philosopher, and minister.

$$
p(A \mid B)=\frac{p(B \mid A) p(A)}{p(B)}=\frac{p(B \mid A) p(A)}{\sum_{A} p(B \mid A) p(A)}
$$
- Derived from product rule: $p(A, B)=p(A \mid B) p(B)=p(B \mid A) p(A)$

Useful in Bayesian statistics:

- $p(A)$ is the prior belief about $A$
- $p(B \mid A)$ is the likelihood of an observation $B$ given those beliefs
- $p(A \mid B)$ is our posterior belief about $A$ given our observation $B$


## Analyzing the Wake Word Detector

'Ne assumed rare wake words: $p(X=1)=0.0001$
A true positive rate (recall) of $99 \%: p(Y=1 \mid X=1)=0.99$
And a false positive rate of $0.1 \%: p(Y=1 \mid X=0)=0.001$
If the detector is positive, what is the chance that it was a wake word?

$$
\begin{gathered}
p(X=1 \mid Y=1)=\frac{p(Y=1 \mid X=1) p(X=1)}{p(Y=1 \mid X=1) p(X=1)+p(Y=1 \mid X=0) p(X=0)} \\
p(X=1 \mid Y=1)=\frac{0.99 * 0.0001}{0.99 * 0.0001+0.001 * 0.9999}=0.09=9 \%
\end{gathered}
$$

Is that good (enough)?
How would we improve it?
![](https://cdn.mathpix.com/cropped/2025_10_01_3f05b45f008d21709034g-41.jpg?height=452&width=485&top_left_y=578&top_left_x=204)

**Image Description:** This image is a graphic representation of a speech bubble. It features a rounded rectangular shape with a pointed tail at the bottom, indicating the source of the message. The inner area contains two horizontal lines representing text or dialogue, while the outer border is a solid color, framing the bubble. The overall design conveys communication or conversation, often used in contexts related to messaging, dialogue, or social interaction in academic lectures.



## How could we improve the Wake Word Detector?

## Demo

Analysis of Recall and False Positive Rates

## Bayesian Updates: Wake Word Detectợs,

Prior: we assumed rare wake words $p(X=1)=0.0001$
After observing the test $Y=1$ we updated our belief about the presence of the wake word using Bayes Rule:

$$
p(X=1 \mid Y=1)=\frac{p(Y=1, X=1)}{p(Y=1)}=\frac{p(Y=1 \mid X=1) p(X=1)}{p(Y=1)}=0.09
$$

Posterior: the updated belief $p(X=1 \mid Y=1)$ after the observation.
Bayesian Updating: the process of starting from a prior belief and then using observations to compute posterior distribution (updated belief) is central to Bayesian machine learning.

## Lecture 4

## K-Means and Probability

Credit: Joseph E. Gonzalez and Narges Norouzi
Reference Book Chapters:

- Clustering: Chapter 15.1 (k-means)
- Probability: Chapter 2.[1-2]

