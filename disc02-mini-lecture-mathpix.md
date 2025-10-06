---
course: CS 189
semester: Fall 2025
type: discussion
title: Discussion 2
source_type: slides
source_file: Discussion Mini Lecture 02.pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Discussion Mini Lecture 2

# Machine Learning Design 

The Supervised Learning Design Process

CS 189/289A, Fall 2025 @ UC Berkeley<br>Sara Pohland

## Machine Learning Lifecycle

![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-02.jpg?height=1481&width=3114&top_left_y=327&top_left_x=89)

**Image Description:** The image is a flowchart illustrating a data science process. It consists of four quadrants labeled: (1) "LEARNING PROBLEM," (2) "MODEL DESIGN," (3) "OPTIMIZATION," and (4) "PREDICT & EVALUATE." Each quadrant contains guiding questions related to its focus area. The flowchart features arrows connecting the quadrants, indicating a cyclical process. Icons representing relevant concepts accompany the text in each quadrant, enhancing understanding through visual cues. The color scheme is yellow and blue, providing a clear distinction between different parts of the process.


## Concepts Covered

1. Data Pre-processing
2. Model Design
a) Feature Engineering
b) Model Families
c) Design Considerations
3. Fitting a Model
4. Making a Prediction

## Understanding \& Preparing our Data

1. Data Pre-processing
2. Model Design
a) Feature Engineering
b) Model Families
c) Design Considerations
3. Fitting a Model
4. Making a Prediction

## Desired Form of Data

What do we want our data to look like for training?

Dataset: $\mathcal{D}=\left\{\left(x_{n}, y_{n}\right)\right\}_{n=1}^{N}$
$N=$ number of samples
Input $\quad x_{n} \in \mathbb{R}^{D}$
${ }^{\text {i }}$ D = number of features
Output $y_{n} \in \mathbb{R}$
:
' $\longrightarrow$ May not be available

Design
$\stackrel{\text { Matrix: }}{-x_{1}^{\prime}}-\left[\begin{array}{ccc}x_{11} & \cdots & x_{1 D} \\ \vdots & \ddots & \vdots \\ -x_{N}^{T} & -\end{array}\right] \in \mathbb{R}^{N \times D}$
Target
Vector:
$Y=\left[\begin{array}{c}\vdots \\ y_{N}\end{array}\right] \in \mathbb{R}^{N}$

## Starting Form of Data

## What does our data initially look like?

![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-06.jpg?height=443&width=592&top_left_y=650&top_left_x=425)

**Image Description:** This image is a miscellaneous depiction of a puppy and a kitten positioned in a grassy area adorned with pink flowers. The golden retriever puppy is sitting on the left, displaying a soft, fluffy coat and a friendly expression. Beside it, a gray tabby kitten is nestled closely, appearing calm and content. The background features vibrant green grass and colorful blossoms, creating a serene and cheerful environment. The overall composition conveys a sense of companionship and playfulness between the two animals.

images

HI, It's your boss. Im stuck in Nigeria with none money. Please wire to TRWIGB2LXXX SOON. text

![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-06.jpg?height=244&width=728&top_left_y=756&top_left_x=2313)

**Image Description:** This image is a table presenting real estate data. It includes three columns labeled "# Bed," "# Bath," and "Location." The row displayed lists numerical values: 4 for the number of bedrooms and 3 for the number of bathrooms, with "Berkeley" indicating the location. The table is formatted with distinct blue headers and gray content cells, likely for clarity in data presentation.

tabular

![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-06.jpg?height=341&width=405&top_left_y=1318&top_left_x=123)

**Image Description:** The image features a graphic representation of a speaker symbol, commonly associated with audio output. The speaker icon is designed with a solid blue outline on the left, showcasing a triangular speaker shape, while emanating curved sound waves depicted by three vertical arcs to the right. The sound waves are progressively larger, indicating the propagation of sound. This image is often used in contexts related to audio, sound systems, or volume control within multimedia presentations.

audio

S\&P 500 Index Historical Chart
![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-06.jpg?height=332&width=712&top_left_y=1348&top_left_x=773)

**Image Description:** The image is a line graph depicting data over time. The x-axis represents "Year," ranging from 1960 to 2020, while the y-axis represents a numerical value, likely indicative of growth or another quantitative measure, ranging from 0 to 4000. The plot exhibits an increasing trend, with a notable surge in values after the year 2000, suggesting significant growth or change during that period. The line shows fluctuations prior to this surge, indicating a gradual increase before a more rapid ascent.


time series

![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-06.jpg?height=460&width=558&top_left_y=1263&top_left_x=1684)

**Image Description:** The image depicts an office or study setting with a gray cabinet featuring three drawers. One drawer is partially open, revealing a can (possibly a drink) placed on a white interior surface. On the adjacent surface, there is a red apple and two cans of energy drink. The environment includes items commonly found in a workspace, but there are no diagrams or equations present. The focus is on the arrangement of objects within a practical context.

video

![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-06.jpg?height=337&width=920&top_left_y=1301&top_left_x=2355)

**Image Description:** The image depicts a stylized representation of DNA, illustrating a double helix structure. The helical backbone is highlighted in blue, with individual nucleotides represented as colored base pairs (adenine in green, thymine in red, cytosine in blue, and guanine in black). At one end of the image, there is a depiction of a chromosome, with scissors symbolizing DNA cutting or editing, possibly referring to genetic engineering techniques. The sequence of nucleotides is displayed at the bottom, emphasizing genetic information.

genomic

## 1) Investigate the Data

Answer the following about your data:

- How many data samples do I have?
-What type of data am I working with?
-What does my data look like?
- Is my data labeled?

If you have categorical labels:

- How many unique labels are there?
- How many samples do you have from each class?

If you have quantitative labels:
-What is the distribution of the target labels?

## 2) Clean up the Data

## For categorical features:

- Remove data with missing values or use mode imputation
- Use one-hot-encoding (see lecture 3)
- This avoids artificial ordering that comes from simply converting a categorical features to a numeric value!


## For numerical features:

- Remove data with missing values or use median imputation
- Normalize each feature (see lecture 3)
- This avoids placing a disproportionate weight on features with larger magnitudes and improves numerical stability!
- Consider removing highly correlated variables


## 2) Clean up the Data

## For text:

- Remove punctuation, special characters, stop words, etc.


## For images:

- Reshape images to a fixed size
- Normalize pixel values


## For all data types:

- Remove "bad" data (low-quality images, anomalies, etc.)
- Consider removing features that we do not expect/want to help us predict our label/target
- Features with poor predictive power can introduce unnecessary variance!


## 3) Divide Data into Three Sets

![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-10.jpg?height=1141&width=2008&top_left_y=518&top_left_x=46)

**Image Description:** The image illustrates a data partitioning process used in machine learning, represented by a flow diagram. It shows a blue vertical bar labeled "Data," which is split into two segments: "Train" (80%) and "Test" (20%). The "Train" segment is further divided into "Train" (75%) and "Val." (Validation, 25%). Arrows indicate the flow from "Data" to "Train" and further to "Val." and "Test," illustrating the sequential process of dividing data for model training and evaluation purposes.


Train: ~60\% of dseldato fit the model during training
Val.: ~20\% of data
Used to select model params. during training
Test: ~20\% of dsedato estimate performance posttraining
*These are commonly used splits, but you can adjust these percentages based on your data.

## Choosing/ Learning Good Features

## 1. Data Pre-processing

2. Model Design
a) Feature Engineering
b) Model Families
c) Design Considerations
3. Fitting a Model
4. Making a Prediction

## Generating Features

## For text data:

- Use a Bag-of-Words model
- i.e., Count the presence of words in a vocabulary (see lecture 3)
- Usually stop words (e.g., the, is, of...) that contain minimal info are dropped
- Count the use of punctuation marks
- Consider text length as a feature
- Look for specific patterns in text
- e.g., Presence of URLs, typos, all caps words, etc.


## Generating Features

## For image data:

- Generate hand-crafted features
- e.g., Use edge detectors to count presence of edges, texture descriptors to identify patterns, or CV tools to measure image properties (e.g., brightness)
- Not common today but could be used to design more interpretable models.
- Find deep learning representations
- e.g., Use (some component of) a pretrained model (ResNet, CLIP, etc.) to generate a feature vector that can be used for their ML task.
- e.g., Use an object detection model to detect the presence of objects that might be relevant to your larger ML task.


## Generating Features

## For tabular data:

-Derive features from existing ones

- e.g., We want to predict worker burnout for staff members who started working in different weeks. Among other things, we're given the total number of hours each worked and total number of weeks. We also might want to derive the average number of hours worked per week.


## For any data:

- Use domain knowledge to identify what information might be useful in predicting the thing we want.


## Selecting a Model Family

1. Data Pre-processing
2. Model Design
a) Feature Engineering
b) Model Families
c) Design Considerations
3. Fitting a Model
4. Making a Prediction

## Determining the ML Paradigm

![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-16.jpg?height=1498&width=3292&top_left_y=361&top_left_x=25)

**Image Description:** The image presents a flowchart categorizing types of machine learning. At the top, a decision node asks, “Do I have observations associated with me?” leading to “Yes” (rewards) for Reinforcement Learning and “No” for further classification. This splits into Supervised Learning (associated with labels) and Unsupervised Learning. It further branches under Supervised Learning into "Categorical" (Classification) and "Quantitative" (Regression), while Unsupervised Learning categorizes into "Group data" (Clustering) and "Reduce features" (Dimensionality Reduction). The flowchart is visually organized with labeled boxes and arrows indicating pathways.


## Examples of ML Problems

1. A botanist gives images of plants and tells me what they are. When I see a plant in the wild, I want to label it myself.
![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-17.jpg?height=243&width=2050&top_left_y=595&top_left_x=340)

**Image Description:** The image depicts a linear diagram illustrating a process or flow from "Categorical labels" on the left to "Classification" on the right. An arrow points from the source label to the classification, indicating a directional relationship. The text is prominently displayed in a bold, clear font, emphasizing the transformation or mapping from categorical labels to a classification system in a conceptual framework.

2. Before testing medical interventions, I want to group together patients that have similar medical backgrounds.
![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-17.jpg?height=154&width=1787&top_left_y=1088&top_left_x=340)

**Image Description:** The image is a diagram illustrating the concept of clustering in data analysis. It features a horizontal arrow that transitions from "Grouping data" on the left to "Clustering" on the right. The diagram emphasizes the relationship between the two concepts, suggesting that clustering is a method of grouping data points based on similarity. The arrow is the focal point, highlighting the direction and connection between the terms, and is presented in a bold, clear font to ensure legibility.

3. I want my robot to fold my laundry but don't know how to encode this. I reward it for every article of clothing it successfully folds.
![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-17.jpg?height=230&width=1736&top_left_y=1569&top_left_x=344)

**Image Description:** The image presents a flow diagram illustrating the concept of "Reinforcement Learning." It features a directional arrow pointing from "Rewards" to "Reinforcement Learning." The axis is not explicitly defined with numerical scales, as the focus is on the relationship between the concept of rewards and reinforcement learning. The visual emphasizes the significance of rewards in the reinforcement learning process, indicating that rewards are a driving factor for learning and decision-making in this context. The color scheme is a single blue font on a white background, ensuring clarity and straightforward comprehension.


## Examples of ML Problems

4. You give me a list of ratings for all of the movies you've watched. I want to predict how you want to rate a movie you haven't seen.
![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-18.jpg?height=243&width=1978&top_left_y=595&top_left_x=340)

**Image Description:** The image is a simple directional diagram illustrating a process. It features a horizontal arrow pointing towards the right, labeled "Regression," and a vertical arrow pointing upwards with the label "Quantitative labels." The arrows indicate a relationship or flow between the two concepts, suggesting that quantitative labels may lead to or inform regression analysis. The diagram is minimalistic, focusing on key terms without additional data or graph axes.

5. I have a table of data with 10,000 columns. I want to remove redundant features before training my model.

![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-18.jpg?height=141&width=333&top_left_y=1080&top_left_x=344)

**Image Description:** The image depicts a step function graph. The horizontal axis represents the independent variable, while the vertical axis shows the dependent variable. The graph starts at a value along the vertical axis and abruptly jumps to a higher value at a specific point on the horizontal axis, then remains constant thereafter. The arrow at the right signifies that the function continues indefinitely in that direction. This visual representation is commonly used in mathematical contexts to illustrate discontinuous functions.

Reducing features

![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-18.jpg?height=188&width=1013&top_left_y=1131&top_left_x=1433)

**Image Description:** The image features an arrow pointing to the right, accompanied by the text "Dimensionality Reduction." The arrow suggests a directional flow or process, indicating a transition towards the concept of dimensionality reduction, which is often used in data analysis and machine learning to simplify datasets by reducing the number of dimensions while retaining essential information. The text is presented in a bold, blue font, emphasizing the importance of the topic.


## Regression Model Families

Simpl
Comple

Linear Regression
(Lecture 5)

Neural Networks
(Lecture 12)

## Classification Model Families

![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-20.jpg?height=851&width=3212&top_left_y=527&top_left_x=59)

**Image Description:** The image is a horizontal gradient bar representing a continuum from "Simple" to "Complex." The left end is labeled "Logistic Regression (Lecture 7)" and the right end is labeled "Neural Networks (Lecture 12)." The bar transitions from a dark blue color on the left to a dark orange color on the right, indicating increasing complexity. An arrow spans the gradient, emphasizing the shift from simpler to more complex models in machine learning.


## Interplay of Features and Model Families

## 1. Data Pre-processing

2. Model Design
a) Feature Engineering
b) Model Families
c) Design Considerations
3. Fitting a Model
4. Making a Prediction

## Selecting Features \& Model Families

## Key Considerations:

- Machine learning is full of trade-offs.
- More features provide more information to our model, but they introduce additional variance, so we're more likely to capture spurious data patterns.
- More complex models can capture more complex input-output relationships, but they're more likely to overfit to our training set.
- Larger datasets can help us learn more generalizable models (assuming our dataset is representative of our population), but they require more memory and computational resources to train with them.
- Engineering new features and increasing model complexity can improve accuracy, but this may also reduce interpretability.


## Selecting Features \& Model Families

## Key Considerations:

- Feature engineering and model selection are complementary.
- If we have a simple model (e.g., a linear classifier), we probably want to choose/learn some new, useful features.
- If we have a complex model (e.g., a neural network), we may want to work with our original, unmodified features.
- Domain knowledge is critical for training/deploying ML models.
-When in doubt, run validation to evaluate your options.


## Learning Model Parameters

1. Data Pre-processing
2. Model Design
a) Feature Engineering
b) Model Families
c) Design Considerations
3. Fitting a Model
4. Making a Prediction

## Scikit-learn is Your Friend!

![](https://cdn.mathpix.com/cropped/2025_10_01_3694e56d33a7904289a3g-25.jpg?height=1493&width=2366&top_left_y=382&top_left_x=484)

**Image Description:** The image is a flowchart illustrating a cheat sheet for scikit-learn algorithms. It features distinct color-coded sections for "Classification," "Regression," "Clustering," and "Dimensionality Reduction." Each section contains key algorithm names and branches based on specific criteria, such as labeled data and output types. Arrows indicate pathways leading to different algorithms based on decision points, like "looking" for features and the nature of the output (e.g., quantity vs. category). The diagram serves to guide users in selecting appropriate machine learning techniques within the scikit-learn library.


## Using Scikit-learn to Fit a Model

- Scikit-learn provides dozens of built-in machine learning algorithms and models, called estimators.
- Each estimator can be fitted to some data using its fit method.

```
>>> from sklearn.ensemble import RandomForestClassifier
>>> clf = RandomForestClassifier(random_state=0)
>>> X = [[ 1, 2, 3], # 2 samples, 3 features
    [11, 12, 13]]
>>> y = [0, 1] # classes of each sample
>>> clf.fit(X, y)
```

Live, love, learn the fit method

## Using Scikit-learn to Fit a Model

The fit method generally accepts 2 inputs:

- The $N \times D$ design matrix, $X$, where samples are represented as rows and features are represented as columns (as in our notation).
- The $N$ target values, $y$, which are real numbers for regression tasks and a discrete set of values for classification. It is usually a 1D array where the $n$th entry corresponds to the target of the $n$th sample (row) of $X$. (For unsupervised learning tasks, $y$ does not need to be specified.)
- Both $X$ and $y$ are usually expected to be NumPy arrays or equivalent array-like data types.


## Making Model Predictions

1. Data Pre-processing
2. Model Design
a) Feature Engineering
b) Model Families
c) Design Considerations
3. Fitting a Model
4. Making a Prediction

## Using Scikit-learn to Make Predictions

- After using Scikit-learn to fit an estimator, it can be used for predicting target values of new data.

```
>>> clf.predict(X) # predict classes of the training data
array([0, 1])
>>> clf.predict [[4, 5, 6], [14, 15, 16]]) # predict classes of new data
array([0, 1])
With fit comes its forever +1 , predict
```


## Discussion Mini Lecture 2

## Machine Learning Design

Contributors: Sara Pohland

## Additional Resources

1. Fitting a model and making predictions

- Getting Started with Scikit-learn

2. We'll go deeper into other things later in the semester!
