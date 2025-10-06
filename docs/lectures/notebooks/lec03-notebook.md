---
course: CS 189
semester: Fall 2025
type: lecture
number: 3
title: Lecture 3
source_type: notebook
source_file: lec03.ipynb
processed_date: '2025-09-30'
---



<h1 class="cal cal-h1">Lecture 03 – CS 189, Fall 2025</h1>

In this lecture, we will cover the basic machine learning lifecycle using a hands-on approach with scikit-learn.
We will work through each stage of the machine learning lifecycle while also introducing standard machine learning tools and techniques.  The machine learning lifecycle consists of four parts:

<div style="text-align: center;">
<img src="https://eecs189.org/fa25/resources/assets/lectures/lec03/images/ml_lifecycle.png" alt="drawing" width="600"/>
</div>

```python
import numpy as np 
import pandas as pd
import plotly.express as px
```


<h2 class="cal cal-h2">The Learning Problem</h2>

Suppose we are launching a new fashion trading website where people can upload pictures of clothing they want to trade. We want to help posters identify the clothing in the images. Suppose we have some training data consisting of clothing pictures with labels describing the type of clothing (e.g., "dress", "shirt", "pants").

**What data do we have?**
* Labeled training examples.

**What do we want to predict?**
* The category label of the clothing in the images.  We may want to predict other things as well.

**How would we evaluate success?**
* We likely want to measure our prediction accuracy. 
* We may eventually want to improve accuracy on certain high-value classes.





<h3 class="cal cal-h3">Looking at the Data</h3>

A key step that is often overlooked in machine learning projects is understanding the data. This includes exploring the dataset, visualizing the data, and gaining insights into its structure and characteristics.

We will be using the Fashion-MNIST dataset, which is a (now) classic dataset with gray scale 28x28 images of articles of clothing.

>[Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.](https://arxiv.org/abs/1708.07747) Han Xiao, Kashif Rasul, Roland Vollgraf.
> https://github.com/zalandoresearch/fashion-mnist

This is an alternative to the even more classic MNIST digits dataset, which contains images of handwritten digits. 
The following block of code will download the Fashion-MNIST dataset and load it into memory. 
```python
# Fetch the Data
import torchvision
data = torchvision.datasets.FashionMNIST(root='data', train=True, download=True)

# Preprocess the data into numpy arrays
images = data.data.numpy().astype(float)
targets = data.targets.numpy() # integer encoding of class labels
class_dict = {i:class_name for i,class_name in enumerate(data.classes)}
labels = np.array([class_dict[t] for t in targets]) # raw class labels
n = len(images)

print("Loaded FashionMNIST dataset with {} samples.".format(n))
print("Classes: {}".format(class_dict))
print("Image shape: {}".format(images[0].shape))
print("Image dtype: {}".format(images[0].dtype))
print("Image 0:\n", images[0])
```



<h4 class="cal cal-h4">Understanding The Raw Features (Images)</h4>
How much data do we have?
```python
images.shape
```
The images are stored in a 60000 by 28 by 28 tensor.  This means we have 60000 images, each of which is 28 pixels wide and 28 pixels tall.  Each pixel is represented by a single value.  What are those values?

```python
counts, bins =  np.histogram(images, bins=255)
fig_pixels = px.bar(x=bins[1:], y=counts,  title="Pixel value distribution", 
       log_y=True, labels={"x":"Pixel value", "y":"Count"})
fig_pixels
```
It is important to learn how to visualize and work with data.  Here we use Plotly express to visualize the image. Note, I am using the 'gray_r' color map to visualize the images, which is a gray scale color map that is reversed (so that black is 1 and white is 0). 
```python
px.imshow(images[0], color_continuous_scale='gray_r') 
```
The following snippet of code visualizes multiple images in a grid.  You are not required to understand this code, but it is useful to know how to visualize images in Python. 

```python
def show_images(images, max_images=40, ncols=5, labels = None):
    """Visualize a subset of images from the dataset.
    Args:
        images (np.ndarray): Array of images to visualize [img,row,col].
        max_images (int): Maximum number of images to display.
        ncols (int): Number of columns in the grid.
        labels (np.ndarray, optional): Labels for the images, used for facet titles.
    Returns:
        plotly.graph_objects.Figure: A Plotly figure object containing the images.
    """
    n = min(images.shape[0], max_images) # number of images to show
    px_height = 220 # height of each image in pixels
    fig = px.imshow(images[:n, :, :], color_continuous_scale='gray_r', 
                    facet_col = 0, facet_col_wrap=ncols,
                    height = px_height * int(np.ceil(n/ncols)))
    fig.update_layout(coloraxis_showscale=False)
    if labels is not None:
        # Extract the facet number and replace with the label.
        fig.for_each_annotation(lambda a: a.update(text=labels[int(a.text.split("=")[-1])]))
    return fig
```
```python
show_images(images, 20, labels=labels)
```
Let's look at a few examples of each class. Here we use Pandas to group images by their labels and sample 2 for each class. You are not required to know Pandas (we won't test you on it), but it is a useful library for data manipulation and analysis and we will use it often in this course.
```python
idx = (
    pd.DataFrame({"labels": labels})
      .groupby("labels", as_index=False)
      .sample(2)
      .index
      .to_numpy())
show_images(images[idx,:,:], labels=labels[idx])
```



<h4 class="cal cal-h4">Understanding the Labels</h4>

New let's examine the labels.  Are they discrete?  What is the distribution? Are there missing values or errors?

In the Fashion-MNIST dataset, each image is labeled with a class corresponding to a type of clothing. There are 10 classes in total. 

However, it is also important to understand the distribution of labels in the dataset. This can help us identify potential issues such as class imbalance, where some classes have significantly more samples than others.
```python
labels
```
The labels are strings (discrete).
What is the distribution of labels?
```python
px.histogram(labels, title="Label distribution")   
```
There appear to be equal proportion of each type of clothing.  We don't have any missing values since all labels are one of the 10 classes (no blank or "missing" label values).

Most real world datasets aren't this balanced or clean.  In fact, it's common to see a long tail distribution, where a few classes are very common and many classes are rare.  



<h3 class="cal cal-h3">Reviewing the Learning Setting</h3>

Having examined the data we can see that we have a large collection of pairs of features and categorical labels (with 10 classes). This is a **supervised learning** problem, where the goal is to learn a mapping from the input features (images) to the output labels (categories). Because the labels are discrete this is a **classification** problem. 

It is also worth noting that because the input features are images this is also a **computer vision** problem.  This means that when we get to the model development stage, we will need to consider techniques that are specifically designed for multi-class classification and in particular computer vision.
<br><br><br>

---

**Return to Slides**

---

<br><br><br>



<h3 class="cal cal-h3">Train-Test-Validation Split</h3>

We will split the dataset into a training set, a validation set, and a test set. The training set will be used to train the model, while the validation set will be used to tune the model's hyperparameters. The test set will be used to evaluate the model's performance. 

Technically, the Fashion-MNIST dataset has a separate test set, but we will demonstrate how to split data in general.
```python
# use sklearn to construct a train test split
from sklearn.model_selection import train_test_split
```
```python
# Construct the train - test split
images_tr, images_te, labels_tr, labels_te = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# Construct the train - validation split
images_tr, images_val, labels_tr, labels_val = train_test_split(
    images_tr, labels_tr, test_size=0.2, random_state=42)

print("images_tr shape:", images_tr.shape)
print("images_val shape:", images_val.shape)
print("images_te shape:", images_te.shape)
```
<br><br><br>

---

**Return to Slides**

---

<br><br><br>



<h2 class="cal cal-h2"> Model Design </h2>

We have already loaded the Fashion-MNIST dataset in the previous section. Now, we will preprocess the data to make it suitable for training a classification model. 






<h3 class="cal cal-h3">Feature Engineering</h3>

**Feature Engineering** is the process of transforming the raw features into a representation that can be used effectively by machine learning techniques.  This will often involve transforming data into vector representations. 
<h4 class="cal cal-h4">Naive Image Featurization</h4>

For this example we are working with low-resolution grayscale image data. Here we adopt a very simple featurization approach -- flatten the image. We will convert the 28x28 pixel images into 784-dimensional vectors.
```python
images_tr.shape
```
```python
def flatten(images):
    return images.reshape(images.shape[0], -1)
```
```python
X_tr = flatten(images_tr)
```
```python
X_tr.shape
```


<h4 class="cal cal-h4">Standardization</h4>

Recall that the pixel intensities are from 0 to 255:

```python
fig_pixels
```
Let's standardize the pixel intensities to have zero mean and unit variance.

Here we use the sklearn StandardScaler
```python
from sklearn.preprocessing import StandardScaler

# 1. Initialize a StandardScaler object
image_scaler = StandardScaler()

# 2. Fit the scaler
image_scaler.fit(flatten(images_tr))
```
What do the mean and variance images tell us about the dataset?
```python
display(px.imshow(image_scaler.mean_.reshape(28,28), 
                  color_continuous_scale='gray_r', title="Mean image"))
display(px.imshow(image_scaler.var_.reshape(28,28), 
                  color_continuous_scale='gray_r', title="Variance image"))
```
Let's create a generic featurization function that we can reuse for different datasets.  Notice that this function uses the image_scaler that we fit to the training data.

```python
def featurizer(images):
    flattened = flatten(images)
    return image_scaler.transform(flattened)

X_tr = featurizer(images_tr)
```
Our new images look similar to the original images but they have been standardized to have zero mean and unit variance. This should help improve the performance of our machine learning models.
```python
show_images(X_tr.reshape(images_tr.shape), max_images=10, labels=labels_tr)
```


<h4 class="cal cal-h4">One-Hot Encoding</h4>

We don't need to one-hot encode the features in this dataset so we will briefly demonstrate on another dataset:
```python
df = pd.DataFrame({"color": ["red", "green", "red", "blue", "blue", "yellow", ""]})
df
```
```python
from sklearn.preprocessing import OneHotEncoder
# 1. Initialize a OneHotEncoder object
ohe = OneHotEncoder()
# 2. Fit the encoder
ohe.fit(df[["color"]])
```
```python
ohe.categories_
```
```python
ohe.transform(df[["color"]]).toarray()
ohe.categories_
```


<h4 class="cal cal-h4">Bag of Words</h4>

We also don't need the bag-of-words representation for this dataset, but we will demonstrate it briefly using another dataset.
```python
df['text'] = [
    "Red is a color.",
    "Green is for green food.",
    "Red reminds me of red food.",
    "Blue is my favorite color!",
    "Blue is for Cal!",
    "Yellow is also for Cal!",
    "I forgot to write something."
]
```
```python
from sklearn.feature_extraction.text import CountVectorizer

# 1. Initialize a CountVectorizer object
vectorizer = CountVectorizer()

# 2. Fit the vectorizer
vectorizer.fit(df["text"])

```
```python
pd.DataFrame(vectorizer.transform(df["text"]).toarray(), 
             columns=vectorizer.get_feature_names_out())
```
<br><br><br>

---

**Return to Slides**

---

<br><br><br>





<h2 class="cal cal-h2">Modeling and Optimization</h2>

In this section, we will go through the modeling process. We will focus on developing a classification model.



<h3 class="cal cal-h3">Training (Fitting) a Classifier </h3>

We will start with the most basic classifier, the logistic regression model, to demonstrate the classification workflow.

Logistic regression is a **linear model** that is commonly used for binary and multi-class classification tasks.  It is also a good starting point for understanding more complex deep learning models that will be covered later in the course.

Here we use `sklearn` to fit a logistic regression model to the training data. The `LogisticRegression` class from `sklearn.linear_model` is used to create an instance of the model. 

The `fit` method is called on the model instance, passing in the training data and labels. This trains the model to learn the relationship between the input features (flattened images) and the target labels (clothing categories).  In scikit-learn, the `fit` method is used to train any model on the provided data.

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
lr_model.fit(X=X_tr, y=labels_tr)
```
Notice that we get a warning that: 
```plaintext
lbfgs failed to converge after 100 iteration(s) (status=1):
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT
```

This warning indicates that the **optimization algorithm** used by the logistic regression model did not converge to a solution within the default number of iterations. This can happen if the model is complex or if the data is not well-scaled.

We can change the optimization algorithm used by the logistic regression model. The default algorithm is `lbfgs`, which is a quasi-Newton method. Other options include `newton-cg`, `sag`, and `saga`. Each of these algorithms has its own strengths and weaknesses, and the choice of algorithm can affect the convergence speed and final performance of the model.


In this class, we will explore variations of stochastic gradient descent like `saga`. Let's try using the `saga` algorithm instead of `lbfgs` to see if it converges faster.  Here we will also set the `tol` parameter since we don't want to wait.


```python
lr_model = LogisticRegression(tol=0.05, solver='saga', random_state=42)
lr_model.fit(X=X_tr, y=labels_tr)
```



<h3 class="cal cal-h3">Parameters </h3>
The **parameters** of a model are the internal variables that the model learns during the training process. For example, in logistic regression, the parameters are the weights assigned to each feature. These weights are adjusted during training to minimize the loss function, which measures how well the model's predictions match the actual labels.
```python
print("model.coef_.shape:", lr_model.coef_.shape)
print("model.intercept_.shape:", lr_model.intercept_.shape)
print(lr_model.coef_)
print(lr_model.intercept_)
```
We can also visualize these coefficients.  This can help us understand which pixels are most important for each class. We will learn more about this model in the future. (You don't have to understand these details now.)
```python
coeffs = lr_model.coef_
show_images(coeffs.reshape(10, 28, 28), labels=lr_model.classes_)
```
#### Neural Networks


Neural networks are often the model of choice for image classification tasks. They can learn complex patterns in the data and often outperform simpler models like logistic regression. However, they also require the correct architecture and significant training data and computational resources. 

Here we will try a simple neural network with two hidden layers.
```python
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50), 
    max_iter=100, tol=1e-3, random_state=42)

mlp.fit(X=X_tr, y=labels_tr)
```



<h3 class="cal cal-h3">Hyperparameters </h3>
The **hyperparameters** are the arguments that are set before the training process begins. These include the choice of optimization algorithm, the learning rate, and the number of iterations, among others. Hyperparameters are typically tuned using techniques like cross-validation to find the best combination for a given dataset.  

Confusingly these hyperparameters are often referred to as "parameters" in the context of machine learning libraries like `sklearn`. For example, the `LogisticRegression` class has hyperparameters like `solver`, `C`, and `max_iter` that can be adjusted to improve model performance.
```python
lr_model
```
To evaluate the model we will use the validation dataset.
```python
X_val = featurizer(images_val)
```
Let's try tuning the regularization parameter `C`.  To make this process more illustrative we will work with a smaller subset of the training data (n=1000).  This will allow us to better demonstrate underfitting and overfitting and significantly speed up the training process.

```python
n_small = 1000
X_tr_small = X_tr[:n_small,:]
labels_tr_small = labels_tr[:n_small]
```
In practice, we need to be careful when tuning regularization parameters against a sample of the data.  In this case, more regularization is likely needed for smaller datasets to prevent overfitting.
```python
from sklearn.metrics import log_loss

C_vals = np.logspace(-5, 1, 20)

logprob_tr = []
logprob_val = []
acc_tr = []
acc_val = []

for C in C_vals:
    print("starting training with C =", C)
    model = LogisticRegression(tol=1e-3, random_state=42, C=C)
    model.fit(X=X_tr_small, y=labels_tr_small)
    
    # compute the logprob accuracy
    logprob_tr.append(-log_loss(labels_tr_small, model.predict_proba(X_tr_small), labels=model.classes_))
    logprob_val.append(-log_loss(labels_val, model.predict_proba(X_val), labels=model.classes_))

    # compute the accuracy
    acc_tr.append(np.mean(model.predict(X_tr_small) == labels_tr_small))
    acc_val.append(np.mean(model.predict(X_val) == labels_val))

```
```python
df_logprob = pd.DataFrame({
    "C_val": C_vals, 
    "Train": logprob_tr, "Validation": logprob_val,
}).set_index("C_val") 

display(
    px.line(df_logprob, 
        labels={"value": "Avg. Log Prob.", "C_val": "Reg. Parameter C"},
        title="LR Classifier Log Prog vs Reg. Parameter",
        markers=True,
        log_x=True,
        width=800, height=500)
)

df_acc = pd.DataFrame({
    "C_val": C_vals, 
    "Train": acc_tr, "Validation": acc_val
}).set_index("C_val") 

display(
    px.line(df_acc, 
        labels={"value": "Accuracy", "C_val": "Reg. Parameter C"},
        title="LR Classifier Accuracy vs Reg. Parameter",
        markers=True,
        log_x=True,
        width=800, height=500
    )
)
```



<h2 class="cal cal-h2">Evaluating the Model</h2>

After training the model, we can use it to make predictions on new data. The `predict` method of the trained model is used to generate predictions based on the input features.
Let's return to our logistic regression model.
```python
lr_model.predict(X_tr[:10,:])
```
Do you agree with the predictions? Let's visualize the predictions on a few test images.
```python
show_images(images_tr[:10,:].reshape(10, 28, 28),
            labels = lr_model.predict(X_tr[:10,:]))
```
Now let's see what the correct labels are for these images. 
```python
k = 10
tmp_labels = labels_tr[:k] + " (pred=" + lr_model.predict(X_tr[:k,:]) + ")"
show_images(images_tr[:k,:].reshape(k, 28, 28), labels=tmp_labels)
```



<h3 class="cal cal-h3">Predicting Probabilities</h3>

Many models can also provide probabilities for each class using the `predict_proba` method. This is useful for understanding the model's confidence in its predictions.  In this class, we will often use a probabilistic framing, where we interpret the output of the model as probabilities of each class.
```python
lr_model.predict_proba(X_tr[:5,:])
```
We can visualize these probabilities for the same images we predicted earlier.
```python
k = 10
df = pd.DataFrame(lr_model.predict_proba(X_tr[:k,:]), columns=lr_model.classes_)
bars = px.bar(df, barmode='stack',orientation='v')
bars.update_layout(xaxis_tickmode='array', xaxis_tickvals=np.arange(k))
display(bars)
tmp_labels = labels_tr[:k] + " (pred=" + lr_model.predict(X_tr[:k,:]) + ") img: " + np.arange(k).astype(str)
show_images(images_tr[:k,:].reshape(k, 28, 28), labels=tmp_labels)
```



<h3 class="cal cal-h3">Accuracy Metrics and Test Performance</h3>

After training the model, we often want to evaluate the model. There are many ways to evaluate a model, and the best method depends on the task and the data. For classification tasks, we often use metrics like accuracy, precision, recall, and F1-score. Let's start with accuracy. 

Accuracy is the simplest metric, which measures the proportion of correct predictions out of the total number of predictions. 
Let's start by computing the accuracy of our model on the training set.
```python
np.mean(lr_model.predict(X_tr) == labels_tr, axis=0)
```
One of the issues with the training set is that the model may have overfit to the training data, meaning it performs well on the training set but poorly on unseen data. Intuitively, this is like practicing on a set of questions and then getting those same questions right on a test, but not being able to answer new questions that are similar but not identical.

To assess the model's performance on unseen data, we will evaluate it on the test set. Recall, the test set is a separate portion of the dataset that was not used during training.
```python
X_te = featurizer(images_te)
```
```python
np.mean(lr_model.predict(X_te) == labels_te, axis=0)
```
```python
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(labels_tr, lr_model.predict(X_tr))
val_acc = accuracy_score(labels_val, lr_model.predict(X_val))
test_acc = accuracy_score(labels_te, lr_model.predict(X_te))

print("Train accuracy:", train_acc)
print("Validation accuracy:", val_acc)
print("Test accuracy:", test_acc)
```
The test accuracy is slightly lower than the training accuracy, which is expected. However, the difference is not too large, indicating that the model has not overfit significantly.
**Is this accuracy good? What would a random guess yield?**
A common way to evaluate a classification model is to compare its accuracy against a baseline. Perhaps the simplest baseline is random guessing, where we randomly assign a class to each image.

**What accuracy does random guessing yield?**

This would depend on the how frequently each class appears in the test dataset. 
```python
np.random.seed(42)
print("Model Accuracy:", np.mean(lr_model.predict(X_val) == labels_val, axis=0))
print("Random Guess Accuracy:", 
      np.mean(np.random.choice(lr_model.classes_, size=len(labels_te)) == labels_te, axis=0))
```
Does our model struggle with any particular class?
```python
isWrong = lr_model.predict(X_val) != labels_val
# make a histogram with frequency of correct and incorrect predictions
fig = px.histogram(labels_val[isWrong], histnorm='percent')
fig.update_layout(xaxis_title="Label", 
                  yaxis_title="Percentage of Incorrect Predictions")
fig.update_xaxes(categoryorder="total descending")
```
For classification tasks, we often want to look at more than just accuracy. We can use a confusion matrix to visualize the performance of the model across different classes. The confusion matrix shows the number of correct and incorrect predictions for each class.
```python
from sklearn.metrics import confusion_matrix

fig = px.imshow(
    confusion_matrix(labels_val, lr_model.predict(X_val)), 
    color_continuous_scale='Blues'
    )
fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        coloraxis_showscale=False,
        xaxis=dict(tickmode='array', tickvals=np.arange(len(model.classes_)), ticktext=model.classes_),
        yaxis=dict(tickmode='array', tickvals=np.arange(len(model.classes_)), ticktext=model.classes_)
    )   

```



<h2 class="cal cal-h2">Last Thoughts</h2>

In the homework, you will have a chance to work with this data and use scikit-learn in more depth.  We recommend reading the documentation and tutorial on the scikit-learn website as we go through the course.  These provide a great resource for understanding the various functions and capabilities of the library as well as machine learning concepts.

