---
course: CS 189
semester: Fall 2025
type: notebook
number: 11
title: Lecture 11 Notebook
source_type: jupyter_notebook
processed_date: '2025-10-04'
---

<link rel="stylesheet" href="berkeley.css">

<h1 class="cal cal-h1">Lecture 11: Logistic Regression – CS 189, Fall 2025</h1>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error as mse
from scipy import stats
import plotly.express as px
import sklearn.linear_model as lm

import warnings
warnings.filterwarnings("ignore")

import plotly.io as pio
pio.renderers.default = "notebook_connected"
```

```python
from sklearn.datasets import load_breast_cancer

# Load the dataset
data_dict = load_breast_cancer()
data = pd.DataFrame(data_dict['data'], columns=data_dict['feature_names'])
data['malignant'] = (data_dict['target'] == 0)  # 1 for malignant, 0 for benign

# Display the first few rows
data.head()
```

```python
# Split the data into training and testing sets
X = data[['mean radius']].to_numpy()
y = data['malignant'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
```

```
Training set size: 455
Testing set size: 114
```

```python
toy_df = pd.DataFrame({
        "x": [-4, -2, -0.5, 1, 3, 5],
        "y": [0, 0, 1, 0, 1, 1]
})
toy_df["str_y"] = toy_df["y"].astype(str)
toy_df.sort_values("x")
```

```python
fig = px.scatter(toy_df, x="x", y="y", color="str_y", width=800)
fig.update_traces(marker_size=20)
```

```python
def sigmoid(z):
    return 1/(1+np.e**-z)

def ss_error_on_toy_data(theta):
    p_hat = sigmoid(toy_df['x'] * theta)
    return 1/2.0 * np.sum((toy_df['y'] - p_hat)**2)

w_error = pd.DataFrame({"w": np.linspace(-10, 10, 100)})
w_error["Error"] = w_error["w"].apply(ss_error_on_toy_data)
fig = px.line(w_error, x="w", y="Error", width=800,
    title="Error on Toy Classification Data")
fig.update_traces(line=dict(width=4))
fig.show()
```

```python
# Set the initial guess as w = 0
from scipy.optimize import minimize

best_w = minimize(ss_error_on_toy_data, x0 = 0)["x"][0]
best_w
```

```python
non_optimal_w = minimize(ss_error_on_toy_data, x0 = -5)["x"][0]
non_optimal_w
```

```python
fig = px.scatter(toy_df, x="x", y="y", color="str_y", width=800)
xs = np.linspace(-10, 10, 100)
fig.add_trace(go.Scatter(
    x=xs, y=sigmoid(xs * best_w),
    mode="lines", line_color="black",
    name=f"LR Model: w = {best_w:.2f}"))
fig.update_traces(line=dict(width=4))
fig.update_traces(marker_size=20)
fig.show()
```

```python
fig = px.scatter(toy_df, x="x", y="y", color="str_y", width=800)
xs = np.linspace(-10, 10, 100)
fig.add_trace(go.Scatter(
    x=xs, y=sigmoid(xs * non_optimal_w),
    mode="lines", line_color="black",
    name=f"LR Model: w = {best_w:.2f}"))
fig.update_traces(line=dict(width=4))
fig.update_traces(marker_size=20)
fig.show()
```

```python
p_hat_error = pd.DataFrame({"p": np.arange(0.001, 0.999, 0.01)})
p_hat_error["Squared Error"] = 1/2.0* (1 - p_hat_error["p"])**2
fig = px.line(p_hat_error, x="p", y="Squared Error", width=800,
        title="Squared Loss for One Individual when y=1")
fig.update_traces(line=dict(width=4))
fig.show()
```

```python
p_hat_error["Neg Log Error"] = -np.log(p_hat_error["p"])
```

```python
fig = px.line(p_hat_error.melt(id_vars="p", value_name="Error"),
        x="p", y="Error", color="variable", width=800,
        title="Error Comparison for One Observation when y = 1")
fig.update_traces(line=dict(width=4))
fig.show()
```

```python
p_hat_error = pd.DataFrame({"p": np.arange(0.001, 0.999, 0.01)})
p_hat_error["Squared Error"] = 1/2.0 * (1 - (1-p_hat_error["p"]))**2
p_hat_error["Neg Log Error"] = -np.log(1 - p_hat_error["p"])
fig = px.line(p_hat_error.melt(id_vars="p", value_name="Error"),
        x="p", y="Error", color="variable", width=800,
        title="Error Comparison for One Observation when y = 0")
fig.update_traces(line=dict(width=4))
fig.show()
```

```python
def cross_entropy(y, p):
    return - y * np.log(p) - (1 - y) * np.log(1 - p)
def mean_cross_entropy_on_toy_data(w):
    p = sigmoid(toy_df["x"] * w)
    return np.mean(cross_entropy(toy_df["y"], p))
```

```python
w_error["Cross-Entropy"] = w_error["w"].apply(mean_cross_entropy_on_toy_data).dropna()
fig = px.line(w_error, x="w", y="Cross-Entropy", width=800,
           title="Cross-Entropy on Toy Classification Data")
fig.update_xaxes(range=[w_error["w"].min(), 4])
fig.update_traces(line=dict(width=4))
fig.show()
```

```python
toy_model = lm.LogisticRegression(C=10)

# We fit to two data points: (-1, 0) and (1, 1).
toy_model.fit([[-1], [1]], [0,1])

# Generate estimated probabilities across a range of x-values.
xtest = np.linspace(-5, 5, 1000)[:, np.newaxis]
p = toy_model.predict_proba(xtest)[:,1]

fig = px.scatter(toy_df, x="x", y="y",
         color="str_y", symbol="str_y",
         symbol_sequence=["circle", "cross"],
         title=f"LR Fit (slope = {toy_model.coef_[0][0]}, intercept = {toy_model.intercept_[0]})",
         render_mode="svg")
fig.update_traces(marker=dict(size=15))
fig.update_layout(
  xaxis_title=dict(font=dict(size=22)),
  yaxis_title=dict(font=dict(size=22))
)
fig.add_scatter(x=np.ravel(xtest), y=p, mode="lines", name="LR Model with C=10",
                line_color="black", opacity=0.5)
```

```python
toy_model = lm.LogisticRegression(C=1000)

# We fit to two data points: (-1, 0) and (1, 1).
toy_model.fit([[-1], [1]], [0,1])

# Generate estimated probabilities across a range of x-values.
xtest = np.linspace(-5, 5, 1000)[:, np.newaxis]
p = toy_model.predict_proba(xtest)[:,1]

fig = px.scatter(toy_df, x="x", y="y",
         color="str_y", symbol="str_y",
         symbol_sequence=["circle", "cross"],
         title=f"LR Fit (slope = {toy_model.coef_[0][0]}, intercept = {toy_model.intercept_[0]})",
         render_mode="svg")
fig.update_traces(marker=dict(size=15))
fig.update_layout(
  xaxis_title=dict(font=dict(size=22)),
  yaxis_title=dict(font=dict(size=22))
)
fig.add_scatter(x=np.ravel(xtest), y=p, mode="lines", name="LR Model with C=1000",
                line_color="black", opacity=0.5)
```

##  Build a Logistic Regression Model


```python
model = lm.LogisticRegression()
model.fit(X_train, y_train)

print("Slope:", model.coef_[0][0])
print("Intercept:", model.intercept_[0])
```

```
Slope: 0.9488287268826906
Intercept: -14.039681312756318
```

Now, rather than predict a numeric output, we predict the *probability* of a datapoint belonging to Class 1. We do this using the `.predict_proba` method.

```python
# Preview the first 10 rows
model.predict_proba(X_train)[:10]
```

By default, `.predict_proba` returns a 2D array.

One column contains the predicted probability that the datapoint belongs to Class 0, and the other contains the predicted probability that it belongs to Class 1 (notice that all rows sum to a total probability of 1).

To check which is which, we can use the `.classes_` attribute.

```python
model.classes_
```

This tells us that the first column contains the probabilities of belonging to Class 0 (benign), and the second column contains the probabilities of belonging to Class 1 (malignant). Let's grab just the probabilities of Class 1.

We then apply a decision rule: Predict Class 1 if the predicted probability of belonging to Class 1 is 0.5 or higher. Otherwise, predict Class 0.

- Remember that 0.5 is a common threshold, but we are not required to always use 0.5

```python
# Obtain P(Y=1|x) from the output.
p = model.predict_proba(X_train)[:, 1]

# Apply decision rule: predict Class 1 if P(Y=1|x) >= 0.5.
(p >= 0.5).astype(int)
```

The `.predict` method of `LogisticRegression` will apply a 0.5 threshold to classify data, by default

```python
# .predict will automatically apply a 0.5 threshold for a logistic regression model.
classes = model.predict(X_train).astype(int)

classes
```

The point where the sigmoid function outputs 0.5 is the **decision boundary**.

- This is the point where the model is indifferent between predicting Class 0 and Class 1.  

- This is also the point where $\theta_0 + \theta_1 x = 0$.

For this one dimensional case we can solve for the $x$ value of the decision boundary:

$$
x = - \frac{\theta_0}{\theta_1} = - \frac{\text{intercept}}{\text{slope}}
$$

Let's visualize our predictions.

```python
# Convert X_train to a DataFrame for compatibility
X_train_df = pd.DataFrame(X_train, columns=["mean radius"])
X_train_df["Predicted Class"] = pd.Categorical(model.predict(X_train))

test_points = pd.DataFrame({"mean radius": np.linspace(5, 30, 100)})
test_points["Predicted Prob"] = model.predict_proba(test_points[["mean radius"]])[:, 1]

fig = px.scatter(X_train_df, x="mean radius", y=y_train.astype(int), color="Predicted Class", opacity=0.6)

# Add the logistic regression model predictions
fig.add_trace(go.Scatter(x=test_points["mean radius"], y=test_points["Predicted Prob"],
                         mode="lines", name="Logistic Regression Model",
                         line_color="black", line_width=5, line_dash="dash"))
fig.add_vline(x = -model.intercept_[0]/model.coef_[0][0], line_dash="dash",
              line_color="black",
              annotation_text="Decision Boundary",
              annotation_position="right")

```

Any time the predicted probability $p$ is less than 0.5, the model predicts Class 0. Otherwise, it predicts Class 1.

A decision boundary describes the **line** that splits the data into classes based on the *features*.

For a model with one feature, the decision boundary is a *point* that separates the two classes. The number of dimensions of the decision boundary plot is the number of features.

- We visualize this using a 1D plot to plot all data points in terms of *just* the feature.

- We cannot define a decision boundary in terms of the predictions, so we remove that axis from our plot.

Notice that all data points to the right of our decision boundary are classified as Class 1, while all data points to the left are classified as Class 0.

```python
fig = px.scatter(X_train_df, x="mean radius", y=np.zeros(len(X_train_df)),
                 symbol="Predicted Class", symbol_sequence=[ "circle-open", "cross"],
                 color="Predicted Class", height=300, opacity=0.6)
# fig.update_traces(marker_symbol='line-ns-open')
fig.update_traces(marker_size=8)
fig.update_layout(
    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
)

decision_boundary =  -model.intercept_[0]/model.coef_[0][0]
fig.add_vline(x = decision_boundary, line_dash="dash",
              line_color="black",
              annotation_text="Decision Boundary",
              annotation_position="top right")
```

## Decision Boundaries


The `LogisticRegression` class of `sklearn.linear_model` behaves very similarly to the `LinearRegression` class. As before, we:

1. Initialize a model object, and
2. Fit it to our data.

You find it helpful to recall the model formulation of a fitted logistic regression model with one input:

$$
\hat{P}_{\hat{w}}(Y=1 \mid X) = \sigma \left( \hat{w}_0 + \hat{w}_1 X \right) = \frac{1}{1 + e^{-(\hat{w}_0 + \hat{w}_1 X)}}
$$

## 🎯 Performance Metrics

Let's return to our data. We'll compute the **accuracy** of our `model` on this data.


```python
def accuracy(X, Y):
    return np.mean(model.predict(X) == Y)

print(model.predict(X_train)[:5].astype(int))
print(y_train[:5].astype(int))
accuracy(X_train, y_train)
```

```
[0 1 0 0 0]
[0 1 0 0 0]
```

```python
model.score(X_train, y_train)
```

> Important Note: `model.predict` and `model.score` use a threshold of 0.5.
To use a different threshold, you must use `model.predict_proba` and work with probabilities directly.

### Confusion matrix

`scikit-learn` has an built-in `confusion_matrix` method.

```python
from sklearn.metrics import confusion_matrix

# Be careful – confusion_matrix takes in y_true as the first parameter and y_pred as the second.
# Don't mix these up!
cm = confusion_matrix(y_train, model.predict(X_train))
cm
```

```python
fig = px.imshow(cm, x=["0", "1"], y=["0", "1"],
          labels=dict(x="Predicted", y="Actual"),
          text_auto=True,
          color_continuous_scale="Blues",
          width=400, height=400)
fig.update_xaxes(side="top")
```

### Precision and Recall

We can also compute the number of TP, TN, FP, and TN for our classifier, and then its precision and recall.

```python
Y_hat = model.predict(X_train)
tp = np.sum((Y_hat == 1) & (y_train == 1))
tn = np.sum((Y_hat == 0) & (y_train == 0))

fp = np.sum((Y_hat == 1) & (y_train == 0))
fn = np.sum((Y_hat == 0) & (y_train == 1))


print("True Positives: ", tp)
print("True Negatives: ", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
```

```
True Positives:  130
True Negatives:  266
False Positives: 20
False Negatives: 39
```

These numbers match what we see in the confusion matrix above.

### Precision and Recall

**Precision** -- How precise are my positive predictions? In other words, what fraction of the things the model predicted positive are actually positive?

```python
precision = tp / (tp + fp)
precision
```

**Recall** -- What proportion of actual positives did my model recall in its predictions? In other words, what proportion of actual positive cases that were correctly identified by the model?

```python
recall = tp / (tp + fn)
recall
```

### True and False Positive Rates

The TP, TN, FP, and TN we just calculated also allow us to compute the true and false positive rates (TPR and FPR). Recall that TPR is the same as recall.

```python
fpr = fp/(fp + tn)
fpr
```

```python
tpr = tp/(tp + fn)
tpr
```

It's important to remember that these values are all for the threshold of $T = 0.5$, which is `scikit-learn`'s default.

## 🎛️ Adjusting the Classification Threshold

Before, we used a threshold of 0.5 in our decision rule: If the predicted probability was greater than 0.5 we predicted Class 1, otherwise, we predicted Class 0.

```python
def plot_predictions(threshold = 0.5):
    # Convert X_train to a DataFrame for compatibility
    X_train_df = pd.DataFrame(X_train, columns=["mean radius"])
    X_train_df["Predicted Class"] = model.predict_proba(X_train)[:, 1] >= threshold

    X_train_df["Predicted Class"] = pd.Categorical(X_train_df["Predicted Class"].astype(int))
    fig = px.scatter(X_train_df,
            x="mean radius", y=y_train.astype(int), color="Predicted Class",
            title=f"Logistic Regression Predictions (Threshold = {threshold})")
    # Add the logistic regression model predictions
    # Make the data points for the LR model curve
    test_points = pd.DataFrame({"mean radius": np.linspace(5, 30, 100)})
    test_points["Predicted Prob"] = model.predict_proba(test_points)[:, 1]
    fig.add_trace(go.Scatter(x=test_points["mean radius"], y=test_points["Predicted Prob"],
                            mode="lines", name="Logistic Regression Model",
                            line_color="black", line_width=5, line_dash="dash"))
    decision_boundary = (-np.log(1/threshold - 1) - model.intercept_[0])/model.coef_[0][0]
    fig.add_vline(x = decision_boundary, line_dash="dash", line_color="black",
                  annotation_text="Decision Boundary", annotation_position="right")
    return fig

plot_predictions(0.5)
```

```python
plot_predictions(0.25)
```

When we **lower the threshold**, we require a lower predicted probability before we predict Class 1. We can think of this as us telling our model that it needs to be less "confident" about a data point being Class 1 before making a positive prediction. The total number of data points predicted to be Class 1 **either stays the same or increases**.

The converse happens if we raise the threshold. Consider setting $T=0.75$. Now, we require a higher predicted probability before we predict Class 1. The total number of data points predicted to be Class 1 decreases.

```python
plot_predictions(0.75)
```

## Thresholds and Performance Metrics

How does changing the threshold impact our performance metrics?

Let's run an experiment: we'll test out several different possible thresholds.

- For each threshold $T$, we'll make a decision rule where we classify any point with a predicted probability equal to or greater than $T$ as being in Class 1.

- Otherwise, we'll predict Class 0.

- We'll then compute the overall accuracy of the classifier when using that threshold.

```python
# Define performance metrics dependent on the threshold value.
def predict_threshold(model, X, T):
    prob_one = model.predict_proba(X)[:, 1]
    return (prob_one >= T).astype(int)

def accuracy_threshold(X, Y, T):
    return np.mean(predict_threshold(model, X, T) == Y)

def precision_threshold(X, Y, T):
    Y_hat = predict_threshold(model, X, T)
    denominator = np.sum(Y_hat == 1)
    if denominator == 0:
        denominator = np.nan
    return np.sum((Y_hat == 1) & (Y == 1)) / denominator

def recall_threshold(X, Y, T):
    Y_hat = predict_threshold(model, X, T)
    return np.sum((Y_hat == 1) & (Y == 1)) / np.sum(Y == 1)

def tpr_threshold(X, Y, T): # Same as recall
    Y_hat = predict_threshold(model, X, T)
    return np.sum((Y_hat == 1) & (Y == 1)) / np.sum(Y == 1)

def fpr_threshold(X, Y, T):
    Y_hat = predict_threshold(model, X, T)
    return np.sum((Y_hat == 1) & (Y == 0)) / np.sum(Y == 0)
```

```python
metrics = pd.DataFrame()
metrics["Threshold"] = np.linspace(0, 1, 1000)
metrics["Accuracy"] = [accuracy_threshold(X_train, y_train, t) for t in metrics["Threshold"]]
metrics["Precision"] = [precision_threshold(X_train, y_train, t) for t in metrics["Threshold"]]
metrics["Recall"] = [recall_threshold(X_train, y_train, t) for t in metrics["Threshold"]]
metrics.head()
```

```python
fig = px.line(metrics,
        x="Threshold", y="Accuracy",
        title="Accuracy vs. Threshold",
        render_mode="svg", width=600, height=600)

fig.add_scatter(x=[metrics.loc[metrics['Accuracy'].idxmax(), 'Threshold']], y=[metrics.loc[metrics['Accuracy'].idxmax(), 'Accuracy']],
                mode='markers', marker=dict(size=10, color='red'),
                name=f"Accuracy Max {metrics.loc[metrics['Accuracy'].idxmax(), 'Accuracy']:.5f}",)
fig.update_layout(
  xaxis_title=dict(font=dict(size=22)),
  yaxis_title=dict(font=dict(size=22))
)
fig.show()
```

```python
# The threshold that maximizes accuracy.
metrics.sort_values("Accuracy", ascending=False).head()
```

It turns out that setting $T=0.5$ does not always result in the best performance! Part of the model design process for classification includes **choosing an appropriate threshold value**.

### Precision-Recall Curves
In the lecture, we noted that there is a tradeoff between precision and recall.

Precision $=\frac{TP}{\text{Positive Predictions}}=\frac{TP}{TP+FP}$ increases as the number of false positives decreases, which occurs as the threshold is raised, since raising the threshold tends to reduce the number of positive predictions.

Recall $=\frac{TP}{\text{Actual Class 1s}}=\frac{TP}{TP+FN}$ increases as the number of false negatives decreases, which occurs as the threshold is lowered, since lowering the threshold tends to decrease number of negative predictions.

We want to keep both precision and recall high. To do so, we'll need to strategically choose a threshold value.

```python
fig = px.line(metrics,
        x="Threshold", y=["Accuracy", "Precision", "Recall"],
        title="Performance Metrics vs. Threshold",
        render_mode="svg", height=600, width=600)
fig.update_layout(
  xaxis_title=dict(font=dict(size=22)),
  yaxis_title=dict(font=dict(size=22))
)
fig.show()
```

A **precision-recall** curve tests out many possible thresholds. Each point on the curve represents the precision and recall of the classifier for a *particular choice of threshold*.

We choose a threshold value that keeps both precision and recall high (usually in the rightmost "corner" of the curve).

```python
fig = px.line(metrics, x="Recall", y="Precision",
        title="Precision vs. Recall",
        width=600, height=600,
        render_mode="svg")
fig.update_layout(
  xaxis_title=dict(font=dict(size=22)),
  yaxis_title=dict(font=dict(size=22))
)
fig.show()
```

One way to balance precision and recall is to compute the **F1 score**. The F1 score is the harmonic mean of precision and recall:

$$F1 = 2 \times \frac{\text{precision} \times \text{recall}}{\text{precision} + \text{recall}}$$

```python
metrics["F1"] = (2 * metrics["Precision"] * metrics["Recall"]
                     / (metrics["Precision"] + metrics["Recall"]))
ind = metrics['F1'].idxmax()
metrics.loc[ind,:]
```

```python
fig = px.line(metrics, x="Threshold", y="F1",
              title="Finding F1 Score Maximum",
              render_mode="svg",
              height=600, width=600)

fig.add_scatter(x=[metrics.loc[ind, 'Threshold']], y=[metrics.loc[ind, 'F1']],
                mode='markers', marker=dict(size=10, color='red'),
                name=f"F1 Max {metrics.loc[ind, 'Threshold']:.5f}",)

fig.update_layout(
  xaxis_title=dict(font=dict(size=22)),
  yaxis_title=dict(font=dict(size=22))
)
fig.show()
```

```python
fig = px.line(metrics, x="Recall", y="Precision",
              title="Precision vs. Recall", width=600, height=600,
              render_mode="svg")
fig.add_scatter(x=[metrics.loc[ind, 'Recall']], y=[metrics.loc[ind, 'Precision']],
                mode='markers', marker=dict(size=10, color='red'),
                name=f"F1 Max {metrics.loc[ind, 'Threshold']:.5f}")
fig.update_layout(legend=dict(x=.5, y=.1))
fig.update_layout(
  xaxis_title=dict(font=dict(size=22)),
  yaxis_title=dict(font=dict(size=22))
)
fig.show()
```

### ROC Curves

We can repeat a similar experiment for the FPR and TPR. Remember that we want to keep FPR *low* and TPR *high*.

```python
metrics["TPR"] = [tpr_threshold(X_train, y_train, t) for t in metrics["Threshold"]]
metrics["FPR"] = [fpr_threshold(X_train, y_train, t) for t in metrics["Threshold"]]
```

```python
fig = px.line(metrics, x="Threshold", y=["TPR", "FPR", "Accuracy"],
        render_mode="svg", width=600, height=600)
fig.update_layout(
  xaxis_title=dict(font=dict(size=22)),
  yaxis_title=dict(font=dict(size=22))
)
fig.show()
```

A **ROC curve** tests many possible decision rule thresholds. For each possible threshold, it plots the corresponding TPR and FPR of the classifier.

"ROC" stands for "Receiver Operating Characteristic". It comes from the field of signal processing.

```python
fig = px.line(metrics, x="FPR", y="TPR", title="ROC Curve",
        width=600, height=600,
        render_mode="svg")
fig.update_layout(
  xaxis_title=dict(font=dict(size=22)),
  yaxis_title=dict(font=dict(size=22))
)
fig.show()
```

 Ideally, a perfect classifier would have a FPR of 0 and TPR of 1. The area under the perfect classifier is 1.

 We often use the area under the ROC curve (abbreviated "AUC") as an indicator of model performance. The closer the AUC is to 1, the better.

```python
fig = px.line(metrics, x="FPR", y="TPR", title="ROC Curve",
              width=600, height=600,
              render_mode="svg")
fig.add_scatter(x=[0,0,1], y=[0,1,1], mode='lines',
                line_dash='dash', line_color='black',
                name="Perfect Classifier")
# move the legend inside the plot
fig.update_layout(legend=dict(x=.5, y=.1))
fig.update_layout(
  xaxis_title=dict(font=dict(size=22)),
  yaxis_title=dict(font=dict(size=22))
)
fig.show()
```

