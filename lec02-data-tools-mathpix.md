---
course: CS 189
semester: Fall 2025
type: lecture
title: Data Tools
source_type: slides
source_file: Lecture 02 -- Data Tools.pptx
processed_date: '2025-10-01'
processor: mathpix
---

## Lecture 2

## Data Tools

Tools we need for ML - pandas and visualization

## EECS 189/289, Fall 2025 @ UC Berkeley

Joseph E. Gonzalez and Narges Norouzi

# III Join at slido.com <br> '1َيْL \#7824888 

## Why Start with Data?

- Data is the foundation of ML

Every model begins and ends with data. It's the raw material for training and evaluation.

- Inputs and Experiments

Success in ML depends on how well we process inputs and outputs from experiments.

- Critical Skill for Research \& Industry

Whether you join a lab or work in industry, strong data wrangling and visualization skills are essential.

## Roadmap

- pandas
- pandas Data Structures
- Exploring DataFrames
- Selecting and Retrieving Data in DataFrames
- Filtering Data
- DataFrame Modification
- Aggregation in DataFrame
- Joining DataFrames
- Visualization
- Matplotlib
- Plotly
- pandas
- pandas Data Structures
- Exploring DataFrames
- Selecting and Retrieving


## pandas Data Structures

Data in DataFrames

- Filtering Data
- DataFrame Modification
- Aggregation in DataFrame
- Joining DataFrames
- Visualization
- Matplotlib
- Plotly


## pandas

Using pandas, we can:

- Arrange data in a tabular format.
- Extract useful information filtered by specific conditions.
- Operate on data to gain new information.
- Apply numerical operations using NumPy to our data.
- Perform vectorized computations to speed up our analysis.
pandas is the standard tool across research and industry for working with tabular data.


## pandas Data Types

- In the language of pandas, tables are referred to as DataFrames.
![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-07.jpg?height=1043&width=2761&top_left_y=637&top_left_x=89)

**Image Description:** The image presents a table formatted as a DataFrame, displaying landmarks with associated attributes. It contains five rows and four columns labeled "Landmark," "Type," "Height," and "Year Built." Notable landmarks include "Sather Gate," "Campanile," "Doe Library," "Memorial Glade," and "Sproul Plaza." The "Height" column indicates numerical values for gates and towers, with "NaN" for open spaces. The table is visually distinguished, indicating the structure of data in a programming context, likely related to data analysis or manipulation using pandas in Python. An index is also highlighted, showing the row identifiers.



## Series

- A Series is a one-dimensional labeled array.
- Components of a Series object:
- Values: The actual data stored in the Series.
- Index: The labels associated with each data point, which allow for easy access and manipulation.

```
import pandas as pd
welcome_series = pd.Series(["welcome", "to", "CS 189"])
```

![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-08.jpg?height=494&width=2803&top_left_y=1263&top_left_x=510)

**Image Description:** The image depicts a Python series object from the Pandas library. It shows the contents of a series called "welcome_series." The diagram outlines two components: "welcome_series.values," which displays an array containing the strings "welcome," "to," and "CS 189," and indicates the data type as "object." The second component is "welcome_series.index," showing a RangeIndex from start=0 to stop=3 with a step of 1, illustrating the indexing of the series. The axis indicates the elements in the series corresponding to their indices.


## DataFrame

![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-09.jpg?height=431&width=124&top_left_y=433&top_left_x=59)

**Image Description:** The image is a diagram representing a vertical display of numerical labels in a single column, showcasing the numbers 0 through 4. Each number is centrally aligned within a white rectangular box bordered by a dark blue outline. The format suggests it could represent an index or a list of discrete values, possibly indicating a hierarchy or levels in data analysis.

![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-09.jpg?height=444&width=2216&top_left_y=433&top_left_x=238)

**Image Description:** The image presents a diagram illustrating a weighted sum of locations on a campus, based on a grid-like structure. Each location is represented as a labeled box with indices indicating their respective coordinates. The locations include "Sather Gate," "Campanile," "Doe Library," "Memorial Glade," "Sproul Plaza," followed by a summation of weights or coordinates. The axis appears to represent both the location names and their associated numerical values, indicating a weighting or scoring system for different campus areas, although specific metrics are not detailed.


Landmark Series
Type Series
Height Series
Year Built Series

|  | Landmark | Type | Height | Year Built |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 |
| 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |

You can create a DataFrame:

- From a CSV file
- Using a dictionary
- Using a list and column names
- From Series


## Creating a DataFrame

```
ata = {
Landmark': ['Sather Gate', 'Campanile', 'Doe Library', 'Memorial Glade', 'Sproul Plaza']
Type': ['Gate', 'Tower', 'Library', 'Open Space', 'Plaza'],
Height': [30, 307, 80, 0, 0],
Year Built': [1910, 1914, 1911, None, 1962]
f = pd.DataFrame(data)
e/ent_data = pd.read_csv("data/uc_berkeley_events.csv", index_col='Year')

\begin{tabular}{|l|l|l|l|l|}
\hline & & & Event & Location \\
\hline \multicolumn{5}{|l|}{Year} \\
\hline 1868 & & & Founding of UC Berkeley & Berkeley, CA \\
\hline 1914 & & & Completion of Campanile & Berkeley, CA \\
\hline 1923 & & & Opening of Memorial Stadium & Berkeley, CA \\
\hline 1964 & & & Free Speech Movement & Berkeley, CA \\
\hline 2000 & Opening of & Hearst & Memorial Mining Building & Berkeley, CA \\
\hline
\end{tabular}
```


## Exploring DataFrames

- pandas
- pandas Data Structures
- Exploring DataFrames
- Selecting and Retrieving Data in DataFrames
- Filtering Data
- DataFrame Modification
- Aggregation in DataFrame
- Joining DataFrames
- Visualization
- Matplotlib
- Plotly


## Utility Functions for DataFrame

- Understanding the structure and content of DataFrame is an essential first step in data analysis. Here are some methods to get a quick overview of DataFrame:
- head()and tail(),
- info(),
- describe(),
- sample(),
- value_counts(), and
- unique().


## head() and tail()

|  | Landmark | Type | Height | Year Built |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 |
| 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |

Use head () to display the first few rows of the DataFrame.
df.head() \# default is 5

|  | Landmark | Type | Height | Year Built |
| :--- | ---: | ---: | ---: | ---: |
| 0 | Sather Gate | Gate | 30 | 1910.0 |
| 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |

Use tail() to display the last few rows
df.tail(3)

|  | Landmark | Type | Height | Year Built |
| :--- | ---: | ---: | ---: | ---: |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |

## shape and size

![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-14.jpg?height=723&width=1655&top_left_y=421&top_left_x=718)

**Image Description:** The image is a table presenting data on various landmarks. It includes the following columns: "Landmark," "Type," "Height," and "Year Built." Each row lists a specific landmark alongside its type (e.g., Gate, Tower), height in feet, and the year it was built. Notable entries include Sather Gate (30 ft, 1910) and Campanile (307 ft, 1914). The table appears under a flowchart structure, indicated by numbers 4 and 5, suggesting an organization within a larger diagram or framework. The heights vary significantly, and the "Year Built" column includes a NaN value, indicating missing data for Memorial Glade.

shape gets the number of rows and columns in the DataFrame.
df.shape
$(5,4) \quad \longrightarrow \quad 5 \times 4=20$
size gets the total number of elements in the DataFrame.
df.size
20

## sample()

To sample a random selection of rows from a DataFrame, we use the .sample() method.
.sample(n=\#)
Randomly sample n rows.
.sample(n=\#, replace=True)
Allows the same row to appear multiple times.
.sample(frac=\#)
Randomly sample frac of rows.
.sample(n=\#, random_state=42)
Using random_state for reproducibility.

## value_counts() and unique()

| Series.value_counts | Series.unique |
| :--- | :--- |
| Counts the number of occurrences of each unique value in a Series. | Returns an array of every unique value in a Series. |
| type_counts = df['Type'].value_counts() | type_counts = df['Type'].unique() |
| Type |  |
| Gate | ['Gate' 'Tower' 'Library' 'Open Space' 'Plaza'] |
| 1 |  |
| Library |  |
| Plaza |  |
| Name: count, dtype: int64 |  |

## Selecting and Retrieving Data

- pandas
- pandas Data Structures
- Exploring DataFrames
- Selecting and Retrieving Data in DataFrames
- Filtering Data
- DataFrame Modification
- Aggregation in DataFrame
- Joining DataFrames
- Visualization
- Matplotlib
- Plotly


## Integer-based Extraction: iloc[]

We want to extract data according to its position.

| Row position | 0 | 1 | 2 | 3 | Column position |
| :--- | :--- | :--- | :--- | :--- | :--- |
|  | Landmark | Type | Height | Year Built |  |
| 0 | 0 | Gate | 30 | 1910.0 | Python |
| 1 | 1 | Tower | 307 | 1914.0 | convention: The |
| 2 | 2 | Library | 80 | 1911.0 |  |
| 4 | 3 | Open Space | 0 | 1962.0 | first position has integer index 0. |

Arguments to .iloc can be:

- A list.
- A slice (syntax is exclusive of the right-hand side of the slice).
- A single value.


## Integer-based Extraction: iloc[]

Arguments to .iloc can be:

- A list.
- A slice (syntax is exclusive of the right-hand side of the slice).
- A single value.

|  |  | 0 | 1 | 2 | 3 |
| :--- | :--- | :--- | :--- | :--- | :--- |
|  |  | Landmark | Type | Height | Year Built |
| 0 | 0 | Sather Gate | Gate | 30 | 1910.0 |
| 1 | 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | 3 | Memorial Glade | Open Space | 0 | NaN |
| 4 | 4 | Sproul Plaza | Plaza | 0 | 1962.0 |

## Integer-based Extraction: iloc[]

Arguments to .iloc can be:

- A list.
- A slice (syntax is exclusive of the right-hand side of the slice).
- A single value.
df.iloc[[1, 2]]

|  | Landmark | Type | Height | Year Built |
| :--- | ---: | ---: | ---: | ---: |
| 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |

df.iloc[[1, 2], [1, 2, 3]]

|  |  | 0 | 1 | 2 | 3 |
| :--- | :--- | :--- | :--- | :--- | :--- |
|  |  | Landmark | Type | Height | Year Built |
| 0 | 0 | Sather Gate | Gate | 30 | 1910.0 |
| 1 | 1 | Doe Library | Library | 307 | 1914.0 |
| 3 | 3 | Memorial Glade | Open Space | 0 | NaN |
| 4 | 4 | Sproul Plaza | Plaza | 0 | 1962.0 |


|  | Type | Height | Year Built |
| :--- | ---: | ---: | ---: |
| 1 | Tower | 307 | 1914.0 |
| 2 | Library | 80 | 1911.0 |

## Integer-based Extraction: iloc[]

Arguments to .iloc can be:

- A list.
- A slice (syntax is exclusive of the right-hand side of the slice).
- A single value.
df.iloc[2:4]

|  | Landmark | Type | Height | Year Built |
| :--- | ---: | ---: | ---: | ---: |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |

$\begin{array}{rrr}\text { df.ilOC[:, 1:2] } & \begin{array}{l}0 \\ 1 \\ 2\end{array} & \text { Gate } \\ 3 & \text { Tibrary } \\ 4 & \text { Open Space } \\ & \text { Name: Type, dtype: object }\end{array}$

|  |  | 0 | 1 | 2 | 3 |
| :---: | :---: | ---: | ---: | ---: | ---: |
|  |  | Landmark | Type | Height | Year Built |
| $\mathbf{0}$ | 0 | Sather Gate | Gate | 30 | 1910.0 |
| $\mathbf{1}$ | 1 | Campanile | ïower | 307 | 1914.0 |
| $\mathbf{2}$ | 2 | Doe Library | Library | 80 | 1911.0 |
| $\mathbf{3}$ | 3 | Memorial Glade | Open Space | 0 | NaN |
| $\mathbf{4}$ | 4 | Sproul Plaza | Plaza | 0 | 1962.0 |

## Integer-based Extraction: iloc[]

Arguments to .iloc can be:

- A list.
- A slice (syntax is exclusive of the right-hand side of the slice).
- A single value.

| df.iloc[:, 0] ${ }_{1}^{0}$ | Sather Gate |
| :--- | :--- |
| 2 | Doe Library |
| 3 | Memorial Glade |
| 4 | Sproul Plaza |
|  | Name: Landmark, dtype: object |
| df.iloc[2] | df.iloc[0, 1] |
| Landmark | Landmark |
| Type | Type |
| Height | Height |
| Year Built | Year Built |
| Name: 2, dtype: object | Name: 2, dtype: object |


|  |  | O | $\mathbf{1}$ | $\mathbf{2}$ | $\mathbf{3}$ |
| :--- | :--- | ---: | ---: | ---: | ---: |
|  |  | Landmark | Type | Height | Year Built |
| $\mathbf{0}$ | 0 | Sather Gate | Gate | 30 | 1910.0 |
| $\mathbf{1}$ | 1 | Campanile | Tower | 307 | 1914.0 |
| $\mathbf{2}$ | 2 | Doe Library | Library | 80 | 1911.0 |
| $\mathbf{3}$ | 3 | Memorial Glade | Open Space | 0 | NaN |
| $\mathbf{4}$ | 4 | Sproul Plaza | Plaza | 0 | 1962.0 |

## Label-based Extraction: loc[]

We want to extract data according to its labels.

| Row Labels | Landmark | Type | Height | Year Built | Column Labels |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 |  |
| 1 | Campanile | Tower | 307 | 1914.0 |  |
| 2 | Doe Library | Library | 80 | 1911.0 |  |
| 3 | Memorial Glade | Open Space | 0 | NaN |  |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |  |

Arguments to .loc can be:

- A list.
- A slice (syntax is inclusive of the right-hand side of the slice).
- A single value.


## Label-based Extraction: loc[]

Arguments to .loc can be:

- A list.
- A slice (syntax is inclusive of the right-hand side of the slice).
- A single value.

Column
Labels

| Landmark | Type | Height | Year Built |
| :--- | :--- | :--- | :--- |
| Sather Gate | Gate | 30 | 1910.0 |
| Campanile | Tower | 307 | 1914.0 |
| Doe Library | Library | 80 | 1911.0 |
| Memorial Glade | Open Space | 0 | NaN |
| Sproul Plaza | Plaza | 0 | 1962.0 |

## Label-based Extraction: loc[]

Arguments to .loc can be:

- A list.
- A slice (syntax is inclusive of the right-hand side of the slice).
- A single value.
df.loc[[1, 2]]

|  | Landmark | Type | Height | Year Built |
| ---: | ---: | ---: | ---: | ---: |
| 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |

df.loc[[1, 2], ['Type', 'Height', 'Year Built']]

| Row |  |  |  | Column Labels |
| :--- | :--- | :--- | :--- | :--- |
| Labels | Landmark | Type | Height | Year Built |
| 0 | Sather Gate | Gate | 30 | 1910.0 |
|  | Doe Library | Library | 307 | 1914.0 |
| ![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-25.jpg?height=147&width=62&top_left_y=1273&top_left_x=2079) | Memorial Glade | Open Space | 0 | NaN |
|  | Sproul Plaza | Plaza | 0 | 1962.0 |


|  | Type | Height | Year Built |
| :--- | :---: | ---: | ---: |
| 1 | Tower | 307 | 1914.0 |
| 2 | Library | 80 | 1911.0 |

## Label-based Extraction: loc[]

Arguments to .loc can be:

- A list.
- A slice (syntax is inclusive of the right-hand side of the slice).
- A single value.
df.loc[2:3]

|  | Landmark | Type | Height | Year Built |
| :--- | ---: | ---: | ---: | ---: |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |

df.loc[:, 'Landmark':'Height']

| Row Labels |  |  |  | Column Labels |
| :--- | :--- | :--- | :--- | :--- |
|  | Landmark | Type | Height | ear Built |
|  | Sather Gate | Gate | 30 | 1910.0 |
|  | Campanile | Tower | 307 | 1914.0 |
|  | Doe Library <br> Memorial Glade | Library | 80 | 1911.0 <br> NaN |
|  |  | Open Space | 0 |  |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |


|  | Landmark | Type | Height |
| :--- | ---: | ---: | ---: |
| 0 | Sather Gate | Gate | 30 |
| 1 | Campanile | Tower | 307 |
| 2 | Doe Library | Library | 80 |
| 3 | Memorial Glade | Open Space | 0 |
| 4 | Sproul Plaza | Plaza | 0 |

## Label-based Extraction: loc[]

Arguments to .loc can be:

- A list.
- A slice (syntax is inclusive of the right-hand side of the slice).
- A single value.

| df.loc[:, 'Landmark' |  | Sather Gate <br> Campanile <br> Doe Library <br> Memorial Glade Sproul Plaza <br> $\begin{array}{lc}4 & \text { Sproul Plaza } \\ \text { Name: } & \text { Landmark, dtyp }\end{array}$ e: object |
| :--- | :--- | :--- |
| df.loc[2] |  | df.loc[0, 'Type'] |
| Landmark <br> Doe Library <br> Type Library <br> Height <br> 80 <br> Year Built <br> 1911.0 <br> Name: 2, dtype: |  | Gate |


| Row |  |  |  | Labels |
| :--- | :--- | :--- | :--- | :--- |
| Labels | Landmark | Type | Height | Year Built |
| 0 | Sather Gate | Gate | 30 | 1910.0 |
| 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |
| ] | Sproul Plaza | Plaza | 0 | 1962.0 |

## Context-Dependent Selection

- [ ] only takes one argument, which may be:
$\bigcirc$ A slice of row numbers.
$\bigcirc$ A list of column labels.
$\bigcirc$ A single column label.
That is, [ ] is context sensitive.


## Context-Dependent Selection

- [ ] only takes one argument, which may be:
$\bigcirc$ A slice of row numbers.
$\bigcirc$ A list of column labels.
$\bigcirc$ A single column label.
df[1:3]

|  | Landmark | Type | Height | Year Built |
| ---: | ---: | ---: | ---: | ---: |
| 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |


|  | Landmark | Type | Height | Year Built |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 |
| 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |

## Context-Dependent Selection

- [ ] only takes one argument, which may be:
- A slice of row numbers.
$\bigcirc$ A list of column labels.
$\bigcirc$ A single column label.
df[['Landmark', 'Year Built']]

|  | Landmark | Year Built |
| :--- | ---: | ---: |
| 0 | Sather Gate | 1910.0 |
| 1 | Campanile | 1914.0 |
| 2 | Doe Library | 1911.0 |
| 3 | Memorial Glade | NaN |
| 4 | Sproul Plaza | 1962.0 |


|  | Landmark | Type | Height | Year Built |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 |
| 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Dpen Space | 0 | NaN |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |

## Context-Dependent Selection

- [ ] only takes one argument, which may be:
- A slice of row numbers.
$\bigcirc$ A list of column labels.
$\bigcirc$ A single column label.

df['Year Built']

| 0 | 1910.0 |
| :--- | :--- |
| 1 | 1914.0 |
| 2 | 1911.0 |
| 3 | NaN |
| 4 | 1962.0 |
| Name: | Year Built, dtype: float64 |


|  | Landmark | Type | Height | Year Built |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 |
| 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |

Which landmark name is returned by each command?

1. df.loc[1, "Landmark"]
2. df.iloc[2, 0]
3. df["Landmark"][3]

## Filtering Data

- pandas

7824888

- pandas Data Structures
- Exploring DataFrames
- Selecting and Retrieving Data in DataFrames
- Filtering Data
- DataFrame Modification
- Aggregation in DataFrame
- Joining DataFrames
- Visualization
- Matplotlib
- Plotly


## Boolean Array for Filtering a DataFram

We learned to extract data according to its integer position (.iloc) or its label (.loc)

What if we want to extract rows that satisfy a given condition?

- loc and [ ] also accept Boolean arrays as input.
- Rows corresponding to True are extracted; rows corresponding to False are not. df[df['Height'] > 50]

| Landmark | Type | Height | Year Built | 7 |  |  |  | Landmark | Type | Height | Year Built |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Gate | 30 | 1910.0 | 0 | 0 | False | 1 | Campanile | Tower | 307 | 1914.0 |
| 1 | Tower | 307 | 1914.0 |  | 1 | True | 2 |  |  | 80 | 1911.0 |
| 2 | Library | 80 | 1911.0 |  |  | False |  |  |  |  |  |
| - Memorial Slate | Spen-Space | 0 | Non | ![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-34.jpg?height=75&width=76&top_left_y=1520&top_left_x=1385) | 4 | False |  |  |  |  |  |
| ![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-34.jpg?height=62&width=429&top_left_y=1598&top_left_x=116) | Plaza | 0 | 1002.0 | ![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-34.jpg?height=114&width=117&top_left_y=1594&top_left_x=1381)

**Image Description:** This image features a stylized prohibition symbol, characterized by a red circle with a diagonal line crossing through it. Adjacent to this symbol is a brown L-shaped figure, suggesting a contextual relationship—potentially implying "not allowed" or "forbidden" in reference to the L-shape. The graphically simplified design emphasizes clarity in communicating a restriction or warning related to the L-shape, commonly used in educational contexts to denote correct or incorrect practices.
 |  | Name: Height, dtype: bool | f | 7 |  |  |  |

df['Height'] > 50 equivalent/df.loc[df['Height'] > 50]

## Combining Boolean Series for Filtering

Boolean Series can be combined using various operators, allowing filtering of results by multiple criteria.

- The \& operator allows us to apply operand_1 and operand_2
- The | operator allows us to apply operand_1 or operand_2
df[(df['Height'] > 50) \& (df['Type'] == 'Library')]

|  | Landmark | Type | Height | Year Built |  |  |  |  |  |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 |  |  |  |  |  |
| 1 | Campanile | Tower | 307 | 1914.0 |  | Landmark | Type | Height | Year Built |
| 2 | Doe Library | Library | 80 | 1911.0 | 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |  |  |  |  |  |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |  |  |  |  |  |

## Bitwise Operators

- \& and | are examples of bitwise operators. They allow us to apply multiple logical conditions.
- If $p$ and $q$ are boolean arrays or Series:

| Symbol | Usage | Meaning |
| :---: | :---: | :---: |
| $\sim$ | $\sim \mathrm{p}$ | Negation of p |
| । | $\mathrm{p} \mid \mathrm{q}$ | p OR q |
| $\&$ | $\mathrm{p} \& \mathrm{q}$ | p AND q |
| $\wedge$ | $\mathrm{p} \wedge \mathrm{q}$ | p XOR q (exclusive or) |

## DataFrame Modification

- pandas
- pandas Data Structures
- Exploring DataFrames
- Selecting and Retrieving Data in DataFrames
- Filtering Data
- DataFrame Modification
- Aggregation in DataFrame
- Joining DataFrames
- Visualization
- Matplotlib
- Plotly


## Adding and Modifying Columns

Adding a column is easy:

1. Use [ ] to reference the desired new column.
2. Assign this column to a Series or array of the appropriate length.
df['Experience'] = [2, 5, 1, 8, 4]

|  | Landmark |  | Type | Height |  | Year Built |  |  | Landmark |  | Type | Height | Year Built | Experience |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate |  | Gate | 30 |  | 1910.0 |  | 0 | Sather Gate |  | Gate | 30 | 1910.0 | 2 |
| 1 | Campanile |  | Tower | 307 |  | 1914.0 |  | 1 | Campanile |  | Tower | 307 | 1914.0 | 5 |
| 2 | Doe Library |  | Library | 80 |  | 1911.0 |  | 2 | Doe Library |  | Library | 80 | 1911.0 | 1 |
| 3 | Memorial Glade |  | Open Space | 0 |  | NaN |  | 3 | Memorial Glade |  | Open Space | 0 | NaN | 8 |
| 4 | Sproul Plaza |  | Plaza | 0 |  | 1962.0 |  | 4 | Sproul Plaza |  | Plaza | 0 | 1962.0 | 4 |
| df['Height_Increase'] = df['Height'] <br> * 0.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |


|  | Landmark | Type | Height | Year Built | Experience | Height_Increase |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 | 2 | 3.0 |
| 1 | Campanile | Tower | 307 | 1914.0 | 5 | 30.7 |
| 2 | Doe Library | Library | 80 | 1911.0 | 1 | 8.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN | 8 | 0.0 |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 | 4 | 0.0 |

## Dropping a Column

df.drop(columns=['Experience'])
print(df)

|  | Landmark | Type | Height | Year Built | Experience | Height_Increase |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 | 2 | 3.0 |
| 1 | Campanile | Tower | 307 | 1914.0 | 5 | 30.7 |
| 2 | Doe Library | Library | 80 | 1911.0 | 1 | 8.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN | 8 | 0.0 |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 | 4 | 0.0 |


|  | Landmark | Type | Height | Year Built | Experience | Height_Increase |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 | 2 | 3.0 |
| 1 | Campanile | Tower | 307 | 1914.0 | 5 | 30.7 |
| 2 | Doe Library | Library | 80 | 1911.0 | 1 | 8.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN | 8 | 0.0 |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 | 4 | 0.0 |

What happened?
Most DataFrame operations are not in-place.

## Dropping a Column

df.drop(columns=['Experience'] inplace=True)
print(df)

|  | Landmark | Type | Height | Year Built | Experience | Height_Increase |  | Landmark | Type | Height | Year Built | Height_Increase |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 | 2 | 3.0 | 0 | Sather Gate | Gate | 30 | 1910.0 | 5.0 |
| 1 | Campanile | Tower | 307 | 1914.0 | 5 | 30.7 | 1 | Campanile | Tower | 307 | 1914.0 | 30.7 |
| 2 | Doe Library | Library | 80 | 1911.0 | 1 | 8.0 | 2 | Doe Library | Library | 80 | 1911.0 | 8.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN | 8 | 0.0 | 3 | Memorial Glade | Open Space | 0 | NaN | 0.0 |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 | 4 | 0.0 | 4 | Sproul Plaza | Plaza | 0 | 1962.0 | 0.0 |

Alternatively we can directly assign:
df = df.drop(columns=['Experience'])

## Sorting DataFrame

- Sorting organizes your data for better analysis.
- We use sort_values() to sort by one or more columns in ascending or descending order.
- Syntax:
- Single column:
df.sort_values(by='Column', ascending=True)
- Multiple columns:
df.sort_values(by=['Col1', 'Col2'], ascending=[True, False])


## Sorting DataFrame by One Column

## -Single column:

df.sort_values(by='Column', ascending=True)
df = df.sort_values(by='Height') The default value for the ascending argument is True.

|  | Landmark | Type | Height | Year Built |  | Landmark | Type | Height | Year Built |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 | 3 | Memorial Glade | Open Space | 0 | NaN |
| 1 | Campanile | Tower | 307 | 1914.0 | 4 | Sproul Plaza | Plaza | 0 | 1962.0 |
| 2 | Doe Library | Library | 80 | 1911.0 | 0 | Sather Gate | Gate | 30 | 1910.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN | 2 | Doe Library | Library | 80 | 1911.0 |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 | 1 | Campanile | Tower | 307 | 1914.0 |

## Sorting DataFrame by One Column

## - Multiple columns:

$$
\begin{array}{r}
\text { df.sort_values(by=['Col1', 'Col2'],' } \\
\text { ascending=[True, True]) }
\end{array}
$$

df = df.sort_values(by=['Height',')' 'Type'], ascending=[True,

|  | Landmark | Type | Height | Year Built |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 |
| 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |


|  | Landmark | Type | Height | Year Built |
| :--- | :--- | :--- | :--- | :--- |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |
| 0 | Sather Gate | Gate | 30 | 1910.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 1 | Campanile | Tower | 307 | 1914.0 |

If you run df.sort_values('Height', ascending=False).iloc[0]['Landmark'], which landmark do you get?

## Handling Missing Values

- Missing values are a common issue in real-world datasets.
-We will explore techniques to:
- Detect missing values.
- Handle missing values by either removing or imputing them.


## Handling Missing Values

- We will explore techniques to:
- Detect missing values.
df_missing.isnull()

|  | Landmark | Type | Height | Year Built |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30.0 | NaN |
| 1 | Campanile | Tower | 307.0 | 1914.0 |
| 2 | Doe Library | Library | NaN | 1911.0 |
| 3 | Memorial Glade | Open Space | 0.0 | NaN |
| 4 | Sproul Plaza | None | 0.0 | 1962.0 |


|  | Landmark | Type | Height | Year Built |
| :--- | :---: | :--- | :---: | :---: |
| 0 | False | False | False | True |
| 1 | False | False | False | False |
| 2 | False | False | True | False |
| 3 | False | False | False | True |
| 4 | False | True | False | False |

## Handling Missing Values

- We will explore techniques to:
- Handle missing values by either removing or imputing them.
df_missing.dropna()
![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-47.jpg?height=643&width=2888&top_left_y=952&top_left_x=153)

**Image Description:** The image presents a table with data on various landmarks. It has five columns: "Landmark," "Type," "Height," "Year Built," and contains entries for landmarks like "Sather Gate," "Campanile," "Doe Library," "Memorial Glade," and "Sproul Plaza." The "Height" column includes units (feet) for some landmarks, while others display "NaN" for missing data. The "Type" column categorizes landmarks as "Gate," "Tower," "Library," "Open Space," and "None." An arrow points to a subset of this data, highlighting the "Campanile" tower with a height of 307.0 and year built 1914.0.



## Handling Missing Values

- We will explore techniques to:
- Handle missing values by either removing or imputing them.
df_missing.fillna()

|  | Landmark | Type | Height | Year Built |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30.0 | NaN |
| 1 | Campanile | Tower | 307.0 | 1914.0 |
| 2 | Doe Library | Library | NaN | 1911.0 |
| 3 | Memorial Glade | Open Space | 0.0 | NaN |
| 4 | Sproul Plaza | None | 0.0 | 1962.0 |


|  | Landmark | Type | Height | Year Built |
| :--- | ---: | ---: | ---: | ---: |
| 0 | Sather Gate | Gate | 30.0 | 0.0 |
| 1 | Campanile | Tower | 307.0 | 1914.0 |
| 2 | Doe Library | Library | 0.0 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0.0 | 0.0 |
| 4 | Sproul Plaza | 0 | 0.0 | 1962.0 |

## Aggregation in DataFrame

- pandas
- pandas Data Structures
- Exploring DataFrames
- Selecting and Retrieving Data in DataFrames
- Filtering Data
- DataFrame Modification
- Aggregation in DataFrame
- Joining DataFrames
- Visualization
- Matplotlib
- Plotly


## Aggregating Data in pandas

-We can perform aggregations on our data, such as calculating means, sums, and other summary statistics.

Basic
sum()
mean()
median()
min()
max()
count ()
nunique()- \# of unique values
prod()
df['Height'].mean()
83.4

Statistical
std()
var()
sem( )- standard error of the mean
skew()
df['Height'].std()
129.2006191935627

Logical and Index-Based
any()- True if any value is True
all()- True if all values are True
first()- first non-null value
last()- last non-null value
idxmin()- index of min value
idxmax()- index of max value
df['Height'].idxmax()
1

## Grouping Data in pandas

Our goal:

- Group together rows that fall under the same category.
- For example, group together all rows representing a Tower landmark.
- Perform an operation that aggregates across all rows in the category.
- For example, sum up the average height of all Tower landmarks.

Grouping is a powerful tool to 1 ) perform large operations, all at once and 2) summarize trends in a dataset.

## Grouping Data in pandas

## A .groupby() operation involves some combination of splitting the object, applying a function, and combining the results.

|  | Landmark | Type | Height | Year Built | Campus |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 | UC Berkeley |
| 1 | Campanile | Tower | 307 | 1914.0 | UC Berkeley |
| 2 | Doe Library | Library | 80 | 1911.0 | UC Berkeley |
| 3 | Memorial Glade | Open Space | 0 | NaN | UC Berkeley |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 | UC Berkeley |
| 5 | North Gate | Gate | 25 | 1909.0 | UC Berkeley |
| 6 | Moffitt Library | Library | 60 | 1970.0 | UC Berkeley |
| 7 | Faculty Glade | Open Space | 0 | NaN | UC Berkeley |
| 8 | Lower Sproul Plaza | Plaza | 0 | 2015.0 | UC Berkeley |
| 9 | 77 Mass Ave Entrance | Gate | 15 | 1939.0 | MIT |
| 10 | Green Building | Tower | 295 | 1964.0 | MIT |
| 11 | Barker Library | Library | 0 | 1916.0 | MIT |
| 12 | Killian Court | Open Space | 0 | 1920.0 | MIT |
| 13 | Stata Center Courtyard | Plaza | 0 | 2004.0 | MIT |
| 14 | Palm Drive Entrance | Gate | 20 | 1890.0 | Stanford |
| 15 | Hoover Tower | Tower | 285 | 1941.0 | Stanford |
| 16 | Green Library | Library | 0 | 1919.0 | Stanford |
| 17 | Main Quad | Open Space | 0 | 1891.0 | Stanford |
| 18 | White Plaza | Plaza | 0 | 1964.0 | Stanford |

## Grouping Data in pandas

augmented_df.groupby('Type')
|  | Landmark | Type | Height | Year Built | Campus |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 | UC Berkeley |
| 1 | Campanile | Tower | 307 | 1914.0 | UC Berkeley |
| 2 | Doe Library | Library | 80 | 1911.0 | UC Berkeley |
| 3 | Memorial Glade | Open Space | 0 | NaN | UC Berkeley |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 | UC Berkeley |
| 5 | North Gate | Gate | 25 | 1909.0 | UC Berkeley |
| 6 | Moffitt Library | Library | 60 | 1970.0 | UC Berkeley |
| 7 | Faculty Glade | Open Space | 0 | NaN | UC Berkeley |
| 8 | Lower Sproul Plaza | Plaza | 0 | 2015.0 | UC Berkeley |
| 9 | 77 Mass Ave Entrance | Gate | 15 | 1939.0 | MIT |
| 10 | Green Building | Tower | 295 | 1964.0 | MIT |
| 11 | Barker Library | Library | 0 | 1916.0 | MIT |
| 12 | Killian Court | Open Space | 0 | 1920.0 | MIT |
| 13 | Stata Center Courtyard | Plaza | 0 | 2004.0 | MIT |
| 14 | Palm Drive Entrance | Gate | 20 | 1890.0 | Stanford |
| 15 | Hoover Tower | Tower | 285 | 1941.0 | Stanford |
| 16 | Green Library | Library | 0 | 1919.0 | Stanford |
| 17 | Main Quad | Open Space | 0 | 1891.0 | Stanford |
| 18 | White Plaza | Plaza | 0 | 1964.0 | Stanford |


.agg(mean)

DataFrameGroupB
![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-53.jpg?height=1375&width=2084&top_left_y=480&top_left_x=1199)

**Image Description:** The image is a data table showing various landmarks at different university campuses, including UC Berkeley, MIT, and Stanford. The columns include "Landmark," "Type," "Height," and "Year Built." Each landmark is categorized into types such as "Gate," "Tower," "Plaza," "Open Space," and "Library." The "Height" and "Year Built" data are numerical, indicating the height in meters and the year of construction. Arrows point from the headings to the respective rows, emphasizing the relationship between the type and its attributes, making the information easily indexable and interpretable.


## Aggregation Functions

What goes inside of .agg( )?Any function that aggregates several values into one summary value.
-Common examples:

| In-Built Python Functions | NumPy Functions |
| :--- | :--- |
| .agg(sum) | .agg(np.sum) |
| .agg(max) | .agg(np.max) |
| .agg(min) | .agg(np.min) |
|  | .agg(np.mean) |
|  | .agg("first") <br> .agg("last") |

Some commonly-used aggregation functions can even be called directly, without the explicit use of .agg( )d:f.groupby("Type").mean()

## What will df.groupby('Type')

['Height'].mean()['Plaza'] return?

## Grouping by Multiple Columns

augmented_df.groupby([ 'Type', 'Campus '[][)' Height']]agg('max')

## Grouping by Multiple Columns: pivot_table

## augmented_df.groupby(['Type', 'Campus '[][)' Height']]agg('max')

![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-57.jpg?height=1349&width=1260&top_left_y=518&top_left_x=276)

**Image Description:** The image is a table comparing the height of different structures (Gate, Tower, Plaza, Open Space, Library) across three campuses: MIT, Stanford, and UC Berkeley. The rows indicate the structure types, while the columns represent the campuses and their corresponding heights in meters. The height values are numerically listed, showing significant variation, particularly for the "Tower" at Stanford (285 m) and "Gate" at MIT (15 m). Colored arrows imply a connection or trend among data points, enhancing visual interpretation of campus height variations.


But we have two index in our DataFrame

```
pivot_table =
    pd.pivot_table(
        augmented_df,
        index='Type',
```

```
columns='Campus',
        values='Height',
        aggfunc='max'
```

)

## Joining DataFrames

- pandas
- pandas Data Structures
- Exploring DataFrames
- Selecting and Retrieving Data in DataFrames
- Filtering Data
- DataFrame Modification
- Aggregation in DataFrame
- Joining DataFrames
- Visualization
- Matplotlib
- Plotly


## Joining Events and Landmarks

Joining DataFrames allows you to combine data from different sources based on a common key or index.

## landmarks

event_data
|  | Year |  | Event | Location |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 1868 |  | Founding of UC Berkeley | Berkeley, CA |
| 1 | 1914 |  | Completion of Campanile | Berkeley, CA |
| 2 | 1923 |  | Opening of Memorial Stadium | Berkeley, CA |
| 3 | 1964 |  | Free Speech Movement | Berkeley, CA |
| 4 | 2000 | Opening of Hearst | Memorial Mining Building | Berkeley, CA |


event_data
|  | Landmark | Type | Height | Year Built |
| :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 |
| 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |


![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-59.jpg?height=456&width=664&top_left_y=1318&top_left_x=42)

**Image Description:** This image depicts a large, classical-style building, likely a university or academic institution. The structure has prominent columns supporting an ornate roof, with multiple large windows symmetrically arranged along the facade. The building's exterior is characterized by white stone and green-tinted glass elements, suggesting a modern adaptation. In the foreground, well-kept lawns and landscaped areas are visible, enhancing the scholarly atmosphere. The clear blue sky indicates a bright, sunny day, further complementing the aesthetic of the architecture.

![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-59.jpg?height=443&width=656&top_left_y=1318&top_left_x=722)

**Image Description:** The image depicts a university campus landscape featuring a large green lawn surrounded by trees and buildings. In the foreground, the open grassy area is visible, with a few people walking or standing. The background includes several buildings, which are multi-storied and modern in design, suggesting an academic environment. The sky is clear with minimal clouds, indicating a bright, sunny day. Overall, the scene conveys a vibrant academic setting conducive to outdoor activities and social interaction.

![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-59.jpg?height=447&width=592&top_left_y=1318&top_left_x=1394)

**Image Description:** The image depicts the iconic Sather Gate, an entrance archway at the University of California, Berkeley. The structure features ornate green metalwork and stone columns, flanked by elaborate lanterns. The background shows clear blue skies and greenery, with people visible in the distance. This image serves as a representation of architectural design and university heritage, encapsulating a significant cultural symbol within an academic environment.

![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-59.jpg?height=447&width=668&top_left_y=1318&top_left_x=2058)

**Image Description:** This is a photograph depicting an outdoor campus area of a university. The image features a brick plaza bordered by trees, with several groups of people walking and congregating. In the background, there are large buildings with classical architectural elements, including columns. There is also a clock tower visible, partially obscured by trees. The sky is partly cloudy, contributing to a vibrant atmosphere. This miscellaneous image showcases a lively educational environment.

![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-59.jpg?height=533&width=405&top_left_y=1275&top_left_x=2781)

**Image Description:** The image depicts the Campanile, a prominent bell tower located at the University of California, Berkeley. It is illuminated at night, standing tall against a dark, cloudy sky. The tower's architectural features include a rectangular base, a tapering body, and a peaked roof with a clock face. Surrounding foliage is visible in the foreground, enhancing the tower's verticality. The contrast between the illuminated structure and the shadowy background creates a dramatic effect, highlighting the tower's significance as a landmark.


## Types of Join: Inner Join

- Inner Join: Returns only the rows with matching keys in both DataFraquesidmarks event_data

|  | Landmark | Type | Height | Year Built |
| :--- | ---: | ---: | ---: | ---: |
| 0 | Sather Gate | Gate | 30 | 1910.0 |
| 1 | Campanile | Tower | 307 | 1914.0 |
| 2 | Doe Library | Library | 80 | 1911.0 |
| 3 | Memorial Glade | Open Space | 0 | NaN |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |


|  | Year | Event | Location |
| :--- | :--- | :--- | :--- |
| 0 | 1868 | Founding of UC Berkeley | Berkeley, CA |
| 1 | 1914 | Completion of Campanile | Berkeley, CA |
| 2 | 1923 | Opening of Memorial Stadium | Berkeley, CA |
| 3 | 1964 | Free Speech Movement | Berkeley, CA |
| 4 | 2000 | Opening of Hearst Memorial Mining Building | Berkeley, CA |


| sult_join_inner = landmarks.join(event_data.set_index('Year'), on= 'Year Built', how='inner') |  |  |  |  |
| :--- | :--- | :--- | :--- | :--- |
| sult_merge_inner = landmarks.merge(event_data, how='inner', left_on='Year Built', <br> right_on='Year') |  |  |  |  |
|  | Landmark | Year Built | Event | Location |
| 0 | Campanile | 307 | 1914 | Berkeley, CA |

## Types of Join: Outer Join

- Outer Join: Returns all rows from both DataFrames, filling missing values with NaN where there is no match.
landmarks event data

|  | Landmark | Type | Height | Year Built | no match |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | Sather Gate | Gate | 30 | 1910.0 |  |
| 1 | Campanile | Tower | 307 | 1914.0 |  |
| 2 | Doe Library | Library | 80 | 1911.0 |  |
| 3 | Memorial Glade | Open Space | 0 | NaN |  |
| 4 | Sproul Plaza | Plaza | 0 | 1962.0 |  |

![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-61.jpg?height=507&width=1541&top_left_y=620&top_left_x=1615)

**Image Description:** The image is a table presenting historical events related to the University of California, Berkeley. It has three columns: "Year," "Event," and "Location." The "Year" column includes values such as 1868, 1914, 1923, 1964, and 2000. The corresponding events are listed beside their respective years, and the location is consistently "Berkeley, CA." A highlighted cell indicates the year 1868, which is the founding year of UC Berkeley. Color-coded outlines are used to differentiate various data points and highlight the absence of a match related to the year.


```
result_merge_outer = landmarks.merge(event_data,
    how='outer',
    left_on='Year Built',
    right_on='Year')
```


## Types of Join

- Inner Join: Returns only the rows with matching keys in both DataFrames.
- Outer Join: Returns all rows from both DataFrames, filling missing values with NaN where there is no match.
- Left Join: Returns all rows from the left DataFrame and matching rows from the right DataFrame.
- Right Join: Returns all rows from the right DataFrame and matching rows from the left DataFrame.
- pandas
- pandas Data Structures
- Exploring DataFrames
- Selecting and Retrieving Data in DataFrames


## Visualization

- Filtering Data
- DataFrame Modification
- Aggregation in DataFrame
- Joining DataFrames
- Visualization
- Matplotlib
- Plotly


## Why Do We Visualize Data?

Insight: Critical to gaining deeper insights into trends in the data.
Communication: Help convey trends in data to others.

In this class we will explore both but focus more on the first.

## Plotting Libraries

There are a wide range of tools for data visualization in machine learning.

- Weights and Biases - Commercial service used for tracking training runs and model artifacts.
- Matplotlib - Python library commonly used for static plots.
- Plotly - Cross-language interactive plotting library.

In this course we will focus on Plotly and Weights and Biases.

## Matplotlib

- pandas
- pandas Data Structures
- Exploring DataFrames
- Selecting and Retrieving Data in DataFrames
- Filtering Data
- DataFrame Modification
- Aggregation in DataFrame
- Joining DataFrames
- Visualization
- Matplotlib
- Plotly


## Matplotlib

- Matplotlib is a versatile Python library for creating static and publication-quality visualizations.

import matplotlib.pyplot as plt

Line Plot
![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-67.jpg?height=482&width=622&top_left_y=833&top_left_x=42)

**Image Description:** The diagram is a line graph titled "Average MPG by Model Year and Origin." The x-axis represents the "Model Year" ranging from 1970 to 1982, while the y-axis indicates "Miles Per Gallon (MPG)." It features three colored lines, each representing the average MPG for cars from the USA (green), Japan (orange), and Europe (blue). The graph includes shaded areas indicating the range of values for each origin, highlighting trends over the model years. Data points are connected by lines, showcasing the increase in MPG for each region over time.


Scatter Plot
![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-67.jpg?height=507&width=644&top_left_y=833&top_left_x=790)

**Image Description:** The image is a scatter plot titled "MPG vs. Weight by Origin." It shows the relationship between vehicle weight (in pounds) on the x-axis and miles per gallon (MPG) on the y-axis. Data points represent individual vehicles, colored by origin: blue for USA, orange for Japan, and green for Europe. The plot exhibits a downward trend, indicating that heavier vehicles generally have lower MPG. There are numerous data points clustered in different regions, highlighting variations in fuel efficiency across vehicle weights and origins.


Heatmap
![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-67.jpg?height=441&width=632&top_left_y=1428&top_left_x=796)

**Image Description:** The image is a correlation heatmap displaying the relationships between five variables: miles per gallon (mpg), cylinders, displacement, weight, and acceleration. The axes represent the variables, with each cell containing a correlation coefficient ranging from -1 to 1. The color gradient indicates the strength of the correlation, with dark red indicating strong positive correlations and dark blue indicating strong negative correlations. For example, mpg shows a strong negative correlation with weight and a strong positive correlation with displacement. The values are arranged in a matrix format for quick visual analysis of variable relationships.


Histogram
![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-67.jpg?height=558&width=618&top_left_y=833&top_left_x=1556)

**Image Description:** The image is a box plot depicting the distribution of miles per gallon (MPG) by car origin. The x-axis represents MPG, ranging from 0 to 45, while the y-axis indicates the count of vehicles. Each color represents different origins: blue for USA, orange for Japan, and green for Europe. The histogram displays the frequency of MPG across different categories, highlighting variations in fuel efficiency among the car origins. The title "Box Plot" is positioned prominently at the bottom.


Bar Plot
![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-67.jpg?height=498&width=580&top_left_y=842&top_left_x=2304)

**Image Description:** The image is a bar chart titled "Average MPG by Origin." The x-axis represents the "Origin" of the vehicles, categorized into three regions: Europe, Japan, and the USA. The y-axis indicates "Average MPG" (miles per gallon). Each bar is colored distinctly: blue for Europe, orange for Japan, and green for the USA. The heights of the bars represent the average fuel efficiency, with Japan having the highest average MPG, followed by Europe, and the USA showing the lowest. The chart provides a visual comparison of fuel efficiency across these regions.


![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-67.jpg?height=464&width=584&top_left_y=1403&top_left_x=1564)

**Image Description:** The image is a box plot showing the distribution of miles per gallon (MPG) for cars based on their origin: USA, Japan, and Europe. The y-axis represents MPG, while the x-axis categorizes the data by origin. Each box displays the interquartile range, median, and potential outliers. The USA box is colored green, Japan orange, and Europe blue. The median line within each box indicates central tendency, while the whiskers represent the range of the data. Outliers are depicted as individual points beyond the whiskers.


- pandas
- pandas Data Structures
- Exploring DataFrames
- Selecting and Retrieving Data in DataFrames
Plotly
- Filtering Data
- DataFrame Modification
- Aggregation in DataFrame
- Joining DataFrames
- Visualization
- Matplotlib
- Plotly


## Why Plotly?

Many would (correctly) argue you should learn MatplotLib.

- Gold standard for python visualization since ~2005.
- Most paper plots are still made with Matplotlib.
- If we had time we would teach both (we may use some in demos).

However, Matplotlib plots are static making it difficult to interactively slice.

- Weights and Bias and Plotly are designed for interaction.
- Quicker to gain insights - a focus of this class.


## Matplotlib vs plotly

## Demo

![](https://cdn.mathpix.com/cropped/2025_10_01_11c1d498f2335de2048cg-70.jpg?height=1612&width=2025&top_left_y=204&top_left_x=1241)

**Image Description:** The image contains two scatter plots related to vehicle fuel efficiency. 

1. The top plot, titled "MPG by Weight and Origin," depicts data points categorized by vehicle origin (USA, Japan, Europe) with the x-axis representing vehicle weight (in pounds) and the y-axis showing miles per gallon (MPG). The points are color-coded by origin. 

2. The bottom plot, titled "MPG vs. Weight by Origin," similarly shows MPG against weight, emphasizing trends and clustering in vehicle performance based on origin. A legend provides additional context on data points.

Both visualizations are useful for analyzing the relationship between vehicle weight, MPG, and origin.


## Three Ways to Plot using Plotly

Easiest: Using pandas built-in plotting functions

- Great place to start.

Easy + Expressive: Use Plotly Express to construct plots quickly

- Like pandas plotting functions but with more options.

Advanced: Build plot from graphics objects (like Matplotlib)

- Need to learn some basic Plotly concepts.


## Using pandas built-in Plotting Function றி <br> 7824888

Configure pandas to use Plotly (done at beginning of notebook).
pd.set_option('plotting.backend', 'plotly')
Call plot directly on your DataFrame:

```
# Make various kinds of plots 'scatter', 'bar', 'hist'
mpg.plot(kind='scatter', x='weight', y='mpg', color='origin',
    title='MPG vs. Weight by Origin',
    width=800, height=600)
```

Notice that we define:

- The kind of plot (e.g., 'scatter', 'bar')
- How columns are attached to visual elements (e.g., $\mathrm{x}, \mathrm{y}$, color, size, shape ...)


## Using Plotly Express

Very similar to calling pandas .plot functions but with a wide range of plotting capabilities (see tutorials):

```
import plotly.express as px
px.scatter(mpg, x='weight', y='mpg', color='origin', size='cylinders',
    hover_data = mpg.columns,
    title='MPG vs. Weight by Origin',
    width=800, height=600)
```

Here we pass the DataFrame (mpg) into the desired plotting function along with how columns are mapped to visual elements.

With:
px.scatter(df, x="Year Built", y="Height", size="Height", hover_name="Landmark")

What happens to the marker for Campanile (Height=307, Year Built=1914)?

## How you Can Learn More

Skim the tutorials (optional but can be fun):

- Plotly Express Overview
- Scatter Plots
- Line Plots
- Bar Plots
- Histograms and Dist Plots
- Facet Plots
- Subplots
- Discrete and Continuous Color
- Formatting Axes
- Graphics Objects

When you can't figure out how to plot something (or just want to learn more) ask an AI agent.

- Most LLMs are very good at Plotly and Matplotlib.


## Lecture 2

## Data Tools

Credit: Joseph E. Gonzalez and Narges Norouzi
Reference Book Chapters: This topic is not covered in the textbook.

