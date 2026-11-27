# BEANWISE

## Exploratory Data Analysis and Data Cleaning Report



**Project Domain:** Data Science and Machine Learning

**Dataset:** Dry Bean Dataset

**Source:** UCI Machine Learning Repository (Dataset ID: 602)



---



## 1. Introduction



BEANWISE is a machine learning-based application developed to classify seven varieties of dry beans using their physical and morphological characteristics.



The project uses numerical measurements extracted from bean images. Exploratory Data Analysis (EDA) and data preprocessing are important initial stages because they help understand the dataset, identify data quality issues, and prepare reliable input for machine learning models.



This document summarizes the dataset, exploratory analysis, data cleaning, and preparation activities performed during the project.



## 2. Dataset Overview



The Dry Bean Dataset contains measurements describing the shape and physical properties of individual bean samples.



### 2.1 Dataset Statistics



| Attribute | Description |

|---|---:|

| Original observations | 13,611 |

| Observations after duplicate removal | 13,543 |

| Duplicate records removed | 68 |

| Numerical input features | 16 |

| Target variable | Class |

| Number of classes | 7 |



### 2.2 Bean Categories



The dataset contains the following seven bean varieties:



1\. SEKER

2\. BARBUNYA

3\. BOMBAY

4\. CALI

5\. HOROZ

6\. SIRA

7\. DERMASON



### 2.3 Description of Input Features



The dataset contains 16 numerical measurements.



| Feature | Description |

|---|---|

| Area | Number of pixels within the bean boundary |

| Perimeter | Length of the bean boundary |

| MajorAxisLength | Length of the major axis |

| MinorAxisLength | Length of the minor axis |

| AspectRation | Ratio of major axis length to minor axis length |

| Eccentricity | Measure of deviation from a circular shape |

| ConvexArea | Area of the convex hull |

| EquivDiameter | Diameter of a circle with equivalent area |

| Extent | Ratio of bean area to bounding-box area |

| Solidity | Ratio of bean area to convex area |

| Roundness | Measure of how close the shape is to a circle |

| Compactness | Measure of shape compactness |

| ShapeFactor1 | First shape-related measurement |

| ShapeFactor2 | Second shape-related measurement |

| ShapeFactor3 | Third shape-related measurement |

| ShapeFactor4 | Fourth shape-related measurement |



## 3. Exploratory Data Analysis



Exploratory Data Analysis was performed to understand the structure, distribution, and relationships within the dataset.



### 3.1 Initial Dataset Inspection



The initial inspection focused on understanding the dataset structure.



The following aspects were examined:



- Number of observations and features.

- Column names and data types.

- Sample records from the dataset.

- Numerical descriptive statistics.

- Target variable categories.



Descriptive statistics such as mean, standard deviation, minimum, maximum, and quartiles help understand the range and variation of the measurements.



### 3.2 Class Distribution



The target variable `Class` represents the variety of each bean sample.



Studying the class distribution helps understand how observations are distributed among the seven categories.



This information is useful when selecting appropriate data splitting and model evaluation strategies.



### 3.3 Numerical Feature Distribution



Numerical feature distributions were examined to understand the variation in bean measurements.



Histograms and distribution plots help identify:



- Skewness in numerical measurements.

- Differences in feature scales.

- Concentration of observations.

- Unusual values and extreme observations.



### 3.4 Correlation Analysis



Correlation analysis helps identify relationships between numerical features.



A correlation matrix and heatmap can be used to examine the strength and direction of relationships.



Highly correlated features may contain overlapping information, which is useful to consider during feature engineering and feature selection.



### 3.5 Outlier Analysis



Outlier analysis helps identify observations with unusually high or low measurements.



Boxplots and statistical summaries can be used to examine extreme values.



However, extreme measurements are not automatically incorrect. Differences in bean size and shape may naturally produce unusual observations.



### 3.6 Comparison Across Bean Varieties



Comparing numerical measurements across the seven classes helps identify physical characteristics that distinguish one variety from another.



These comparisons provide useful information for selecting features for the classification process.



## 4. Data Cleaning and Preprocessing



Data cleaning was performed to improve the quality and consistency of the dataset before machine learning development.



### 4.1 Missing Value Inspection



The dataset was inspected for missing values.



Missing value analysis is important because incomplete observations may affect statistical analysis and model training.



### 4.2 Duplicate Record Removal



Duplicate observations were identified and removed to avoid repeated records influencing the analysis.



| Description | Number of Records |

|---|---:|

| Original dataset | 13,611 |

| Duplicate records removed | 68 |

| Final cleaned dataset | 13,543 |



The resulting dataset contains 13,543 observations.



### 4.3 Data Type Verification



Feature data types were examined to ensure that numerical measurements were represented using appropriate numerical formats.



### 4.4 Data Consistency Checks



The dataset was reviewed for consistency and suitability for subsequent machine learning operations.



The input features and target variable were checked before proceeding to the next stages.



### 4.5 Cleaned Dataset Preparation



After duplicate removal and data validation, the cleaned observations were prepared for feature engineering and model development.



## 5. Dataset Partitioning



The cleaned dataset was divided into three subsets for model development and evaluation.



| Dataset | Number of Records | Purpose |

|---|---:|---|

| Training | 9,479 | Training machine learning models |

| Validation | 2,032 | Model comparison and tuning |

| Testing | 2,032 | Independent final evaluation |

| **Total** | **13,543** | |



Stratified partitioning was used to maintain the distribution of the seven bean varieties across the subsets.



The testing subset was kept separate for independent evaluation of the selected model.



## 6. Key Findings



The exploratory analysis and data preparation stages provided the following observations:



- The dataset contains seven distinct dry bean varieties.

- Each sample is represented using 16 numerical measurements.

- The features describe different physical and morphological properties.

- Numerical features have different measurement ranges.

- Relationships between numerical features provide useful information for feature selection.

- Duplicate removal reduced the dataset from 13,611 to 13,543 observations.

- The cleaned dataset was prepared for subsequent feature engineering and classification.



## 7. Outcome of EDA and Data Cleaning



The EDA and data cleaning stages established a foundation for the remaining machine learning workflow.



The prepared dataset contains:



- 13,543 observations.

- 16 numerical input features.

- One target variable named `Class`.

- Seven dry bean varieties.



The dataset was subsequently used for feature engineering, model training, comparison, and final evaluation.



## 8. Related Project Files



The following notebooks contain the corresponding project work:



| File | Description |

|---|---|

| `02_Exploratory_Data_Analysis.ipynb` | Exploratory analysis and visualization |

| `03_Data_Cleaning_Preprocessing.ipynb` | Data cleaning and preprocessing |



## 9. Conclusion



Exploratory Data Analysis and data cleaning are essential steps in developing a reliable machine learning system.



In the BEANWISE project, these stages helped understand the Dry Bean Dataset, examine numerical measurements, investigate relationships between features, and remove duplicate observations.



The resulting cleaned dataset provides the foundation for feature engineering, machine learning model development, and performance evaluation.



---



**BEANWISE â€“ AI-Powered Dry Bean Classification Platform**

*Data Science Project Documentation*

