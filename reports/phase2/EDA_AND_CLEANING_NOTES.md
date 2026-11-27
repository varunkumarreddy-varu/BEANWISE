\# BEANWISE – Exploratory Data Analysis and Data Cleaning



\## 1. Project Overview



BEANWISE is a machine learning-based dry bean classification project. The objective is to classify dry bean varieties using their physical and morphological characteristics.



The project uses the Dry Bean Dataset obtained from the UCI Machine Learning Repository. It contains numerical measurements extracted from bean images.



\## 2. Dataset Description



The dataset consists of 13,611 original observations and 16 numerical input features.



Each observation represents one dry bean sample. The target variable is `Class`, which identifies the bean variety.



The dataset contains seven varieties:



1\. SEKER

2\. BARBUNYA

3\. BOMBAY

4\. CALI

5\. HOROZ

6\. SIRA

7\. DERMASON



\### Important Features



| Feature | Description |

|---|---|

| Area | Total number of pixels inside the bean |

| Perimeter | Circumference of the bean |

| MajorAxisLength | Length of the major axis |

| MinorAxisLength | Length of the minor axis |

| AspectRation | Ratio of major to minor axis |

| Eccentricity | Measure of deviation from circular shape |

| ConvexArea | Area of the convex hull |

| EquivDiameter | Equivalent diameter of the bean |

| Extent | Ratio of bean area to bounding-box area |

| Solidity | Ratio of area to convex area |

| Roundness | Measure of circularity |

| Compactness | Measure of compactness |

| ShapeFactor1–4 | Additional shape-related measurements |



\## 3. Exploratory Data Analysis



Exploratory Data Analysis (EDA) was performed to understand the dataset, identify patterns, examine feature relationships, and study differences between bean varieties.



\### 3.1 Dataset Inspection



\- Examined the number of rows and columns.

\- Identified numerical and categorical attributes.

\- Checked feature data types.

\- Reviewed sample records.

\- Examined descriptive statistics such as mean, median, standard deviation, minimum, and maximum.



\### 3.2 Class Distribution Analysis



The target variable `Class` was examined to understand the distribution of the seven bean varieties.



This analysis helps identify class imbalance and understand how many observations belong to each variety.



\### 3.3 Feature Distribution Analysis



Histograms and distribution plots were used to study numerical feature distributions.



The analysis helped identify:



\- Skewed distributions.

\- Differences in measurement ranges.

\- Unusual observations.

\- Features with varying levels of dispersion.



\### 3.4 Correlation Analysis



Correlation analysis was used to examine relationships between numerical features.



A correlation heatmap helped identify strongly related measurements and potential multicollinearity.



\### 3.5 Outlier Analysis



Boxplots and statistical summaries were used to investigate extreme observations.



Outliers were examined carefully because some extreme measurements may represent genuine differences in bean sizes and shapes rather than incorrect data.



\### 3.6 Feature Comparison Across Classes



Feature comparisons were used to understand how physical characteristics differ among the seven bean varieties.



These comparisons helped identify potentially useful features for classification.



\## 4. Data Cleaning and Preprocessing



Data cleaning was performed to improve dataset quality and prepare the observations for machine learning.



\### 4.1 Missing Value Inspection



The dataset was checked for missing values across all columns.



Missing values were examined before proceeding with model development.



\### 4.2 Duplicate Record Removal



Duplicate observations were identified and removed to avoid repeated records affecting the analysis.



| Description | Records |

|---|---:|

| Original dataset | 13,611 |

| Duplicate records removed | 68 |

| Cleaned dataset | 13,543 |



\### 4.3 Data Type Validation



Feature data types were examined to ensure that numerical measurements were stored in suitable numerical formats.



\### 4.4 Data Validation



The cleaned dataset was checked for consistency and suitability for subsequent analysis.



The target variable and input features were reviewed before model development.



\### 4.5 Dataset Preparation



The cleaned dataset was prepared for subsequent feature engineering, model training, and evaluation.



\## 5. Dataset Partitioning



The cleaned dataset was divided into three subsets.



| Dataset | Records | Purpose |

|---|---:|---|

| Training | 9,479 | Model training |

| Validation | 2,032 | Model comparison and tuning |

| Testing | 2,032 | Independent final evaluation |

| \*\*Total\*\* | \*\*13,543\*\* | |



Stratified partitioning was used to maintain the class distribution across the subsets.



\## 6. Key Observations



The EDA process provided the following insights:



\- The dataset contains seven different dry bean varieties.

\- Each variety has distinct physical and morphological characteristics.

\- Numerical features have different measurement scales.

\- Some features show strong correlations with other measurements.

\- Feature distributions vary across bean varieties.

\- Data cleaning reduced the dataset from 13,611 to 13,543 observations after duplicate removal.



\## 7. Conclusion



Exploratory Data Analysis helped develop a better understanding of the Dry Bean Dataset and its underlying patterns.



The data cleaning process removed duplicate observations and prepared the dataset for further analysis.



The resulting dataset contains 13,543 observations and 16 numerical input features. It was subsequently used for feature engineering, machine learning model development, and independent evaluation.



\## 8. Related Project Files



\- `notebooks/02\_Exploratory\_Data\_Analysis.ipynb`

\- `notebooks/03\_Data\_Cleaning\_Preprocessing.ipynb`

\- `reports/phase2/`



\---



\*\*Project:\*\* BEANWISE – AI-Powered Dry Bean Classification Platform  

\*\*Domain:\*\* Data Science and Machine Learning  

\*\*Dataset Source:\*\* UCI Machine Learning Repository – Dry Bean Dataset (ID: 602)

