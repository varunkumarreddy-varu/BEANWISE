# **BEANWISE – AI-Powered Dry Bean Classification Platform**

BEANWISE is a machine learning-based web application developed to classify dry beans into seven different varieties using their physical and morphological characteristics. The system uses 16 numerical measurements obtained through computer vision techniques to identify the variety of a given bean sample.

The project is developed using Python, Scikit-learn, XGBoost, and Streamlit. It provides an easy-to-use interface where users can enter bean measurements and get predictions. It also supports batch classification through CSV file uploads, displays prediction probabilities, and provides information and images of the seven bean varieties.

## **1. Key Features of the Application**

- **Single Bean Prediction:** Users can enter all 16 morphological measurements manually and obtain the predicted bean variety.
- **Input Validation:** The application checks whether the entered values are valid. It verifies conditions such as positive measurements, major axis length being greater than or equal to minor axis length, and convex area being greater than or equal to area.
- **Out-of-Range Detection:** The system identifies values that fall outside the ranges observed in the training data and displays a warning when necessary.
- **Bean Variety Images:** After prediction, the application displays an image of the corresponding bean variety.
- **Prediction Probabilities:** Interactive Plotly charts show the estimated probabilities for all seven bean varieties.
- **Batch Prediction:** Users can upload a CSV file containing multiple bean samples and receive predictions for all records. The results can also be downloaded.
- **Variety Information:** A separate gallery provides information about all seven varieties, including their physical characteristics, botanical details, and common uses.
- **Model Evaluation:** The application uses evaluation results obtained from an independent test dataset containing 2,032 samples.

## **2. Dataset Description**

The project uses the Dry Bean Dataset from the UCI Machine Learning Repository (Dataset ID: 602), introduced by Koklu and Ozkan (2020).

The dataset contains 13,611 original records, each representing a dry bean sample described using 16 numerical features.

After removing duplicate records, the cleaned dataset contains 13,543 observations.

**Dataset Details:**

- Original records: 13,611
- Records after duplicate removal: 13,543
- Input features: 16 numerical features
- Additional engineered features: 6
- Target variable: Class
- Total bean varieties: 7

**The seven bean varieties are:**

1. SEKER
2. BARBUNYA
3. BOMBAY
4. CALI
5. HOROZ
6. SIRA
7. DERMASON

**Dataset Partitioning:**

| Dataset    | Records    |
| ---------- | ---------- |
| Training   | 9,479      |
| Validation | 2,032      |
| Testing    | 2,032      |
| **Total**  | **13,543** |

The dataset is divided into training, validation, and testing sets to train the models, compare their performance, and evaluate the final model using separate data.

## **3. Machine Learning Approach and Results**

Several machine learning algorithms were trained and compared to identify a suitable model for classifying the seven dry bean varieties.

The final classification system uses a Scikit-learn Pipeline consisting of:

- StandardScaler for feature scaling.
- Support Vector Classifier (SVC) with an RBF kernel for classification.
- Regularization parameter: C = 2.0.
- Gamma: scale.

The model was selected after comparing nine machine learning algorithms using stratified 5-fold cross-validation and hyperparameter tuning.

**Final Model Performance**

The selected model was evaluated on an independent test dataset containing 2,032 samples.

| Evaluation Metric                | Result         |
| -------------------------------- | -------------- |
| Accuracy                         | 92.22%         |
| Balanced Accuracy                | 93.26%         |
| Macro F1-Score                   | 0.9334         |
| Weighted F1-Score                | 0.9224         |
| Matthews Correlation Coefficient | 0.9060         |
| Cohen's Kappa                    | 0.9059         |
| Multiclass ROC-AUC (One-vs-Rest) | 0.9901         |
| Average Inference Latency        | 0.43 ms/sample |

These results indicate that the model performs well across the seven bean varieties. Balanced accuracy and macro F1-score also provide a better understanding of its performance across individual classes.

## **4. Technologies and Libraries Used**

| Category                   | Technologies          |
| -------------------------- | --------------------- |
| Programming Language       | Python 3.12           |
| Machine Learning           | Scikit-learn, XGBoost |
| Data Processing            | Pandas, NumPy         |
| Statistical Analysis       | SciPy                 |
| Model Saving and Loading   | Joblib                |
| Web Application            | Streamlit             |
| Interactive Visualizations | Plotly                |
| Data Visualization         | Matplotlib, Seaborn   |
| Automated Testing          | Pytest                |

The project uses the global Python environment on Windows.

## **5. How to Run the Application**

To start the BEANWISE application, open a terminal in the project folder and execute:

```
python -m streamlit run app.py
```

After the application starts, open the following address in your browser:

[http://localhost:8501](http://localhost:8501?utm_source=chatgpt.com)

To execute the automated test suite, run:

```
python -m pytest -v
```

## **6. Project Folder Structure**

The project is organized into separate folders to make the code easier to understand, maintain, and extend.

```
BEANWISE/
│
├── app.py
├── requirements.txt
├── README.md
│
├── config/
│   └── settings.py
│
├── dataset/
│   ├── raw/
│   └── processed/
│
├── assets/
│   ├── bean_images/
│   │   ├── barbunya.jpg
│   │   ├── bombay.jpg
│   │   ├── cali.jpg
│   │   ├── dermason.jpg
│   │   ├── horoz.jpg
│   │   ├── seker.jpg
│   │   ├── sira.jpg
│   │   └── IMAGE_SOURCES.md
│   │
│   └── templates/
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── prediction/
│
├── notebooks/
│   ├── 01_Project_Initialization.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_Data_Cleaning_Preprocessing.ipynb
│   ├── 04_Feature_Engineering_Selection.ipynb
│   ├── 05_Machine_Learning_Model_Training.ipynb
│   └── 06_Final_Model_Evaluation.ipynb
│
├── artifacts/
│   ├── models/
│   ├── encoders/
│   └── metadata/
│
├── reports/
│   ├── phase2/
│   ├── phase3/
│   ├── phase4/
│   ├── phase5/
│   ├── phase6/
│   └── final_report/
│
└── tests/
```

Each folder has a separate purpose. The notebooks contain the analysis and experiments, `src` contains the main Python modules, `artifacts` stores trained models and related files, and `tests` contains automated tests used to check the project.

## **7. Limitations and Future Improvements**

Although BEANWISE provides good classification performance, there are some limitations:

- The model is trained using the available UCI Dry Bean dataset, so its performance on samples from different geographical regions or growing conditions may vary.
- The classification depends on the accuracy of the numerical measurements supplied by the user.
- Some varieties have similar physical characteristics, which can make classification difficult.
- Predictions for measurements outside the training data range may be less reliable.
- The probability values displayed by the model should be treated as estimates and not as guaranteed confidence.

In the current results, BOMBAY, SEKER, and HOROZ achieve F1-scores of at least 0.95, while some confusion remains between DERMASON and SIRA because of similarities in their physical characteristics.

**Future Improvements:**

- Adding image-based bean classification using computer vision and deep learning.
- Testing the model with additional datasets.
- Improving prediction probability calibration.
- Developing a mobile-friendly version.
- Evaluating the system using real-world agricultural samples.

## **Conclusion**

BEANWISE demonstrates how machine learning can be used to classify dry bean varieties based on their physical measurements. By combining data preprocessing, feature engineering, model comparison, and an interactive Streamlit application, the project provides a practical approach to automated bean classification.

The project also helped in understanding the complete Data Science workflow, from exploring and preparing a dataset to training machine learning models, evaluating their performance, and building a working application.