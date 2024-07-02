# Breast Cancer Detection

This project uses machine learning algorithms to detect breast cancer using a dataset containing various features of cell nuclei present in the digitized image of a fine needle aspirate (FNA) of a breast mass. The primary goal is to classify whether the breast cancer is malignant or benign.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Data Set. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The dataset includes the following columns:

- ID number
- Diagnosis (M = malignant, B = benign)
- 30 features (mean, standard error, and worst of ten real-valued features)

## Preprocessing

The preprocessing steps include:
1. Loading the dataset.
2. Converting the diagnosis column to binary values: 'M' is mapped to 1, and 'B' is mapped to 0.
3. Dropping unnecessary columns.
4. Splitting the data into training and testing sets.

## Models Used

Four machine learning models were evaluated:
1. Decision Tree Classifier
2. Support Vector Machine (SVM)
3. Gaussian Naive Bayes
4. K-Nearest Neighbors (KNN)

## Best Model and Performance

The best performing model was found to be the Support Vector Machine (SVM) with the following parameters:
- C: 2.0
- Kernel: rbf

### Confusion Matrix

The confusion matrix for the SVM model is as follows:
- [[74 1]
[ 0 39]]



- **True Positives (TP):** 39
- **True Negatives (TN):** 74
- **False Positives (FP):** 1
- **False Negatives (FN):** 0

### Accuracy

The accuracy of the SVM model is calculated as:

Accuracy = (TP + TN) / (TP + TN + FP + FN)
= (39 + 74) / (39 + 74 + 1 + 0)
= 113 / 114
= 0.9912 (approximately 99.12%)


## How to Use

1. Ensure you have the necessary libraries installed:
    ```sh
    pip install numpy pandas scikit-learn matplotlib
    ```

2. Run the Python script to load the data, preprocess it, train the models, and evaluate their performance.

3. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

## Conclusion

This project demonstrates the application of machine learning algorithms for the classification of breast cancer as malignant or benign. The SVM model achieved an accuracy of approximately 99.12%, making it a reliable choice for this classification task.

## References

- [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
