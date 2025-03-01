# Credit Score Prediction using Random Forest Classifier

## Overview

This project focuses on predicting credit scores using a **Random Forest Classifier** and visualizing the results using **Plotly**. The dataset contains various financial and demographic factors that influence a person's creditworthiness.

## Features

- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling.
- **Model Training**: Using Random Forest Classifier for prediction.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score.
- **Visualization**: Plotly is used to generate interactive graphs for better data insights.

## Requirements

Before running the project, install the required dependencies:

```bash
pip install pandas numpy scikit-learn plotly matplotlib seaborn
```

## Dataset

The dataset used for this project includes various features such as:

- **Age**
- **Annual Income**
- **Credit Utilization**
- **Number of Late Payments**
- **Debt-to-Income Ratio**
- **Credit History Length**

Ensure the dataset is in CSV format and properly cleaned before proceeding.

## Implementation Steps

1. **Load the dataset**: Using Pandas to read and preprocess the data.
2. **Data Cleaning**: Handle missing values and perform encoding.
3. **Feature Selection**: Selecting relevant features for better accuracy.
4. **Train-Test Split**: Splitting data into training and testing sets.
5. **Model Training**: Training Random Forest Classifier on the dataset.
6. **Model Evaluation**: Checking performance using accuracy and confusion matrix.
7. **Visualization**: Plotly graphs to represent feature importance, prediction distribution, and evaluation metrics.

## Results

- **Accuracy Achieved**: \~82% (varies based on dataset and preprocessing in test dataset), 100% accuracy in train dataset
- **Feature Importance Analysis**: Helps in understanding key factors affecting credit scores.
- **Interactive Visualizations**: Plotly provides an engaging way to analyze data.

## Conclusion

This project demonstrates the application of machine learning in **credit score prediction** using **Random Forest Classifier**. With **Plotly**, we gain deeper insights into the data, making predictions more interpretable.

## Future Enhancements

- Try **Hyperparameter Tuning** to improve model performance.
- Implement **other ML models** like XGBoost or Neural Networks for comparison.
- Deploy the model as a web app using Flask or Streamlit.

## Author

**Name** : Rudranarayan Sahu
**Email** : rudranarayansahu.tech@gmail.com

## License

This project is open-source and available under the MIT License.


Add some feature ✨✨
 
