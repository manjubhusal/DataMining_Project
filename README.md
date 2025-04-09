# BMI Analysis Based on Calorie Intake

This project investigates whether Body Mass Index (BMI) classification and value can be predicted using daily calorie intake and demographics, without directly measuring weight.

## 🧠 Project Overview

BMI is a widely used metric to assess body fat based on weight and height. The goal of this project is to evaluate if daily calorie consumption patterns—along with timing and demographic features—can accurately classify or predict a person's BMI category.

We utilized both classification and regression machine learning techniques on structured weekday and weekend dietary datasets.

---

## 📁 Data Description

Two datasets were used:

Each contains 9494 individuals and 206 features:
- **Meal times and calorie intake** (61 columns each for time and calories)
- **BMI class (1: Obesity, 2: Overweight, 3: Healthy Weight)**
- **BMI values for regression**
- **Demographic data**: age, gender, height, waist circumference, etc.

---

## 🧹 Data Preprocessing

Key steps included:
- Grouping meal time and calorie columns into intervals and meal-based segments
- Handling missing data (assumed as zero intake)
- Feature selection for regression and classification
- Splitting datasets into 80% training and 20% testing

---

## 🧪 Machine Learning Methods

### Classification
- **Gaussian Naive Bayes** (simple, fast, assumes feature independence)
- **Random Forest** (ensemble of decision trees, better accuracy)

### Regression
- **Polynomial Regression** (for non-linear modeling)
- **Gradient Boosting Regressor** (sequential tree-based model with improved accuracy)

---

## 📊 Results Summary

### Using Calorie + Demographic Features:
- **Random Forest** outperformed Naive Bayes for classification (Accuracy up to ~80%)
- **Gradient Boosting** gave best regression performance (R² up to ~0.90)
  
### Using Calories Only:
- Significantly lower performance
- Accuracy around ~35% and R² near 0.0 or negative

### Insight:
> Demographic features (age, gender, etc.) greatly improve model performance.

---

## 📈 Evaluation Metrics

- Classification: **Accuracy, Precision, Recall**
- Regression: **Mean Squared Error (MSE), R² Score**
- Execution time also tracked for each method

---

## 📌 Key Takeaways

- Calorie intake **alone** isn't sufficient to estimate BMI.
- Including demographics makes models much more accurate.
- Random Forest and Gradient Boosting are well-suited for such complex datasets.
- Future work may include lifestyle and physical activity data, or using deep learning models like Neural Networks.

---


## 📚 References

This project utilizes Python libraries such as:
- `scikit-learn`
- `pandas`
- `numpy`

---


## 📌 Note

The dataset used in this project is **not included in this repository** due to privacy and sensitivity concerns. It was provided for academic use only and contains personally identifiable or sensitive information. 

If you're interested in replicating or exploring this work, we recommend using publicly available nutrition and BMI-related datasets (e.g., from [Kaggle](https://www.kaggle.com), [UCI Machine Learning Repository](https://archive.ics.uci.edu/), or [NHANES](https://www.cdc.gov/nchs/nhanes/index.htm)) with similar features.
