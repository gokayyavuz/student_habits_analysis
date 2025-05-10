# 📚 Student Habits & Performance Analysis

This project investigates the impact of student lifestyle and study habits on exam performance using statistical analysis and predictive modeling. The goal is to identify which behavioral and contextual factors most significantly affect academic success.

---

## 🎯 Objectives

- Explore and visualize student behavior data (sleep, diet, study time, job, internet quality, etc.)
- Encode categorical variables and scale numeric data
- Build and compare regression models to predict exam scores
- Interpret key features using correlation heatmaps and model outputs

---

## 📊 Dataset Overview

The dataset includes the following categories:
- **Numerical Features:** `sleep_hours`, `study_hours`, `exercise_hours`, `screen_time`, `exam_score`
- **Categorical Features:** `gender`, `part_time_job`, `extracurricular_participation`, `diet_quality`, `parental_education_level`, `internet_quality`

---
## 🧰 Technologies Used

- **Programming Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn
- **Models:** Linear Regression, Decision Tree Regression

---

## 🛠 Workflow Summary

1. **Data Cleaning:** Check for nulls, drop unnecessary columns
2. **Encoding:**  
   - Label Encoding: `diet_quality`, `parental_education_level`, `internet_quality`  
   - One-Hot Encoding: `gender`, `part_time_job`, `extracurricular_participation`
3. **Visualization:** Histograms for distributions, heatmap for correlation
4. **Feature Scaling:** StandardScaler applied to all numeric input features
5. **Modeling:**  
   - Linear Regression  
   - Ridge Regression  
   - Lasso Regression  
   - Random Forest Regressor
6. **Evaluation:** R² and RMSE on test set

---

## 🧪 Model Comparison (Results)

| Model           | R² Score | RMSE   |
|----------------|----------|--------|
| Linear          | ~0.899    | ~5.09  |
| Ridge           | ~0.899    | ~5.09  |
| Lasso           | ~0.898    | ~5.10  |
| Random Forest   | ~0.810    | ~6.98  |

🔍 *Linear Regression performed the best, suggesting non-linear relationships between variables and exam performance.*

---

## 🔍 Key Insights

- **Study hours** and **sleep duration** show positive correlation with performance
- **High-quality internet** and **parental education** may boost results
- **Poor diet quality** and high **screen time** are negatively associated
- **Extracurricular participation** appears neutral or slightly beneficial

---

## 🧠 What I Learned

- How to handle mixed-type datasets with both categorical and numerical variables
- The importance of encoding strategies for ML models
- Model comparison using evaluation metrics (R², RMSE)
- The strength of ensemble models for capturing non-linear patterns
