# 💎 Diamond Price Predictor

An end-to-end Machine Learning project that predicts diamond prices based on their physical characteristics and quality attributes. The project includes data cleaning, exploratory data analysis, model comparison, hyperparameter tuning, and an interactive web application built using **Streamlit**.

## 🚀 Live Demo

**Try the application:** [Diamond Price Predictor](https://diamond-price-predictor-vnlp9hyxchx5ab95pbi6wc.streamlit.app/)

## 📸 Application Preview

### Prediction Interface
<img width="1465" height="791" alt="image" src="https://github.com/user-attachments/assets/f79860e1-2bfb-4dac-80d6-a782fc7eed49" />


### Prediction Results
<img width="1458" height="804" alt="image" src="https://github.com/user-attachments/assets/8531a344-a973-4f0a-9ef3-b0998545aa5e" />

## 📌 Project Overview

The objective of this project is to build a machine learning model that estimates the price of a diamond using its characteristics.

The dataset was cleaned and explored, and multiple regression algorithms were trained and compared to identify the best-performing model. The selected model was then integrated into an interactive Streamlit application.

## ✨ Features

- Predicts diamond prices based on user-provided attributes.
- Interactive and user-friendly Streamlit interface.
- Supports different diamond cuts, colors, and clarity grades.
- Uses preprocessing techniques for numerical and categorical features.
- Provides an estimated price range along with the predicted price.
- Includes an optional INR price conversion.
- Displays a plain-English summary of the diamond.

## 🧠 Machine Learning Workflow

1. **Data Collection:** Loaded the diamond dataset.
2. **Data Cleaning:** Handled data quality issues and prepared the dataset.
3. **Exploratory Data Analysis:** Explored relationships between diamond characteristics and prices.
4. **Feature Engineering:** Prepared numerical and categorical features.
5. **Model Training:** Compared six regression algorithms:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Bagging Regressor
6. **Model Evaluation:** Compared model performance and tuned the best-performing model.
7. **Deployment:** Integrated the trained model into a Streamlit web application.

## 💎 Input Features

The application uses the following diamond characteristics:

| Feature | Description |
|---|---|
| Carat | Weight of the diamond |
| Cut | Quality of the diamond's cut |
| Color | Diamond color grade |
| Clarity | Diamond clarity grade |
| Depth | Total depth percentage |
| Table | Width of the top facet relative to the widest point |
| X | Length in mm |
| Y | Width in mm |
| Z | Depth in mm |

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Web Application:** Streamlit
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
diamond-price-predictor/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── config.toml
│
├── diamond_price_model.pkl
├── diamond_encoder.pkl
├── diamond_scaler.pkl
├── diamond_numerical_features.pkl
├── diamond_categorical_features.pkl
├── diamond_feature_columns.pkl
│
├── diamonds.csv
├── diamond_prediction (1).ipynb
│
├── assets/
│   ├── home-page.png
│   ├── prediction-page.png
│   └── prediction-result.png
│
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/vedika-kulkarni/diamond-price-predictor.git
```

### 2. Navigate to the project directory

```bash
cd diamond-price-predictor
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📦 Model Artifacts

The following serialized files are used by the application:

- `diamond_price_model.pkl` – Trained regression model.
- `diamond_encoder.pkl` – Categorical feature encoder.
- `diamond_scaler.pkl` – Numerical feature scaler.
- `diamond_numerical_features.pkl` – Numerical feature names.
- `diamond_categorical_features.pkl` – Categorical feature names.
- `diamond_feature_columns.pkl` – Model feature column information.

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

To deploy your own version:

1. Fork or clone this repository.
2. Connect your GitHub account to Streamlit Community Cloud.
3. Select this repository and the `main` branch.
4. Set the main file path to `app.py`.
5. Configure the Python version and deploy.

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- End-to-end machine learning development.
- Data preprocessing and feature engineering.
- Comparing and evaluating regression algorithms.
- Hyperparameter tuning and model selection.
- Saving and loading machine learning models using Joblib.
- Building interactive applications using Streamlit.
- Deploying machine learning applications using Streamlit Community Cloud.

## 👩‍💻 Author

**Vedika Kulkarni**

GitHub: [@vedika-kulkarni](https://github.com/vedika-kulkarni)

---

⭐ If you find this project interesting, feel free to star the repository!
