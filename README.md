# Machine Learning Regression Project 📊

This project involves building and deploying a machine learning regression model using various algorithms. The model is integrated with a frontend application developed with **Streamlit**, providing an interactive interface to predict outcomes based on user inputs.

## Table of Contents 📝
- [Machine Learning Regression Project 📊](#machine-learning-regression-project-)
  - [Table of Contents 📝](#table-of-contents-)
  - [Project Overview](#project-overview)
  - [Technologies Used](#technologies-used)
  - [Setup Instructions 🛠️](#setup-instructions-️)
  - [Model 🧠](#model-)
    - [Model Training Steps 🔧:](#model-training-steps-)
    - [Model Evaluation 📈](#model-evaluation-)
    - [Model Saving 💾](#model-saving-)
  - [Frontend (Streamlit App) 🌐](#frontend-streamlit-app-)
    - [Key Features ⚙️:](#key-features-️)
  - [Usage 📅](#usage-)
    - [Step 1: Input Data](#step-1-input-data)
    - [Step 2: Make Predictions](#step-2-make-predictions)
    - [Step 3: View Results](#step-3-view-results)
  - [Data 📊](#data-)
  - [Contributing 🤝](#contributing-)
  - [License 📝](#license-)

## Project Overview 

This project aims to predict a continuous target variable based on input features using various regression models such as **Ridge**, **Lasso**, and **Random Forest**. The backend is powered by **scikit-learn** and **XGBoost**. The user interacts with the model through a **Streamlit** application, which provides real-time predictions based on user input.

## Technologies Used 

- **Backend**:
  - Python 3.x 🐍
  - `pandas` 📊 (for data manipulation and cleaning)
  - `numpy` ➗ (for numerical calculations)
  - `scikit-learn` 🔍 (for machine learning models like Ridge, Lasso, etc.)
  - `xgboost` 🚀 (for XGBoost regression model)
  - `pickle` 📦 (for serializing the trained model)

- **Frontend**:
  - `Streamlit` 🌐 (to build the interactive web application interface)
  - `matplotlib` & `seaborn` 📉 (for visualizations)

- **Data Processing & Model Training**:
  - Jupyter Notebooks 📓 for model development and training

## Setup Instructions 🛠️

To set up the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BlackShort/mental_health.git
   cd mental_health
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate 

   # On Windows, use 
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Running the Streamlit App**:
   To start the Streamlit app (frontend), run the following command:
   ```bash
   streamlit run app.py
   ```

   This will open a local development server where you can interact with the model via the web interface.

5. **Jupyter Notebook**:
   To train the models and explore data analysis, open the `.ipynb` file:
   ```bash
   jupyter notebook model.ipynb
   ```

## Model 🧠

The machine learning model has been developed and trained using the following regression algorithms:

- **Ridge Regression** 🔒
- **Lasso Regression** 🧹
- **ElasticNet Regression** 🧬
- **DecisionTreeRegressor** 🌳
- **RandomForestRegressor** 🌲
- **GradientBoostingRegressor** ⚡
- **XGBoost (XGBRegressor)** 🏆
- **KNeighborsRegressor** 🏃
- **MLPRegressor** 💻

### Model Training Steps 🔧:

1. Data preprocessing using `pandas` (e.g., handling missing values, encoding categorical features, etc.).
2. Splitting the dataset into training and test sets using `train_test_split`.
3. Model selection and training using `scikit-learn`'s regression algorithms.
4. Evaluation of model performance using metrics like Mean Squared Error (MSE) and R² score.
5. Saving the trained model using `pickle` for deployment.

### Model Evaluation 📈

The models are evaluated using the following metrics:

- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
- **R² Score**: Indicates how well the model explains the variability in the data.

After evaluating the models, the one with the best performance is selected for deployment.

### Model Saving 💾

The trained model is serialized using `pickle` to be loaded and used in the Streamlit app for predictions. This allows the app to use the pre-trained model without needing to retrain it every time.

## Frontend (Streamlit App) 🌐

The frontend is built using **Streamlit**, a Python library for creating interactive web applications. It allows users to input data, trigger predictions, and view results in real-time.

### Key Features ⚙️:

- **Input Fields**: Users can input numerical features for prediction.
- **Model Prediction**: Once data is input, the model predicts the output in real-time.
- **Data Visualizations**: Includes plots and graphs to visualize trends and results.

To run the Streamlit app, use the following command:
```bash
streamlit run app.py
```

Once the app is running, visit `http://localhost:8501` in your browser to interact with it.

## Usage 📅

### Step 1: Input Data
- Enter the required numerical data into the input fields on the web interface.

### Step 2: Make Predictions
- After entering the data, click the "Predict" button to get the model's prediction.

### Step 3: View Results
- The predicted result will be displayed along with visualizations of the data.

## Data 📊

The dataset used in this project is [Dataset Name], which contains [number] rows and [number] features. It includes data like:

- [List of features]
- The dataset was preprocessed by handling missing values, scaling numerical features, and encoding categorical variables.

## Contributing 🤝

We welcome contributions to the project! To contribute:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes and commit them
4. Push your changes to your fork
5. Open a pull request to merge your changes into the main branch

Please ensure your code follows the project’s coding style and add tests if applicable.

## License 📝

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 
