
# NeuralNest

This repository contains files for a project focused on predicting resale flat prices using machine learning models. The main components of the project include a Neural Network, Random Forest, and XGBoost models.

## Project Structure

- **Neural Network**
  - **NeuralNest_App.py**: This is the main Python file for the NeuralNest application, which utilizes the trained neural network model to make predictions.
  - **Neural_Network.ipynb**: Jupyter notebook for training the neural network model on the resale flat price dataset. The notebook includes data preprocessing, model architecture, training, and evaluation steps.
  - **NN_model.h5**: The saved neural network model in `.h5` format.
  - **NN_scaler.pkl**: Scaler used for normalizing the data before feeding it into the neural network model.
  - **requirements.txt**: Lists all necessary packages and libraries required for running the project.
  - **Resale Flat Price.csv**: Dataset used for training the neural network.
  - **street_name_categories.pkl** & **town_categories.pkl**: Encoded categories for `street_name` and `town` used in the neural network.
  - **resale_price_trend.png**: Visual representation of the resale price trends over time.

- **Random Forest and XGBoost**
  - **RF_XGB.ipynb**: Jupyter notebook for training the Random Forest and XGBoost models on the resale flat price dataset. This notebook also includes hyperparameter tuning, model evaluation, and feature importance visualization.
  - **Resale Flat Price.csv**: Dataset used for training the Random Forest and XGBoost models.

## Instructions

1. **Setting up the environment**:
   - Ensure that all dependencies are installed by running:
     ```bash
     pip install -r requirements.txt
     ```

2. **Running NeuralNest App**:
   - To use the NeuralNest application, run `NeuralNest_App.py`. This script loads the trained neural network model and scaler to make predictions on new data.

3. **Model Training Notebooks**:
   - The `Neural_Network.ipynb` and `RF_XGB.ipynb` notebooks can be used to train and evaluate the respective models. Follow the steps within each notebook to understand data preprocessing, model training, and evaluation procedures.

## Dataset

The dataset used in this project (`Resale Flat Price.csv`) contains historical resale prices of flats, which has been preprocessed and used to train the models. For further details, refer to the respective model training notebooks.

## Results

- **Neural Network Model**: The neural network model architecture and training process are detailed in `Neural_Network.ipynb`.
- **Random Forest and XGBoost Models**: Feature importance and hyperparameter tuning for Random Forest and XGBoost models are documented in `RF_XGB.ipynb`.

## Authors

- **Low Jow Loon Jovian** - A0218112W
- **Sugimoto Shoujin** - A0265946M
- **Chang Xinzhou Leslie** - A0233010H
- **Cheng Zhibin, Nicholas** - A0217486W
- **Mai Youlian** - A0222998M

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
