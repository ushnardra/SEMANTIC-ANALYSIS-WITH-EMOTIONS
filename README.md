# Semantic Analysis with Emotion Detection

## Project Overview
This project is a web-based "Semantic Analysis with Emotion Detection" application built using **Python** and **Streamlit**. It utilizes Machine Learning (Logistic Regression) to analyze text input and predict the underlying emotion.

The application takes a user's text input, processes it through a trained model, and outputs the predicted emotion along with a corresponding emoji.

## Key Features
-   **Real-time Emotion Detection**: Instantly analyzes text to detect emotions like Joy, Fear, Anger, Love, Sadness, and Surprise.
-   **Interactive UI**: Built with Streamlit for a clean and responsive user experience.
-   **Machine Learning Model**: Powered by a Logistic Regression model trained on a balanced dataset of emotions.
-   **Confidence Score**: (Optional Future Feature) Displays the probability/confidence of the prediction.

## Technologies Used
-   **Python 3.x**
-   **Streamlit**: For the web application interface.
-   **Scikit-learn**: For the machine learning model building and pipelining.
-   **Pandas & NumPy**: For data manipulation and analysis.
-   **NeatText**: For text cleaning and preprocessing.
-   **Altair**: For visualizations (if applicable in extended versions).
-   **Joblib**: For serializing and loading the trained model.

## Installation & Setup

### Prerequisites
-   Python 3.7 or higher installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/ushnardra/SEMANTIC-ANALYSIS-WITH-EMOTIONS.git
cd SEMANTIC-ANALYSIS-WITH-EMOTIONS
```

### Step 2: Create a Virtual Environment (Recommended)
It is good practice to create a virtual environment to manage dependencies.

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
Install the required Python packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application
To start the Streamlit app, run the following command in your terminal:

```bash
streamlit run app.py
```

The application will launch in your default web browser (usually at `http://localhost:8501`).

### How to Use
1.  Enter a sentence or paragraph in the text area provided.
2.  Click the **Submit** button.
3.  The application will display:
    -   The original text.
    -   The predicted emotion (e.g., "joy", "sadness") with an emoji.

## Project Structure
```
SEMANTIC-ANALYSIS-WITH-EMOTIONS/
├── app.py                  # Main Streamlit application file
├── requirements.txt        # List of Python dependencies
├── balancing.py            # Utility script for data balancing (if used)
├── Untitled-1.ipynb        # Jupyter Notebook for data exploration & model training
├── data/
│   ├── balanced_emotion.csv      # Processed dataset
│   └── (other data files)
├── model/
│   ├── balanced_emotion.pkl      # Trained Machine Learning Model
│   └── (other model files)
└── README.md               # Project documentation
```

## Model Training
The model is trained using the `Untitled-1.ipynb` notebook. It preprocesses the data using `NeatText` to remove noise (handles, stopwords) and uses a `CountVectorizer` paired with `Logistic Regression` for classification.

## Contributing
Contributions are welcome! If you have suggestions for improvements or want to add new features, feel free to fork the repository and submit a pull request.
