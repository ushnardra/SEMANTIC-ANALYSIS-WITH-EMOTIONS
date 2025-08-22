# SEMANTIC-ANALYSIS-WITH-EMOTIONS
# Sentiment Analysis with Emotions

## Overview
This project performs sentiment analysis on text input, classifying it into positive, negative, or neutral sentiments while also identifying associated emotions. The analysis is conducted using Python, Pandas, and NLTK's VADER sentiment analysis tool.

## Features
- **Sentiment Classification:** Classifies text as Positive, Negative, or Neutral.
- **Emotion Detection:** Identifies emotions such as joy, sadness, anger, surprise, fear, and disgust.
- **Comparison with Dataset:** Analyzes user input against a dataset of predefined reviews to find similar sentiments and emotions.
- **CSV Data Processing:** Reads and processes datasets containing text, sentiment, and emotion labels.

## Technologies Used
- **Python**: Main programming language
- **NLTK (VADER)**: Sentiment analysis tool
- **Pandas**: Data processing and manipulation
- **Matplotlib & Seaborn** (optional): Data visualization

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/sentiment-analysis-emotions.git
   cd sentiment-analysis-emotions
   ```
2. Install required dependencies:
   ```bash
   pip install pandas nltk matplotlib seaborn
   ```
3. Download NLTK resources:
   ```python
   import nltk
   nltk.download('vader_lexicon')
   ```

## Usage
1. Prepare your dataset in CSV format with the following columns:
   - `text`: The review or input text.
   - `sentiment`: Label as Positive, Negative, or Neutral.
   - `emotion`: The associated emotion (e.g., Joy, Sadness, Anger, etc.).

2. Run the sentiment analysis script:
   ```bash
   python sentiment_analysis.py
   ```
3. Input a custom text for sentiment and emotion analysis when prompted.
4. View results, including sentiment classification and detected emotions.

## Example Output
```
Enter your text: I love this product! It's amazing.
Sentiment: Positive
Emotion: Joy
```

## Visualization
(Optional) You can generate visualizations of sentiment distribution from your dataset using:
```bash
python visualize_data.py
```

## License
This project is open-source under the MIT License.

## Author
Developed by Ushnardra

