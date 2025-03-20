# 📈 Stock Search & Forecasting Web App

This is a Streamlit-based web application built as a side project to showcase skills in Python, data visualization, and machine learning. The app is designed for financial analysis and wealth management, allowing users to:

- **Search for Stocks:** Retrieve and display key financial indicators and historical data for any given stock.
- **Forecast Stock Prices:** Experiment with different machine learning models and tuning parameters to forecast future stock prices.


[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stockforecastingwebapp-jackmao.streamlit.app/)


## Features

- **Interactive Stock Dashboard:**  
  - Search for a stock by its symbol (e.g., TSLA).
  - Display key indicators such as Open, High, Low, Close, and Volume.
  - Visualize historical price trends using interactive charts.

- **Machine Learning Forecasting:**  
  - Choose from multiple machine learning models for forecasting.
  - Adjust tuning parameters to optimize predictions.
  - Compare forecasted stock prices against historical data.

## Technologies Used

- **[Python](https://www.python.org/):** Core programming language.
- **[Streamlit](https://streamlit.io/):** Framework for building the interactive web app.
- **[Pandas](https://pandas.pydata.org/):** Data manipulation and analysis.
- **[Matplotlib](https://matplotlib.org/)/[Plotly](https://plotly.com/):** Data visualization libraries.
- **[yfinance](https://pypi.org/project/yfinance/) & [Alpha Vantage](https://www.alphavantage.co/) APIs:** Data sources for stock information.
- **Scikit-learn/TensorFlow/Keras:** Machine learning libraries for forecasting models.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/stock-forecasting-app.git
   cd stock-forecasting-app
   ```

2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment:**

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

4. **Install the Required Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application:**

   ```bash
   streamlit run run_streamlit.py
   ```

2. **Interact with the App:**
   - Use the search bar to enter a stock ticker.
   - Explore the dashboard to view key indicators and historical trends.
   - Switch to the forecasting section to select a machine learning model, adjust its parameters, and view the forecasted stock prices.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please:
- Fork the repository.
- Create a new branch for your feature or bug fix.
- Commit your changes.
- Open a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Thanks to the developers of [Streamlit](https://streamlit.io/), [yfinance](https://pypi.org/project/yfinance/), and [Alpha Vantage](https://www.alphavantage.co/) for providing powerful tools and APIs that make this project possible.
- Special thanks to the community for their continuous support and feedback.

---

Feel free to adjust any sections to better match your project's specifics or personal style.
