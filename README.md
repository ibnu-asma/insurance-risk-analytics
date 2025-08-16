# Insurance Risk Analytics

Exploratory Data Analysis (EDA) and Machine Learning for AlphaCare Insurance Solutions  
**Data:** `MachineLearningRating_v3.csv` (converted from pipe-separated .txt, Feb 2014–Aug 2015)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Key Features](#key-features)
- [Data Notes](#data-notes)
- [Results & Outputs](#results--outputs)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project analyzes insurance policy and claims data to:
- Calculate and visualize loss ratios by province, vehicle type, and gender
- Explore distributions and outliers in premiums, claims, and vehicle values
- Identify temporal trends in claim frequency and severity
- Highlight vehicle makes/models associated with high or low claim amounts

---

## Folder Structure

```
insurance-risk-analytics/
│
├── data/                   # Raw and processed data files
│   └── MachineLearningRating_v3.csv
├── notebooks/              # Jupyter notebooks for EDA and analysis
│   └── eda_analysis.ipynb
├── src/                    # Source code (data loading, preprocessing, utils)
│   ├── data_loader.py
│   ├── data_preprocessor.py
│   └── utils.py
├── plots/                  # Generated plots and visualizations
├── README.md
└── requirements.txt
```

---

## Setup & Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/insurance-risk-analytics.git
    cd insurance-risk-analytics
    ```

2. **Create and activate a virtual environment (optional but recommended):**
    ```sh
    python -m venv venv
    venv\Scripts\activate  # On Windows
    ```

3. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

---

## Usage

- **Run EDA notebook:**
    1. Open `notebooks/eda_analysis.ipynb` in Jupyter or VS Code.
    2. Run all cells to generate summary statistics and plots.

- **Convert raw TXT to CSV (if needed):**
    ```sh
    python src/data_loader.py
    ```

- **Preprocess data for modeling:**
    ```sh
    python src/data_preprocessor.py
    ```

---

## Key Features

- **Data Cleaning:** Handles missing values, mixed types, and converts columns to appropriate types.
- **Descriptive Statistics:** Summarizes key numerical and categorical features.
- **Visualization:** Generates histograms, bar charts, scatter plots, box plots, and heatmaps.
- **Loss Ratio Analysis:** Calculates and visualizes loss ratios by region and other factors.
- **Temporal Analysis:** Examines trends in claims over time.
- **Vehicle Analysis:** Identifies makes/models with highest and lowest claim amounts.

---

## Data Notes

- `CapitalOutstanding` had comma-separated numbers, converted to floats.
- `MaritalStatus`, `Gender`, `CrossBorder` had mixed types, set as categories.
- Negative `TotalPremium` and `TotalClaims` may indicate refunds or errors.
- `CustomValueEstimate` has 78% missing values.
- Some columns may have mixed types; see code for dtype handling.

---

## Results & Outputs

- **Plots:** Saved in the `plots/` directory (e.g., `total_claims_hist.png`, `loss_ratio_province.png`).
- **Summary statistics:** Printed in the EDA notebook output.
- **Cleaned data:** Saved as CSV after preprocessing.

---

## Contributing

Pull requests and suggestions are welcome!  
Please open an issue to discuss your ideas or report bugs.

---

## License

[MIT License](LICENSE)

