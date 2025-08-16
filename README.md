Insurance Risk Analytics Project
Overview
This project analyzes historical car insurance data (Feb 2014–Aug 2015) for AlphaCare Insurance Solutions (ACIS) to optimize marketing strategies and identify low-risk segments for premium reductions. It involves Git setup, EDA (via Jupyter notebooks), A/B testing, and predictive modeling.
Data Structure

Policy: UnderwrittenCoverID, PolicyID
Transaction: TransactionMonth
Client: IsVATRegistered, Citizenship, LegalType, Title, Language, Bank, AccountType, MaritalStatus, Gender
Location: Country, Province, PostalCode, MainCrestaZone, SubCrestaZone
Car: ItemType, Mmcode, VehicleType, RegistrationYear, Make, Model, etc.
Plan: SumInsured, TermFrequency, CalculatedPremiumPerTerm, etc.
Financial: TotalPremium, TotalClaims

Setup

Clone the repository:git clone https://github.com/your-username/insurance-risk-analytics.git


Install dependencies:pip install -r requirements.txt


Run EDA notebook:jupyter notebook notebooks/eda_analysis.ipynb


Run tests:pytest tests/



Project Structure

data/: Stores insurance_data.csv.
notebooks/: Contains eda_analysis.ipynb for EDA.
src/: Python modules for data loading and utilities.
tests/: Unit tests for Python modules.
plots/: Output directory for visualizations.

CI/CD

GitHub Actions runs flake8 for linting and pytest for tests on push/PRs to main and task-* branches.

Tasks

Task 1: Git setup, EDA, and statistical analysis (see notebooks/eda_analysis.ipynb).
Task 2: Data versioning with DVC.
Task 3: A/B hypothesis testing.
Task 4: Predictive modeling.

