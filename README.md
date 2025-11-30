# covid-data-analysis
Exploratory Data Analysis (EDA) of global COVID-19 statistics using pandas, matplotlib, seaborn, and Plotly/Cufflinks. The `covid_data_analysis.ipynb` notebook demonstrates data loading and cleaning, summary statistics, and a variety of visualizations including bar charts, scatter plots, heatmaps, and interactive Plotly charts.

## Quick start

- Requirements: Python 3.8+, `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, and `cufflinks`.
- Install dependencies with pip:

```bash
python -m pip install --user pandas numpy matplotlib seaborn plotly cufflinks jupyterlab
```

## Run the notebook

- Open the notebook in Jupyter or VS Code and run cells interactively:

```bash
jupyter notebook covid_data_analysis.ipynb
# or open in VS Code
code covid_data_analysis.ipynb
```

## Dataset

- The notebook expects a CSV (example from Worldometer). In the notebook the path used is:

```
/Users/kajal.parmar/Documents/worldometer_data.csv
```

- For reproducible runs, place a CSV at `data/worldometer_data.csv` relative to the repo root, or update the path in the notebook before running.

## Examples (extracted from `covid_data_analysis.ipynb`)

Below are representative code snippets you can run in the notebook to reproduce analyses and visualizations.

1) Imports and Plotly/Cufflinks setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
import plotly.offline as pyo
import plotly.express as px

pyo.init_notebook_mode(connected=True)  # For Plotly offline mode
cf.go_offline()
```

2) Load the data

```python
data = pd.read_csv('/path/to/worldometer_data.csv')  # update the path for your environment
data.head()
```

3) Inspect and clean

```python
data.info()
data.describe()
data.isnull().sum()

# Fill NA for some columns and convert comma-formatted numbers to numeric
data['Population'] = data['Population'].fillna(0)
for col in ['TotalCases', 'NewCases', 'TotalDeaths', 'NewDeaths']:
	data[col] = data[col].astype(str).str.replace(',', '')
	data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
```

4) Compute a derived column

```python
# DeathRate as a percentage
data['DeathRate'] = (data['TotalDeaths'] / data['TotalCases'] * 100).round(2)
```

5) Top 10 countries by cases and deaths (barplots)

```python
top10_cases = data.sort_values(by='TotalCases', ascending=False).head(10)
sns.barplot(x='TotalCases', y='Country/Region', data=top10_cases, palette='Reds_r')
plt.title('Top 10 Countries by Total COVID-19 Cases')
plt.xlabel('Total Cases')
plt.ylabel('Country')
plt.show()

top10_deaths = data.sort_values(by='TotalDeaths', ascending=False).head(10)
sns.barplot(x='TotalDeaths', y='Country/Region', data=top10_deaths, palette='Greys_r')
plt.title('Top 10 Countries by Total COVID-19 Deaths')
plt.xlabel('Total Deaths')
plt.ylabel('Country')
plt.show()
```

6) Scatterplots (Seaborn)

```python
# Cases vs Deaths for Top 10
sns.scatterplot(x='TotalCases', y='TotalDeaths', data=top10_cases, hue='Country/Region', s=100)
plt.title('Total Cases vs Total Deaths (Top 10 Countries)')
plt.xlabel('Total Cases')
plt.ylabel('Total Deaths')
plt.show()

# Cases vs Recovered
sns.scatterplot(x='TotalCases', y='TotalRecovered', data=top10_cases, hue='Country/Region', s=100)
plt.title('Total Cases vs Total Recovered (Top 10 Countries)')
plt.xlabel('Total Cases')
plt.ylabel('Total Recovered')
plt.show()
```

7) Correlation heatmap

```python
corr = data[['TotalCases','TotalDeaths','TotalRecovered']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation between Cases, Deaths, and Recovered')
plt.show()
```

8) Interactive Plotly scatter with sizes and hover

```python
top10 = data.sort_values('TotalCases', ascending=False).head(10)
fig = px.scatter(
	top10,
	x='TotalCases',
	y='TotalRecovered',
	size='TotalCases',
	hover_name='Country/Region',
	title='Top 10 Countries: Cases vs Recoveries',
	color='Country/Region'
)
fig.show()
```

## Notes

- The notebook uses a CSV stored locally in the author's environment; update the path or copy the CSV into this project directory (e.g., `data/worldometer_data.csv`) before running.
- If you prefer to keep a smaller, versioned dataset in the repo, you can add a sample CSV subset and update the notebook path accordingly.
- I can also add a `requirements.txt`, generate PNGs for the README, or extract these examples into a `scripts/` directory for command-line runs.

## See also
- `pandas-practice-examples` — Useful for data cleaning and operations used in this notebook.
- `seaborn-practice-examples` — Helpful for advanced plotting and attractive default themes.
- `matplotlib-practice-examples` — More low-level control examples for plotting.
