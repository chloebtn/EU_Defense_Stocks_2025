import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

# Load the Data
stocks = ['HO.PA', 'AM.PA', 'RHM.DE', 'CHG.L','LMT']
names = ['Thales(FR)', 'Dassault(FR)', 'Rheinmetall(DE)', 'Chemring(EN)', 'Lockheed Martin(US)']

end_date = '2025-03-04'
start_date = '2023-03-03'

data = yf.download(stocks, start=start_date, end=end_date)['Close']

ticker_to_name = dict(zip(stocks, names))   # make sure names align with tickers
data.rename(columns=ticker_to_name, inplace=True)

# Returns and Cumulative Returns
returns = data.pct_change().dropna()

cumulative_returns = (1 + returns).cumprod() - 1

cumulative_returns.plot()
plt.title("Cumulatitive Returns of Defense Stocks")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.ylabel("Cumulative Returns")
plt.legend(loc='upper left')
plt.show()

cumulative_ret_6_months = cumulative_returns.loc[cumulative_returns.index >= cumulative_returns.index[-1] - pd.DateOffset(months=6)]

norm_cumul_ret_6_months = (1 + cumulative_ret_6_months) / (1 + cumulative_ret_6_months.iloc[0])

norm_cumul_ret_6_months.plot()
plt.title("Normalized Cumulatitive Returns of Defense Stocks (Last 6 Months)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.ylabel("Cumulative Returns")
plt.legend(loc='upper left')
plt.show()


# Correlation
correlation = returns.corr()

plt.figure(figsize=(10,6))
sns.heatmap(correlation, annot=True)
plt.title("Correlation Matrix of Defense Stock Returns")
plt.gca().xaxis.set_ticks_position('top')
plt.xlabel('')
plt.ylabel('')

plt.show()

# Big Picture of Data and Returns
annualized_returns = (1 + returns.mean())**252 - 1
volatility = returns.std() * np.sqrt(252)
sharpe_ratio = returns.mean() / volatility
max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

summary = pd.DataFrame({
    'Annualized Return': annualized_returns,
    'Volatility': volatility,
    'Sharpe Ratio': sharpe_ratio,
    'Max Drawdown': max_drawdown
})

summary_sorted = summary.sort_values('Sharpe Ratio', ascending=False)

# Plot the Summary in Subplots (one for each metric)
metrics = ['Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']

fig, axes = plt.subplots(2, 2, figsize=(15,12))

for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    bars = summary_sorted[metric].plot(kind='bar', ax=ax)
    ax.set_title(metric)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('')

    for bar in bars.patches:
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            bar.get_height(),  
            f"{bar.get_height():.4f}", 
            ha='center', va='bottom', fontsize=10
        )

plt.tight_layout()
plt.show()

# Monte Carlo Simulation
np.random.seed(82)      # to get the same results after repeating the process
# Function for Generating Simulations
def monte_carlo_simulation(stock_data, num_simulations=1000, num_days=252):
    last_price = stock_data.iloc[-1]
    daily_volatility = stock_data.pct_change().std()
   
    simulations = np.zeros((num_days, num_simulations))

    for i in range(num_simulations):
        price_series = [last_price]

        for _ in range(num_days):
            price = price_series[-1] * (1 + np.random.normal(0, daily_volatility))
            price_series.append(price)

        simulations[:,i] = price_series[1:]
    
    return simulations

# Monte Carlo Results
monte_carlo_results = {}

for stock in names:
    monte_carlo_results[stock] = monte_carlo_simulation(data[stock])

# Plot the Monte Carlo Simulations in Subplots (one for each stock)
fig, axes = plt.subplots(2, 3, figsize=(20, 15))
axes = axes.flatten()

for i, (stock, simulation) in enumerate(monte_carlo_results.items()):
    ax = axes[i]
    ax.plot(simulation, alpha=0.1, color='blue')
    ax.set_title(f"{stock} Monte Carlo Simulation")
    ax.set_xlabel("Days")
    ax.set_ylabel("Stock Price")

if len(monte_carlo_results) < 6:
    axes[-1].axis('off')

plt.tight_layout()
plt.show()

# Value at Risk (Var 95)
confidence_level = 5

var_results = {}
mean_results= {}

for stock in names:
    final_prices = monte_carlo_results[stock][-1,:]
    mean_results[stock] = np.mean(final_prices)
    var_results[stock] = np.percentile(final_prices, confidence_level)

mean_df = pd.DataFrame.from_dict(mean_results, orient='index', columns=['Average Final Price'])
var_df = pd.DataFrame.from_dict(var_results, orient='index', columns=['Volatility'])

last_prices = data.iloc[-1]
var_df['VaR %'] = ((last_prices - var_df['Volatility']) / last_prices) * 100
var_df['Average Final Price'] = mean_df['Average Final Price']
var_df = var_df[['Average Final Price', 'Volatility', 'VaR %']]

print(f"Value at Risk (95% confidence): \n{var_df}")

