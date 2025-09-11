using DataFrames, Statistics, Plots

# --------------------------
# 1. Load and clean data
# --------------------------
# Replace this with your .sav loading if needed
df = DataFrame(load("Du bao bang mo hinh nhan qua san luong _ CPQC Hoahong quy.sav"))

# Example: assume df is already loaded
# Ensure no missing values
df_clean = copy(df)
df_clean.Sanluong .= coalesce.(df_clean.Sanluong, mean(skipmissing(df_clean.Sanluong)))

# Ensure sequential time index
df_clean.time_index = 1:nrow(df_clean)

y = df_clean.Sanluong
n = length(y)
m = 4  # Quarterly seasonality

# --------------------------
# 2. Holt-Winters additive model
# --------------------------
alpha, beta, gamma = 0.2, 0.1, 0.3  # smoothing parameters

level = zeros(n)
trend = zeros(n)
seasonal = zeros(n)

# Initialize
level[1] = y[1]
trend[1] = y[2] - y[1]
seasonal[1:m] .= y[1:m] .- mean(y[1:m])

# Fit Holt-Winters
for t in m+1:n
    level[t] = alpha * (y[t] - seasonal[t-m]) + (1 - alpha) * (level[t-1] + trend[t-1])
    trend[t] = beta * (level[t] - level[t-1]) + (1 - beta) * trend[t-1]
    seasonal[t] = gamma * (y[t] - level[t]) + (1 - gamma) * seasonal[t-m]
end

# In-sample fitted values
hw_fitted = level + trend .+ seasonal

# --------------------------
# 3. Forecast future periods
# --------------------------
forecast_steps = 4
hw_forecast = zeros(forecast_steps)

for h in 1:forecast_steps
    hw_forecast[h] = (level[end] + h*trend[end]) + seasonal[(n + h - m - 1) % m + 1]
end

# --------------------------
# 4. Plot actual, fitted, forecast
# --------------------------
plot(df_clean.time_index, y, label="Actual", lw=2, marker=:circle)
plot!(df_clean.time_index, hw_fitted, label="HW Fitted", lw=2, linestyle=:dash)
plot!(df_clean.time_index[end]+1:df_clean.time_index[end]+forecast_steps, hw_forecast,
      label="HW Forecast", lw=2, linestyle=:dot)
xlabel!("Time Index")
ylabel!("Sản lượng")
title!("Holt-Winters Forecasting")

# --------------------------
# 5. Evaluate performance
# --------------------------
function mape(y_true, y_pred)
    return mean(abs.((y_true .- y_pred) ./ y_true)) * 100
end

mape_value = mape(y, hw_fitted)
println("In-sample MAPE: ", round(mape_value, digits=2), "%")