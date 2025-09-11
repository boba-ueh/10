
using DataFrames, StatFiles, StatsPlots, GLM, Statistics  

# # Load the .sav file
# df = DataFrame(load("Du bao bang mo hinh nhan qua san luong _ CPQC Hoahong quy.sav"))

# # Create a new column with year.quarter format
# df.year_quarter = string.(df.nam) .* ".Q" .* string.(df.quy)

# # Convert year and quarter into a continuous time variable
# df.time_index = df.nam .+ (df.quy .- 1) ./ 4

# # Fit a linear regression model: quantity ~ time_index
# model = lm(@formula(Sanluong ~ time_index), df)

# # Predict values
# df.predicted = predict(model)

# # Plot actual vs predicted
# @df df plot(:time_index, [:Sanluong :predicted],
#     xlabel = "Time Index (Year + Quarter)",
#     ylabel = "Quantity",
#     title = "Regression of Quantity over Time",
#     label = ["Actual" "Predicted"],
#     seriestype = :line,
#     markershape = :circle)


#######

#Load the .sav file
df = DataFrame(load("Du bao bang mo hinh nhan qua san luong _ CPQC Hoahong quy.sav"))

# Sort by year and quarter to ensure correct order
sort!(df, [:nam, :quy])

# Create a sequential time index from 1 to n
df.time_index = 1:nrow(df)

# Fit a linear regression model: Sanluong ~ time_index
model = lm(@formula(Sanluong ~ time_index), df)

# Predict values
df.predicted = predict(model)

# Plot actual vs predicted
@df df plot(:time_index, [:Sanluong :predicted],
    xlabel = "Time Index (1 to n)",
    ylabel = "Sản lượng",
    title = "Regression of Sản lượng over Sequential Time",
    label = ["Actual" "Predicted"],
    seriestype = :line,
    markershape = :circle)



println("Regression Summary:")
display(coeftable(model))
println("R²: ", round(r2(model), digits=4))
println("Adjusted R²: ", round(adjr2(model), digits=4))



# Define upper and lower bounds (10% around predicted)
df.upper = df.predicted .* 1.20
df.lower = df.predicted .* 0.80

@df df plot(:time_index, [:Sanluong :predicted],
    xlabel="Time Index (1 to n)",
    ylabel="Sản lượng",
    title="\Sản lượng theo thời gian",
    label=["Actual" "Predicted"],
    seriestype=:line,
    markershape=:circle)

# Add outer lines
@df df plot!(:time_index, :upper, label="Upper Bound", color=:gray, linestyle=:dash)
@df df plot!(:time_index, :lower, label="Lower Bound", color=:gray, linestyle=:dash)

##

# using Metrics

# # Assuming df.Sanluong is the true values and df.predicted is the predicted values
# mape_value = mape(df.Sanluong, df.predicted)
# println("MAPE: ", mape_value)

function mape(y_true, y_pred)
    return mean(abs.((y_true .- y_pred) ./ y_true)) * 100
end

mape_value = mape(df.Sanluong, df.predicted)
println("MAPE: ", round(mape_value, digits=2), "%")



