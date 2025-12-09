# Thư viện
library(readxl)
library(forecast)
library(ggplot2)
library(reshape2)
library(prophet)
library(lightgbm)
library(xgboost)
library(lubridate)   

set.seed(42)

# nạp data 
data <- read.csv("data.csv", stringsAsFactors = FALSE)

# xử lý cột ngày
if (!inherits(data$date, "Date")) {
  data$date <- ymd(data$date)  
}
if (any(is.na(data$date))) {
  stop("Date parsing failed. Check the 'date' format in CSV.")
}

btc_prices <- as.numeric(data$price)
if (any(is.na(btc_prices))) {
  stop("Price contains NA. Clean data before modeling.")
}
n <- length(btc_prices)

# tách 105 quan sát cuối
h <- 105
train <- btc_prices[1:(n - h)]
test  <- btc_prices[(n - h + 1):n]

# tạo dữ liệu chuỗi thời gian
train_ts <- ts(train, frequency = 1)

#Chạy mô hình 
#  1. ARIMA 
fit_arima     <- auto.arima(train_ts)
forecast_arima <- forecast(fit_arima, h = length(test))

#  2. ETS 
fit_ets       <- ets(train_ts)
forecast_ets  <- forecast(fit_ets, h = length(test))

#  3. TBATS 
train_log <- log(train)
train_msts <- msts(train_log, seasonal.periods = c(7, 365.25))
fit_tbats      <- tbats(train_msts)
forecast_tbats_log <- forecast(fit_tbats, h = length(test))
# Back-transform
forecast_tbats <- forecast_tbats_log
forecast_tbats$mean  <- exp(forecast_tbats_log$mean)
forecast_tbats$lower <- exp(forecast_tbats_log$lower)
forecast_tbats$upper <- exp(forecast_tbats_log$upper)

#  4. NNETAR 
fit_nnetar      <- nnetar(train_ts)                  
forecast_nnetar <- forecast(fit_nnetar, h = length(test))

#  5. Prophet 
btc_df <- data.frame(
  ds = as.Date(data$date[1:(n - h)]),  
  y  = log(train)                       
)

btc_df <- btc_df[complete.cases(btc_df), ]
if (nrow(btc_df) < (length(train) - 5)) {
  warning("Training rows reduced due to NA/Inf. Check data cleanliness.")
}

fit_prophet <- prophet(
  btc_df,
  interval.width     = 0.95,
  daily.seasonality  = TRUE,
  weekly.seasonality = TRUE,
  yearly.seasonality = TRUE
)

future <- data.frame(
  ds = seq.Date(from = max(btc_df$ds) + 1, by = "day", length.out = h)
)
forecast_prophet <- predict(fit_prophet, future)

forecast_prophet_values  <- exp(forecast_prophet$yhat)
forecast_prophet_lower   <- exp(forecast_prophet$yhat_lower)
forecast_prophet_upper   <- exp(forecast_prophet$yhat_upper)


#  6. LightGBM 
lag_features <- function(x, lags = 5) {
  df <- as.data.frame(embed(x, lags + 1))
  colnames(df) <- c("y", paste0("lag", 1:lags))
  return(df)
}

lags <- 5
train_ml <- lag_features(train, lags)
test_ml_full <- lag_features(c(train[(length(train) - lags + 1):length(train)], test), lags)
test_ml <- test_ml_full[(lags + 1):nrow(test_ml_full), ]

dtrain_lgb <- lgb.Dataset(as.matrix(train_ml[,-1]), label = train_ml$y)
params_lgb <- list(objective = "regression", metric = "rmse")
fit_lgb    <- lgb.train(params = params_lgb, data = dtrain_lgb, nrounds = 200)
forecast_lgb <- as.numeric(predict(fit_lgb, as.matrix(test_ml[,-1])))

# 7. XGBoost
dtrain_xgb <- xgb.DMatrix(as.matrix(train_ml[,-1]), label = train_ml$y)
params_xgb <- list(objective = "reg:squarederror", eval_metric = "rmse")
fit_xgb    <- xgb.train(params = params_xgb, data = dtrain_xgb, nrounds = 200)
forecast_xgb <- as.numeric(predict(fit_xgb, as.matrix(test_ml[,-1])))

# Bảng kết quả
rmse_prophet <- sqrt(mean((forecast_prophet_values - test)^2))
mae_prophet  <- mean(abs(forecast_prophet_values - test))
