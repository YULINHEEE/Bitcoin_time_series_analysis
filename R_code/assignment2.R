# Time Series Analysis Assignment 2
# Data: Monthly historical performance of the Bitcoin index (in USD) 
#     between August 2011 and January 2025
rm(list=ls())

# upload the following libraries 
library(TSA)
library(fUnitRoots)
library(forecast)
library(tseries)
library(lmtest)

# ================================
# Descriptive Statistic analysis
# ================================

# Set working directory and load data
setwd("C:/Users/Yulin He/Documents/Datascience/9.Time Series Analysis/Assignment 2")
bitcoin<-read.csv('assignment2Data2025.csv')
length(bitcoin$Bitcoin)

# Convert data to time series object
bitcoin.ts<-ts(bitcoin$Bitcoin, 
               start = c(2011,8),
               end = c(2025,1),
               frequency = 12)
class(bitcoin.ts)
head(bitcoin.ts)

# ================================
# Descriptive Statistic analysis
# ================================
plot(bitcoin.ts, type = "l",
     ylab = "Bitcoin Index (US)",
     main = "Time series plot of monthly Bitcoin index")


# Autocorrelation Analysis
# First-order autocorrelation
y<-bitcoin.ts
x<- zlag(bitcoin.ts)
index <- 2:length(x)
cor(y[index],x[index])
# Lag plot for first-order autocorrelation
plot(y[index], x[index],
     ylab = "Bitcoin Index",
     xlab = "Previous Bitcoin Index",
     main = "Scatter plot of the Bitcoin Index in consecutive months")

summary(bitcoin.ts)

# ACF & PACF plot
par(mfrow=c(1,2))
acf(bitcoin.ts,main = "ACF plot",lag.max = 42) 
pacf(bitcoin.ts, main = 'PACF plot')
par(mfrow=c(1,1))

# ADF test to check whether bitcoin.ts is stationary or not
adf.test(bitcoin.ts) #p-value = 0.99, non-stationarity

# Check whether bitcoin.ts is normally distribution or not
qqnorm(bitcoin.ts, xlab = 'Normal Scores', ylab = 'Bitcoin Index')
qqline(bitcoin.ts, col='red')
shapiro.test(bitcoin.ts) #not normally distributed

# ================================
# Data Transformation
# ================================

# Box-Cox
BC <- BoxCox.ar(bitcoin.ts,lambda = seq(-1, 0.5, 0.01)) #get an error.

# log transformation
bitcoin.log<-log(bitcoin.ts)
plot(bitcoin.log)
adf.test(bitcoin.log) #p-value = 0.4262, bitcoin.log is still non-stationarity

# ACF and PACF plot of bitcoin.log
par(mfrow=c(1,2))
acf(bitcoin.log,main = "ACF plot",lag.max = 42) 
pacf(bitcoin.log, main = 'PACF plot')
par(mfrow=c(1,1))

# differences = 1
bitcoin_log_diff<-diff(bitcoin.log, differences = 1)
plot(bitcoin_log_diff)

# ACF and PACF plot of bitcoin_log_diff
par(mfrow=c(1,2))
acf(bitcoin_log_diff,main = "ACF plot",lag.max = 42) 
pacf(bitcoin_log_diff, main = 'PACF plot')
par(mfrow=c(1,1))

# ADF test to check whether bitcoin_log_diff is stationary or not
adf.test(bitcoin_log_diff) 
pp.test(bitcoin_log_diff) 
kpss.test(bitcoin_log_diff) #results show it's stationarity

# Check whether bitcoin_log_diff is normally distribution or not
qqnorm(bitcoin_log_diff)
qqline(bitcoin_log_diff,col='red') # almost normally distributed
shapiro.test(bitcoin_log_diff) 

# ================================
# Model Identification
# ================================

#ACF and PACF plot of bitcoin_log_diff
par(mfrow = c(1,2))
acf(bitcoin_log_diff, main= "ACF plot")
pacf(bitcoin_log_diff, main= "PACF plot")
par(mfrow = c(1,1))
#ARIMA(p,d,q) p=1/2, d=1, q=1/2
#possible {ARIMA (1,1,1) , ARIMA (1,1,2), ARIMA (2,1,1), ARIMA (2,1,2)}

# EACF of bitcoin_log_diff 
eacf(bitcoin_log_diff, ar.max = 5, ma.max = 5)
#{ARIMA(0,1,1),ARIMA(0,1,2),ARIMA(1,1,0),ARIMA(1,1,1),ARIMA(1,1,2),ARIMA (2,1,1),ARIMA(2,1,2)}

# BIC table
par(mfrow=c(1,1))
res = armasubsets(y=bitcoin_log_diff,nar=5,nma=5,y.name='p',ar.method='ols')
plot(res)
#first two rows: {ARIMA(1,1,0), ARIMA(1,1,4)}

# Final set of possible models: 
# {ARIMA(0,1,1),ARIMA (0,1,2),ARIMA(1,1,0),ARIMA(1,1,1) , 
# ARIMA (1,1,2),ARIMA (2,1,1),ARIMA(2,1,2),ARIMA(1,1,4)}

# ================================
# Model Identification
# ================================

# build a model fitting function
fit_arima_models <- function(ts_data, orders, include.mean = FALSE) {
   
   #using ML and CSS methods to fit ARIMA model
   #print the coeftest results
   results <- list()
   
   for (order in orders) {
      order_str <- paste(order, collapse = "")
      
      cat("\n===== ARIMA(", paste(order, collapse = ","), ") with ML =====\n")
      model_ml <- Arima(ts_data, order = order, method = "ML", include.mean = include.mean)
      print(coeftest(model_ml))
      results[[paste0("model", order_str, "_ML")]] <- model_ml
      
      cat("\n===== ARIMA(", paste(order, collapse = ","), ") with CSS =====\n")
      model_css <- Arima(ts_data, order = order, method = "CSS", include.mean = include.mean)
      print(coeftest(model_css))
      results[[paste0("model", order_str, "_CSS")]] <- model_css
   }
   
   return(results)
}


orders_to_test <- list(c(0,1,1),c(0,1,2),c(1,1,0),c(1,1,1), 
                       c(1,1,2), c(1,1,4), c(2,1,1), c(2,1,2),
                       c(0,0,1)) # add one more no-differenced modol

results <- fit_arima_models(log(bitcoin.ts), orders_to_test)

# ================================
# Comparised AIC and BIC
# ================================

# use ML method to fit models
model.011 = Arima(bitcoin.log,order=c(0,1,1), method='ML')
model.012 = Arima(bitcoin.log,order=c(0,1,2), method='ML')
model.110 = Arima(bitcoin.log,order=c(1,1,0), method='ML')
model.111 = Arima(bitcoin.log,order=c(1,1,1), method='ML')
model.112 = Arima(bitcoin.log,order=c(1,1,2), method='ML')
model.114 = Arima(bitcoin.log,order=c(1,1,4), method='ML')
model.211 = Arima(bitcoin.log,order=c(2,1,1), method='ML')
model.212 = Arima(bitcoin.log,order=c(2,1,2), method='ML')

# sort AIC or BIC score
sort.score <- function(x, score = c("bic", "aic")){
   if (score == "aic"){
      x[with(x, order(AIC)),]
   } else if (score == "bic") {
      x[with(x, order(BIC)),]
   } else {
      warning('score = "x" only accepts valid arguments ("aic","bic")')
   }
}

#AIC results
sort.score(
   AIC(model.011, model.012, model.110, model.111,
       model.112, model.114, model.211,model.212), 
   score = "aic")

#BIC results
sort.score(
   BIC(model.011, model.012, model.110, model.111, 
       model.112, model.114, model.211,model.212), 
   score = "bic")

# results show that ARIMA(1,1,0) have lowest AIC and BIC values

# ================================
# Accuracy Evaluation
# ================================

# use CSS method to fit models
model.011_css = Arima(bitcoin.log,order=c(0,1,1), method='CSS')
model.012_css = Arima(bitcoin.log,order=c(0,1,2), method='CSS')
model.110_css = Arima(bitcoin.log,order=c(1,1,0), method='CSS')
model.111_css = Arima(bitcoin.log,order=c(1,1,1), method='CSS')
model.112_css = Arima(bitcoin.log,order=c(1,1,2), method='CSS')
model.114_css = Arima(bitcoin.log,order=c(1,1,4), method='CSS')
model.211_css = Arima(bitcoin.log,order=c(2,1,1), method='CSS')
model.212_css = Arima(bitcoin.log,order=c(2,1,2), method='CSS')

# get accuracy results
Smodel_011_css <- accuracy(model.011_css)[1:7]
Smodel_012_css <- accuracy(model.012_css)[1:7]
Smodel_110_css <- accuracy(model.110_css)[1:7]
Smodel_111_css <- accuracy(model.111_css)[1:7]
Smodel_112_css <- accuracy(model.112_css)[1:7]
Smodel_114_css <- accuracy(model.114_css)[1:7]
Smodel_211_css <- accuracy(model.211_css)[1:7]
Smodel_212_css <- accuracy(model.212_css)[1:7]

# cllect the results in one data frame
df.Smodels <- data.frame(
   rbind(Smodel_011_css, Smodel_012_css, Smodel_110_css, Smodel_111_css,
         Smodel_112_css, Smodel_114_css, Smodel_211_css, Smodel_212_css)
)

colnames(df.Smodels) <- c("ME", "RMSE", "MAE", "MPE", "MAPE", "MASE", "ACF1")
rownames(df.Smodels) <- c("ARIMA(0,1,1)", "ARIMA(0,1,2)","ARIMA(1,1,0)", 
                          "ARIMA(1,1,1)", "ARIMA(1,1,2)",
                          "ARIMA(1,1,4)", "ARIMA(2,1,1)", "ARIMA(2,1,2)")
round(df.Smodels,  digits = 3)

# ================================
# Overparameterized models
# ================================

# The best model is ARIMA(1,1,0). So, the overparameterized models are 
# ARIMA(2,1,0) and ARIMA(1,1,1)

#only fit ARIMA(2,1,0)
model.210 = Arima(bitcoin.log,order=c(2,1,0),method='ML')
coeftest(model.210) 

model.210_CSS = Arima(bitcoin.log,order=c(2,1,0),method='CSS')
coeftest(model.210_CSS) 

model.210_CSS_ML = Arima(bitcoin.log,order=c(2,1,0),method='CSS-ML')
coeftest(model.210_CSS_ML) 

# ================================
# Residual Diagnostics of ARIMA(1,1,0)
# ================================

# model.110 <- Arima(bitcoin.log, order = c(1,1,0), method = 'ML')

# Function to plot residuals for a model
plot_residuals <- function(residuals, model_name) {
   par(mfrow = c(2, 2))
   
   # Time series plot of residuals
   plot(y = residuals,
        x = as.vector(time(bitcoin.ts)),
        xlab = 'Time',
        ylab = 'Standardized Residuals',
        type = "l",
        main = paste("Residuals Time Series Plot for", model_name))
   points(y = residuals,
          x = as.vector(time(bitcoin.ts)))
   abline(h = 0, col = "red", lty = 2)
   
   # Histogram
   hist(residuals,
        xlab = "Standardized Residuals",
        main = paste("Histogram of standardized residuals for", model_name))
   
   # Q-Q plot
   qqnorm(y = residuals,
          main = paste("Q-Q plot of standardized residuals for", model_name))
   qqline(y = residuals, col = 2, lwd = 1, lty = 2)
   
   # ACF plot
   acf(residuals, main = paste("ACF of standardized residuals for", model_name))
   
   par(mfrow = c(1, 1))
   
   # Shapiro-Wilk test
   print(paste("Shapiro-Wilk test for", model_name))
   print(shapiro.test(residuals))
}

# Analyze residuals for each model
res.model1 <- residuals(model.110)
plot_residuals(res.model1, "ARIMA(1,1,0)")








