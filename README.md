# Kaggle_Credit_Default_Loan
 My work using Python on data from a Kaggle competition on credit scoring to predict defaults 

 I have some interest in working on data sets used by banks to determine the credit score of their customers to predict the likelihood that the customer would default on a potential loan. The data set was obtained from: https://www.kaggle.com/c/GiveMeSomeCredit

I used Python to work on this data set. I start with Logistic Regression.  Because the training set has lots of NA values, I will get rid of the entries that contain any NA values

The "30-59DaysPastDueNotWorse" variable contains values 96 and 98, which are typos, so I will replace them with the median of 30-59DaysPastDueNotWorse

I then generate histograms of the 'age' (hist_age.pdf) and 'NumberOfTime30-59DaysPastDueNotWorse' (hist_late.pdf)  variables

I can also visualize the data with a KDE plot, which is similar to histogram in that it treats each data point as Gaussian distribution and then takes cumulative probability function. The resulting plot is "KDE.pdf"

I also want to generate plots that show how many of the entries contained defaults and non-defaults, along with factor plots (using the Seaborn package) that shows how the defaults varies depending on the age and number of dependents of each person. The plot that just shows the number of defaults vs non-defaults is "default.pdf", the plot showing the defaults vs age is "default_age.pdf", and the plot showing the defaults vs number of dependents is "default_dependents.pdf"

I then use linear plots from Seaborn to see how the number of defaults correlates with the 'NumberOfTime30-59DaysPastDueNotWorse' variable. The plot showing the defaults vs "NumberOfTime30-59DaysPastDueNotWorse" is "late.pdf", the plot with bins is "late_bins.pdf", and the plot of defaults vs age with NumberOfTime30-59DaysPastDueNotWorse as the hue is "late_age_linear.pdf"

Now I will proceed with Logistic Regression. We first set the dependent variable as the defaults and convert it into a 1-d array as required by Scikit-learn
Y=trainDF.SeriousDlqin2yrs print "Y.head is ", Y.head() #need to convert this to 1-d array to use with scikit-learn Y = np.ravel(Y) print "Y is ", Y
We then compute the score when using Logistic Regression on the entire training set. It turns out to be 93.06%. However, this is only a marginal improvement from the actual percentage of non-defaults in the dataset, which is 93.05%
.To improve on this score, I will try Regularization with the Lasso l1 penalty. We now split the training set into a training and validation/test set. Python automatically converts 75% of the original set into a new training set and the remaining 25% becomes the validation set. The resulting plot showing the coefficients as a function of the log of C (where C=1/lambda, where lambda is the penalty term. The greater the lambda, the more the coefficients of the predictors tends towards 0, thus eliminating the irrelevant predictors)
Many of the coefficients go towards 0 when C=0 (or lambda = inf). The accuracy scores are 1.0 for C values = 1, 316.2, 100000, 3.16e7, etc. However, the score is .9301 when C=1e-5 and .99963 when C=.003 (logC = -2.5). From the plot ("l1_coefs.pdf"), it is hard to determine which predictors become 0 due to increasing C. It seems that the most relevant predictors appear to be DebtRatio, age, NumberRealEstateLoansOrLines, and NumberOfOpenCreditLinesAndLoans.

I will then use only these relevant predictors in another Logistic Regression. However, with just these predictors, the accuracy drops to .9277
