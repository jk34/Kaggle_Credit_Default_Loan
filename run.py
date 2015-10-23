import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics

	
trainDF = pd.read_csv('cs-training.csv')
trainDF.head()
trainDF.info()

#can drop NaN values
trainDF = trainDF.dropna() 

#30-59DaysPastDueNotWorse contains values 96 and 98, which are typos
#will replace them will median of 30-59DaysPastDueNotWorse

#trainDF=trainDF.groupby('NumberOfTime30-59DaysPastDueNotWorse').transform(getmedian)
#trainDF['NumberOfTime30-59DaysPastDueNotWorse'].max()
#both don't work

#use loc, which picks out the entries in trainDF['NumberOfTime30-59DaysPastDueNotWorse] that contain 98 or 96
trainDF['NumberOfTime30-59DaysPastDueNotWorse'].loc[(trainDF['NumberOfTime30-59DaysPastDueNotWorse']==98) | (trainDF['NumberOfTime30-59DaysPastDueNotWorse']==96)] = trainDF['NumberOfTime30-59DaysPastDueNotWorse'].median()
print "max is ", trainDF['NumberOfTime30-59DaysPastDueNotWorse'].max()


#histogram of the ages
fig1 = plt.figure()
fig2 = plt.figure()
ax1 = fig1.add_subplot(1,1, 1) #1 rows, 1 column, 1st plot
ax2 = fig2.add_subplot(1,1, 1) #1 rows,1 column, 1st plot

n, bins, patches = ax1.hist(trainDF['age'])
#fig = hplot.get_figure()
ax1.set_xlabel('age')
ax1.set_ylabel('Frequency')
fig1.savefig("hist_age.pdf", format='pdf')

#histogram of the NumberOfTime30-59DaysPastDueNotWorse
n, bins, patches = ax2.hist(trainDF['NumberOfTime30-59DaysPastDueNotWorse'])
ax2.set_xlabel('Num Times Late 30-59 Days')
ax2.set_ylabel('Frequency')
fig2.savefig("hist_late.pdf", format='pdf')

#can also visualize data with KDE plot
#similar to histogram in that it treats each data point as Gaussian distribution
#and then takes cumulative probability function

# Set the figure equal to a facetgrid with the training set as data and change the aspect ratio
fig = sns.FacetGrid(trainDF,aspect=4)
# Next use map to plot all the possible kdeplots for the 'age' values
sns_plot = fig.map(sns.kdeplot,'age',shade= True)

oldest_age = trainDF['age'].max()
#minimum age is 0
fig.set(xlim=(0,oldest_age))
sns_plot.savefig("KDE.pdf", format='pdf')

#see defaults vs nondefaults by creating new column called 'Default'
trainDF['Default'] = trainDF.SeriousDlqin2yrs.map({0:'nodefault', 1:'default'})
#categorize ages 0-5, 5-15, 15-25, ...
trainDF['age_rounded'] = np.round(trainDF['age'],-1)
categories_dict = {0 : '0-4', 10 : '5-14', 20 : '15-24', 30: '25-34', 40: '35-44',
                    50 : '45-54',60 : '55-64', 70 : '65-74', 80 : '75-84',
					90 : '85-94',100 : '95-104', 110 : '105+'}
trainDF['age_category'] = trainDF['age_rounded'].map(categories_dict)

defaultplot = sns.factorplot('Default', data=trainDF, kind="count", palette='Set1')
defaultplot.savefig("default.pdf",format="pdf")
#defaults vs age
defaultplot = sns.factorplot('age_category', data=trainDF, hue='SeriousDlqin2yrs',kind="count",palette='coolwarm')
defaultplot.set_xticklabels(rotation=45)
defaultplot.savefig("default_age.pdf",format="pdf")
#defaults vs # of dependents
defaultplot = sns.factorplot('NumberOfDependents', data=trainDF, hue='SeriousDlqin2yrs',kind="count",palette='coolwarm')
defaultplot.set_xticklabels(rotation=45)
defaultplot.savefig("default_dependents.pdf",format="pdf")

#try to see if number of times borrower was 30-59days late in payment
#affected whether they defaulted or not
times=[1,3,5,7,9,11,13]
sns_plot=sns.lmplot('NumberOfTime30-59DaysPastDueNotWorse','SeriousDlqin2yrs',data=trainDF)
sns_plot.savefig("late.pdf", format='pdf')
sns_plot=sns.lmplot('NumberOfTime30-59DaysPastDueNotWorse','SeriousDlqin2yrs',data=trainDF,palette='winter',x_bins=times)
sns_plot.savefig("late_bins.pdf", format='pdf')

#can visualize this with linear plot
generations=[10,20,30,40,50,60,70, 80]
sns_plot =sns.lmplot('age','SeriousDlqin2yrs',hue='NumberOfTime30-59DaysPastDueNotWorse',data=trainDF,palette='winter',x_bins=generations)
sns_plot.savefig("late_age_linear.pdf", format='pdf')

#logistic regression
Y=trainDF.SeriousDlqin2yrs
print "Y.head is ", Y.head()
#need to convert this to 1-d array to use with scikit-learn
Y = np.ravel(Y)
print "Y is ", Y

X=trainDF.drop(['Default', 'age_category', 'age_rounded'], axis=1)
print "X.head is ", X.head()

log_model = LogisticRegression()
log_model.fit(X,Y)
#check accuracy
print "accuracy is ", log_model.score(X,Y) #mult by 100% to get %accuracy
#gives .930672

#compare to null error rate, which is the % of people who defaulted
print "Y.mean is ", Y.mean()
print "1- Y.mean is ", 1-Y.mean() #gives .930514
#compare log_model score to 1- Y.mean()

#check coefficients of predictors to see which predictors were strongest
coeff_df = DataFrame(zip(X.columns, np.transpose(log_model.coef_)))
print "coeff_df is ", coeff_df
#positive coefficient means an increase in likelihood of default, negative coef means decrease in default

#train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
#this automatically splits them into training and test sets, 75% and 25% respectively

#regularization
#cs = l1_min_c(X_train,Y_train, loss='log')*np.logspace(-2.0, 2.0, num=5)
cs = np.logspace(-5, 15, num=9) 
#np.logspace(2,3, num=4) gives 100, 215.4, 464.1, 1000
print "cs: ", cs
#C is 1/lambda

#log_model2 = LogisticRegression()
log_model2 = LogisticRegression(C=.1, penalty='l1', tol=1e-6)
coefs=[]
c1=[]
c2=[]
c3=[]
c4=[]
c5=[]
c6=[]
c7=[]
c8=[]
c9=[]
c10=[]

a=0
for c in cs:
    log_model2.set_params(C=c)
    log_model2.fit(X_train, Y_train)
    coefs.append(log_model2.coef_.ravel().copy())
    #print "coefs append is", coefs
    #print "1st coef ", coefs[0][1]
    c1.append(coefs[a][2])
    c2.append(coefs[a][3])
    c3.append(coefs[a][4])
    c4.append(coefs[a][5])
    c5.append(coefs[a][6])
    c6.append(coefs[a][7])
    c7.append(coefs[a][8])
    c8.append(coefs[a][9])
    c9.append(coefs[a][10])
    c10.append(coefs[a][11])
    class_predict = log_model2.predict(X_test)
    #now compare predicted classes to actual classes
    print "C is ", c
    print "acc is ", metrics.accuracy_score(Y_test,class_predict)
    a=a+1
	
coefs = np.array(coefs)
#print "the coefs are", coefs
c1 = np.array(c1)
c2 = np.array(c2)
c3 = np.array(c3)
c4 = np.array(c4)
c5 = np.array(c5)
c6 = np.array(c6)
c7 = np.array(c7)
c8 = np.array(c8)
c9 = np.array(c9)
c10 = np.array(c10)

figC = plt.figure()
axC = figC.add_subplot(1,1, 1)
axC.plot(np.log10(cs), c1)
#np.log10(1e-15) returns -15
axC.plot(np.log10(cs), c2)
axC.plot(np.log10(cs), c3)
axC.plot(np.log10(cs), c4)
axC.plot(np.log10(cs), c5)
axC.plot(np.log10(cs), c6)
axC.plot(np.log10(cs), c7)
axC.plot(np.log10(cs), c8)
axC.plot(np.log10(cs), c9)
axC.plot(np.log10(cs), c10)
ymin, ymax = axC.set_ylim()
axC.set_xlabel('log(C)')
axC.set_ylabel('Coefficients')
axC.set_title('Logistic Regression Path')
plt.tight_layout()
axC.legend(['RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents'], loc='best', prop={'size':8})
figC.savefig("l1_coefs.pdf", format='pdf')
#can use Regularization to improve the accuracy score

#many of the coefficients go towards 0 when C=0 (or lambda = inf)
#1-Y.mean() was .930514
#the accuracy scores are 1.0 for C values = 1, 316.2, 100000, 3.16e7, etc
#but is .9301 when C=1e-5 and .99963 when C=.003 (logC = -2.5)

#from plot, hard to determine which predictors become 0 due to decreasing C 
#most relevant predictors appear to be DebtRatio, age, NumberRealEstateLoansOrLines, and NumberOfOpenCreditLinesAndLoans

X=X.drop(['RevolvingUtilizationOfUnsecuredLines','MonthlyIncome', 'NumberOfDependents', 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTimes90DaysLate'], axis=1)
log_model3 = LogisticRegression()

#train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
log_model3.fit(X_train, Y_train)
class_predict = log_model3.predict(X_test)
#now compare predicted classes to actual classes
print "log_model3 acc is ", metrics.accuracy_score(Y_test,class_predict)
#accuracy was .930672 in original set and 1-Y.mean() was .930514 originally
#now, accuracy drops to .9277

#check coefficients of predictors to see which predictors were strongest
coeff_df = DataFrame(zip(X.columns, np.transpose(log_model.coef_)))
print "coeff_df is ", coeff_df
#positive coefficient means an increase in likelihood of default, negative coef means decrease in default
