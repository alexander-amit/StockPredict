import csv
import numpy as np
from  sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
    with open(filename,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
                dates.append(int(row[0].split('-')[0]))
                prices.append(float(row[1]))
    return

def predict_price(dates,prices,x):
    dates = np.reshape(dates,(len(dates),1))
    
    svr_lin = SVR(kernel = 'linear', C=1e3)
    svr_poly = SVR(kernel = 'poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel = 'rbf', C=1e3, gamma = 0.1)
    print('reached 1')
    svr_lin.fit(dates,prices)
    svr_poly.fit(dates,prices)
    svr_rbf.fit(dates,prices)
    print('reached 2')
    
    plt.scatter(dates,prices, color='black', label = 'Data')
    print('reached 3')
    plt.plot(dates, svr_rbf.predict(dates), color = 'red', label = 'rbf model')
    print('reached 4')
    plt.plot(dates, svr_lin.predict(dates), color = 'green', label = 'linear model')
    print('reached 5')    
    #plt.plot(dates, svr_poly.predict(dates), color = 'blue', label = 'poly model')    
    #plt.xlabel('Date')
    #plt.xlabel('Price')
    #plt.title('Support Vector Regression')
    #plt.legend()
    #plt.show()
    print('about to return from function')
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]
    
        

get_data('aapl.csv')
print('called')
predicted_price = predict_price(dates,prices,2)
print('going to print price')
print(predicted_price)
