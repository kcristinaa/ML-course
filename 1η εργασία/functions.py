import numpy as np
import matplotlib.pyplot as plt

def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones(len(y_true)), np.abs(y_true))))*100

def plot_train_test_errors_ridge(train_errors, test_errors, lambd_values):
    plt.plot(np.log10(lambd_values), train_errors, 'r--', label = 'Train error')
    plt.plot(np.log10(lambd_values), test_errors, 'b--', label = 'Test error')
    plt.xlabel('logα')
    plt.ylabel('MSE',size=10)
    plt.legend()
    plt.title('MSE (with Ridge Regression) vs alpha', fontsize = 10)
    plt.show()

def plot_train_test_errors_lasso(train_errors, test_errors, lambd_values):
    plt.plot(np.log10(lambd_values), train_errors, 'r--', label = 'Train error')
    plt.plot(np.log10(lambd_values), test_errors, 'b--', label = 'Test error')
    plt.xlabel('logα')
    plt.ylabel('MSE',size=10)
    plt.legend()
    plt.title('MSE (with Lasso) vs alpha', fontsize = 10)
    plt.show()