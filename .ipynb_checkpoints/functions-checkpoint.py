import numpy as np
import matplotlib.pyplot as plt

def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones(len(y_true)), np.abs(y_true))))*100

def plot_train_test_errors_ridge(train_errors, test_errors, lambd_values, error, model):
    plt.plot(np.log10(lambd_values), train_errors, 'r--', label = 'Train error')
    plt.plot(np.log10(lambd_values), test_errors, 'b--', label = 'Test error')
    plt.xlabel('logα')
    plt.ylabel('%s'%error)
    plt.legend()
    plt.title('%s (with %s Regression) vs alpha'%(error,model))
    plt.savefig("visualizations/assignment2/%s_errors_%s.png"%(error, model),bbox_inches='tight')
    plt.show()
    
def plot_train_test_errors_lasso(train_errors, test_errors, lambd_values, s):
    plt.plot(np.log10(lambd_values), train_errors, 'r--', label = 'Train error')
    plt.plot(np.log10(lambd_values), test_errors, 'b--', label = 'Test error')
    plt.xlabel('logα')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE (with Lasso) vs alpha')
    plt.show()
    
def display_result(result):
    #print("MSE: ",result['test_mse'])
    print("mean MSE: ",-result['test_mse'].mean().round(2))
    print("standard deviation MSE: ", np.std(-result['test_mse']).round(2))
    #print("MAE: ",result['test_mae'])
    print("mean MAE: ",result['test_mae'].mean().round(2))
    print("standard deviation MAE: ", np.std(result['test_mae']).round(2))
    #print("MAPE: ",result['test_mape'])
    print("mean MAPE: ",result['test_mape'].mean().round(2))
    print("standard deviation MAPE: ", np.std(result['test_mape']).round(2))