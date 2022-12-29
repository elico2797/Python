import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
from pandas_datareader import data
from datetime import date
import math
from typing import List
from pandas_datareader._utils import RemoteDataError
import itertools


class PortfolioBuilder:

    def get_daily_data(self, tickers_list: List[str],  # get daily data from yahoo finance
                       start_date: date,
                       end_date: date = date.today()
                       ) -> pd.DataFrame:
        self.list_1 = int(len(tickers_list))
        try:
            self.stock_data = data.DataReader(tickers_list,
                                              'yahoo',
                                              start_date,
                                              end_date)
            if self.stock_data['Adj Close'].isnull().values.any():
                raise ValueError
            else:
                return self.stock_data['Adj Close']
        except RemoteDataError:  # raise errors
            raise ValueError

    def find_universal_portfolio(self, portfolio_quantization: int = 20) -> List[float]:
        # find the universal portfolio
        p = np.longdouble(1)
        result = [np.longdouble(1)]
        O = np.asarray(self.Omega(portfolio_quantization), dtype=float)
        denominator_1 = np.zeros(self.stock_data['Adj Close'].shape[0], dtype=float)
        Numerator_1 = np.zeros((self.stock_data['Adj Close'].shape[0], self.list_1), dtype=float)
        X_j = np.asarray(self.Xj_matrix().copy(), dtype=float)
        B_t_matrix = np.zeros((self.stock_data['Adj Close'].shape[0], self.list_1), dtype=float)
        Porto_matrix = np.ones((O.shape[0], X_j.shape[0] + 1), dtype=float)
        for j in range(0, O.shape[0]):
            for t in range(0, X_j.shape[0]):
                Porto_matrix[j][t + 1] = np.dot(O[j], X_j[t]) * Porto_matrix[j][t]
                # create the matrix represent the wealth assuming B-w if choose to day t
        for t in range(0, X_j.shape[0]):
            for i in range(0, O.shape[0]):
                Numerator_1[t] += np.multiply(O[i], Porto_matrix[i][t])
                denominator_1[t] += Porto_matrix[i][t]
            B_t_matrix[t] = np.divide(Numerator_1[t], denominator_1[t])
            p *= np.dot(X_j[t], B_t_matrix[t])
            result.append(p)
        return result


    def X_t_matrx(self):  # create X_j matrix
        A = np.asarray(self.stock_data['Adj Close'], dtype=float)
        X = np.ones((A.shape[0], self.list_1), dtype=np.float)
        X[0] = float(1)
        for i in range(1, A.shape[0] - 1):
            X[i] = np.true_divide(A[i + 1], A[i])
        return X

    def Omega(self, p_q):
        a = [0]
        Omega = []
        for i in range(1, p_q + 1):
            a.append(i * (1 / p_q))
        l = list(itertools.product(a, repeat=self.list_1))
        for j in range(0, len(l)):
            if abs(sum(l[j]) - 1) < 0.001:
                Omega.append(list(l[j]))
        return Omega

    def find_exponential_gradient_portfolio(self, learn_rate: float = 0.5) -> List[float]:

        self.Bj_matrix = np.zeros((self.stock_data['Adj Close'].shape[0] - 1, self.list_1), dtype=float)
        self.Bj_matrix[0] = [float(1 / (int(self.list_1)))] * self.list_1
        X_j = np.asarray(self.Xj_matrix().copy(), dtype=float)
        Numerator = np.zeros((self.stock_data['Adj Close'].shape[0], self.list_1), dtype=float)
        denominator = np.zeros(self.stock_data['Adj Close'].shape[0], dtype=float)
        for j in range(1, self.Bj_matrix.shape[0]):
            for i in range(0, self.list_1):
                x = math.exp(np.true_divide(learn_rate * X_j[j - 1][i], np.dot(self.Bj_matrix[j - 1], X_j[j - 1])))
                Numerator[j][i] = np.multiply(self.Bj_matrix[j - 1][i], x)
                denominator[j - 1] += Numerator[j][i]
            self.Bj_matrix[j] = np.true_divide(Numerator[j], denominator[j - 1])
        return self.summarize()

    def Xj_matrix(self):  # create X_j matrix
        A = np.asarray(self.stock_data['Adj Close'].copy(), dtype=float)
        X = np.ones((A.shape[0] - 1, self.list_1), dtype=float)
        for i in range(0, A.shape[0] - 1):
            X[i] = np.true_divide(A[i + 1], A[i])
        return X

    def summarize(self, p: np.longdouble = 1.0):  # summerize the wealth for each day
        M = [float(1)]
        for i in range(0, self.stock_data['Adj Close'].shape[0] - 1):
            p *= np.dot(self.Xj_matrix().copy()[i], self.Bj_matrix[i])
            M.append(p)
        return M


if __name__ == '__main__':  # You should keep this line for our auto-grading code.
    print('write your tests here')  # don't forget to indent your code here!
    pd =PortfolioBuilder()
    df = pd.get_daily_data(['GOOG','AAPL','v','NFLX'], date(2018,5,15) ,date(2020,5,23))
    print(pd.find_exponential_gradient_portfolio(0.7))
