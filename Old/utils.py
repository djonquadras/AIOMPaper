from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

class Product():
    """
    Class for each product. It has 3 attributes:
     - `dataframe`: the dataframe with all the historical data of this product
     - `length`: the number of observations for this product
     - `code`: the product itself
     - `model`: the classification model that best fit the dataset
     - `error`: the error of the best model
     - `features`: the dataset of the variables used to the classification
     - `label`: the dataset of the label variable
     - `X_train`: the features train dataset
     - `X_test`: the features test dataset
     - `y_train`: the label train dataset
     - `y_test`: the label test dataset
     - `paretoClassification`: the Pareto's classification of the product (A, B or C)
    """
    def __init__(self, dataframe = 0, material = 0):
        self.material = material
        self.dataframe = dataframe
        self.size = 0
        self.model = 0
        self.error = 0
        self.features = 0
        self.label = 0
        self.X_train = 0
        self.X_test = 0
        self.y_train = 0
        self.y_test = 0
        self.pareto = 0
        self.dataset = 0

    def processDataframe(self, pareto):
        self.size = len(self.dataframe)
        self.setPareto(**pareto)
        self.defineDataset()
        self.splitDataset()
        self.trainAndTest()

    def defineDataset(self):
        dt = self.dataframe[["Date", "Demand"]]
        dt = dt.groupby(pd.Grouper(key='Date', axis=0, 
                                    freq='2D')).sum()
        self.dataset = dt
        
    def getDataframe(self):
        return self.dataframe

    def setMaterial(self, material):
        self.material = material
    
    def getMaterial(self):
        return self.material

    def setModel(self,model):
        self.model = model

    def setError(self, error):
        self.error = error

    def getError(self):
        return self.error

    def setSize(self):
        self.size = len(self.dataframe)
    
    def getSize(self):
        return self.size

    def splitDataset(self):
        self.features = self.dataframe.iloc[:, :-1].values
        self.label = self.dataframe.iloc[:, -1].values
    
    def getFeatures(self):
        return self.features

    def getLabel(self):
        return self.label

    def trainAndTest(self):
        if self.pareto == "A" or self.pareto == "B":
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.label, test_size = 0.2, random_state = 0)
        else:
            self.X_train = 0
            self.X_test = 0
            self.y_train = 0
            self.y_test = 0
    
    def get_X_train(self):
        return self.X_train

    def get_y_train(self):
        return self.y_test

    def get_X_test(self):
        return self.X_test

    def get_y_test(self):
        return self.y_test

    def setPareto(self, a, b):
        if(self.size >= a):
            self.pareto = "A"
        elif(self.size >= b):
            self.pareto = "B"
        else:
            self.pareto = "C"
    
    def getPareto(self):
        return self.pareto
        

    def report(self):
        print(f"<-------- Report: Product {self.material} -------->")
        print(f"Nº of observations = {self.size}")
        print(f"Pareto = {self.pareto}")
        print(f"Selected Model = {self.model}")
        print(f"Error: {self.error}")

def selectModel(X_train, y_train, X_test, y_test):
    
    # Linear Regression
    linearReg = LinearRegression()
    
    # Polynomial Regression
    polyReg = PolynomialFeatures(degree = 4)
    X_Poly = polyReg.fit_transform(X_train)
    linRegPoly = LinearRegression()

    # Support Vector Regression (SVR)
    SVRReg = SVR(kernel = 'rbf')
       

    # Training
    linearReg.fit(X_train, y_train)
    linRegPoly.fit(X_Poly, y_train)
    SVRReg.fit(X_train, y_train)


    # Forecasting
    forecastLinReg = linearReg.predict(X_test)
    forecastPolyReg = linRegPoly.predict(X_test)
    forecastSVR = SVR.predict(X_test)


    
