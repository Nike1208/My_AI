import numpy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pandas
from pathlib import Path

#Variables
"""
scale = StandardScaler()
data_file = Path(__file__).parent / "data.csv"
df = pandas.read_csv(data_file)

X = df[['Weight', 'Volume']]
y = df['CO2']

scaledX = scale.fit_transform(X)

regr = linear_model.LinearRegression()
regr.fit(scaledX, y)

scaled = scale.transform([[2300, 1.3]])

#predcti the value depentent on the CO2 Emission
predictedCO2 = regr.predict([scaled[0]])
print(predictedCO2)
"""

numpy.random.seed(2)

x = numpy.random.normal(3,1,100)
y = numpy.random.normal(150,40,100)/x

Time = input("Enter the time in minutes: ")


train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))
r2 = r2_score(test_y, mymodel(test_x))

myline = numpy.linspace(0, 6, 100)

print(r2)
print(mymodel(float(Time)))

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()

exit()
