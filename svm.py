from sklearn import svm

klasyfikator = svm.SVC()
X = [[2,1],[3,4],[5,7]]
c = [1,0,1]
klasyfikator.fit(X,c)

print(klasyfikator.predict([3,2]))