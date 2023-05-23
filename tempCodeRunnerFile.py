sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)