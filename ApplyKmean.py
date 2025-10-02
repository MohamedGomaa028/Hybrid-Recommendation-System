from BuildKmean import *

##########################Training set##############################
Centroids1,Address=Kmean_Fit(X,K=8,iteration=1000)

pca=PCA(n_components=2)
X_PCA=pca.fit_transform(X)
Centroids_PCA=pca.transform(Centroids1)

plt.figure(figsize=(12, 10))

plt.scatter(X_PCA[:,0],X_PCA[:,1],c=Address,cmap="plasma", s=5)
plt.scatter(Centroids_PCA[:,0],Centroids_PCA[:,1],c="red",marker="X", s=200)
plt.title("K-mean for users")
plt.show()

##########################Test set##############################
UserDatascale_test=X_Kmean_test[["Occupation","Age"]]
X_Kmean_test=X_Kmean_test.drop(["UserId","Occupation","Age"],axis=1)
X_Kmean_test=X_Kmean_test.to_numpy()
scaler=StandardScaler()
xscaled_test=scaler.fit_transform(UserDatascale_test)
X_Kmean_test=np.hstack((X_Kmean_test,xscaled_test))

Address=kmean_prediction(X_Kmean_test,Centroids1)

Xtest_PCA=pca.transform(X_Kmean_test)
plt.figure(figsize=(12, 10))
plt.scatter(Xtest_PCA[:,0],Xtest_PCA[:,1],c=Address,cmap="plasma", s=5)
plt.scatter(Centroids_PCA[:,0],Centroids_PCA[:,1],c="red", s=100)
