from DataPreprocessing import *

def init_Centroids(X,K):
    idx=np.random.choice(len(X),K,replace=False)
    centroids=X[idx]
    return centroids


def new_Centroids(X,K,Address):
    Centroids=[]
    for k in range (K):
        ClusterPoints=X[Address==k]
        if len(ClusterPoints)>0:
           newcentroids=ClusterPoints.mean(axis=0)
        else:
            newcentroids=X[np.random.choice(len(X))]
        Centroids.append(newcentroids)
    centroids=np.array(Centroids)
    return centroids


def Kmean_Fit(X,K,iteration=1000):
    centroids=init_Centroids(X,K)
    for i in range (iteration):
        distance =np.linalg.norm(X[:,np.newaxis]-centroids,axis=2)
        Address=np.argmin(distance,axis=1)
        newcentroids=new_Centroids(X,K,Address)
        if np.all(centroids==newcentroids):
            break
        centroids=newcentroids
    return centroids,Address


def kmean_prediction(X,centroids):
    distance=np.linalg.norm(X[:,np.newaxis]-centroids,axis=2)
    cluster=np.argmin(distance,axis=1)
    return cluster

##########################Elbow Method##############################
def compute_cost(X,centroids,Address):
    cost=0
    for i in range(len(X)):
        centroid=centroids[Address[i]]
        cost+=np.sum((X[i]-centroid)**2)
    return cost/len(X)

Kvalues=range(1,11)
def Elbow_Method (X):
    costlist=[]
    for k in Kvalues:
        centroids,Address=Kmean_Fit(X,k)
        cost=compute_cost(X,centroids,Address)
        costlist.append(cost)
    return costlist


##########################Find K Value##############################

WCSS=Elbow_Method(X)
plt.plot(Kvalues, WCSS)
plt.xlabel('K')
plt.ylabel('Cost')
plt.title('Elbow Method')
plt.show()
