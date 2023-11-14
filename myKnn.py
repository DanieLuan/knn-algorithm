import numpy as np

class knn:
    """Implementação de algoritmo de classificação k-NN. 
    """
    def __init__(self, k=5, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
    
    def fit(self, data, targets):
        self.data = data
        self.targets = targets
    
    
    def predict(self, data_sample):
        predctions = []
        
        for data in data_sample:
            distance = []
            for X_train in self.data:
                distance.append(self.distance(data, X_train))

            k_indices = np.argsort(distance)[:self.k]
            
            k_nearest_labels = [self.targets[i] for i in k_indices]
            
            most_common = self.most_common(k_nearest_labels)
            
            predctions.append(most_common) 
            
        return predctions
    
    def distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            distance = np.sqrt(np.sum((x1 - x2)**2))
            return distance
        elif self.distance_metric == 'manhattan':
            distance = np.sum(np.abs(x1 - x2))
            return distance
        else:
            raise ValueError('Metric not implemented')
    
    def most_common(self, k_nearest_labels):
        labels, label_counts = np.unique(k_nearest_labels, return_counts=True)
        return labels[np.argmax(label_counts)]