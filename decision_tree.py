import numpy as np



class Node :
    def __init__(self,feature=None,threshold = None, left=None,right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left=left
        self.right=right
        self.value = value
    def __str__(self):
        attributes = {
            "feature": self.feature,
            "threshold": self.threshold,
            "left": self.left,
            "right": self.right,
            "value": self.value
        }
        return str(attributes)
    def is_leaf_node(self):
        return self.value is None
n = Node(4)
print(n)
n1= Node(4,value=4)
print(n1)