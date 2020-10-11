"""Assignment 2 ID3 Decision Implementation""" 


import pandas as pd
import numpy as np

_raw_data = """Outlook,Temperature,Humidity,Wind,Play
Sunny,Hot,High,Weak,No
Sunny,Hot,High,Strong,No
Overcast,Hot,High,Weak,Yes
Rain,Mild,High,Weak,Yes
Rain,Cool,Normal,Weak,Yes
Overcast,Cool,Normal,Strong,Yes
Sunny,Cool,Normal,Strong,Yes
Sunny,Mild,High,Weak,No
Rain,Cool,Normal,Weak,Yes
Sunny,Mild,Normal,Weak,Yes
Overcast,Mild,Normal,Strong,Yes
Overcast,Hot,Normal,Strong,Yes
"""
with open("sport_data.csv", "w") as f:
    f.write(_raw_data)
df = pd.read_csv("sport_data.csv")

"""the above is sample crafted dataset for testing trial"""










"""Algorithm implementation below"""

def compute_entropy(y):
    if len(y) < 2:
        return 0
    freq = np.array( y.value_counts(normalize=True) )
    return -(freq * np.log2(freq + 1e-6)).sum() 

def compute_info_gain(samples, attr, target):
    values = samples[attr].value_counts(normalize=True)
    split_ent = 0
    for v, fr in values.iteritems():
        index = samples[attr]==v
        sub_ent = compute_entropy(target[index])
        split_ent += fr * sub_ent
    
    ent = compute_entropy(target)
    return ent - split_ent


class TreeNode:
    def __init__(self):
        self.children = {} 
        self.decision = None 
        self.split_feat_name = None 

    def pretty_print(self, prefix=''):
        if self.split_feat_name is not None:
            for k, v in self.children.items():
                v.pretty_print(f"{prefix}:When {self.split_feat_name} is {k}")
                #v.pretty_print(f"{prefix}:{k}:")
        else:
            print(f"{prefix}:{self.decision}")

    def predict(self, sample):
        if self.decision is not None:
            print("Decision:", self.decision)
            return self.decision
        else: 
            attr_val = sample[self.split_feat_name]
            child = self.children[attr_val]
            print("Testing ", self.split_feat_name, "->", attr_val)
            return child.predict(sample)

    def fit(self, X, y):
        if len(X) == 0:
            self.decision = "Yes"
            return
        else: 
            unique_values = y.unique()
            if len(unique_values) == 1:
                self.decision = unique_values[0]
                return
            else:
                info_gain_max = 0
                for a in X.keys(): 
                    aig = compute_info_gain(X, a, y)
                    if aig > info_gain_max:
                        info_gain_max = aig
                        self.split_feat_name = a
                print(f"Split by {self.split_feat_name}, IG: {info_gain_max:.2f}")
                self.children = {}
                for v in X[self.split_feat_name].unique():
                    index = X[self.split_feat_name] == v
                    self.children[v] = TreeNode()
                    self.children[v].fit(X[index], y[index])








"""Testing implementation using sample crafted dataset"""
attrs = ['Outlook', 'Temperature', 'Humidity', 'Wind']
data = df[attrs]
target = df["Play"]

t = TreeNode()
t.fit(data, target)

"""Testing for tree prediction"""
for i in [0,2,4]:
    print(f"Test predict sample[{i}]: \n{data.iloc[i]}\n\tTarget: {target.iloc[i]}")
    print(f"Making decision ...")
    pred = t.predict(data.iloc[i])







"""Model Evaluation below"""
    
path = "C:/Users/gn-dy/birthwtDT.csv"
data_set = pd.read_csv(path)
print(data_set)

ai = ["race", "smoke", "ui"]
to = data_set[ai]
the_target = data_set["low"]

t = TreeNode()
t.fit(to, the_target)


"""Model Evaluation results"""
for i in [0,2,4]:
    print(f"Test predict sample[{i}]: \n{data.iloc[i]}\n\tTarget: {target.iloc[i]}")
    print(f"Making decision ...")
    pred = t.predict(to.iloc[i])
    
    