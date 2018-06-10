import pickle


with open('./line.pkl', 'rb') as f:
    data = pickle.load(f)

for data_ in data:
    print(data_)