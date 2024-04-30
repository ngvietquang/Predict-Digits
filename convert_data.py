import pandas as pd

data = pd.read_csv("./train.csv")

df = pd.DataFrame(data)

df.to_csv('modified_data.csv', index=False)

print(df)
