import pandas as pd

CSV_PATH = 'xogta_ardeyda.csv'

df = pd.read_csv(CSV_PATH)
# print(df.head())
# print(df.shape)
# print(df.isnull().sum())
print(df["Windows10"].count())