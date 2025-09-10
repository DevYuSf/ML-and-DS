import pandas as pd

CSV_PATH = 'house_l3_dataset.csv'

df = pd.read_csv(CSV_PATH)
# print(df.head())
# print(df.shape)
# print("Missiing values")
# print(df.isnull().sum())
# print(df.info())
# print("show counts of location")
# print(df["Location"].value_counts(dropna=False))

# df["Price"] = df["Price"].replace(r"[\$,]","", regex=True).astype(float)
# print(df.info())

# df["Location"] = df["Location"].replace({"Subrb":"Suburb", "??":pd.NA})
# print(df["Location"].value_counts(dropna=False))
df["Size_sqft"] = df["Size_sqft"].fillna(df["Size_sqft"].median(0))
df["Bedrooms"] = df["Bedrooms"].fillna(df["Bedrooms"].mode()[0])
df["Location"] = df["Location"].fillna(df["Location"].mode()[0])
# print(df.info())
before = df.shape
df = df.drop_duplicates()
after = df.shape

print('before:',before, 'after:',after)