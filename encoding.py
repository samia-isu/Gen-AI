import pandas as pd

# pd.set_option('display.max_colwidth',100)

# # df = pd.read_csv("physics.csv", encoding="utf-8", errors="ignore")
# df = pd.read_csv("physics.csv", encoding="utf-8", errors="ignore")  # Ignores problematic characters

# print(df.shape)
import chardet

# Detect the encoding
with open("physics.csv", "rb") as f:
    result = chardet.detect(f.read())
    print(result)

# Use the detected encoding
df = pd.read_csv("physics.csv", encoding=result["encoding"])
