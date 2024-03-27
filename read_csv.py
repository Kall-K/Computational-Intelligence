import pandas as pd

file_path = 'C:/Users/kalli/Desktop/iphi2802.csv'
csv2excel_file = 'C:/Users/kalli/Desktop/out.xlsx'

try:
    df = pd.read_csv(file_path, sep='\t')
    df.to_excel (csv2excel_file, index = None, header=True)

except pd.errors.ParserError as e:
    print("Error parsing CSV file:", e)
