"""
Convert results csvs from 
q3map = {'0': 'transportation_urban',
         '1': 'restaurant',
         '2': 'recreation',
         '3': 'domestic',
         '4': 'work_education',
         '5': 'athletics',
         '6': 'shop',
         '7': 'other_unclear',
         'NA': 'NA'}
to
q3map = {'0': 'transportation_urban',
         '1': 'restaurant',
         '2': 'recreation',
         '3': 'domestic',
         '4': 'work_education',
         '5': 'other_unclear',
         'NA': 'NA'}
"""
import sys
import csv
import pandas as pd

def convert_df(oldcsv):
    # row[4] is q3 response
    df = pd.read_csv(oldcsv)
    cmap = {5: 2, 6: 4, 7: 5}
    convert_q3 = lambda x: cmap[x] if cmap.get(x) is not None else x
    df['q3'] = df['q3'].map(convert_q3)
    return df

def write_new_csv(df, new_csv):
    df.to_csv(new_csv, na_rep='NA', index=False, float_format='%.0f')

def convert_field_forest(oldcsv, newcsv):
    with open(oldcsv, 'r') as old, open(newcsv, 'w') as new:
        reader = csv.reader(old)
        writer = csv.writer(new)
        next(reader)
        for row in reader:
            new_row = row
            if new_row[5] == '3':
                new_row[5] = '1'
            elif new_row[5] == '4':
                new_row[5] = '3'
            writer.writerow(new_row)

if __name__ == '__main__':
    df = convert_df(sys.argv[1])
    write_new_csv(df, sys.argv[2])
