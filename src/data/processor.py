from pathlib import Path
import pandas as pd
import argparse
import csv
import os

def formatting(filename):

    with open(filename, "r") as in_file:
        buf = in_file.readlines()

    with open(filename, "w") as out_file:
        for line in buf:
            string = "."
            if string in line:
                line = line + "\n"
            out_file.write(line)


def csv_to_tsv(up_file_name):

    new_file_name = os.path.splitext(up_file_name)[0]+".tsv"

    with open(up_file_name,'r', encoding='utf-8') as csvin, open(new_file_name, 'w', newline='', encoding='utf-8') as tsvout:
        csvin = csv.reader(csvin)
        tsvout = csv.writer(tsvout, delimiter='\t')

        for row in csvin:
            tsvout.writerow(row)

    formatting(new_file_name)




def process_data(file_path , file_name, up_file_name=None):

    df = pd.read_csv(file_path, delimiter = ' ' ,
                header = None,
                skiprows=1,
                names = ['word','POS', 'POS_BILOU', 'O_BILOU'], 
                index_col = False)
    
    if up_file_name is not None:

        df_updated = df.drop(labels=['POS',  'POS_BILOU'], axis = 1)
        
        df_updated.to_csv(up_file_name, index=False)
        csv_to_tsv(up_file_name)

    else:
        df.to_csv(file_name, index=False)
        csv_to_tsv(file_name)


