import re
import glob
import numpy as np
import pandas as pd

path = './'


parameter_list = ['vs', 'am','fw_L','fw_G',  'p','vs_rev','c0', 'c1','c2','c3','c4']
df = pd.DataFrame(data=None,columns=['Name']+parameter_list)
df['p'] = np.zeros(len(glob.glob(path + '*result.txt')))


for i,fn in enumerate(glob.glob(path + '*result.txt')):

  starname = fn.split('/')[-1].split('_')[0]

  df['Name'].loc[i] = starname

  with open(fn, 'r') as f:
    lines = f.readlines()
    for line in lines:

        for parameter in parameter_list:
            print(parameter)
            if re.search(parameter+':', line):
                
                print(starname)
                print(line)

                if "(fixed)" not in line:

                    value = float(line.split(' ')[-7])
                    print(value)

                else:


                    value = float(line.split(' ')[-2])

                    
                df[parameter].loc[i] = value



df.to_csv("parameter_values.csv")
