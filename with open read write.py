

import bitarray

def write_the_bit_tothefile(bits, fname):   """obtain bits from other self-made function or other, 
                                              filename shoube be "XXXXX"     """
    with open(fname, "wb") as f:          """as f, the f is served as a file"""
        b = bitarray.bitarray(bits)
        f.write(b.tobytes())             """use tobytes() function to encode"""
        """f.write()       use write() function to put the result generated from tobytes into f"""
        
        
"""with open() method 

r	Opens a file for reading only.
rb	Opens a file for reading only in binary format.
r+	Opens a file for both reading and writing.
rb+	Opens a file for both reading and writing in binary format.
w	Opens a file for writing only.
wb	Opens a file for writing only in binary format.
w+	Opens a file for both writing and reading.
wb+	Opens a file for both writing and reading in binary format.
a	Opens a file for appending.
ab	Opens a file for appending in binary format.
a+	Opens a file for both appending and reading.
ab+	Opens a file for both appending and reading in binary format.

"""



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
with open("sport_data.csv", "w") as f:       """here, firstly give the name of the file by "XXXXXXXXXX.csv", 
                                                    with the mode "w" as write in these input data, as secondly give a 
                                                      a name as f"""
    f.write(_raw_data)                  
df = pd.read_csv("sport_data.csv")             """in pandas use read_csv() function to view the dataframe"""
# check the data
df

df["Play"]