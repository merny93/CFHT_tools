import numpy as np
import os
import argparse

##start with the read in utility:
def data_reader(file_name, n_rows = -1, n_cols = -1, head_return = False, sort_data = True):
    '''
    Data_reader for CFHT
        Read data from ascii text files
    INPUT
        file_name: path; path to data file
        n_cols: int; number of columns to read. Defult -1 for all data
        n_rowss: int; number of rows to read. Defult -1 for all data
    OUTPUT
        
    '''
    with open(file_name, "r") as data_file: #open up the file
        #get the name of the data and cut off the * at the start
        header = data_file.readline().split("*")[-1]

        #get the size in string 
        data_size_string = data_file.readline().split()
        #switch to integers
        data_size = [int(val) for val in data_size_string]
        print(data_size)
        #set the number of rows to read
        if n_rows < 0:
            rows_read = data_size[0] #cause of how python does things
        else:
            rows_read = min(n_rows, data_size[0])
        
        #set cols to read
        if n_cols < 0:
            cols_read = data_size[1]
        else:
            cols_read = min(n_cols, data_size[1])

        #now for the read
        #init the array
        data_ar = np.zeros((rows_read, cols_read))
        ##read all the data (this is wasteful but modern computers dont care)
        full_data = data_file.readlines()

        #move to a numpy data structure
        for n_row, row_string in enumerate(full_data[:rows_read]):
            row_string_list = row_string.split()
            data_ar[n_row, :] = np.array([np.float(val) for val in row_string_list[:cols_read]])
    
    if sort_data:
        #busy line to sort by frequency
        data_ar = data_ar[data_ar[:,0].argsort()]
    
    if head_return:
        return header, data_ar
    else:
        return data_ar


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Detect the absorption and emmision in the data sets. Run with -h for help')
    parser.add_argument("fname", type=str, help="Path to data file to analyize")
    ##add more data later
    args = parser.parse_args()
    file_name = args.fname

    header, data = data_reader(file_name, n_cols= 2, head_return=True)
    import matplotlib.pyplot as plt
    plt.plot(np.fft.rfft(data[:,1]-1))
    plt.show()
    plt.plot(data[:,0], data[:,1])
    plt.show()
    print(header)
    #print(data)