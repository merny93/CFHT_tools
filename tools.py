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
    parser.add_argument("save_file", type=str, help="where to save the output")
    parser.add_argument("--up_thresh", type=float, default=1.4, help="Threshhold to detect peaks")
    parser.add_argument("--down_thresh", type=float, default=0.5, help="Threshhold to detect troths")
    parser.add_argument("--plot", action="store_true", default=False, help="generate a plot")
    
    ##add more data later
    args = parser.parse_args()
    file_name = args.fname

    header, data = data_reader(file_name, n_cols= 2, head_return=True)

    #the treshholds for detection
    thr_up = args.up_thresh
    thr_down = args.down_thresh

    #find where they are happening
    down_arg = np.argwhere(data[:,1] < thr_down).T
    up_arg = np.argwhere(data[:,1]> thr_up).T
    
    #split into continous blocks (acts as a bit of filterin)
    up_candidate_blocks = np.split(up_arg, np.where(np.diff(up_arg) != 1)[1]+1, axis = -1)
    down_candidate_blocks = np.split(down_arg, np.where(np.diff(down_arg) != 1)[1]+1, axis = -1)
    
    #check that the size is resonable
    up_loc = []
    for up_can in up_candidate_blocks:
        if up_can.size < 10 or up_can.size > 500:
            pass #rejected cause too big or too small
        else:
            #its good so lets find the center
            up_loc.append((np.mean(up_can)))
    down_loc = []
    for down_can in down_candidate_blocks:
        if down_can.size < 10 or down_can.size > 30:
            pass
        else:
            down_loc.append((np.mean(down_can)))

    #now the arrays contain the info so we simply save to a file.
    #we need to switch to freqeuncy 
    up_freq = np.interp(up_loc, np.arange(0, data.shape[0]), data[:,0])
    down_freq = np.interp(down_loc, np.arange(0, data.shape[0]), data[:,0])

    with open(args.save_file, "w") as save_file:
        save_file.write("Data from: " +header)
        save_file.write("Emmisions (nm): \n")
        string_val = ""
        for val in up_freq:
            string_val = string_val + str(val) + ", "
        save_file.write(string_val)
        save_file.write("\n")
        save_file.write("Absorptions (nm): \n")
        string_val = ""
        for val in down_freq:
            string_val = string_val + str(val) + ", "
        save_file.write(string_val)


    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(data[:,0],data[:,1], label = "data")
        plt.axvline(x = up_freq[0], color="red", label = "emmision")
        for val in up_freq[1:]:
            plt.axvline(x = val, color="red")
        plt.axvline(x = down_freq[0], color = "green", label="absorption")
        for val in down_freq[1:]:
            plt.axvline(x = val, color = "green")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized power")
        plt.title("Plot with features")
        plt.legend()
        plt.show()