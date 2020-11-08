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



def my_window(N, option= None):
    """
    Return a window of length N 
    
    Defaults to hanning but if called with options ["Tukey", frac]
    Will give a wider window such that it only windows the outer frac 
    of the data

    """
    if option is None:
        return np.hanning(N)
    if option[0] == "Tukey": ##Tukey window this like the cos window but has flat 1 on top so is good shit
        window = np.hanning(int(N * option[1]))
        window_full = np.ones(N)
        window_full[:window.size//2] = window[: window.size //2]
        window_full[-window.size//2:] = window[-window.size //2 :]
        return window_full 



def colgate(data, window = my_window, window_params=["Tukey", 0.02], spec= None, plot_model =False):
    '''
    Pre-whiten data to get it ready for match filtering

    If called with no spectrum it will assume that you want to generate the noise spectrum and do so 
    It will use a window function which defualts to a hanning but can be changed to wider functions 
    Without a spectrum it will reutrn the PS and whitened data

    Calling with spec it will only return the whitened spectrum (it works with a recusrive call)
    '''
    if spec is None:
        ##get the ft to look at noise
        ft_noise = np.abs(((np.fft.rfft(data * window(data.size, window_params), norm="ortho"))))**2
    
        if plot_model:
            import matplotlib.pyplot as plt
            plt.clf()
            plt.loglog(ft_noise, label= "before")

        ##what we want to do is draw a line along the peaks to make sure its not under represented 
        ## to do this i will make a rolling window of 3 elements and if the middle is smaller than both of its neigbors 
        ## i will replace it with the average of the two neigbhors.
        ## And then do this a bunch of times. 
        ##the construction here will mean that we never loose the peaks (cause we always take maximums) 
        # but the valleys will get pulled up to the tops around them
        # if we run this long enough it will slowly become a straight line between the two biggest peaks with lines from the edges going there
        #but it will never get there since we dont run it for long enough
        
        niter = 500 #number of iterations (this is a guess but works ok)
        for _ in range(niter):
            #I started with a python loop implementation and it took forever so sorry for the hard code 
            # I want to make a rolling window array. So 3 wide such that the first row is 1st ,2nd ,3rd element 
            #second row is 2nd, 3rd, 4th element and so on so forth 
            #numpy does not have a funciton to do this so lets tell it to read the array differently with strides!

            stride = (ft_noise.strides[0], ft_noise.strides[0])
            #this line tells python that a move to the right or a move to down is the same equivalent 
            #since the elemnt to the right and the element bellow are both indexed 1 away in the original array

            ft_rolling = np.lib.stride_tricks.as_strided(ft_noise, shape = (ft_noise.size - 2, 3), strides=stride, writeable=False)
            ##that line generated the array representation (not actually writable)

            ft_argmin = np.argmin(ft_rolling, axis=-1)#check which is the minimum 

            ft_reset = np.where(ft_argmin == 1, (ft_rolling[:,0] + ft_rolling[:,2])/2, ft_rolling[:,1])
            #another hard line. This one checks if the min happened in the middle and if that is the case fills an array
            #of size ft_noise.size -2 with the average of the points around or just copies the value if it wasnt the min

            ft_noise[1:-1] = ft_reset #finally adding it to the array for the next itteration
            
            
            continue
            ##here is the original (almost) identical code
            print("this is demonstration shouldnt run")
            for i in range(ft_noise.size -2):
                if ft_noise[i] > ft_noise[i+1] and ft_noise[i+2] > ft_noise[i+1]:
                    ft_noise[i+1] = (ft_noise[i] + ft_noise[i+2])/2


        PS = ft_noise
        if plot_model:
            plt.loglog(ft_noise, label = "after")
            plt.legend()
            plt.savefig("model_dome.png")
        
        return PS, colgate(data, window_params=window_params, spec = np.sqrt(PS))
    
    ft_white = (np.fft.rfft(data * window(data.size, window_params), norm="ortho")) / spec
    data_white = np.fft.irfft(ft_white, norm="ortho")
    return data_white

def match_filter(y, model):
    '''
    a matching service for pre-whitened signal

    '''
    yft=np.fft.rfft(y, norm="ortho")
    modft=np.fft.rfft(model, norm="ortho")
    mycorr=np.fft.irfft(yft*np.conj(modft), norm="ortho")
    return mycorr

def gaus_temp(N, sig):
    x = np.linspace(-N/2 / sig, N/2 / sig, num= N)
    return np.roll(np.exp(-x**2), N//2)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Detect the absorption and emmision in the data sets. Run with -h for help')
    parser.add_argument("fname", type=str, help="Path to data file to analyize")
    parser.add_argument("save_file", type=str, help="where to save the output")
    parser.add_argument("--up_thresh", type=float, default=1.4, help="Threshhold to detect peaks")
    parser.add_argument("--down_thresh", type=float, default=0.5, help="Threshhold to detect troths")
    parser.add_argument("--plot", action="store_true", default=False, help="generate a plot")
    
    temp_widths = [200,400, 800, 1200]

    import matplotlib.pyplot as plt
    ##add more data later
    args = parser.parse_args()
    file_name = args.fname

    header, data = data_reader(file_name, n_cols= 2, head_return=True)

    ps, data_white = colgate(data[:,1])
    matched_total = []
    for sigma in temp_widths:
        template_white = colgate(gaus_temp(data_white.size, sigma), spec=np.sqrt(ps))
        matched_out = match_filter(data_white, template_white)
        matched_total.append(matched_out)
    matched_total = np.sum(np.array(matched_total), axis = 0)

    from scipy.signal import find_peaks
    peaks,_ = find_peaks(matched_total, prominence= 0.002)
    anti_peaks,_ = find_peaks(-1* matched_total, prominence= 0.002)

    
    down_freq = np.interp(peaks, np.arange(0, data.shape[0]), data[:,0])
    up_freq = np.interp(anti_peaks, np.arange(0, data.shape[0]), data[:,0])

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












  # #the treshholds for detection
    # thr_up = args.up_thresh
    # thr_down = args.down_thresh

    # #find where they are happening
    # down_arg = np.argwhere(data[:,1] < thr_down).T
    # up_arg = np.argwhere(data[:,1]> thr_up).T
    
    # #split into continous blocks (acts as a bit of filterin)
    # up_candidate_blocks = np.split(up_arg, np.where(np.diff(up_arg) != 1)[1]+1, axis = -1)
    # down_candidate_blocks = np.split(down_arg, np.where(np.diff(down_arg) != 1)[1]+1, axis = -1)
    
    # #check that the size is resonable
    # up_loc = []
    # for up_can in up_candidate_blocks:
    #     if up_can.size < 10 or up_can.size > 500:
    #         pass #rejected cause too big or too small
    #     else:
    #         #its good so lets find the center
    #         up_loc.append((np.mean(up_can)))
    # down_loc = []
    # for down_can in down_candidate_blocks:
    #     if down_can.size < 10 or down_can.size > 30:
    #         pass
    #     else:
    #         down_loc.append((np.mean(down_can)))

    # #now the arrays contain the info so we simply save to a file.
    # #we need to switch to freqeuncy 