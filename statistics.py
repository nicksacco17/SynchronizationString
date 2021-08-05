
import numpy as np
import os as os
import matplotlib.pyplot as plt
import math as math
import csv as csv

BITS_PER_SYMBOL = 8

NUM_MESSAGE_SIZES = 21

# Default RS Alphabet Size 
q = 8

# e_r=2^(-q/4 (1-1/p) ),|Sigma_C|=2^q,n=2^p

def epsilon_stats():
    n = np.arange(2, int(2**20), dtype = int)
    critical_epsilon = np.zeros(shape = len(n), dtype = float)
    equivalent_rate_epsilon = np.zeros(shape = len(n), dtype = float)
    minimum_epsilon = np.zeros(shape = len(n), dtype = float)

    for i in range(0, len(critical_epsilon)):
        critical_epsilon[i] = float(n[i]) ** -0.25

    for i in range(0, len(equivalent_rate_epsilon)):
        p = np.log2(n[i])
        equivalent_rate_epsilon[i] = 2 ** (-q/4 * (1 - 1.0/p))

    for i in range(0, len(minimum_epsilon)):
        minimum_epsilon[i] = 2 ** (-q/4)

    plt.semilogx(n[1:], critical_epsilon[1:], '-r', linewidth = 1.5, label = r'$\varepsilon_L$')
    plt.semilogx(n[1:], equivalent_rate_epsilon[1:], '-b', linewidth = 1.5, label = r'$\varepsilon_R$')
    plt.semilogx(n[1:], minimum_epsilon[1:], '-g', linewidth = 1.5, label = r'$\varepsilon_P$')
    plt.xlim(2, 2**20)

    er_el = np.argwhere(np.diff(np.sign(critical_epsilon - equivalent_rate_epsilon))).flatten()

    print("INTERSECTION POINT: n[%d] = %d" % (er_el[1], n[er_el[1]]))

    plt.axvline(x = n[er_el[1]], color = 'k', linestyle = '--')
    plt.plot(n[er_el[1]], critical_epsilon[er_el[1]], 'kx', markersize = 4.5)

    el_ep = np.argwhere(np.diff(np.sign(critical_epsilon - minimum_epsilon))).flatten()
    plt.plot(n[el_ep[1]], critical_epsilon[el_ep[1]], 'kx', markersize = 4.5)

    plt.fill_between(n[er_el[1]:], equivalent_rate_epsilon[er_el[1]:], 1, where = equivalent_rate_epsilon[er_el[1]:] < 1, color = 'blue', alpha = 0.5, interpolate = True)
    
    plt.legend()
    plt.xlabel("Transmission length n")
    plt.ylabel("Epsilon Value")
    plt.title("Equal Length Epsilon,  Equal Rate Epsilon, Minimum Epsilon vs. Transmission Length n")
    plt.show()

def histogram_stats():

    PATH = "d:\\RPC\\log_data"

    for folder_name in os.listdir(PATH):

        tokens = folder_name.split('_')
        
        if len(tokens) > 1 and int(tokens[3]) == 32 and tokens[1] == "FIXED":
            
            print(tokens)
            scheme = tokens[0]
            for file_name in os.listdir(os.path.join(PATH, folder_name)):
                
                more_tokens = file_name.split('_')
                delta = float(more_tokens[1]) / (1e5)
                if delta == 0.76:
                    
                    row_list = []
                    with open(os.path.join(PATH, folder_name, file_name), newline = '') as csvfile:
                        reader = csv.reader(csvfile, delimiter = ',')
                        header = next(reader)
                        for row in reader:

                            for char in row[6]:
                                if char == '[':
                                    new_list = []
                                    x = ""
                                elif char.isdigit():
                                    if not x:
                                        x = x.join(char)
                                    else:
                                        x = x + char
                                elif char == ',':
                                    new_list.append(x)
                                    x = ""
                                elif char == ']':
                                    new_list.append(x)

                            row_list.append([int(y) for y in new_list])
                        csvfile.close()

                    all_data = []
                    count_above_threshold = 0
                    total_occurrences = 0
                    for row in row_list:
                        for entry in row:
                            all_data.append(entry)
                            total_occurrences += 1
                            if entry > 16:
                                count_above_threshold += 1
                    print(count_above_threshold)
                    print(total_occurrences)

                    plt.hist(all_data, bins = 30, color = 'b', align = 'mid')
                    plt.axvline(x = 16, color = 'r', linestyle = '--', linewidth = 2.5)
                    plt.xlabel("Number of Erasures")
                    plt.ylabel("Count")
                    plt.title("%s: Number of Induced Erasures in Fixed-Error Model for channel noise \N{greek small letter delta} = 0.76, threshold $\N{greek small letter delta}_t$ = 0.5" % scheme)
                    plt.show()


def decoding_stats():

    PATH = "d:\\RPC\\log_data\\DECODING"

    num_list = set()

    for folder_name in os.listdir(PATH):
        
        tokens = folder_name.split('_')

        scheme = tokens[0]
        error_mode = tokens[1]
        n = int(tokens[3])
        num_list.add(n)

    n_list = np.zeros(shape = len(num_list), dtype = int)

    for i in range(0, len(num_list)):
        n_list[i] = list(num_list)[i]
    n_list = np.sort(n_list)

    unique_avg_decode_times = np.zeros(shape = len(num_list), dtype = float)
    sync_avg_decode_times = np.zeros(shape = len(num_list), dtype = float)

    unique_std_devs = np.zeros(shape = len(num_list), dtype = float)
    sync_std_devs = np.zeros(shape = len(num_list), dtype = float)

    for i in range(0, len(n_list)):
        print("n = %d" % n_list[i])

        for folder_name in os.listdir(PATH):
        
            tokens = folder_name.split('_')

            scheme = tokens[0]
            error_mode = tokens[1]
            n = int(tokens[3])

            if n_list[i] == n:
                for csv_name in os.listdir(os.path.join(PATH, folder_name)):

                    more_tokens = csv_name.split('_')
                    delta = float(more_tokens[1]) / (1e5)

                    if delta == 0.5:
                        print("SCHEME %s, FILE NAME %s" % (scheme, csv_name))
                        row_list = []
                        with open(os.path.join(PATH, folder_name, csv_name), newline = '') as csvfile:
                            reader = csv.reader(csvfile, delimiter = ',')
                            header = next(reader)
                            for row in reader:
                                row_list.append(row)
                        csvfile.close()

                        local_decoding_times = np.zeros(shape = len(row_list), dtype = float)
                        for row in enumerate(row_list):
                            local_decoding_times[row[0]] = row[1][5]

                        print(len(local_decoding_times))
                        if scheme == "UNIQUE":
                            if len(local_decoding_times) == 100:
                                prefactor = 1.984
                            else:
                                prefactor = 2.131
                            unique_avg_decode_times[i] = np.average(local_decoding_times)
                            unique_std_devs[i] = prefactor * (np.std(local_decoding_times) / np.sqrt(len(local_decoding_times)))
                        elif scheme == "SYNC":
                            if len(local_decoding_times) == 100:
                                prefactor = 1.984
                            else:
                                prefactor = 2.131
                            sync_avg_decode_times[i] = np.average(local_decoding_times)
                            sync_std_devs[i] = prefactor * (np.std(local_decoding_times) / np.sqrt(len(local_decoding_times)))
    
    plt.errorbar(n_list, unique_avg_decode_times, yerr = unique_std_devs, color = 'b', fmt = "o", linestyle = 'solid', linewidth = 1.5, markersize = 2.0, ecolor = 'r', elinewidth = 2.5, capsize = 2.5, label = "Unique Indexing")
    plt.errorbar(n_list, sync_avg_decode_times, yerr = sync_std_devs, color = 'g', fmt = "o", linestyle = 'solid', linewidth = 1.5, markersize = 2.0, ecolor = 'k', elinewidth = 2.5, capsize = 2.5, label = "Synchronization Strings")
    plt.yscale('log')
    plt.xlabel("Transmission length n")
    plt.ylabel("(log) Decoding Time")
    plt.title("(Unique, " + r'$\varepsilon_R$' " Sync. String) Decoding Time vs. Transmission Length")
    plt.legend()
    plt.show()

def sync_str_stats():

    PATH = "d:\\RPC\\sync_string_data"
    plot_file_names = ["sync_str_n_4_epsilon_50000", "sync_str_n_8_epsilon_39685", 
                        "sync_str_n_16_epsilon_35355", "sync_str_n_32_epsilon_32988", 
                        "sync_str_n_64_epsilon_31498", "sync_str_n_128_epsilon_30475"]

    n_list = np.zeros(shape = len(plot_file_names), dtype = int)
    avg_construction_times = np.zeros(shape = len(plot_file_names), dtype = float)
    std_devs = np.zeros(shape = len(plot_file_names), dtype = float)

    for base_name in enumerate(plot_file_names):

        tokens = base_name[1].split('_')
        n = int(tokens[3])
        n_list[base_name[0]] = n

        file_name = os.path.join(PATH, base_name[1])
        row_list = []
        with open(file_name, newline = '') as csvfile:
            reader = csv.reader(csvfile, delimiter = ',')
            header = next(reader)
            for row in reader:
                row_list.append(row)
        csvfile.close()

        construction_times = np.zeros(shape = len(row_list), dtype = float)
        for row in enumerate(row_list):
            construction_times[row[0]] = row[1][-1]
        
        avg_construction_times[base_name[0]] = np.average(construction_times)
        print(len(construction_times))
        std_devs[base_name[0]] = 2.009 * (np.std(construction_times) / np.sqrt(len(construction_times)))

    print(avg_construction_times)
    print(std_devs)

    plt.errorbar(n_list[0:5], avg_construction_times[0:5], yerr = std_devs[0:5], color = 'b', fmt = "o", linestyle = 'solid', linewidth = 1.5, markersize = 2.0, ecolor = 'r', elinewidth = 2.5, capsize = 2.5)
    plt.plot(n_list[4:], avg_construction_times[4:], 'bx', markersize = 5.0, linestyle = '--', linewidth = 1.5)
    plt.yscale('log')
    plt.xlabel("Transmission length n")
    plt.ylabel("(log) Construction Time")
    plt.title(r'$\varepsilon_R$' + "-Synchronization String Construction Time vs. Transmission Length ")
    plt.show()    

if __name__ == '__main__':

    histogram_stats()
    #epsilon_stats()
    #sync_str_stats()
    #decoding_stats()


    



