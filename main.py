
# python main.py -directory d:\\RPC -n 30 -num_rates 10 -num_steps 50 -num_iterations 1
#import hamming
import numpy as np
import math as math
import random as rand
import string as string
import matplotlib.pyplot as plt

import csv as csv
import os as os
import sys as sys
import argparse as argparse
import time as time

import reed_solomon as rs
import alphabet as a
import channel as chn
import transmission as tx
import decoder as dec
import sync_string as sync

def pause(flag):
    if flag:
        input("Press Enter to continue...")

DEBUG_FLAG = False

def insdel_communication(message, n, k, delta, epsilon = 0, scheme = "UNIQUE", error_model = "FIXED", load_str = False, load_str_path = None, debug = False, log = False, log_path = None):
    
    # REED-SOLOMON PARAMETERS
    
    prim = 0x011D
    rs.init_tables(prim)
    
    message_list = []                               # List of parsed messages of length k

    rs_tx_codewords = []                            # List of transmitted RS codewords (list of ints in range (0, 255))

    codeword_list = []                              # List of transmitted codewords formatted as strings

    symbol_codeword_list = []                       # List of transmitted codewords formatted as symbols

    tx_tuple_list = []                              # Transmitted tuples <data, index>

    rx_tuple_list = []                              # Received tuples <data, index> (Corrupted by insdel channel)

    corrupted_codewords_list = []                   # Dictionary of corrupted codewords produced from index decoding 
                                                    # scheme along w/corresponding location of induced erasures

    rs_rx_codewords = []                            # List of received RS codewords (Corrupted by Index Decoding Scheme)

    recovered_message_list = []                     # List of recovered messages of length k

    RC = (1.0 * k) / n
    delta_crit = 1.0 - RC
    if log:
        log_file_name = os.path.join(log_path, "alpha_%s_epsilon_%s.csv" % ("{:0.5f}".format(delta)[2 : ], "{:0.5f}".format(epsilon)[2 : ]))
    #log_file_name = os.path.join(log_path, "%s_%s_rate_%s_delta_crit_%s_alpha_%s_epsilon_%s.csv" % (scheme, error_model, "{:0.5f}".format(RC)[2 : ], "{:0.5f}".format(delta_crit)[2 : ], "{:0.5f}".format(delta)[2 : ], "{:0.5f}".format(epsilon)[2 : ]))
    log_header = ["FORMATTING TIME", "RS-ENC TIME", "SYMBOL ENCODING TIME", "ATTACHING INDEX TIME", "TRANSMISSION TIME", "INDEX DECODING TIME", "NUMBER ERASURES", "RS-DEC TIME", "NUMBER OF INVALID DECODES", "OVERALL VALID DECODE"]
    log_row = []
    # ------------------------------------------------------------------------------------------------------------------
    # STEP 1: FORMAT THE ORIGINAL MESSAGE AS A LIST OF MESSAGES OF LENGTH k 
    # ------------------------------------------------------------------------------------------------------------------

    start_time = time.time()
    for i in range(0, len(message), k):
        msg = "{:<{str_width}}".format(message[i : i + k], str_width = k)
        message_list.append(msg)

        # This string is virtually identical to base message, just includes trailing spaces for alignment purposes.
        original_message = "".join(msg for msg in message_list)
    stop_time = time.time()
    if debug:
        print("FORMATTING TIME = %lf sec" % (stop_time - start_time))
    if log:
        log_row.append("{:0.5f}".format(stop_time - start_time))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 2: ENCODED EACH MESSAGE USING THE REED-SOLOMON CODING SCHEME
    # ------------------------------------------------------------------------------------------------------------------

    start_time = time.time()
    for msg in message_list:
        mesecc = rs.rs_encode_msg([ord(x) for x in msg], n-k)
        rs_tx_codewords.append(mesecc)
        codeword = "".join(chr(x) for x in mesecc)
        codeword_list.append(codeword)
    stop_time = time.time()
    if debug:
        print("RS-ENC TIME = %lf sec" % (stop_time - start_time))
    if log:
        log_row.append("{:0.5f}".format(stop_time - start_time))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 3: CONVERT THE ENCODED DATA INTO SYMBOL REPRESENTATION
    # ------------------------------------------------------------------------------------------------------------------
    
    start_time = time.time()
    source_alphabet = a.Alphabet(size = a.MAX_PRINT_SIZE)

    for codeword in codeword_list:
        symbol_codeword = source_alphabet.convert_string_to_symbols(codeword)
        symbol_codeword_list.append(symbol_codeword)
    stop_time = time.time()
    if debug:
        print("SYMBOL ENCODING TIME = %lf sec" % (stop_time - start_time))
    if log:
        log_row.append("{:0.5f}".format(stop_time - start_time))
    
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 4: ATTACH INDEXING ELEMENTS
    # ------------------------------------------------------------------------------------------------------------------

    start_time = time.time()
    txmitter = tx.Transmitter(transmission_length = n, epsilon = epsilon, indexing_scheme = scheme, load_str = load_str, load_str_path = load_str_path)

    for codeword in symbol_codeword_list:
        tx_tuples = txmitter.create_transmission_tuple(codeword)
        tx_tuple_list.append(tx_tuples)
    stop_time = time.time()
    if debug:
        print("ATTACHING INDEX TIME = %lf sec" % (stop_time - start_time))
    if log:
        log_row.append("{:0.5f}".format(stop_time - start_time))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 5: TRANSMIT OVER CHANNEL
    # ------------------------------------------------------------------------------------------------------------------

    if scheme == "UNIQUE":
        indexing_alphabet = a.Alphabet(size = n)
    elif scheme == "SYNC":
        indexing_alphabet = a.Alphabet(size = math.ceil(epsilon ** -4))

    start_time = time.time()
    channel = chn.InsDelChannel(delta = delta, n = n, insertion_prob = 0.5, data_alphabet = source_alphabet, index_alphabet = indexing_alphabet, error_model = error_model)

    for tx_stream in enumerate(tx_tuple_list):
        rx_tuples = channel.transmit(tx_tuple_array = tx_stream[1])
        rx_tuple_list.append(rx_tuples)

    stop_time = time.time()
    if debug:
        print("TRANSMISSION TIME = %lf sec" % (stop_time - start_time))
    if log:
        log_row.append("{:0.5f}".format(stop_time - start_time))

    #for tuple_stream in enumerate(rx_tuple_list):
    #    print("---------- STREAM %d ----------, LENGTH = %d" % (tuple_stream[0], len(tuple_stream[1])))
    #    for t in tuple_stream[1]:
    #        t.print_tuple(get_char = False, get_byte = True)
    
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 7: INDEXING DECODING
    # ------------------------------------------------------------------------------------------------------------------

    start_time = time.time()
    decoder = dec.Decoder(transmission_length = n, indexing_sequence = txmitter.get_indexing_sequence(), indexing_scheme = scheme, epsilon = epsilon)

    #corrupted_datastream = decoder.decode(tx_tuple_list[0])
    
    #for s in corrupted_datastream:
    #    s.print_symbol(print_char = False, print_byte = False, new_line = False)

    overall_erasure_counts = []
    for rx_stream in enumerate(rx_tuple_list):
        if debug or log and scheme == "SYNC":
            print("Decoding %d/%d..." % (rx_stream[0]+1, len(rx_tuple_list)))
        corrupted_datastream, erasure_count = decoder.decode(rx_stream[1])
        overall_erasure_counts.append(erasure_count)
        #print("Transmitted...")
        #for s in symbol_codeword_list[rx_stream[0]]:
        #    s.print_symbol(print_char = True, print_byte = False, new_line = False)

        #print()
        #print("Received...")
        #for s in corrupted_datastream:
        #    s.print_symbol(print_char = True, print_byte = False, new_line = False)
        #print()
        corrupted_codeword, _, erasure_locations = source_alphabet.convert_symbols_to_string(corrupted_datastream)
        corrupted_codewords_list.append((corrupted_codeword, erasure_locations))

    #print()
    stop_time = time.time()
    if debug:
        print("INDEX DECODING TIME = %lf sec" % (stop_time - start_time))
    if log:
        log_row.append("{:0.5f}".format(stop_time - start_time))
        log_row.append(overall_erasure_counts)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 8: REED-SOLOMON DECODING
    # ------------------------------------------------------------------------------------------------------------------

    DECODING_ERRORS = 0
    VALID_RESULT = 1

    start_time = time.time()
    for entry in corrupted_codewords_list:
        corrupted_codeword = entry[0]
        erasure_locations = entry[1]

        rs_formatted_word = [ord(x) for x in corrupted_codeword]
        rs_rx_codewords.append(rs_formatted_word)

        # Try to decode each codeword - if successful, continue
        try:
            corrected_message, corrected_ecc = rs.rs_correct_msg(rs_formatted_word, n-k, erase_pos = erasure_locations)
            recovered_message_list.append("".join(chr(x) for x in corrected_message))

        # Else the codeword could not be decoded, so mark the overall result as invalid and count the number of codewords that could not be decoded
        except:
            VALID_RESULT = 0
            DECODING_ERRORS += 1
    stop_time = time.time()
    if debug:
        print("RS-DEC TIME = %lf sec" % (stop_time - start_time))
    if log:
        log_row.append("{:0.5f}".format(stop_time - start_time))
        log_row.append(DECODING_ERRORS)
        log_row.append(VALID_RESULT)

    if log:

        # If the file does not exist yet, create it and append the header
        if not os.path.exists(log_file_name):

            with open(log_file_name, 'a', newline = '') as csvfile:
                writer = csv.writer(csvfile, delimiter = ',')
                writer.writerow(log_header)
            csvfile.close()

        # If here, the file should exist now so just append data rows, no need to append header
        with open(log_file_name, 'a', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow(log_row)
        csvfile.close()
        
    # ------------------------------------------------------------------
    # STEP 9: RETURN ORIGINAL DATA
    # ------------------------------------------------------------------

    recovered_message = "".join(s for s in recovered_message_list)
    return original_message, recovered_message, VALID_RESULT

    # ----------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------

def main():
    rand.seed(0x66023C)
    LOAD_DIR = os.path.join(sys.argv[1], "sync_string_data")

    #S = sync.load_sync_str(n = 30, epsilon = 0.5, directory = LOAD_DIR)
    #S.print_string(print_char = False, print_byte = False)

    #pause(True)

    # Base message
    message = rs.DARTH_PLAGUEIS_SCRIPT
    #message = "hello world"
    #message = "Te saluto. Augustus sum, imperator et pontifex maximus romae. Si tu es Romae amicus, es gratus."
    # Reed-Solomon Parametesr
    n = 30
    k = 5
    assert n > k, "[ERROR] PARAMETERS NOT VALID, n MUST BE GREATER THAN k!"

    # Synchronization Parameters
    delta_rs_max = 1 - k/n
    delta = (5.0/36)
    epsilon = 0.05

    assert delta < delta_rs_max, "[ERROR] CHANNEL FIDELITY TOO HIGH"

    start_time = time.time()
    original_message, recovered_message = insdel_communication(message, n, k, delta, epsilon, scheme = "SYNC", error_model = "FIXED", load_str = True, load_str_path = LOAD_DIR, debug = True, log = False, log_path = None)
    stop_time = time.time()
    total_time = stop_time - start_time

    print("COMMUNICATION OVER INSDEL CHANNEL W/FIDELITY %0.3lf COMPLETE; TOTAL TIME = %lf sec" % (delta, total_time))

    if original_message != -1 and recovered_message != -1:

        print("--> TRANSMITTED MESSAGE: " + original_message)
        print("--> RECEIVED MESSAGE: " + recovered_message)

        if original_message == recovered_message:
            print("COMMUNICATION SUCCESSFUL!")
    else:
        print("********** COULD NOT DECODE **********")
        pause(True)
    
def plot_prob_error(data_directory, output_figure_directory, display = False, save = True):

    print(output_figure_directory)
    base_name_tokens = os.path.basename(data_directory).split(sep = '_')

    scheme = base_name_tokens[0]
    error_mode = base_name_tokens[1]
    n = int(base_name_tokens[3])
    rate = float(base_name_tokens[5]) / (1e5)
    delta_crit = float(base_name_tokens[-1]) / (1e5)

    if scheme == "SYNC":
        delta_crit = (delta_crit * (1 - 0.32988) / (5 - 0.32988))

    num_alphas = len(os.listdir(data_directory))
    alpha_list = np.zeros(shape = num_alphas, dtype = float)
    num_invalid_decodes = np.zeros(shape = num_alphas, dtype = float)
    standard_deviations = np.zeros(shape = num_alphas, dtype = np.float64)
    p_error = np.zeros(shape = num_alphas, dtype = float)

    for data in enumerate(os.listdir(data_directory)):

        i = data[0]

        name = data[1].split(sep = '.')
  
        file_name = os.path.join(data_directory, data[1])

        tokens = name[0].split(sep = '_')

        alpha = float(tokens[1]) / (1e5)
        epsilon = float(tokens[-1]) / (1e5)

        alpha_list[i] = alpha

        row_list = []
        with open(file_name, newline = '') as csvfile:
            reader = csv.reader(csvfile, delimiter = ',')
            header = next(reader)
            for row in reader:
                row_list.append(row)
        csvfile.close()

        data_set = np.zeros(shape = len(row_list))
        m = 0
        for row in row_list:
            data_set[m] = row[-1]
            m += 1
            if int(row[-1]) == 0:
                num_invalid_decodes[i] += 1
        num_it = len(row_list)
        
        p = (np.sum(data_set) / len(data_set)) * 100

        standard_deviations[i] = (np.sqrt(p * (100-p) / len(data_set))) / 100

    for m in range(0, len(num_invalid_decodes)):
        p_error[m] = num_invalid_decodes[m] / num_it
    
    paired_list = []
    sorted_alpha = []
    sorted_perror = []
    sorted_std = []
    assert len(alpha_list) == len(p_error), "[ERROR], ARRAYS DO NOT AGREE IN LENGTH"
    for x in range(0, len(alpha_list)):
        paired_list.append((alpha_list[x], p_error[x], standard_deviations[x]))

    paired_list.sort()

    for x in paired_list:
        sorted_alpha.append(x[0])
        sorted_perror.append(x[1])
        sorted_std.append(x[2])

    plt.figure()
    plt.errorbar(sorted_alpha, sorted_perror, yerr = sorted_std, color = 'b', linewidth = 1.5, markersize = 1.0, ecolor = 'r', elinewidth = 1.5, capsize = 1.5)
    plt.axvline(x = delta_crit, color = 'g', linestyle = '--')
    plt.xlabel('\N{greek small letter delta}')
    plt.ylabel("Prob. of error")
    plt.title("Prob. of error vs. Channel noise \N{greek small letter delta} for critical threshold $\N{greek small letter delta}_t$ = %0.5lf" % (delta_crit))

    plt.xlim(0, 1)
    plt.ylim(0, 1.1)

    if save:     
        print("SAVING FIGURE " + os.path.basename(data_directory))
        figure_out = os.path.join(output_figure_directory, os.path.basename(data_directory) + "_figure.png")
        plt.savefig(figure_out)

    if display:
        plt.show()
    
def test_suite(n, epsilon, number_rates, number_alpha_steps, number_iterations, scheme = "UNIQUE", error_model = "FIXED", load_str = False, load_str_path = None, debug = False, log = False, log_path = None):

    for rate in range(number_rates - 1, 0, -1):

        RC = (1.0 / number_rates) * rate
        delta = 1 - RC

        print("RATE = %0.3lf; CRITICAL DELTA = %0.3lf" % (RC, delta))

        #file_name = os.path.join(data_directory, "p_error_delta_%s_rate_%s.csv" % ("{:0.3f}".format(delta)[2 : ], "{:0.3f}".format(RC)[2 : ]))
        single_test(n, epsilon, RC, number_alpha_steps, number_iterations, scheme, error_model, load_str, load_str_path, debug, log, log_path)

def single_test(n, epsilon, RC, num_steps, num_iterations, scheme = "UNIQUE", error_model = "FIXED", load_str = False, load_str_path = None, debug = False, log = False, log_path = None):

    k = int(n * RC)
    delta_t = 1 - RC

    if log_path:
        TEST_DIRECTORY = os.path.join(log_path, "%s_%s_n_%d_rate_%s_delta_crit_%s" % (scheme, error_model, n, "{:0.5f}".format(RC)[2 : ], "{:0.5f}".format(delta_t)[2 : ]))
        #print(TEST_DIRECTORY)
        if not os.path.exists(TEST_DIRECTORY):
            os.makedirs(TEST_DIRECTORY)
    else:
        TEST_DIRECTORY = None

    #num_invalid_decodes = np.zeros(shape = num_steps, dtype = int)
    #prob_error = np.zeros(shape = num_steps, dtype = float)
    alpha_list = np.zeros(shape = num_steps, dtype = float)

    for i in range(0, num_steps):

        alpha = (1.0 / num_steps) * i
        alpha_list[i] = "{:0.5f}".format(alpha)

        print("--> ALPHA = %0.3lf" % alpha)

        if alpha == 0.5:

            for j in range(0, num_iterations):
                msg = "".join(rand.choices(string.printable, k = 50))
                orig_msg, rec_msg, error_count = insdel_communication(msg, n, k, alpha, epsilon, scheme, error_model, load_str, load_str_path, debug, log, TEST_DIRECTORY)
    if TEST_DIRECTORY:
        return TEST_DIRECTORY
    else:
        return "-1"

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-directory")
    parser.add_argument("-n")

    parser.add_argument("-num_rates")
    parser.add_argument("-num_steps")
    parser.add_argument("-num_iterations")

    args = parser.parse_args()

    return args

def get_critical_epsilon(n):
    return float(n) ** -0.25

def get_equivalent_epsilon(n, q):
    p = np.log2(n)
    return 2 ** (-q/4.0 * (1 - 1.0/p))

def generate_sync_string(n, epsilon = 0.5, num_strings = 50, data_directory = None, debug = False):
    
    header = ["STRING ID", "SYNC-STRING", "NUM ITERATIONS", "AVG NUM INVALID INTERVALS PER ITERATION", "CONSTRUCTON TIME"]
    #row_list = []

    if debug:
        print("---------- STRING LENGTH = %d, EPSILON = %0.5lf ----------" % (n, epsilon))

    file_name = os.path.join(data_directory, "sync_str_n_%d_epsilon_%s" % (n, "{:0.5f}".format(epsilon)[2:]))

    with open(file_name, 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(header)
    csvfile.close()

    for i in range(0, num_strings):
        row = [str(i)]

        S = sync.Synchronization_String(epsilon = epsilon, n = n)
                
        start_time = time.time()
        total_iterations, avg_invalid_intervals = S.verify_synchronization()
        stop_time = time.time()

        str_rep = S.get_string_representation()

        construction_time = stop_time - start_time

        row.append(str_rep)
        row.append(total_iterations)
        row.append("{:0.4f}".format(avg_invalid_intervals))
        row.append("{:0.5e}".format(construction_time))
        
        with open(file_name, 'a', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow(row)

        csvfile.close()

        if debug:
            print("--> STRING %d, CONSTRUCTION TIME = %lf sec" % (i, construction_time))
    
    
if __name__ == '__main__':

    args = parse_arguments()    

    MAIN_DIR = args.directory
    if args.n:
        n = int(args.n)
    
    if args.num_rates:
        NUM_RATES = int(args.num_rates)

    if args.num_steps:
        NUM_ALPHA_STEPS = int(args.num_steps)
    
    if args.num_iterations:
        NUM_ITERATIONS = int(args.num_iterations)

    #main()
    
    rand.seed(0x66023C)

    PLOT_DIR = os.path.join(MAIN_DIR, "plot_data")
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    FIGURE_DIR = os.path.join(MAIN_DIR, "figures")      
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)

    SYNC_STR_DIR = os.path.join(MAIN_DIR, "sync_string_data")
    if not os.path.exists(SYNC_STR_DIR):
        os.makedirs(SYNC_STR_DIR)

    LOG_DIR = os.path.join(MAIN_DIR, "log_data")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    #start_time = time.time()
    #original_message, recovered_message, error_decode = insdel_communication(message = rs.DARTH_PLAGUEIS_SCRIPT, n = 32, k = 16, delta = (5.0/36), epsilon = 0.5, scheme = "SYNC", error_model = "FIXED", load_str = False, load_str_path = None, debug = True, log = False, log_path = None)
    #stop_time = time.time()
    #total_time = stop_time - start_time

    #print("COMMUNICATION OVER INSDEL CHANNEL W/FIDELITY %0.3lf COMPLETE; TOTAL TIME = %lf sec" % (delta, total_time))

    #if original_message != -1 and recovered_message != -1:

    #    print("--> TRANSMITTED MESSAGE: " + original_message)
    #    print("--> RECEIVED MESSAGE: " + recovered_message)

    #    if original_message == recovered_message:
    #        print("COMMUNICATION SUCCESSFUL!")
    #else:
    #    print("********** COULD NOT DECODE **********")
    #    pause(True)

    #single_test(n, epsilon, RC, num_steps, num_iterations, scheme = "UNIQUE", error_model = "FIXED", load_str = False, load_str_path = None, debug = False, log = False, log_path = None)

    #generate_sync_string(n = n, epsilon = get_equivalent_epsilon(n = n, q = 8), num_strings = 1, data_directory = SYNC_STR_DIR, debug = True)
    #generate_sync_string(n = n, epsilon = 0.32988, num_strings = 50, data_directory = SYNC_STR_DIR, debug = True)
    #generate_sync_string(n = n, epsilon = 0.31498, num_strings = 50, data_directory = SYNC_STR_DIR, debug = True)
    #generate_sync_string(n = n, epsilon = 0.30475, num_strings = 50, data_directory = SYNC_STR_DIR, debug = True)

    #test_suite(n = n, epsilon = 0, number_rates = NUM_RATES, number_alpha_steps = NUM_ALPHA_STEPS, number_iterations = NUM_ITERATIONS, 
    #            scheme = "UNIQUE", error_model = "FIXED", load_str = False, load_str_path = None, debug = False, log = True, log_path = LOG_DIR)
    #TEST_DIRECTORY = single_test(n = 4, epsilon = 0.50000, RC = 0.50, num_steps = NUM_ALPHA_STEPS, num_iterations = NUM_ITERATIONS, 
    #                            scheme = "SYNC", error_model = "FIXED", load_str = True, load_str_path = SYNC_STR_DIR, 
    #                            debug = True, log = True, log_path = LOG_DIR + "\\DECODING")
    #TEST_DIRECTORY = single_test(n = 8, epsilon = 0.39685, RC = 0.50, num_steps = NUM_ALPHA_STEPS, num_iterations = NUM_ITERATIONS, 
    #                            scheme = "UNIQUE", error_model = "FIXED", load_str = True, load_str_path = SYNC_STR_DIR, 
    #                            debug = True, log = True, log_path = LOG_DIR + "\\DECODING")
    #TEST_DIRECTORY = single_test(n = 16, epsilon = 0.35355, RC = 0.50, num_steps = NUM_ALPHA_STEPS, num_iterations = NUM_ITERATIONS, 
    #                            scheme = "UNIQUE", error_model = "FIXED", load_str = True, load_str_path = SYNC_STR_DIR, 
    #                            debug = True, log = True, log_path = LOG_DIR + "\\DECODING")
    #TEST_DIRECTORY = single_test(n = 32, epsilon = 0.32988, RC = 0.50, num_steps = NUM_ALPHA_STEPS, num_iterations = NUM_ITERATIONS, 
    #                            scheme = "UNIQUE", error_model = "FIXED", load_str = True, load_str_path = SYNC_STR_DIR, 
    #                            debug = True, log = True, log_path = LOG_DIR + "\\DECODING")
    #TEST_DIRECTORY = single_test(n = 64, epsilon = 0.31498, RC = 0.50, num_steps = NUM_ALPHA_STEPS, num_iterations = NUM_ITERATIONS, 
    #                            scheme = "UNIQUE", error_model = "FIXED", load_str = True, load_str_path = SYNC_STR_DIR, 
    #                            debug = True, log = True, log_path = LOG_DIR + "\\DECODING")
    #TEST_DIRECTORY = single_test(n = 128, epsilon = 0.30475, RC = 0.50, num_steps = NUM_ALPHA_STEPS, num_iterations = NUM_ITERATIONS, 
    #                            scheme = "UNIQUE", error_model = "FIXED", load_str = True, load_str_path = SYNC_STR_DIR, 
    #                            debug = True, log = True, log_path = LOG_DIR + "\\DECODING")

    #plot_prob_error(data_directory = TEST_DIRECTORY, output_figure_directory = FIGURE_DIR, display = False, save = True)
    #plot_prob_error(data_directory = "/home/sacco/Documents/SynchronizationString/log_data/SYNC_FIXED_n_32_rate_50000_delta_crit_50000", output_figure_directory = FIGURE_DIR, display = False, save = True)
    
    plot_prob_error(data_directory = "d:\\RPC\\log_data\\UNIQUE_FIXED_n_32_rate_50000_delta_crit_50000", output_figure_directory = FIGURE_DIR, display = False, save = True)

    plot_prob_error(data_directory = "d:\\RPC\\log_data\\UNIQUE_IID_n_32_rate_50000_delta_crit_50000", output_figure_directory = FIGURE_DIR, display = False, save = True)

    plot_prob_error(data_directory = "d:\\RPC\\log_data\\SYNC_FIXED_n_32_rate_50000_delta_crit_50000", output_figure_directory = FIGURE_DIR, display = False, save = True)

    plot_prob_error(data_directory = "d:\\RPC\\log_data\\SYNC_IID_n_32_rate_50000_delta_crit_50000", output_figure_directory = FIGURE_DIR, display = False, save = True)
