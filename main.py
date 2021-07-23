
#import hamming
import numpy as np
import math as math
import random as rand
import string as string
import matplotlib.pyplot as plt

import csv as csv
import os as os
import sys as sys
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

def old():
    # Test string to transmit
    m_string = "Te saluto.  Augustus sum, imperator et pontifex maximus romae.  Si tu es Romae amicus, es gratus."

    # Convert the text string to binary
    m_string_in_binary = "".join(format(ord(i), 'b').zfill(8) for i in m_string)
    
    # Determine the number of bits in the binary representation of the length of the original transmission string
    num_bits = len(np.binary_repr(len(m_string_in_binary)))

    # If the number of bits is not a multiple of 4 (i.e. the message size in Hamming(7, 4)), need to prepend extra 0's
    if num_bits % 4 != 0:
        num_bits_to_add = 4 - (num_bits % 4)
        leading_zeroes_str = "".join(str(0) for i in ([0] * num_bits_to_add))
        transmission_length = leading_zeroes_str + np.binary_repr(len(m_string_in_binary))

    assert len(transmission_length) % 4 == 0, "[ERROR] SIZE OF TRANSMISSION IS NOT A MULTIPLE OF THE MESSAGE LENGTH k = 4"

    # Create the Hamming(7, 4) encoder/channel/decoder structure 
    h = hamming.HammingCode(k = 4, n = 7, channel_fidelity = 1, hard_code = True)
  
    # Populate the codebook
    h.generate_codebook()

    # First step: Transmit the length so the receiver "knows" the length of the intended transmission
    # This is exact since we have already preprocessed the transmission length to be a multiple of 4, so it can be
    # easily and exactly parsed into several discrete blocks, each of size 4 bits.

    rebuilt_message = []

    # For each block of length 4 bits
    for k in range(0, len(transmission_length), 4):

        # Get the block
        partial_message = np.asarray(list(transmission_length[k : k + 4]), dtype = int)

        # Encode the block using Hamming(7, 4) code
        tx_codeword = h.encode(partial_message)

        # Transmit the block over the erasure channel
        rx_codeword, _ = h.transmit_erasure(tx_codeword)

        # Recover the original data using minimumg Hamming distance decoding
        recovered_partial_message = h.minimum_distance_decode(rx_codeword)

        # Append the recovered data to the list of rebuilt messages
        rebuilt_message.append(recovered_partial_message)

    # Convert the binary representation of the transmission size to decimal
    recovered_transmission_length = 0
    received_num_bits = len(rebuilt_message) * 4
    
    for i in range(0, len(rebuilt_message)):
        for j in range(0, len(rebuilt_message[i])):
            recovered_transmission_length += int(rebuilt_message[i][j]) * 2 ** (received_num_bits - 4 * i - j - 1)
 
    # If the number of bits in the message is not a multiple of 4 (i.e. the message size in Hamming(7, 4)), need to append extra 0's
    if len(m_string_in_binary) % 4 != 0:
        num_bits_to_add = 4 - (len(m_string_in_binary) % 4)
        #appended_zeroes_str = "".join(str(0) for i in ([0] * num_bits_to_add))
        #m_string_in_binary += appended_zeroes_str

    # Now that we actually have the length of the transmission, we can transmit the actual message
    rebuilt_message = []

    total_num_errors = 0
    for k in range(0, len(m_string_in_binary), 4):

        # Get the block
        partial_message = np.asarray(list(m_string_in_binary[k : k + 4]), dtype = int)
    
        # If length of partial message is not 4, then we have reached the last few remaining bits of the binary string
        # Append required number of bits to bring length of partial message to 4, the required message length
        if len(partial_message) != 4:
            partial_message = np.append(partial_message, [0] * num_bits_to_add)

        # Encode the block using Hamming(7, 4) code
        tx_codeword = h.encode(partial_message)
        #print(tx_codeword)

        # Transmit the block over the erasure channel
        rx_codeword, local_num_errors = h.transmit_erasure(tx_codeword)
        total_num_errors += local_num_errors

        # Recover the original data using minimumg Hamming distance decoding
        recovered_partial_message = h.minimum_distance_decode(rx_codeword)

        # Append the recovered data to the list of rebuilt messages
        rebuilt_message.append(recovered_partial_message)

    recovered_binary_string = ""

    for block in rebuilt_message:
        recovered_binary_string += "".join(str(e) for e in block)

    # Parse out the actual data string, ignoring the bits appended at the beginning of the transmission
    recovered_binary_string = recovered_binary_string[0 : recovered_transmission_length]

    assert recovered_binary_string == m_string_in_binary, "[ERROR] RECOVERED BINARY STRING DOES NOT MATCH TRANSMITTED BINARY STRING"

    recovered_string = ""
    for i in range(0, len(recovered_binary_string), 8):

        bin_rep = recovered_binary_string[i : i + 8]
        dec_value = 0
        for j in range(0, len(bin_rep)):
            dec_value += int(bin_rep[j]) * 2 ** (len(bin_rep) - j - 1)
        recovered_string += chr(dec_value)

    assert recovered_string == m_string, "[ERROR] RECOVERED STRING DOES NOT MATCH TRANSMITTED STRING"

    print("---------- TRANSMITTED MESSAGE ----------")
    print(m_string)

    print("---------- RECOVERED STRING ----------")
    print(recovered_string)

    print("TOTAL NUMBER OF INTRODUCED ERRORS = %d" % total_num_errors)

def insdel_communication(message, n, k, delta, epsilon = 0, scheme = "UNIQUE", load_str = False, path = None, error_model = "FIXED"):
    
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
    if DEBUG_FLAG:
        print("FORMATTING TIME = %lf sec" % (stop_time - start_time))

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
    if DEBUG_FLAG:
        print("RS-ENC TIME = %lf sec" % (stop_time - start_time))


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
    if DEBUG_FLAG:
        print("SYMBOL ENCODING TIME = %lf sec" % (stop_time - start_time))
    
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 4: ATTACH INDEXING ELEMENTS
    # ------------------------------------------------------------------------------------------------------------------

    start_time = time.time()
    txmitter = tx.Transmitter(transmission_length = n, epsilon = epsilon, indexing_scheme = scheme, load_str = load_str, path = path)

    for codeword in symbol_codeword_list:
        tx_tuples = txmitter.create_transmission_tuple(codeword)
        tx_tuple_list.append(tx_tuples)
    stop_time = time.time()
    if DEBUG_FLAG:
        print("ATTACHING INDEX TIME = %lf sec" % (stop_time - start_time))

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
    if DEBUG_FLAG:
        print("TRANSMISSION TIME = %lf sec" % (stop_time - start_time))

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

    for rx_stream in enumerate(rx_tuple_list):
        #print("Decoding %d/%d..." % (rx_stream[0]+1, len(rx_tuple_list)))
        corrupted_datastream = decoder.decode(rx_stream[1])

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
    if DEBUG_FLAG:
        print("INDEX DECODING TIME = %lf sec" % (stop_time - start_time))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 8: REED-SOLOMON DECODING
    # ------------------------------------------------------------------------------------------------------------------

    start_time = time.time()
    for entry in corrupted_codewords_list:
        corrupted_codeword = entry[0]
        erasure_locations = entry[1]

        rs_formatted_word = [ord(x) for x in corrupted_codeword]
        rs_rx_codewords.append(rs_formatted_word)

        try:
            corrected_message, corrected_ecc = rs.rs_correct_msg(rs_formatted_word, n-k, erase_pos = erasure_locations)
            recovered_message_list.append("".join(chr(x) for x in corrected_message))

        except:
            return -1, -1
    stop_time = time.time()
    if DEBUG_FLAG:
        print("RS-DEC TIME = %lf sec" % (stop_time - start_time))

    # ------------------------------------------------------------------
    # STEP 9: RETURN ORIGINAL DATA
    # ------------------------------------------------------------------

    recovered_message = "".join(s for s in recovered_message_list)
    return original_message, recovered_message

    # ----------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 9: RETURN ORIGINAL DATA
    # ------------------------------------------------------------------------------------------------------------------

    
    
    
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # STEP 10: STATISTICS

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
    original_message, recovered_message = insdel_communication(message, n, k, delta, epsilon, scheme = "SYNC", load_str = True, path = LOAD_DIR, error_model = "FIXED")
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
    
def fixed_error_model(message, n, k, alpha, epsilon = 0, scheme = "UNIQUE", load_str = False, path = None, error_model = "FIXED"):

    original_message, recovered_message = insdel_communication(message, n, k, delta = alpha, epsilon = epsilon, scheme = scheme, load_str = load_str, path = path, error_model = error_model)

    if original_message != -1 and recovered_message != -1 and original_message == recovered_message:
        return 0
    else:
        return 1

def plot_error_curves(plot_directory, figure_directory, display = False, save = True):

    for plot_data in os.listdir(plot_directory):

        delta_t = plot_data.split(sep = '_')[3]
        file_name = os.path.join(plot_directory, plot_data)
        
        with open(file_name, newline = '') as csvfile:
            reader = csv.reader(csvfile, delimiter = ',')
            alpha_row = next(reader)
            data_row = next(reader)
        csvfile.close()

        alpha_row = [float(x) for x in alpha_row]
        data_row = [float(x) for x in data_row]

        plt.plot(alpha_row, data_row, 'or-', linewidth = 1.5, markersize = 3.0)
        plt.xlabel('\N{greek small letter delta}')
        plt.ylabel("Prob. of error")
        plt.title("Prob. of error vs. \N{greek small letter delta} for critical value $\N{greek small letter delta}_t$ = %s" % delta_t)
    
        plt.xlim(0, 1)
        plt.ylim(0, 1.1)

        if display:
            plt.show()
        if save:
            print("SAVING FIGURE " + os.path.splitext(plot_data)[0])
            figure_out = os.path.join(figure_directory, os.path.splitext(plot_data)[0] + "_figure.png")
            plt.savefig(figure_out)

    #with open(file_name, newline = '') as csvfile:
    #    reader = csv.reader(csvfile, delimiter = ', ')
    #    for row in reader:
    #        print(" ".join(row))
    #csvfile.close()

    #plt.plot([(1.0 / NUM_STEPS) * x for x in range(0, NUM_STEPS)][1:-1] , values[1:-1] / NUM_ITERATIONS, 'or-', linewidth = 1.5, markersize = 3.0)
    #plt.xlabel("$\delta$")
    #plt.ylabel("Prob. of error")
    #plt.title("Prob. of error vs. $\delta$")
    
    #plt.xlim(0, 1)
    #plt.ylim(0, 1.1)

    #plt.show()
        #with open(sync_string_dictionary, newline = '') as csvfile:
    #    reader = csv.reader(csvfile, delimiter = ',')
    #    for row in reader:
    #        print(" ".join(row))

def test_suite_unique_indexing(n, plot_directory, number_rates, number_alpha_steps, number_iterations, error_model = "FIXED", debug = False):

    for rate in range(number_rates - 1, 0, -1):

        RC = (1.0 / number_rates) * rate
        delta = 1 - RC

        if debug:
            print("RATE = %0.3lf; CRITICAL DELTA = %0.3lf" % (RC, delta))

        file_name = os.path.join(plot_directory, "p_error_delta_%s_rate_%s.csv" % ("{:0.3f}".format(delta)[2 : ], "{:0.3f}".format(RC)[2 : ]))
        single_test_unique_indexing(n = n, RC = RC, num_steps = number_alpha_steps, num_iterations = number_iterations, write_file = file_name, error_model = error_model, debug = debug)

def single_test_unique_indexing(n, RC, num_steps, num_iterations, write_file, error_model = "FIXED", debug = False):

    k = int(n * RC)
    delta_t = 1 - RC

    num_invalid_decodes = np.zeros(shape = num_steps, dtype = int)
    alpha_list = np.zeros(shape = num_steps, dtype = float)

    for i in range(0, num_steps):

        alpha = (1.0 / num_steps) * i
        alpha_list[i] = "{:0.5f}".format(alpha)

        if debug:
            print("--> ALPHA = %0.3lf" % alpha)

        for j in range(0, num_iterations):
            msg = "".join(rand.choices(string.printable, k = 500))
            num_invalid_decodes[i] += fixed_error_model(message = msg, n = n, k = k, alpha = alpha, epsilon = 0, scheme = "UNIQUE", load_str = False, path = None, error_model = error_model)

        for k in range(0, len(num_invalid_decodes)):
            num_invalid_decodes[k] /= num_iterations 

        with open(write_file, 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow(alpha_list)
            writer.writerow(num_invalid_decodes)
        csvfile.close()

NUM_RATES = 10
NUM_ALPHA_STEPS = 25
NUM_ITERATIONS = 50

if __name__ == '__main__':

    #main()
    
    rand.seed(0x66023C)
    n = 30



    PLOT_DIR = os.path.join(sys.argv[1], "plot_data")
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    FIGURE_DIR = os.path.join(sys.argv[1], "figures")      
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)

    test_suite_unique_indexing(n = n, plot_directory = PLOT_DIR, number_rates = NUM_RATES, number_alpha_steps = NUM_ALPHA_STEPS, number_iterations = NUM_ITERATIONS, error_model = "IID", debug = True)

    plot_error_curves(plot_directory = PLOT_DIR, figure_directory = FIGURE_DIR, display = False, save = True)
    
    
'''
    for d in range(1, 20):
        num_invalid_decodes = np.zeros(shape = NUM_STEPS, dtype = int)
        delta = 0.05 * d
        k = int(n * (1 - delta))

        
        for i in range(0, NUM_STEPS):
            alpha = (1.0 / NUM_STEPS) * i

            if alpha > 0:
                print("--> ALPHA = %0.3lf" % alpha)
                for j in range(0, NUM_ITERATIONS):
                    msg = "".join(rand.choices(string.printable, k = 500))
                    num_invalid_decodes[i] += fixed_error_model(message = msg, n = n, k = k, alpha = alpha, epsilon = 0, scheme = "UNIQUE", load_str = False, path = None, error_model = "IID")
        p_error_dict["{:0.2f}".format(delta)] = num_invalid_decodes

    for key, values in p_error_dict.items():

        plt.plot([(1.0 / NUM_STEPS) * x for x in range(0, NUM_STEPS)][1:-1] , values[1:-1] / NUM_ITERATIONS, 'or-', linewidth = 1.5, markersize = 3.0)
        plt.xlabel("$\delta$")
        plt.ylabel("Prob. of error")
        plt.title("Prob. of error vs. $\delta$")
        
        plt.xlim(0, 1)
        plt.ylim(0, 1.1)

        plt.show()
    
    #fixed_error_model(delta = 0)
'''
