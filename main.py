
#import hamming
import reed_solomon as rs
import alphabet as a
import channel as chn
import transmission as tx
import decoder as dec
import numpy as np
import time as time

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

def insdel_comm_unique_index(message, n, k, delta):
    
    # REED-SOLOMON PARAMETERS

    prim = 0x011D
    rs.init_tables(prim)
    
    message_list = []                               # List of parsed messages of length k

    rs_tx_codewords = []                            # List of transmitted RS codewords (list of ints in range (0, 255))

    codeword_list = []                              # List of transmitted codewords formatted as strings

    symbol_codeword_list = []                       # List of transmitted codewords formatted as symbols

    tx_tuple_list = []                              # Transmitted tuples <data, index>

    rx_tuple_list = []                              # Received tuples <data, index> (Corrupted by insdel channel)

    corrupted_codewords_dict = {}                   # Dictionary of corrupted codewords produced from index decoding 
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
    print("SYMBOL ENCODING TIME = %lf sec" % (stop_time - start_time))
    
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 4: ATTACH INDEXING ELEMENTS
    # ------------------------------------------------------------------------------------------------------------------

    start_time = time.time()
    txmitter = tx.Transmitter(transmission_length = n, indexing_scheme = "UNIQUE")

    for codeword in symbol_codeword_list:
        tx_tuples = txmitter.create_transmission_tuple(codeword)
        tx_tuple_list.append(tx_tuples)
    stop_time = time.time()
    print("ATTACHING INDEX TIME = %lf sec" % (stop_time - start_time))
    
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 5: TRANSMIT OVER CHANNEL
    # ------------------------------------------------------------------------------------------------------------------

    start_time = time.time()
    channel = chn.InsDelChannel(delta = delta, n = n, insertion_prob = 0.5, data_alphabet = source_alphabet, index_alphabet = a.Alphabet(size = n))

    for tx_stream in enumerate(tx_tuple_list):

        rx_tuples = channel.transmit(tx_tuple_array = tx_stream[1])
        rx_tuple_list.append(rx_tuples)
    
    stop_time = time.time()
    print("TRANSMISSION TIME = %lf sec" % (stop_time - start_time))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 7: INDEXING DECODING
    # ------------------------------------------------------------------------------------------------------------------

    start_time = time.time()
    decoder = dec.Decoder(transmission_length = n, indexing_alphabet = None, indexing_scheme = "UNIQUE")

    for rx_stream in enumerate(rx_tuple_list):
        corrupted_datastream = decoder.decode(rx_stream[1])
        corrupted_codeword, erasure_locations = source_alphabet.convert_symbols_to_string(corrupted_datastream)
        corrupted_codewords_dict[corrupted_codeword] = erasure_locations
    stop_time = time.time()
    print("INDEX DECODING TIME = %lf sec" % (stop_time - start_time))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 8: REED-SOLOMON DECODING
    # ------------------------------------------------------------------------------------------------------------------

    start_time = time.time()
    for corrupted_codeword, erasure_locations in corrupted_codewords_dict.items():

        rs_formatted_word = [ord(x) for x in corrupted_codeword]
        rs_rx_codewords.append(rs_formatted_word)

        corrected_message, corrected_ecc = rs.rs_correct_msg(rs_formatted_word, n-k, erase_pos = erasure_locations)

        recovered_message_list.append("".join(chr(x) for x in corrected_message))
    stop_time = time.time()
    print("RS-DEC TIME = %lf sec" % (stop_time - start_time))

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    # STEP 9: RETURN ORIGINAL DATA
    # ------------------------------------------------------------------------------------------------------------------

    recovered_message = "".join(s for s in recovered_message_list)
    return original_message, recovered_message
    
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # STEP 10: CALCULATE NUMBER OF ERRORS IN TRANSMISSION (BER)

if __name__ == '__main__':

    message = rs.DARTH_PLAGUEIS_SCRIPT
    n = 20
    k = 11

    assert n > k, "[ERROR] PARAMETERS NOT VALID, n MUST BE GREATER THAN k!"
    delta = 1 - k/n
    
    start_time = time.time()
    original_message, recovered_message = insdel_comm_unique_index(message, n, k, delta)
    stop_time = time.time()
    total_time = stop_time - start_time

    print("COMMUNICATION OVER INSDEL CHANNEL W/FIDELITY %0.3lf COMPLETE; TOTAL TIME = %lf sec" % (delta, total_time))
    print("--> TRANSMITTED MESSAGE:")
    print(original_message)
    print("--> RECEIVED MESSAGE:")
    print(recovered_message)

    if original_message == recovered_message:
        print("COMMUNICATION SUCCESSFUL!")
    

