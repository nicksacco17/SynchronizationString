
import alphabet as a
import distance as dist
import sync_string as sync
import numpy as np
from collections import Counter

class Decoder():

    def __init__(self, transmission_length, indexing_sequence, indexing_scheme = "UNIQUE", epsilon = 0):
        self.transmission_length = transmission_length
        self.indexing_sequence = indexing_sequence
        self.indexing_scheme = indexing_scheme
        self.epsilon = epsilon

    def decode(self, rx_tuple_array):

        corrupted_word = np.empty(shape = self.transmission_length, dtype = a.Symbol)
        erasure_count = 0

        #for k in range(0, len(rx_tuple_array)):
        #    print(rx_tuple_array[k].index_symbol.get_id())

        if self.indexing_scheme == "UNIQUE":

            # For each expected index
            for s in enumerate(self.indexing_sequence):

                expected_index_counter = 0
                last_valid_index = 0
                
                # Check for the expected value in the list of received tuples
                for j in range(0, len(rx_tuple_array)):
                    
                    # If the expected value is in the list of received tuples, increment the counter and store the last valid index
                    if rx_tuple_array[j].index_symbol.get_id() == s[1].get_id():
                        expected_index_counter += 1
                        last_valid_index = j

                # If the expected count is exactly 1, there is no apparent error - just append the symbol
                if expected_index_counter == 1:
                    corrupted_word[s[0]] = rx_tuple_array[last_valid_index].message_symbol
                # Else if the count is NOT 1 then there was a deletion (counter == 0) or an insertion (counter > 1), so insert erasure
                else:
                    corrupted_word[s[0]] = a.Symbol(symbol_id = -1, erasure = True)
                    erasure_count += 1
        
        # Minimum RSD decoding via synchroniztion strings
        elif self.indexing_scheme == "SYNC":

            # Convert the actual index sequence (TX) and corrupted index sequence (RX) to synchronization strings
            self.tx_sync_str = sync.Synchronization_String(epsilon = self.epsilon, n = len(self.indexing_sequence))
            self.rx_sync_str = sync.Synchronization_String(epsilon = self.epsilon, n = len(rx_tuple_array))

            for i in range(0, len(self.indexing_sequence)):
                self.tx_sync_str.str[i] = self.indexing_sequence[i]
            for j in range(0, len(rx_tuple_array)):
                self.rx_sync_str.str[j] = rx_tuple_array[j].index_symbol

            decoding_best_guesses = np.empty(shape = len(rx_tuple_array), dtype = int)

            # For each received substring
            for j in range(1, self.rx_sync_str.n + 1):

                #print("SUBSTR j = %d" % j)
                rx_substr = self.rx_sync_str.get_substring(0, j)

                min_RSD = 1
                most_likely_index = -1

                #print("---------- SUBSTRING S'[0:%d] ----------" % j)
                #for s in rx_substr:
                #    s.print_symbol(print_char = False, print_byte = False, new_line = False)
                #print()

                # Compare to all expected codewords - i.e. prefixes of S
                for i in range(1, self.tx_sync_str.n + 1):

                    tx_substr = self.tx_sync_str.get_substring(0, i)
                    #print("--> SUBSTRING S[0:%d]: " % i, end = "")
                    #for s in tx_substr:
                    #    s.print_symbol(print_char = False, print_byte = False, new_line = False)

                    # Calculate RSD{S[0, i], S'[0, j]}
                    current_rsd = dist.symbol_rsd(rx_substr, tx_substr)

                    if current_rsd < min_RSD:
                        min_RSD = current_rsd
                        most_likely_index = i

                    #print(" RSD{S[0, i], S'[0, j] = %0.3lf" % current_rsd)
               
                #print("MOST LIKELY SUBSTRING S[0:%d]: " % most_likely_index, end = "")
                #best_codeword = self.tx_sync_str.get_substring(0, most_likely_index)
                #for s in best_codeword:
                #        s.print_symbol(print_char = False, print_byte = False, new_line = False)
                #print()
                decoding_best_guesses[j - 1] = most_likely_index - 1  
            
            # Check each expected index...
            for k in range(0, len(corrupted_word)):

                expected_index_counter = 0
                last_valid_index = 0

                # ...In the list of guessed indices.  If found, increment the count and keep track of the index
                for m in range(0, len(decoding_best_guesses)):

                    if decoding_best_guesses[m] == k:
                        expected_index_counter += 1
                        last_valid_index = m

                # If one and only one valid index found, attach the corresponding data symbol to the recovered codeword
                if expected_index_counter == 1:
                    corrupted_word[k] = rx_tuple_array[last_valid_index].message_symbol
                # Else insertion or deletion, attach erasure
                else:
                    corrupted_word[k] = a.Symbol(symbol_id = -1, erasure = True)
                    erasure_count += 1
             
        #for s in corrupted_word:
            #s.print_symbol(print_char = True, print_byte = False, new_line = False)
        #print()
        return corrupted_word, erasure_count
    
