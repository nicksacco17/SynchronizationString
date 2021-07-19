
import alphabet as a
import numpy as np

class Decoder():

    def __init__(self, transmission_length, indexing_alphabet = None, indexing_scheme = "UNIQUE"):
        self.transmission_length = transmission_length
        self.indexing_alphabet = indexing_alphabet
        self.indexing_scheme = indexing_scheme

    def decode(self, rx_tuple_array):

        corrupted_word = np.empty(shape = self.transmission_length, dtype = a.Symbol)

        #for k in range(0, len(rx_tuple_array)):
        #    print(rx_tuple_array[k].index_symbol.get_id())

        if self.indexing_scheme == "UNIQUE":

            indexing_sequence = [a.Symbol(x) for x in range(0, self.transmission_length)]

            # For each expected index
            for s in enumerate(indexing_sequence):

                expected_index_counter = 0
                last_valid_index = 0
                
                # Check for the expected value in the list of received tuples
                for j in range(0, len(rx_tuple_array)):
                    
                    # If the expected value is in the list of received tuples, increment the counter and store the last valid index
                    if rx_tuple_array[j].index_symbol.get_id() == s[1].get_id():
                        expected_index_counter += 1
                        last_valid_index = j

                # If the count is 0, there was a deletion
                if expected_index_counter == 0:
                    corrupted_word[s[0]] = a.Symbol(symbol_id = -1, erasure = True)
                
                # Else if the count is greater than 1, there was an insertion
                elif expected_index_counter > 1:
                    corrupted_word[s[0]] = a.Symbol(symbol_id = -1, erasure = True)      

                # Else if the count is EXACTLY 1, then there probably was no error, so get the data symbol at the last valid index
                elif expected_index_counter == 1:
                    corrupted_word[s[0]] = rx_tuple_array[last_valid_index].message_symbol

        return corrupted_word
