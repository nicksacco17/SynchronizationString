
import numpy as np
import alphabet as a
import sync_string as sync

class Tuple():

    def __init__(self, msg_symbol, idx_symbol):
        self.message_symbol = msg_symbol
        self.index_symbol = idx_symbol

    def print_tuple(self, get_char = False, get_byte = False):
        print("< %s | %s >" % (self.message_symbol.get_id(get_char = get_char, get_byte = get_byte), self.index_symbol.get_id(get_char = False, get_byte = False)))

class Transmitter():

    def __init__(self, transmission_length, epsilon = 0.5, indexing_scheme = "UNIQUE", load_str = False, path = None):

        self.transmission_length = transmission_length
        self.indexing_scheme = indexing_scheme
        self.indexing_sequence = np.empty(shape = self.transmission_length, dtype = a.Symbol)

        # If we are using a unique indexing scheme, index each element with a unique id
        if self.indexing_scheme == "UNIQUE":
            self.indexing_alphabet = a.Alphabet(size = transmission_length)
            for i in range(0, self.transmission_length):
                self.indexing_sequence[i] = self.indexing_alphabet.get_symbol_by_index(index = i)
            
        # Else if we are using synchronization scheme, generate required synchronization string & append those elements
        elif self.indexing_scheme == "SYNC":
            
            if load_str:
                self.sync_str = sync.load_sync_str(n = self.transmission_length, epsilon = epsilon, directory = path)
                self.sync_str.print_string(print_char = False, print_byte = False)
            else:
                self.sync_str = sync.Synchronization_String(epsilon = epsilon, n = self.transmission_length)
                self.sync_str.verify_synchronization()

            for i in range(0, self.transmission_length):
                self.indexing_sequence[i] = self.sync_str.get_symbol(index = i)

    # Assume input_data_stream is an array of Symbols
    def create_transmission_tuple(self, input_data_stream):

        tx_tuple_array = np.empty(shape = self.transmission_length, dtype = Tuple)

        if len(input_data_stream) == len(self.indexing_sequence):
            for i in range(0, self.transmission_length):
                tx_tuple_array[i] = Tuple(input_data_stream[i], self.indexing_sequence[i])
        else:
                print("[ERROR] LENGTHS DO NOT AGREE")

        return tx_tuple_array

    def get_indexing_sequence(self):
        return self.indexing_sequence
            

'''
class StreamFormatter():

    def __init__(self, transmission_length, index_alphabet = None, indexing_scheme = "UNIQUE"):
        
        self.transmission_length = transmission_length
        self.indexing_scheme = indexing_scheme

        if self.indexing_scheme == "UNIQUE":
            self.index_alphabet = alpha.Alphabet(size = self.transmission_length, create_ascii = False)
        else:
            self.index_alphabet = index_alphabet

    def create_transmission_tuple(self, base_word):

        tx_tuple_array = np.empty(shape = self.transmission_length, dtype = Tuple)


class Transmitter():

    def __init__(self, encoder, transmission_length, index_alphabet = None, indexing_scheme = "UNIQUE"):
        self.encoder = encoder
        self.transmission_length = transmission_length
        self.indexing_scheme = indexing_scheme
        
        if self.indexing_scheme == "UNIQUE":
            self.index_alphabet = alpha.Alphabet(size = self.transmission_length, create_ascii = False)
        else:
            self.index_alphabet = index_alphabet
    
    def create_transmission_tuple(self, base_string):

        tx_tuple_array = np.empty(shape = self.transmission_length, dtype = Tuple)
        codeword = self.encoder.encode(base_string)

        if self.indexing_scheme == "UNIQUE":
            indexing_sequence = np.empty(shape = self.transmission_length, dtype = alpha.Symbol)

            for i in range(0, len(indexing_sequence)):
                indexing_sequence[i] = self.index_alphabet.get_symbol_by_index(index = i)

            assert len(codeword) == len(indexing_sequence), "[ERROR] THE LENGTH OF THE CODEWORD AND INDEXING SEQUENCE DO NOT AGREE"

            for i in range(0, len(codeword)):
                tx_tuple_array[i] = Tuple(codeword[i], indexing_sequence[i])
            return tx_tuple_array
'''

if __name__ == '__main__':

    m_string = "Te saluto.  Augustus sum, imperator et pontifex maximus romae.  Si tu es Romae amicus, es gratus."
    
    m_alphabet = a.Alphabet(size = a.NUM_ASCII_CHARACTERS)

    data_stream = m_alphabet.convert_string_to_symbols(input_str = m_string)

    for s in data_stream:
        s.print_symbol(print_char = True, print_byte = True)

    m_transmitter = Transmitter(transmission_length = len(m_string), indexing_scheme = "UNIQUE")

    tx_tuple = m_transmitter.create_transmission_tuple(data_stream)

    for t in tx_tuple:
        t.print_tuple(get_char = True, get_byte = False)


   