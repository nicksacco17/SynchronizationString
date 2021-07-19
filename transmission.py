
import numpy as np
import alphabet as a

class Tuple():

    def __init__(self, msg_symbol, idx_symbol):
        self.message_symbol = msg_symbol
        self.index_symbol = idx_symbol

    def print_tuple(self, get_char = False, get_byte = False):
        print("< %s | %s >" % (self.message_symbol.get_id(get_char = get_char, get_byte = get_byte), self.index_symbol.get_id(get_char = False, get_byte = False)))

class Transmitter():

    def __init__(self, transmission_length, index_alphabet = None, indexing_scheme = "UNIQUE"):

        self.transmission_length = transmission_length
        self.indexing_scheme = indexing_scheme

        if self.indexing_scheme == "UNIQUE":
            self.index_alphabet = a.Alphabet(size = transmission_length)
        else:
            self.index_alphabet = index_alphabet

    # Assume input_data_stream is an array of Symbols
    def create_transmission_tuple(self, input_data_stream):

        tx_tuple_array = np.empty(shape = self.transmission_length, dtype = Tuple)

        if self.indexing_scheme == "UNIQUE":
            indexing_sequence = np.empty(shape = self.transmission_length, dtype = a.Symbol)
        
            if len(indexing_sequence) == len(input_data_stream):
                for i in range(0, len(tx_tuple_array)):
                    tx_tuple_array[i] = Tuple(input_data_stream[i], self.index_alphabet.get_symbol_by_index(index = i))
            else:
                print("[ERROR] LENGTHS DO NOT AGREE")

        return tx_tuple_array
            

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


   