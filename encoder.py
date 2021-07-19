
import numpy as np
import alphabet as a

# This module formats input string data into symbols from the supported alphabet
# This module is unncessary...encoding is only on the coding side of the data...
# We just need to get a Symbol representation of the data
class Encoder():

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def encode(self, base_string):
        codeword = np.empty(shape = len(base_string), dtype = a.Symbol)
        
        for i in range(0, len(codeword)):
            codeword[i] = self.alphabet.get_symbol_by_letter(ord(base_string[i]))

        return codeword

if __name__ == '__main__':

    m_alphabet = a.Alphabet(a.NUM_ASCII_CHARACTERS)
    m_string = "Te saluto.  Augustus sum, imperator et pontifex maximus romae.  Si tu es Romae amicus, es gratus."

    e = Encoder(alphabet = m_alphabet)
    m_codeword = e.encode(m_string)

    for s in m_codeword:
        s.print_symbol(print_char = True, print_byte = True)