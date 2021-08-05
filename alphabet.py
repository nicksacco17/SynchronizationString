
import numpy as np
import string as string

MAX_PRINT_SIZE = 256
NUM_ASCII_CHARACTERS = 128
PRINTABLE_CHARACTERS = list(string.printable)
NUM_PRINTABLE_CHARACTERS = len(PRINTABLE_CHARACTERS)
ERASURE_SYMBOL = "[*]"

class Symbol():

    def __init__(self, symbol_id, erasure = False):

        if erasure or symbol_id == -1:
            self.id = -1
        elif not erasure and symbol_id != -1:
            self.id = symbol_id

    def print_symbol(self, print_char = False, print_byte = False, new_line = True):

        if new_line:
            LAST_CHAR = "\n"
        else:
            LAST_CHAR = ""

        # Printing character representation of the symbol if possible
        if print_char and 0 <= self.id <= 255:
            print("(%c)" % chr(self.id), end = LAST_CHAR)
    
        # Printing byte-wise representation of the symbol
        elif print_byte and 0 <= self.id <= 255:
            print("(0x%02X)" % self.id, end = LAST_CHAR)

        # Printing numerical representation of the symbol
        elif self.id >= 0:
            print("(%d)" % self.id, end = LAST_CHAR)

        # Else erasure symbol
        else:
            print(ERASURE_SYMBOL, end = LAST_CHAR)

    def get_id(self, get_char = False, get_byte = False):
        
        # Get the character representation of the symbol if possible
        if get_char and 0 <= self.id <= 255:
            return "%c" % chr(self.id)

        # Get the byte representation of the symbol if possible
        elif get_byte and 0 <= self.id <= 255:
            return "0x%02X" % self.id

        # Get the numerical representation of the symbol if possible
        elif self.id >= 0:
            return "%d" % self.id

        # Else erasure symbol
        else:
            return ERASURE_SYMBOL

class Alphabet():

    def __init__(self, size = NUM_PRINTABLE_CHARACTERS):
        self.size = size
        self.dictionary = np.empty(shape = self.size, dtype = Symbol)
        self.create_dictionary()

    def create_dictionary(self):
        for i in range(0, self.size):
            self.dictionary[i] = Symbol(i)

    def get_symbol_by_index(self, index):
        return self.dictionary[index]

    def get_symbol_by_letter(self, letter):
        for i in range(0, self.size):
            if self.dictionary[i].id == letter:
                return self.dictionary[i]

    def get_random_symbol_from_alphabet(self):
        rand_index = np.random.randint(low = 0, high = self.size)
        return self.dictionary[rand_index]

    def convert_string_to_symbols(self, input_str):
        symbol_str = np.empty(shape = len(input_str), dtype = Symbol)
        for i in range(0, len(input_str)):
            symbol_str[i] = self.get_symbol_by_letter(ord(input_str[i]))

        return symbol_str    

    def convert_symbols_to_string(self, input_symbol_arr):

        erasure_locations = []

        string_repr = ""
        for symbol in enumerate(input_symbol_arr):
            char_representation = symbol[1].get_id(get_char = True, get_byte = False)
            if char_representation == ERASURE_SYMBOL:
                erasure_locations.append(symbol[0])
                string_repr += "0"
            else:
                string_repr += char_representation

        return string_repr, [ord(x) for x in string_repr], erasure_locations

    def print_dictionary(self, print_char = False, print_byte = False):
        print("-------------------- ALPHABET --------------------")
        print("SIZE OF ALPHABET = %d" % len(self.dictionary))
        for s in self.dictionary:
            s.print_symbol(print_char = print_char, print_byte = print_byte)

if __name__ == '__main__':

    #a = Alphabet(62, create_ascii = True)
    #a.print()
    #print(PRINTABLE_CHARACTERS)

    #for c in PRINTABLE_CHARACTERS:
    #    s = Symbol(ord(c))
    #    s.print_symbol(print_byte = True)
    #    s.print_symbol(print_byte = False)

    for i in range(0, 300):
        print("---------- i = %d ----------" % i)
        s = Symbol(i)

        print("BYTE: ", end = "")
        s.print_symbol(print_char = True, print_byte = True)

        print("CHAR: ", end = "")
        s.print_symbol(print_char = True, print_byte = False)

        print("DEC: ", end = "")
        s.print_symbol(print_char = False, print_byte = False)

    print("---------- ERASURE ----------")
    erasure = Symbol(-1, erasure = True)
    erasure.print_symbol()

    a = Alphabet(300)
    a.print_dictionary(print_char = True, print_byte = False)

    m_string = "Te saluto.  Augustus sum, imperator et pontifex maximus romae.  Si tu es Romae amicus, es gratus."

    m_codeword = a.convert_string_to_symbols(m_string)

    for s in m_codeword:
        s.print_symbol(print_char = True, print_byte = True)

