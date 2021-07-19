
import numpy as np
import alphabet as a
import binascii
import string

hex_letters = "0123456789abcdef"

def convert_char_to_bin(c):

    index = hex_letters.find(c.lower())
    assert index != -1, "[ERROR] NOT A VALID HEX DIGIT"
    return np.binary_repr(index, width = 4)

class HammingCode():

    def __init__(self, k, n, channel_fidelity, hard_code = False):
        self.hard_code = hard_code
        self.k = k                                                              # Message length
        self.n = n                                                              # Block length
        self.channel_fidelity = channel_fidelity
        self.G = np.zeros(shape = (self.k, self.n), dtype = int)               # Generator matrix
        self.H = np.zeros(shape = (self.n - self.k, self.n), dtype = int)      # Parity check matrix
        self.H_rows = self.n - self.k
        self.codebook = {}

        # If using the hard code (i.e. default) (7, 4) Hamming code
        if hard_code and self.k == 4 and self.n == 7:

            self.G = np.asarray([
                                    [1, 1, 1, 0, 0, 0, 0], 
                                    [1, 0, 0, 1, 1, 0, 0], 
                                    [0, 1, 0, 1, 0, 1, 0], 
                                    [1, 1, 0, 1, 0, 0, 1]
                                ]) 
            self.GT = np.transpose(self.G)      

            self.H = np.asarray([
                                    [1, 0, 1, 0, 1, 0, 1], 
                                    [0, 1, 1, 0, 0, 1, 1], 
                                    [0, 0, 0, 1, 1, 1, 1]
                                ])  

            self.R = np.asarray([
                                    [0, 0, 1, 0, 0, 0, 0], 
                                    [0, 0, 0, 0, 1, 0, 0], 
                                    [0, 0, 0, 0, 0, 1, 0], 
                                    [0, 0, 0, 0, 0, 0, 1]
                                ]) 

        else:
        # Populate the parity-check matrix

            # For each column
            for i in range(1, self.n + 1):
                
                # Get the binary representation of the column
                #formatting_string = "{0:0%sb}" % (self.n - self.k)
                #bin_i = formatting_string.format((i))
                #bin_i_arr = list(bin_i)

                bin_i_arr = np.binary_repr(i, width = self.n - self.k)

                assert len(bin_i_arr) == self.H_rows, "[ERROR] DIMENSIONS DO NOT AGREE"

                # Populate each row in the current column using the binary representation of i
                for k in range(0, len(bin_i_arr)):
                    if bin_i_arr[k] == '1':
                        self.H[k][i - 1] = True
                    elif bin_i_arr[k] == '0':
                        self.H[k][i - 1] = False

    def generate_codebook(self):

        if self.hard_code:

            for i in range(0, 16):
                bin_i = np.binary_repr(i, width = 4)
                message = np.asarray(list(bin_i), dtype = int)
                message_str = "".join(str(j) for j in message)
                self.codebook[message_str] = np.matmul(self.GT, message) % 2

    def convert_to_standard_form(self):

        max_index = (0, 0)
        max_value = 0

        mat_size = self.k * self.n

        # Look at the pivot
        pivot_row = 0
        pivot_col = self.n - self.k + 1

        pivot_index = (pivot_row, pivot_col)

        while pivot_row < self.H_rows and pivot_col < self.n:

            max_value = 0
            max_index = (0, 0)

            pivot_index = (pivot_row, pivot_col)

            for i in range(pivot_row, self.H_rows):

                if np.abs(self.H[i][pivot_col]) > max_value:
                    max_row = i
                    max_value = np.abs(self.H[i][pivot_col])

            if max_value == 0:
                pivot_col += 1

            else:
                
                # Swap the pivot row and the max row
                self.H[[pivot_row, max_row], : ] = self.H[[max_row, pivot_row], : ]

                for i in range(pivot_row + 1, self.H_rows):

                    f = self.H[i][pivot_col] / self.H[pivot_row][pivot_col]

                    self.H[i][pivot_col] = 0

                    for j in range(pivot_col + 1, self.n):

                        self.H[i][j] = (self.H[i][j] + self.H[pivot_row][j] * f) % 2


                # Increment the pivot
                pivot_row += 1
                pivot_col += 1

        
        #for row in range(0, self.n - self.k):
        #    print("---------- %d ----------" % row)
        #    for col in range(self.n - self.k + 1, self.n):
        #        print(self.H[row][col])

    def transmit_erasure(self, codeword):

        modified_codeword = np.copy(codeword)
        num_errors = 0

        error = np.random.uniform(low = 0.0, high = 1.0)
        if error < self.channel_fidelity:

            error_index1 = np.random.randint(low = 0, high = self.n)

            error_index2 = np.random.randint(low = 0, high = self.n)

            while error_index2 == error_index1:
                error_index2 = np.random.randint(low = 0, high = self.n)

            assert error_index1 != error_index2, "[ERROR] INDICES DO NOT AGREE! (%d, %d)" % (error_index1, error_index2)

            modified_codeword[error_index1] = -1
            modified_codeword[error_index2] = -1

            num_errors += 2
        
        return modified_codeword, num_errors

    def transmit(self, codeword):

        modified_codeword = np.copy(codeword)

        error = np.random.uniform(low = 0.0, high = 1.0)
        if error < self.channel_fidelity:

            error_index = np.random.randint(low = 0, high = self.n)
            modified_codeword[error_index] = not modified_codeword[error_index]

        return modified_codeword

    def print_generator(self):
        print("---------- GENERATOR MATRIX ----------")
        print("DIMENSIONS: (%d x %d)" % (self.k, self.n))
        
        for i in range(0, self.k):
            print("| ", end = "")
            for j in range(0, self.n):
                print("%d " % self.G[i][j], end = "")
            print("|")

        print("---------- GENERATOR TRANSPOSE MATRIX ----------")
        print("DIMENSIONS: (%d x %d)" % (self.n, self.k))
        
        for i in range(0, self.n):
            print("| ", end = "")
            for j in range(0, self.k):
                print("%d " % self.GT[i][j], end = "")
            print("|")

    def print_parity_check(self):
        print("---------- PARITY-CHECK MATRIX ----------")
        print("DIMENSIONS: (%d x %d)" % (self.H_rows, self.n))
        
        for i in range(0, self.H_rows):
            print("| ", end = "")
            for j in range(0, self.n):
                print("%d " % self.H[i][j], end = "")
            print("|")

    def encode(self, message):

        codeword = np.matmul(self.GT, message) % 2
        return codeword

    def calc_Hamming_distance(self, codeword1, codeword2):

        assert len(codeword1) == len(codeword2), "[ERROR] DIMENSIONS DO NOT AGREE"

        hamming_distance = 0
        for i in range(0, len(codeword1)):
            if codeword1[i] != codeword2[i]:
                hamming_distance += 1
        return hamming_distance

    def minimum_distance_decode(self, codeword):

        minimum_distance = np.Inf
        minimum_message = None
        minimum_codeword = None

        for message_in_dict in self.codebook.keys():

            codeword_in_dict = self.codebook[message_in_dict]
            hamming_distance = self.calc_Hamming_distance(codeword, codeword_in_dict)

            if hamming_distance < minimum_distance:
                minimum_distance = hamming_distance
                minimum_codeword = codeword_in_dict
                minimum_message = message_in_dict

        return np.asarray(list(minimum_message), dtype = int)

    def syndrome_decode(self, codeword):

        message = np.zeros(shape = (self.k, 1), dtype = int)
        checked_codeword = np.matmul(self.H, codeword) % 2
        
        modified_codeword = np.copy(codeword)
        if sum(checked_codeword) == 0:
            message = np.matmul(self.R, codeword)

        else:

            index_in_error = np.flip(checked_codeword)

            dec_value = 0
            for i in range(0, len(index_in_error)):
                dec_value += index_in_error[i] * 2 ** (len(index_in_error) - i - 1)

            modified_codeword[dec_value - 1] = not modified_codeword[dec_value - 1]      
            message = np.matmul(self.R, modified_codeword) % 2

        return message

def hamming_test_case():

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
    h = HammingCode(k = 4, n = 7, channel_fidelity = 1, hard_code = True)
  
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

if __name__ == '__main__':

    # Test string to transmit
    # m_string = "Te saluto.  Augustus sum, imperator et pontifex maximus romae.  Si tu es Romae amicus, es gratus."
    
    #byte_string = binascii.hexlify(m_string.encode('utf-8'))
    #bit_string = ""
    #for nibble in byte_string:
    #    bit_string += convert_char_to_bin(chr(nibble))

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
    h = HammingCode(k = 4, n = 7, channel_fidelity = 1, hard_code = True)
  
    # Populate the codebook
    h.generate_codebook()

    for i in range(0, 16):

        bin_i = np.binary_repr(i, width = 4)
        message = np.asarray(list(bin_i), dtype = int)
        
        tx_codeword = h.encode(message)

        # Convert the binary to Symbols

        rx_codeword, _ = h.transmit_erasure(tx_codeword)

        recovered_message = h.minimum_distance_decode(rx_codeword)
        
        #recovered_message = h.syndrome_decode(rx_codeword)

        print("MESSAGE = %s, TRANSMITTED CODEWORD = %s, RECEIVED CODEWORD = %s, RECOVERED MESSAGE = %s" % (message, tx_codeword, rx_codeword, recovered_message))

        #print("---------- CODEWORD %s ----------" % tx_codeword)
        #for j in range(0, len(h.codebook)):

        #    bin_j = np.binary_repr(j, width = 4)
        #    message = np.asarray(list(bin_j), dtype = int)
        #    message_str = "".join(str(k) for k in message)

        #    codeword_from_dict = h.codebook[message_str]

        #    hamming_distance = h.calc_Hamming_distance(tx_codeword, codeword_from_dict)
        #    print("TX CODEWORD = %s, CODEWORD FROM CODEBOOK = %s, HAMMING DISTANCE = %d" % (tx_codeword, codeword_from_dict, hamming_distance))

        #rx_codeword = h.transmit(tx_codeword)



        

        #print("MESSAGE = %s, CODEWORD = %s, RECOVERED MESSAGE = %s" % (message, tx_codeword, recovered_message))

    #h.H[0] = np.asarray([0, 0, 0, 1, 1, 1, 1])
    #h.H[1] = np.asarray([1, 0, 1, 0, 1, 0, 1])
    #h.H[2] = np.asarray([0, 1, 1, 0, 0, 1, 1])

    #h.print_parity_check()

    #h.convert_to_standard_form()

    #h.print_parity_check()
    

        
    