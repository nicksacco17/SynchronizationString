
import alphabet as a
import transmission as tx
import numpy as np
import math as math
import random as rand

#class ErasureChannel():

#    def __init__(self, delta):
#        self.delta = delta

#    def transmit(self, )


class InsDelChannel():

    def __init__(self, delta, n, insertion_prob, data_alphabet, index_alphabet, error_model = "FIXED"):
        self.delta = delta
        self.n = n

        self.max_errors = math.floor(self.delta * self.n)

        self.insertion_prob = insertion_prob
        self.data_alphabet = data_alphabet
        self.index_alphabet = index_alphabet
        self.error_model = error_model

    def transmit(self, tx_tuple_array):

        rx_tuple_array = []

        if self.max_errors == 0:
            return tx_tuple_array

        if self.error_model == "FIXED":

            available_indices = [x for x in range(0, len(tx_tuple_array))]

            error_indices = rand.sample(list(available_indices), k = self.max_errors)

            assert len(error_indices) == self.max_errors, "[ERROR], SOME ENTRIES ARE DUPLICATES!"

            actions = [0] * len(tx_tuple_array)
            
            for i in range(0, len(actions)):
                if i in error_indices:
                    operation = np.random.uniform(low = 0.0, high = 1.0)
                    if operation < self.insertion_prob:
                        actions[i] = 1
                    else:
                        actions[i] = -1

            modified_length = len(tx_tuple_array) + sum(actions)

            for j in range(0, len(actions)):

                # No error
                if actions[j] == 0:
                    rx_tuple_array.append(tx_tuple_array[j])
                
                # Insertion
                elif actions[j] == 1:
                    inserted_message = self.data_alphabet.get_random_symbol_from_alphabet()
                    inserted_index = self.index_alphabet.get_random_symbol_from_alphabet()

                    inserted_tuple = tx.Tuple(msg_symbol = inserted_message, idx_symbol = inserted_index)

                    # Append the inserted tuple
                    rx_tuple_array.append(inserted_tuple)

                    # Now append the following tuple
                    rx_tuple_array.append(tx_tuple_array[j])

                # Deletion
                elif actions[j] == -1:
                    pass

        elif self.error_model == "IID":
            #print("IID HERE")
            num_errors = 0
            actions = [0] * len(tx_tuple_array)

            for i in range(0, len(tx_tuple_array)):

                apply_error = np.random.uniform(low = 0.0, high = 1.0)

                if apply_error < self.delta:
                    
                    num_errors += 1
                    operation = np.random.uniform(low = 0.0, high = 1.0)
                    if operation < self.insertion_prob:
                        actions[i] = 1
                    else:
                        actions[i] = -1
            #if num_errors != self.max_errors:
            #    print("VARIABLE ERRORS!")

            modified_length = len(tx_tuple_array) + sum(actions)

            for j in range(0, len(actions)):

                # No error
                if actions[j] == 0:
                    rx_tuple_array.append(tx_tuple_array[j])
                
                # Insertion
                elif actions[j] == 1:
                    inserted_message = self.data_alphabet.get_random_symbol_from_alphabet()
                    inserted_index = self.index_alphabet.get_random_symbol_from_alphabet()

                    inserted_tuple = tx.Tuple(msg_symbol = inserted_message, idx_symbol = inserted_index)

                    # Append the inserted tuple
                    rx_tuple_array.append(inserted_tuple)

                    # Now append the following tuple
                    rx_tuple_array.append(tx_tuple_array[j])

                # Deletion
                elif actions[j] == -1:
                    pass

        return rx_tuple_array
        


EPSILON = 0.5

def old():
    m_string = "Te saluto.  Augustus sum, imperator et pontifex maximus romae.  Si tu es Romae amicus, es gratus."
    
    m_alphabet = alpha.Alphabet(create_ascii = True)

    m_enc = enc.Encoder(m_alphabet)

    m_transmitter = tx.Transmitter(encoder = m_enc, transmission_length = len(m_string), indexing_scheme = "UNIQUE")

    tx_tuple_array = m_transmitter.create_transmission_tuple(m_string)

    tx_msg_str = ""
    tx_idx_str = ""
    for i in range(0, len(tx_tuple_array)):
        tx_msg_str += chr(tx_tuple_array[i].message_symbol.id)
        tx_idx_str += chr(tx_tuple_array[i].index_symbol.id) + " "
    print(tx_msg_str)
    print(tx_idx_str)

    chn = InsDelChannel(delta = 0.3, ins_prob = 0.5, msg_alphabet = m_alphabet, idx_alphabet = m_transmitter.index_alphabet)
    rx_tuple_array = chn.transmit(tx_tuple_array)

    rx_msg_str = ""
    rx_idx_str = ""
    for i in range(0, len(rx_tuple_array)):
        rx_msg_str += chr(rx_tuple_array[i].message_symbol.id)
        rx_idx_str += chr(rx_tuple_array[i].index_symbol.id) + " "
    print(rx_msg_str)
    print(rx_idx_str)


if __name__ == '__main__':

    print("SALVE MUNDI")