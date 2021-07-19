
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

    def __init__(self, delta, n, insertion_prob, data_alphabet, index_alphabet):
        self.delta = delta
        self.n = n

        self.max_errors = math.floor(self.delta * self.n)

        self.insertion_prob = insertion_prob
        self.data_alphabet = data_alphabet
        self.index_alphabet = index_alphabet

    def transmit(self, tx_tuple_array):

        available_indices = [x for x in range(0, len(tx_tuple_array))]
        #print(available_indices)

        error_indices = rand.sample(list(available_indices), k = self.max_errors)

        assert len(error_indices) == self.max_errors, "[ERROR], SOME ENTRIES ARE DUPLICATES!"

        #print(error_indices)

        actions = [0] * len(tx_tuple_array)
        
        for i in range(0, len(actions)):
            if i in error_indices:
                operation = np.random.uniform(low = 0.0, high = 1.0)
                if operation < self.insertion_prob:
                    actions[i] = 1
                else:
                    actions[i] = -1


        #print(actions)

        modified_length = len(tx_tuple_array) + sum(actions)

        rx_tuple_array = []

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
        


'''
class InsDelChannel():

    def __init__(self, delta, ins_prob, msg_alphabet, idx_alphabet):
        self.delta = delta
        self.insertion_prob = ins_prob
        self.message_alphabet = msg_alphabet
        self.index_alphabet = idx_alphabet

    def transmit(self, tx_tuple_array):

        insdel_flags = [0] * len(tx_tuple_array)
        # For each tuple in the stream
        for i in range(0, len(tx_tuple_array)):

            apply_error = np.random.uniform(low = 0.0, high = 1.0)

            if apply_error < self.delta:

                operation = np.random.uniform(low = 0.0, high = 1.0)

                if operation < self.insertion_prob:
                    insdel_flags[i] = 1
                else:
                    insdel_flags[i] = -1

        num_errors = 0

        for i in range(0, len(insdel_flags)):

            if insdel_flags[i] != 0:

                num_errors += 1

        print(num_errors)


        modified_length = len(tx_tuple_array) + sum(insdel_flags)

        rx_tuple_list = []

        for j in range(0, len(insdel_flags)):

            if insdel_flags[j] == 0:
                rx_tuple_list.append(tx_tuple_array[j])

            elif insdel_flags[j] == 1:
                inserted_message = self.message_alphabet.get_random_symbol_from_alphabet()
                inserted_index = self.index_alphabet.get_random_symbol_from_alphabet()

                inserted_tuple = tx.Tuple(msg_symbol = inserted_message, idx_symbol = inserted_index)
                rx_tuple_list.append(inserted_tuple)
                rx_tuple_list.append(tx_tuple_array[j])
            elif insdel_flags[j] == -1:
                pass

        rx_tuple_array = np.array(rx_tuple_list)
        return rx_tuple_array

    def transmit_string(self, tx_tuple):

        channel_str = list(tx_str)
        insdel_flags = [0] * len(channel_str)

        for i in range(0, len(channel_str)):

            apply_error = np.random.uniform(low = 0.0, high = 1.0)

            if apply_error < self.delta:

                operation = np.random.uniform(low = 0.0, high = 1.0)

                # Apply insertion
                if operation < self.insertion_prob:
                    insdel_flags[i] = 1
                   # channel_str.insert(i, rand.choice(string.ascii_letters + string.digits))
                # Else apply deletion
                else:
                    insdel_flags[i] = -1

        build_str = []
        for j in range(0, len(insdel_flags)):

            # If no error, just append the character
            if insdel_flags[j] == 0:
                build_str.append(channel_str[j])
            # Else if there is insertion, append the insertion then the base character
            elif insdel_flags[j] == 1:
                build_str.append(rand.choice(string.ascii_letters + string.digits))
                build_str.append(channel_str[j])
            # Else if there is deletion, do not append anything
            elif insdel_flags[j] == -1:
                pass

        rx_str = "".join(build_str)
        return rx_str
'''
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