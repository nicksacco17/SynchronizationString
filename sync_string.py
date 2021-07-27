import random as rand
import string as string
import numpy as np
import matplotlib.pyplot as plt
import time as time
import math as math
import csv as csv
import os as os
import sys as sys


import alphabet as a
import distance as dist

DEBUG_FLAG = False

NUM_ITERATIONS = 25
MAX_WORD_SIZE = 50

def pause(flag):
    if flag:
        input("Press Enter to continue...")

class Synchronization_String():

    def __init__(self, epsilon, n):
        self.epsilon = epsilon
        self.ed_prefactor = 1.0 - self.epsilon
        self.n = n

        self.alphabet_size = int(math.ceil(1.0 / self.epsilon ** 4))
        self.index_alphabet = a.Alphabet(size = self.alphabet_size)

        self.str = np.empty(shape = self.n, dtype = a.Symbol)
        self.create_random_string()

        self.intervals_to_check = []
        self.num_itervals = 0

        self.get_intervals()

    def create_random_string(self):

        for i in range(0, len(self.str)):
            self.str[i] = self.index_alphabet.get_random_symbol_from_alphabet()

    def print_string(self, print_char = True, print_byte = True):
        print("---------- SYNCHRONIZATION STRING ----------")
        print("LENGTH = %d" % self.n)

        for s in self.str:
            s.print_symbol(print_char = print_char, print_byte = print_byte, new_line = False)
        print()

    def get_symbol(self, index):
        return self.str[index]

    def get_intervals(self):
        
        for i in range(0, self.n + 1):
            for j in range(0, self.n + 1):
                for k in range(0, self.n + 1):
                    if 0 <= i < j < k <= self.n:
                        self.intervals_to_check.append((i, j, k))
                        self.num_itervals += 1

    def get_substring(self, start_index, stop_index):
        return self.str[start_index : stop_index]

    def print_interval(self, interval, print_char = True, print_byte = True, new_line = False):

        i = interval[0]
        k = interval[2]

        for s in self.str[i : k]:
            s.print_symbol(print_char = print_char, print_byte = print_byte, new_line = new_line)
        print()

    def verify_intervals(self):

        num_invalid_intervals = 0
        for interval in enumerate(self.intervals_to_check):

            #if interval[0] % int(self.num_itervals * 0.05) == 0:
            #    print("INTERVAL [%d]" % interval[0])
            valid_interval, EDIT_DISTANCE, MINIMUM = self.check_interval(interval = interval[1])
            #print("ED = %lf, MIN = %lf" % (EDIT_DISTANCE, MINIMUM))
            if not valid_interval:

                for m in range(0, interval[1][2] - interval[1][0]):
                    self.str[interval[1][0] + m] = self.index_alphabet.get_random_symbol_from_alphabet()

                #print("INTERVAL BEFORE: ")
                #self.print_interval(interval[1])
                #VALID_INTERVAL = False
                #while not VALID_INTERVAL:
                #    for m in range(0, interval[1][2] - interval[1][0]):
                #        self.str[interval[1][0] + m] = self.index_alphabet.get_random_symbol_from_alphabet()
                #    VALID_INTERVAL, EDIT_DISTANCE, MINIMUM = self.check_interval(interval = interval[1])
                    #print("ED = %lf, MIN = %lf" % (EDIT_DISTANCE, MINIMUM))
                #print("INTERVAL AFTER: ")
                #self.print_interval(interval[1])
                # Regenerate interval
                

                #print("--> INTERVAL [%d]: (i = %d, j = %d, k = %d) \tEPSILON-SYNC VALID: %s" % (interval[0], interval[1][0], interval[1][1], interval[1][2], valid_interval))
                num_invalid_intervals += 1
        
        return num_invalid_intervals

    def check_interval(self, interval):

        i = interval[0]
        j = interval[1]
        k = interval[2]

        interval_ed = dist.symbol_ed(self.str[i : j], self.str[j : k])
        return interval_ed > self.ed_prefactor * (k - i), interval_ed, self.ed_prefactor * (k - i)

    def verify_synchronization(self):
        
        SYNCHRONIZATION_COMPLETE = False

        #print("NUMBER OF INTERVALS = %d" % self.num_itervals)
        iteration = 0
        total_number_invalid_intervals = 0
        average_number_invalid_intervals_per_iteration = 0
        while not SYNCHRONIZATION_COMPLETE:

            number_invalid_intervals = self.verify_intervals()
            total_number_invalid_intervals += number_invalid_intervals

            #if iteration % 10 == 0:
            #    print("--> ITERATION = %d, NUMBER OF INVALID INTERVALS = %d" % (iteration, number_invalid_intervals))

            iteration += 1
            if number_invalid_intervals == 0:
                SYNCHRONIZATION_COMPLETE = True

        # Verify one more time
        num_invalid_intervals = self.verify_intervals()
        assert num_invalid_intervals == 0, "[ERROR], VERIFICATION NOT COMPLETE!"

        average_number_invalid_intervals_per_iteration = total_number_invalid_intervals / iteration

        #print("VERIFICATION COMPLETE; NUMBER OF INVALID INTERVALS = %d" % num_invalid_intervals)
        #print("TOTAL NUMBER OF ITERATIONS = %d" % iteration)
        #print("AVERAGE NUMBER OF INVALID INTERVALS PER ITERATION = %d" % average_number_invalid_intervals_per_iteration)
        return iteration, average_number_invalid_intervals_per_iteration
        
    def get_string_representation(self):
        #return self.index_alphabet.convert_symbols_to_string(self.str)
        return "".join("(%s)" % s.get_id(get_char = False, get_byte = False) for s in self.str)

def main_old():

    word_size_array = np.arange(10, MAX_WORD_SIZE)
    comp_time_by_size_array = np.zeros(MAX_WORD_SIZE - 10, dtype = float)
    for j in range(0, len(comp_time_by_size_array)):

        run_time = 0
        print("---------- WORD SIZE = %d ----------" % word_size_array[j])
        for i in range(0, NUM_ITERATIONS):

            if i % (NUM_ITERATIONS / 10) == 0:
                print("ITERATION %d" % i)
            # Generate a new synchronization string
            S = Synchronization_String(epsilon = 0.5, n = j)
            pre_string = S.str

            # Now actually construct the string from the starting random string
            start_time = time.time()
            S.construct()
            stop_time = time.time()
            total_time = stop_time - start_time
            run_time += total_time
            post_string = S.str

            assert len(pre_string) == len(post_string), "ERROR: STRINGS DO NOT HAVE THE SAME LENGTH"

            #print(S.average_rsd)
        comp_time_by_size_array[j] = (total_time / NUM_ITERATIONS)
    plt.plot(word_size_array, comp_time_by_size_array, 'r-', linewidth = 1.5, markersize = 1.5)
    plt.xlabel("Word Size")
    plt.ylabel("Compute Time (sec)")
    plt.title("Computation Time vs. String Size")
    plt.legend()
    plt.show()


    epsilon_array = np.linspace(0, 1, 50)
    epsilon_array = epsilon_array[1:-2]

    avg_num_diff_char_array_per_eps = np.zeros(len(epsilon_array), dtype = float)
    avg_num_resamples_array_per_eps = np.zeros(len(epsilon_array), dtype = float)
    avg_rsd_array_per_eps = np.zeros(len(epsilon_array), dtype = float)

    for m in range(0, len(epsilon_array)):
        
        print("---------- EPSILON = %lf ----------" % epsilon_array[m])

        num_different_char_array = np.zeros(NUM_ITERATIONS, dtype = int)
        num_resamples_array = np.zeros(NUM_ITERATIONS, dtype = int)
        avg_rsd_per_interval_array = np.zeros(NUM_ITERATIONS, dtype = float)

        for i in range(0, NUM_ITERATIONS):

            if i % (NUM_ITERATIONS / 10) == 0:
                print("ITERATION %d" % i)
            # Generate a new synchronization string
            S = Synchronization_String(epsilon = epsilon_array[m], n = 25)
            pre_string = S.str

            # Now actually construct the string from the starting random string
            S.construct()
            post_string = S.str

            assert len(pre_string) == len(post_string), "ERROR: STRINGS DO NOT HAVE THE SAME LENGTH"

            num_different_char = 0
            for j in range(0, len(pre_string)):
                if pre_string[j] != post_string[j]:
                    num_different_char += 1
            num_different_char_array[i] = num_different_char
            num_resamples_array[i] = S.num_resamples
            S.calculate_codeword_rsd()
            avg_rsd_per_interval_array[i] = S.average_rsd
            #print(S.average_rsd)

        #print(avg_rsd_per_interval_array)
        avg_num_of_different_characters = np.average(num_different_char_array)
        avg_num_of_resamples = np.average(num_resamples_array)
        avg_avg_rsd = np.average(avg_rsd_per_interval_array)

        avg_num_diff_char_array_per_eps[m] = avg_num_of_different_characters
        avg_num_resamples_array_per_eps[m] = avg_num_of_resamples
        avg_rsd_array_per_eps[m] = avg_avg_rsd

    plt.plot(epsilon_array, avg_num_diff_char_array_per_eps, 'r-', linewidth = 1.5, markersize = 1.5)
    plt.title("Average number of different characters vs. epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Average number of different characters")
    plt.legend()
    plt.show()


    plt.plot(epsilon_array, avg_num_resamples_array_per_eps, 'g-', linewidth = 1.5, markersize = 1.5)
    plt.title("Average number of different resamples vs. epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Average number of resamples")
    plt.legend()
    plt.show()

    plt.plot(epsilon_array, avg_rsd_array_per_eps, 'b-', linewidth = 1.5, markersize = 1.5)
    plt.title("Average RSD of synchronization codewords vs. epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Average RSD")
    plt.ylim([0, 2])
    plt.legend()
    plt.show()


    #print("AVERAGE NUMBER OF DIFFERENT CHARACTERS = %0.2lf" % avg_num_of_different_characters)
    #print("AVERAGE NUMBER OF RESAMPLES = %0.2lf" % avg_num_of_resamples)
    #print("AVERAGE INTERVAL RSD = %0.2lf" % avg_avg_rsd)

    #I = InsDelChannel(delta = 0.25, ins_prob = 0.5)
    #S_prime = I.process_string(S.str)

    #print(S_prime)

    #S.decode(S_prime)



    #print(S.alphabet_size)
    #print(S.intervals_to_check)
    #S.verify_synchronization()

    #S.print_interval(interval = (0, 0, n))

    #str_rep = S.get_string_representation()
    #print(str_rep)

    #pause(True)

    

    #header_epsilon = ['x'] + ['{0:.2f}'.format(0.05 * e) for e in range(1, 15)]



    

    #DATA_PATH = CSV_PATH + "sync_string_data"
    #DATA_PATH = os.path.join(main_directory, "sync_string_data")
    #sync_string_dictionary = DATA_PATH + "\\sync_str_dict.csv"

    

    #with open(sync_string_dictionary, 'w', newline = '') as csvfile:
    #    writer = csv.writer(csvfile, delimiter = ',')
    #    writer.writerow(header_epsilon)
    #csvfile.close()

    #with open(sync_string_dictionary, newline = '') as csvfile:
    #    reader = csv.reader(csvfile, delimiter = ',')
    #    for row in reader:
    #        print(" ".join(row))

    #row_list = []
    #for n in range(10, 25, 5):
    #    row = [str(n)]
    #    for e in range(1, 15):

    #        print("---------- STRING LENGTH = %d, EPSILON = %0.2lf ----------" % (n, 0.05 * e))
    #        S = Synchronization_String(epsilon = 0.05 * e, n = n)
    #        S.verify_synchronization()

            #S.print_interval(interval = (0, 0, n))

    #        str_rep = S.get_string_representation()

            #string_representation, byte_representation, _ = S.get_string_representation()
    #        row.append(str_rep)

    #    row_list.append(row)

    #for row in row_list:
    #    print(row)

    #with open(sync_string_dictionary, 'a', newline = '') as csvfile:
    #    writer = csv.writer(csvfile, delimiter = ',')
    #    for row in row_list:
    #        writer.writerow(row)
    #csvfile.close()

    #for n in range(10, 256, 5):
        
    #for e in range(1, 20):

    #    epsilon = 0.05 * e
    #    print(epsilon)

    # Generate the synchronization string
    #S = Synchronization_String(epsilon = 0.5, n = 10)
    #S.print_string(print_char = False, print_byte = False)
    #S.verify_synchronization()

    #string_representation, byte_representation, _ = S.get_string_representation()

    #print(string_representation)
    #print(byte_representation)
    #S.index_alphabet.print_dictionary(print_char = True, print_byte = True)

    # Verify the synchronziation property holds

    # Transmit the synchronization string through the channel

    # Decode the synchronization string

    #I = InsDelChannel(delta = 0.25, ins_prob = 0.5)
    #S_prime = I.process_string(S.str)

    #print(S_prime)

    #S.decode(S_prime)

def load_sync_str(n, epsilon, directory):

    epsilon_str = "{:0.4f}".format(epsilon)
    file_name = os.path.join(directory, "sync_str_%d_%s" % (n, epsilon_str[2 : ]))

    row_list = []

    with open(file_name, newline = '') as csvfile:
        reader = csv.reader(csvfile, delimiter = ',')
        next(reader, None)
        
        for row in reader:
            row_list.append(row)
    csvfile.close()

    rand_index = rand.randint(a = 0, b = len(row_list)-1)
    str_sample = row_list[rand_index][1]
    string_repr_list = []

    i = 0
    while i < len(str_sample):

        if str_sample[i] == "(":
            temp = ""
        elif str_sample[i] == ")":
            string_repr_list.append(temp)
        else:
            temp += str_sample[i]
        i += 1
    
    S = Synchronization_String(epsilon = epsilon, n = n)
    A = a.Alphabet(size = int(epsilon ** -4))

    for i in range(0, n):
        S.str[i] = A.get_symbol_by_index(index = int(string_repr_list[i]))

    return S

if __name__ == '__main__':

    rand.seed(0x66023C)
    MAIN_DIRECTORY = os.path.join(sys.argv[1], "sync_string_data")

    S = load_sync_str(n = 30, epsilon = 0.5, directory = MAIN_DIRECTORY)
    #generate_string_repo()

    #S = Synchronization_String(epsilon = 0.5, n = 50)
    #S.print_string()

    #pause(True)
    