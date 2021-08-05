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

        self.alphabet_size = int(self.epsilon ** -4)
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

            valid_interval, EDIT_DISTANCE, MINIMUM = self.check_interval(interval = interval[1])
            #print("ED = %lf, MIN = %lf" % (EDIT_DISTANCE, MINIMUM))
            if not valid_interval:

                # Regenerate interval
                for m in range(0, interval[1][2] - interval[1][0]):
                    self.str[interval[1][0] + m] = self.index_alphabet.get_random_symbol_from_alphabet()

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

def load_sync_str(n, epsilon, directory):

    epsilon_str = "{:0.5f}".format(epsilon)
    file_name = os.path.join(directory, "sync_str_n_%d_epsilon_%s" % (n, epsilon_str[2 : ]))

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

    S = Synchronization_String(epsilon = 0.5, n = 6)
    S.verify_synchronization()

    S.print_string(print_char = False, print_byte = False)

    #rand.seed(0x66023C)
    #MAIN_DIRECTORY = os.path.join(sys.argv[1], "sync_string_data")

    #S = load_sync_str(n = 30, epsilon = 0.5, directory = MAIN_DIRECTORY)
    #generate_string_repo()

    #S = Synchronization_String(epsilon = 0.5, n = 50)
    #S.print_string()

    #pause(True)
    