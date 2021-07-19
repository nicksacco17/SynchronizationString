import alphabet as a
import numpy as np

def symbol_rsd(str1, str2):

    min_len = min(len(str1), len(str2))

    rsd = 0.0
    for k in range(1, min_len + 1):
        substring1 = str1[len(str1) - k :]
        substring2 = str2[len(str2) - k :]

        rsd_k = edit_distance(substring1, substring2) / (2*k)

        print("SUBSTRING 1: ", end = "")
        for s in substring1:
            s.print_symbol(print_char = True, print_byte = True, new_line = False)
        print()
        print("SUBSTRING 2: ", end = "")
        for s in substring2:
            s.print_symbol(print_char = True, print_byte = True, new_line = False)
        print()
        print("RSD(str1, str2) = %0.3lf\n" % rsd_k)
        #print("---> SUBSTRING 1 = %s, SUBSTRING 2 = %s, RSD = %0.2lf" % (substring1, substring2, rsd_k))

        if rsd_k > rsd:
            rsd = rsd_k
        
    return rsd


def relative_suffix_distance(str1, str2):

    min_len = min(len(str1), len(str2))

    rsd = 0.0
    for k in range(1, min_len + 1):

        substring1 = str1[len(str1) - k :]
        substring2 = str2[len(str2) - k :]

        rsd_k = edit_distance(substring1, substring2) / (2*k)

        print("---> SUBSTRING 1 = %s, SUBSTRING 2 = %s, RSD = %0.2lf" % (substring1, substring2, rsd_k))

        #new_max = edit_distance(str1[len(str1)-k : len(str1)], str2[len(str2)-k : len(str2)]) / (2 * k)
        #print(str1[len(str1)-k : len(str1)])
        #print(str2[len(str2)-k : len(str2)])
        #print("EDIT DISTANCE = %d" % edit_distance(str1[len(str1)-k : len(str1)], str2[len(str2)-k : len(str2)]))
        if rsd_k > rsd:
            rsd = rsd_k
            #print("k = %d, MAX substring 1 = S[%d:%d]" % (k, len(str1)-k, len(str1)))
            #print("k = %d, MAX substring 2 = S'[%d:%d]" % (k, len(str2)-k, len(str2)))
    return rsd

def symbol_ed(str1, str2):

    n = len(str1) + 1
    m = len(str2) + 1

    str_mat = np.zeros((m, n), dtype = int)
    str_mat[0] = [i for i in range(0, n)]
    str_mat[:,0] = [i for i in range(0, m)]
    cost = 0

    # For each character in the first string
    for i in range(0, len(str1)):

        # For each character in the second string
        for j in range(0, len(str2)):

            # If the characters are the same, there is no net cost in the string transformation str1 --> str2
            if str1[i].get_id() == str2[j].get_id():
                cost = 0
            # Else if the characters are different, a substitution is required in the string transformation.
            # A substitution is a deletion followed by an insertion.
            else: 
                cost = 2
            
            # Populate the next entry in the matching matrix
            str_mat[j + 1, i + 1] = min(str_mat[j, i + 1] + 1, str_mat[j + 1, i] + 1, str_mat[j, i] + cost)
           
    return str_mat[-1, -1]  


# Wagner-Fischer dynamic programming algorithm
# Edit distance = Levenshtein distance
# Many definitions provide the substitutions, we will consider a substitution as a deletion followed by insertion
# which accumulates a cost of 2.
def edit_distance(str1, str2):

    n = len(str1) + 1
    m = len(str2) + 1

    str_mat = np.zeros((m, n), dtype = int)
    str_mat[0] = [i for i in range(0, n)]
    str_mat[:,0] = [i for i in range(0, m)]
    cost = 0
    
    # For each character in the first string
    for i in range(0, len(str1)):

        # For each character in the second string
        for j in range(0, len(str2)):

            # If the characters are the same, there is no net cost in the string transformation str1 --> str2
            if str1[i] == str2[j]:
                cost = 0
            # Else if the characters are different, a substitution is required in the string transformation.
            # A substitution is a deletion followed by an insertion.
            else: 
                cost = 2
            
            # Populate the next entry in the matching matrix
            str_mat[j + 1, i + 1] = min(str_mat[j, i + 1] + 1, str_mat[j + 1, i] + 1, str_mat[j, i] + cost)
           
    return str_mat[-1, -1]    

if __name__ == '__main__':

    source_alphabet = a.Alphabet(size = a.MAX_PRINT_SIZE)
    str1, str2 = input("Enter two strings: ").split()
    #ed = edit_distance(str1, str2)
    #rsd = relative_suffix_distance(str1, str2)

    ed_sym = symbol_ed(source_alphabet.convert_string_to_symbols(str1), source_alphabet.convert_string_to_symbols(str2))
    rsd_sym = symbol_rsd(source_alphabet.convert_string_to_symbols(str1), source_alphabet.convert_string_to_symbols(str2))
    

    print("STR1: %s, STR2: %s" % (str1, str2))
    print("EDIT DISTANCE = %d, RELATIVE SUFFIX DISTANCE = %0.2lf" % (ed_sym, rsd_sym))
    