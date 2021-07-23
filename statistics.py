
import math as math

BITS_PER_SYMBOL = 8

if __name__ == '__main__':

    # Default RS Alphabet Size
    q = 8       

    for p in range(4, 21):
        
        n = 2 ** p
        epsilon_critical = (n ** (-0.25))
        epsilon_equivalent_rates = 2 ** ((q * -0.25) * (1 - 1/p))

        print("n = %d: EC = %0.5lf, ER = %0.5lf" % (n, epsilon_critical, epsilon_equivalent_rates), end = "")

        if epsilon_equivalent_rates < epsilon_critical:
            print(" UNIQUE INDEXING IS MORE EFFICIENT")
        else:
            print(" SYNCHRONIZATION STRINGS IS MORE EFFICIENT")        
        print()



    #p_list = [x for x in range(4, 21)]
    #q_list = [x for x in range(8, 21)]

    #n = 32
    #p = math.log2(n)

    #for q in q_list:

    #    epsilon = 2 ** ((q * -0.25) * (1 - 1/p))
    #    print(epsilon)

    #index_bits_per_symbol = math.log2(n)
    #total_num_data_bits = BITS_PER_SYMBOL * n
    #total_num_index_bits = index_bits_per_symbol * n

    #total_num_bits = total_num_data_bits + total_num_index_bits

    #useful_bandwidth = total_num_data_bits / total_num_bits

    #epsilon_critical = (n ** -0.25)



