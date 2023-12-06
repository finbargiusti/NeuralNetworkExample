# here we convert the CSV file given into acceptable format for the model

# read from stdin

import sys

# read from stdin

for line in sys.stdin:
    line = line.strip()
    words = line.split(',')

    # since this is a classification problem, we need to convert the letter into an array
    # of 26 elements, where the index of the capital letter is 1 and the rest are 0

    letter = words[0]
    letter_array = [0] * 26
    letter_array[ord(letter) - ord('A')] = 1

    # now we need to convert the rest of the data into floats

    data = [float(x) for x in words[1:]]

    # now we need to print the data in the format that the model expects

    row = data + letter_array

    row = [str(x) for x in row]

    print(' '.join(row))



