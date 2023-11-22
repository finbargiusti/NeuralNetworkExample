import sys
import random
import math

if (len(sys.argv) < 2):
    print("Usage: python generate_examples.py <number of examples>")
    sys.exit()

n = int(sys.argv[1])

# random number between -1.0 and 1.0

randvalue = lambda : (random.random() * 2) - 1

for i in range(n):
    x1, x2, x3, x4 = randvalue(), randvalue(), randvalue(), randvalue()

    y = math.sin(x1 - x2 + x3 - x4)
    
    print(x1, x2, x3, x4, y)
