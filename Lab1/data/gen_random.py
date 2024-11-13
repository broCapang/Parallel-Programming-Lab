import random

# Generate a list of 50 random float values between 1.0 and 100.1
random_floats = [round(random.uniform(1.0, 100.1), 1) for _ in range(100)]

for i in random_floats:
    print(i)
