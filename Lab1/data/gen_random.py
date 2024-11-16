import random

random_floats = [round(random.uniform(1.0, 100.1), 1) for _ in range(100000000)]

# Specify the file name
file_name = "data100000000.txt"

# Write the random floats to a text file
with open(file_name, "w") as file:
    for value in random_floats:
        file.write(f"{value}\n")

print(f"Random floats written to {file_name}")

