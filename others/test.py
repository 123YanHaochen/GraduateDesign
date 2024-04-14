import random

def select_random_numbers(numbers, m):
    if m > len(numbers):
        print("Error: m should be less than or equal to n")
        return []

    random_numbers = random.sample(numbers, m)
    return random_numbers

# Example usage
n = 19   # Total number of elements
m = 5    # Number of elements to select

numbers = list(range(1, n+1))  # Create a list of numbers from 1 to n
random_selection = select_random_numbers(numbers, m)
print(f"Randomly selected {m} numbers: {random_selection}")
