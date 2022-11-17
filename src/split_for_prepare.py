"""
Splits training_and_test.txt into two separate files. One for training, and one for testing.
Both contain data for a command injection attack.
"""

RAW_DATA_FILE = "../data/training_and_test.txt"

def pre_prepare():
    with open(RAW_DATA_FILE, "r") as f:
        lines = f.readlines()
    test_data = []
    training_data = []
    for i in range(len(lines)):
        if i % 5 == 0:
            test_data.append(lines[i])
        else:
            training_data.append(lines[i])
    with open("../data/train/is_command_injection.txt", "w") as f:
        f.write("".join(training_data))
    with open("../data/test/is_command_injection.txt", "w") as f:
        f.write("".join(test_data))

if __name__ == "__main__":
    print("This function is directly called from prepare.py; feel free to run that directly instead!")
    pre_prepare()