"""
Splits training_and_test.txt into two separate files. One for training, and one for testing.
Both contain data for a command injection attack.

Additionally generates a file containing non-CI data and adds "noise" to CI attacks. This way,
the net hopefully focuses more on finding what makes a CI a CI, rather than looking at the random
characters of a "non-CI" and using that instead.
"""
import random

RAW_DATA_FILE = "data/training_and_test.txt"

random.seed(62)  # Constant seed for consistent data generation

# Random characters. Biased towards letters and numbers since humans tend to use those the most.
RANDOM_CHARS_STR = "abcdefghijklmnopqrstvwuxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"*8 + "`-=~!@#$%^&*()_+[]\\}{|;':\",./<>?"
RANDOM_CHARS = [letter for letter in RANDOM_CHARS_STR]

def pre_prepare():
    """Pre-Prepare.

    Generates the .txt files in data/test and data/train
    """
    with open(RAW_DATA_FILE, "r") as f:
        lines = f.readlines()
    test_data = []
    training_data = []
    for i in range(len(lines)):
        if i % 5 == 0:
            test_data.append(noiseify_line(lines[i]))
        else:
            training_data.append(noiseify_line(lines[i]))
    with open("data/train/is_command_injection.txt", "w") as f:
        f.write("".join(training_data))
    with open("data/test/is_command_injection.txt", "w") as f:
        f.write("".join(test_data))


    training_real = [gen_legit_line() for _ in range(8000)]
    test_real = [gen_legit_line() for _ in range(1500)]
    with open("data/train/not_command_injection.txt", "w") as f:
        f.write("\n".join(training_real))
    with open("data/test/not_command_injection.txt", "w") as f:
        f.write("\n".join(test_real))

def gen_legit_line(max_length = 100) -> str:
    """Generate Non-Command Injection Line.

    Args:
        max_length (int, optional): Maximum length of non-CI line. Defaults to 100.

    Returns:
        str: Random line that acts as a line without command injection
    """
    return gen_random_chars(random.randint(1, 100))


def noiseify_line(line: str) -> str:
    """Potentially Add Noise to Line.

    Args:
        line (str): Line to potentitally add noise to

    Returns:
        str: Line with noise in it to make it less distinguishable from a "legitimate" line
    """
    if random.randint(0, 10) < 9:  # High but not guaranteed chance to noisefiy line
        noise_left = random.choice([True, False])
        noise_right = random.choice([True, False])
        if noise_left:
            line = gen_random_chars(random.randint(1, 15)) + line
        if noise_right:
            line = line + gen_random_chars(random.randint(1, 15))
    return line

def gen_random_chars(num_chars: int) -> str:
    """Generate Random Characters String.

    Args:
        num_chars (int): Number of characters in the random string

    Returns:
        str: A string of length num_chars consisting of random characters from RANDOM_CHARS
    """
    ret = ""
    for _ in range(num_chars):
        ret += random.choice(RANDOM_CHARS)
    return ret

if __name__ == "__main__":
    print("This function is directly called from prepare.py; feel free to run that directly instead!")
    pre_prepare()