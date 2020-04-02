import string
import glob
from tqdm import tqdm

def ProcessLargeTextFile():
    count = 0
    with open("preprocessed.txt", "w") as w:
        for path in tqdm(glob.glob('split/*')):
            with open(path, 'r') as r:
                tokens = []
                for line in r:
                    l = line[21:].lower().translate(str.maketrans('', '', string.punctuation)).split()
                    for token in l:
                        if (len(token) >= 2) and (token[0] not in string.digits):
                            tokens.append(token)
                    
                    tokens.append('\n')
                w.writelines(' '.join(tokens))

if __name__ == "__main__":
    ProcessLargeTextFile()