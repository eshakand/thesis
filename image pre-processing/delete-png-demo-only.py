import fnmatch
import os


def main():
    matches = []
    for root, dirnames, filenames in os.walk(os.getenv('dataset_path')):
        for filename in fnmatch.filter(filenames, '*.png'):
            os.remove(os.path.join(root, filename))


main()