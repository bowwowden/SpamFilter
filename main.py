import argparse
import math
import os, re
import collections
import random


def preprocess(line):

    # get rid of the stuff at the end of the line (spaces, tabs, new line, etc.)
    line = line.rstrip()
    # lower case
    line = line.lower()
    # remove everything except characters and white space
    line = re.sub("[^a-z ]", '', line)
    # tokenized, not done "properly" but sufficient for now
    tokens = line.split(' ')

    # adding $ before and after each token because we are working with bigrams
    tokens = ['$' + token + '$' for token in tokens]

    return tokens


def create_model(path):
    # This is just some Python magic ...
    # unigrams[key] will return 0 if the key doesn't exist
    unigrams = collections.defaultdict(int)
    # and then you have to figure out what bigrams will return
    bigrams = collections.defaultdict(lambda: collections.defaultdict(int))

    f = open(path, 'r')


    for l in f.readlines():
        tokens = preprocess(l)
        if len(tokens) == 0:
            continue

        for token in tokens:

            for c in token:
                # including $
                if not c in unigrams:
                    unigrams[c] = 1
                elif c in unigrams:
                    unigrams[c] += 1

            # count bigrams
            left = token[0]
            for right in token[1:]:
                if not left in bigrams:
                    bigrams[left] = {}
                if not right in bigrams[left]:
                    bigrams[left][right] = 1
                else:
                    bigrams[left][right] += 1

                left = right

    # Smooth the bigrams for what doesn't appear
    for c in unigrams:

        # loop
        for k in unigrams:
            # if bigram never seen, equal 1
            if not k in bigrams[c]:
                bigrams[c][k] = 1
            # if bigram seen before, add 1
            else:
                bigrams[c][k] += 1

    # Create the probability models
    for c in unigrams:

        for k in unigrams:
            number_of_xy = bigrams[c][k]
            number_of_x = unigrams[c]

            # bigram probability
            bigrams[c][k] = (number_of_xy + 1) / (number_of_x + len(unigrams))

    # return the actual model: bigram (smoothed log) probabilities and unigram counts (the latter to smooth
    # unseen bigrams in predict(...)
    return [bigrams, unigrams]


def predict(file, model_en, model_es):

    bigrams_en = model_en[0]
    unigrams_en = model_en[1]

    bigrams_es = model_es[0]
    unigrams_es = model_es[1]

    spam = 0
    non_spam = 0

    # going through the file, with each of the probabilities...
    f = open(file, 'r')
    for l in f.readlines():
        tokens = preprocess(l)
        if len(tokens) == 0:
            continue
        for token in tokens:

            # count bigrams
            left = token[0]
            for right in token:
                # left is one before, right is next
                # adding sequences of log probabilites, doing the negative because
                # multiplying negative logs is impossible
                spam += bigrams_es[left][right]
                non_spam += bigrams_en[left][right]
                left = right

    if non_spam >= spam:
        print("non_spam probability: " + str((non_spam)))
        print("spam probability: " + str(spam))
        return 'Non-Spam'
    else:
        print("non_spam probability: " + str(non_spam))
        print("spam probability: " + str(spam))
        return 'Spam'



def main(non_spam, spam, folder_te):

    # STEP 1: create a model for non spam
    model_non_spam = create_model(non_spam)

    # STEP 2: create a model for spam
    model_spam = create_model(spam)

    # STEP 3: loop through all the files in folder_te and print prediction
    folder = os.path.join(folder_te, "non-spam")
    print("Prediction for Non-Spam documents in test:")
    for f in os.listdir(folder):
        f_path = os.path.join(folder, f)
        print(f"{f}\t{predict(f_path, model_non_spam, model_spam)}")

    folder = os.path.join(folder_te, "spam")
    print("\nPrediction for Spam documents in test:")

    for f in os.listdir(folder):
        f_path = os.path.join(folder, f)
        print(f"{f}\t{predict(f_path, model_non_spam, model_spam)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("PATH_NON_SPAM",
                        help="Path to file with Non-spam training files")
    parser.add_argument("PATH_SPAM",
                        help="Path to file with Spam training files")
    parser.add_argument("PATH_TEST",
                        help="Path to folder with test files")
    args = parser.parse_args()

    main(args.PATH_NON_SPAM, args.PATH_SPAM, args.PATH_TEST)
