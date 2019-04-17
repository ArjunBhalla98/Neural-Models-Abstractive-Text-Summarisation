from keras.layers.embeddings import Embedding
import pickle
import csv
import string
import sys


def dataset_to_index(datasets, n_tokens, outfile_inv, outfile_word, outfile_lst):
    """
    Takes in a dataset in the form of "all the news" and generates a word index
    and inverse index, then pickles and saves it. It will also generate headline 
    / article pairs in the form of integer sequences. N.B: Does NOT stem due to 
    semantics of different word forms being important

    dataset: list of csv's (ATN form)
    n_tokens: int on how many word features / tokens to use
    """
    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    inv_idx = {}  # idx to word
    word_to_idx = {}
    hd_article = (
        []
    )  # list of dicts with keys (headline, article, id), where each is a list of ints except id which is an int
    idx = 0
    ds_num = 1  # tracking progress, nothing else
    for dataset in datasets:
        data = csv.DictReader(
            open(dataset, newline=""),
            fieldnames=[
                "a",
                "id",
                "title",
                "publication",
                "author",
                "d",
                "y",
                "m",
                "u",
                "content",
            ],
        )
        print("Dataset: " + str(ds_num), end="\r")
        for line in data:
            if line["id"] == "id":
                continue
            ex_id = line["id"]

            if line["title"].find("- The") != -1:
                title = line["title"][: line["title"].index("-")].lower().strip()
            else:
                title = line["title"].lower().strip()

            content = line["content"].lower().strip()

            title = title.translate(str.maketrans("", "", string.punctuation))
            content = content.translate(str.maketrans("", "", string.punctuation))

            # insert into index if not done already
            for word in title.split():
                if word not in word_to_idx.keys() and idx < n_tokens:
                    word_to_idx[word] = idx
                    inv_idx[idx] = word
                    idx += 1

            for word in content.split():
                if word not in word_to_idx.keys() and idx < n_tokens:
                    word_to_idx[word] = idx
                    inv_idx[idx] = word
                    idx += 1

            # build integer sequences and append to list
            title_sub = []
            for word in title.split():
                if word in word_to_idx:
                    title_sub.append(word_to_idx[word])
                else:
                    title_sub.append(-1)

            content_sub = []
            for word in content.split():
                if word in word_to_idx:
                    content_sub.append(word_to_idx[word])
                else:
                    content_sub.append(-1)
            hd_article.append((title_sub, content_sub, ex_id))
        ds_num += 1

    pickle.dump(inv_idx, open(outfile_inv, "wb"))

    pickle.dump(word_to_idx, open(outfile_word, "wb"))

    pickle.dump(hd_article, open(outfile_lst, "wb"))


def load_atn():
    f1 = open("all-the-news/articles1.csv", "r")
    f2 = open("all-the-news/articles2.csv", "r")
    f3 = open("all-the-news/articles3.csv", "r")
    a = [
        "\n".join(f1.readlines()),
        "\n".join(f2.readlines()),
        "\n".join(f3.readlines()),
    ]
    f1.close()
    f2.close()
    f3.close()
    return a


def split_bigboi(train, test, valid):
    """
    literally using this to split the big file into 3 or so parts so we can use
    it (data_int_stream). Customizeable parameters for train/test/valid split.
    """

    with open("data_int_stream.pickle", "rb") as f:
        data = pickle.load(f)
        tot_len = len(data)
        train_size = int(tot_len * train)
        test_size = int(test * train)
        valid_size = int(valid * train)

        train_set = data[:train_size]
        test_set = data[train_size : train_size + test_size]
        valid_set = data[train_size + test_size :]

        # split train_set into 3 because github file size restrictions
        chunk_size = int(train_size / 3)

        for i in range(3):
            with open("train_" + str(i) + ".pickle", "wb") as f:
                pickle.dump(train_set[chunk_size * i : chunk_size * (i + 1)], f)

        with open("test.pickle", "wb") as f:
            pickle.dump(test_set, f)

        with open("validation.pickle", "wb") as f:
            pickle.dump(valid_set, f)


if __name__ == "__main__":
    # dataset_list = [
    #     "all-the-news/articles1.csv",
    #     "all-the-news/articles2.csv",
    #     "all-the-news/articles3.csv",
    # ]
    # dataset_to_index(
    #     dataset_list,
    #     20000,
    #     "inv_idx.pickle",
    #     "reg_idx.pickle",
    #     "data_int_stream.pickle",
    # )
    split_bigboi(0.8, 0.1, 0.1)
    print("Run completed.")
