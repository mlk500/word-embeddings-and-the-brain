from learn_decoder import *
import matplotlib.pyplot as plt


def eval_decoder_cv(data, vectors, concepts, k=18):
    test_num = len(data) // k
    avg_ranks = []
    concept_rankings = []
    for i in range(k):
        start, end = int(test_num * i), int(test_num * (i + 1))
        x_train, y_train, x_test, y_test = split_data(data, vectors, start, end)

        M = learn_decoder(x_train, y_train)
        pred_vectors = x_test @ M
        ranks = 0
        test_indices = list(range(start, end))
        for i in range(len(pred_vectors)):
            true_index = test_indices[i]
            r = calc_rank(pred_vectors[i], true_index, vectors)
            ranks += r
            concept_name = concepts[true_index]
            concept_rankings.append((concept_name, r))

        avg_ranks.append((ranks / test_num))
    return avg_ranks, concept_rankings


def split_data(data, vectors, start, end):
    x_train = [x for i, x in enumerate(data) if i < start or i >= end]
    y_train = [y for i, y in enumerate(vectors) if i < start or i >= end]
    x_test = data[start:end]
    y_test = vectors[start:end]

    return np.array(x_train), np.array(y_train), x_test, y_test


def calc_rank(y_hat, true_index, vectors):
    sims = {}

    for idx, v in enumerate(vectors):
        sims[idx] = cosine_similarity(y_hat, v)
    sorted_sims = sorted(sims.items(), key=lambda item: -item[1])
    rank = 1
    for rank, (idx, _) in enumerate(sorted_sims, start=1):
        if idx == true_index:
            return rank


def cosine_similarity(y_hat, v):
    """
    Calculate cosine similarity & handle zero vectors
    """
    norm_y_hat = np.linalg.norm(y_hat)
    norm_v = np.linalg.norm(v)
    if norm_y_hat == 0 or norm_v == 0:
        return 0.0
    return np.dot(y_hat, v) / (norm_y_hat * norm_v)


# def cosine_similarity(y_hat, v):
#     return np.dot(y_hat, v) / (np.linalg.norm(y_hat) * np.linalg.norm(v))

# def plot_ranks(avg_ranks, model_name="Glove"):
#     x_values = list(range(1, len(avg_ranks) + 1))
#     plt.figure(figsize=(10, 6))
#     plt.plot(x_values, avg_ranks, marker='o')
#     plt.xlabel('Fold Number')
#     plt.ylabel('Average Rank')
#     plt.title('Performance per Fold {} Model'.format(model_name))
#     plt.xticks(x_values)
#     plt.grid(True)
#     plt.show()


def plot_ranks(avg_ranks_1, avg_ranks_2, model_name_1="Glove", model_name_2="Word2Vec"):
    x_values = list(range(1, len(avg_ranks_1) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, avg_ranks_1, marker="o", label=model_name_1)
    plt.plot(x_values, avg_ranks_2, marker="s", label=model_name_2, linestyle="--")

    plt.xlabel("Fold Number")
    plt.ylabel("Average Rank")
    plt.title("Performance per Fold")
    plt.xticks(x_values)
    plt.grid(True)
    plt.legend()
    plt.show()
