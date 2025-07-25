import numpy as np
from learn_decoder import *
from pset3_functions import cosine_similarity
from sklearn.model_selection import KFold


def train_decoder(data, vectors):
    M = learn_decoder(data, vectors)
    return M


def test_decoder(fmri_data, vector_sentences, decoder):
    """
    Returns:
        dict with:
        - 'mean_accuracy': overall mean
        - 'sentence_accuracies': list of accuracies per sentence
        - 'ranks': list of ranks per sentence
    """
    pred_sentences = fmri_data @ decoder
    rank_scores = []
    ranks = []

    for i, pred_vector in enumerate(pred_sentences):
        similarities = []
        for candidate_vector in vector_sentences:
            sim = cosine_similarity(pred_vector, candidate_vector)
            similarities.append(sim)

        rank = get_rank(similarities, i)
        rank_accuracy = 1 - (rank - 1) / (len(vector_sentences) - 1)

        rank_scores.append(rank_accuracy)
        ranks.append(rank)

    return {
        "mean_accuracy": np.mean(rank_scores),
        "sentence_accuracies": rank_scores,
        "ranks": ranks,
    }


def get_rank(similarities, correct_idx):
    """Get rank of item"""
    sorted_indices = np.argsort(similarities)[::-1]
    rank = np.where(sorted_indices == correct_idx)[0][0] + 1
    return rank


def cross_validate_decoder(fmri_data, semantic_vectors, n_folds=10, random_state=42):
    """
    Cross-validate decoder performance

    Returns:
        tuple: (mean_accuracy, std_accuracy, fold_results)
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(fmri_data)):
        # Handle both DataFrame and numpy array
        if hasattr(fmri_data, "iloc"):
            train_fmri = fmri_data.iloc[train_idx].values
            test_fmri = fmri_data.iloc[test_idx].values
        else:
            train_fmri = fmri_data[train_idx]
            test_fmri = fmri_data[test_idx]

        train_vectors = semantic_vectors[train_idx]
        test_vectors = semantic_vectors[test_idx]

        # Train decoder
        decoder = train_decoder(train_fmri, train_vectors)

        # Test decoder
        results = test_decoder(test_fmri, test_vectors, decoder)

        fold_results.append(
            {
                "fold": fold + 1,
                "accuracy": results["mean_accuracy"],
                "test_indices": test_idx,
            }
        )

    accuracies = [r["accuracy"] for r in fold_results]
    return np.mean(accuracies), np.std(accuracies), fold_results
