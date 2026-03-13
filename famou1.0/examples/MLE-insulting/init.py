# EVOLVE-BLOCK-START
import os
import numpy as np
import pandas as pd
import warnings
import random
import gc

# (All sklearn and torch imports removed)

# Suppress all warnings
warnings.filterwarnings("ignore")


# --- Configuration ---
class CFG:
    # Seed for reproducibility of random guesses
    SEED = 42


# --- Reproducibility ---
def set_seed(seed):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


set_seed(CFG.SEED)


# --- (All model-related functions removed) ---


def main(train_df, test_df, sample_submission_df, save_path):
    """Main function for a minimal 'random guess' submission."""
    print("Starting minimal pipeline (Random Guess Baseline)...")

    # 1. Get the number of test samples
    num_test_samples = len(test_df)
    print(f"Generating {num_test_samples} random predictions...")

    # 2. Generate random predictions
    # np.random.rand() generates random numbers in the [0, 1] range,
    # which satisfies the submission requirement for probabilities.
    test_preds = np.random.rand(num_test_samples)

    # 3. Create Submission
    print("\nCreating submission file...")

    # Use the original test_df to preserve 'Date' and 'Comment'
    submission_df = test_df.copy()
    submission_df["Insult"] = test_preds

    # Ensure correct column order as per submission specs
    final_submission_df = submission_df[["Insult", "Date", "Comment"]]

    final_submission_path = os.path.join(save_path, "submission.csv")
    final_submission_df.to_csv(final_submission_path, index=False, header=True)
    print(
        f"\nFinal (random) submission file created at: {final_submission_path}"
    )

    # Clean up
    del submission_df, final_submission_df, test_preds
    gc.collect()


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # fixed file paths fixed
    DATA_ROOT = "./detecting-insults-in-social-commentary/prepared/public/"
    # A placeholder for saving outputs.
    SAVE_PATH = "/tmp/mle_save_dir/detecting-insults-in-social-commentary/"

    # Create save directory if it doesn't exist
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Load the datasets
    train_df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
    sample_submission_df = pd.read_csv(
        os.path.join(DATA_ROOT, "sample_submission_null.csv")
    )

    # Call the main function with the loaded data
    main(train_df, test_df, sample_submission_df, SAVE_PATH)