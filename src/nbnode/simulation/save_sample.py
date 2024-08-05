import os


def save_sample(df, save_dir, sample_name, save_type, verbose):
    if save_type == "csv":
        current_filepath = os.path.join(save_dir, sample_name + ".csv")
        df.to_csv(current_filepath, index=False)
    elif save_type == "feather":
        current_filepath = os.path.join(save_dir, sample_name + ".feather")
        df.to_feather(current_filepath)
    else:
        raise ValueError(f"save_type {save_type} not implemented")

    if verbose:
        print(f"Saved {current_filepath}")
