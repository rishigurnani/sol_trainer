import re
import pickle
import pandas as pd
from os.path import exists
from math import nan


def parse_to_error_df(
    train_out_path,
    group_map_path,
    save_path,
    n_fold=None,
    split_test_on=None,
):
    """
    Create a .csv file that summarizes the error metrics of the models logged
    in train_out_path.

    Keyword arguments:
        train_out_path (str, list): The path(s) to load the training output from
        group_map_path (str): The path to load the group map from
        save_path (str): The path to save the model metrics to
        n_fold (int, None): The number of folds in each model. Usually, this
            can be left as None. However this value will need to be filled
            in those cases where the number of folds cannot be inferred from
            the path(s) in train_out_path
        split_test_on (str, None): A string that denotes where the test data
            metrics begin. This line should be filled in if train_out_path
            contains metrics on unseen data
    """
    # ###################################
    # error handle the argparse arguments
    # ###################################
    if exists(save_path):
        raise ValueError("The save_path entered already exists.")
    # ###############
    # Massage inputs
    # ###############
    # massage train_out_path into a list
    if isinstance(train_out_path, str):
        train_out_path = [train_out_path]
    # #################
    # Define functions
    # #################
    def group_chunks(total_chunk):
        try:
            return [
                total_chunk.split(f"Working on group {group}")[1]
                for group in group_map.keys()
            ]
        except IndexError:
            return [
                total_chunk.split(f"Working on property {group}")[1]
                for group in group_map.keys()
            ]

    def ensemble_chunk(group_chunk):
        """
        Return the chunk of text that correspond to the training of the ensemble
        """
        split_possibilities = [
            re.compile("Fold 0: training"),
            re.compile("Optimal hyperparameters"),
            re.compile("Length of train"),
        ]
        for split_str in split_possibilities:
            try:
                result = re.split(split_str, group_chunk)[1]
                return result
            except IndexError:
                pass
        end = min(1000, len(group_chunk) - 1)
        print(group_chunk[0:end])
        raise ValueError

    def test_chunk(ensemble_chunk):
        pattern_start = re.compile(split_test_on)
        pattern_end = re.compile("Fold 0: training")
        result = re.split(pattern_start, ensemble_chunk)[1]
        result = re.split(pattern_end, result)[0]
        return result

    def split_and_add(chunk, prefix):
        """
        Split chunk on prefix and then add prefix back to each
        split
        """
        return [prefix + sub for sub in chunk.split(prefix)]

    def submodel_chunks(ensemble_chunk):
        """
        Return a list of chunks. Each chunk corresponds to one
        submodel.

        Keyword arguments
            ensemble_chunk (str): The output of ensemble_chunk()
        """
        return split_and_add(ensemble_chunk, "Epoch 0")[1:]

    def epoch_chunk(submodel_chunk):
        """
        Return a list of chunks. Each chunk corresponds to one epoch
        in submodel_chunk.

        Keyword arguments
            submodel_chunk (str): One element of the output of submodel_chunks()
        """
        return split_and_add(submodel_chunk, "Epoch")[1:]

    def epoch_no_grad_chunk(epoch_chunks):
        """
        Return a list of chunks with gradient information removed.

        Keyword arguments
            epoch_chunks (list): A list of epoch chunks
        """
        return [chunk.split("\n\n..Ave_grads")[0] for chunk in epoch_chunks]

    def best_epoch_chunks(epoch_chunks):
        """
        Filter a list of chunks to only retain those chunks that correspond to
        a model checkpoint.

        Keyword arguments
            epoch_chunks (list): A list of epoch chunks
        """
        pattern = "Best model saved"
        return [chunk.split(pattern)[0] for chunk in epoch_chunks if pattern in chunk]

    def find_all_floats(string):

        return re.findall("-?\d+\.\d+", string)

    def get_float(string, property, metric):
        """
        Return a numerical metric

        Keyword arguments
            string (str): The string to parse
            property (str): The target property corresponding to the desired
                metric
            metric (str): The name of the numerical value that is desired
        """
        try:
            result = string.split(f"{property} orig. scale val {metric}")[1].strip()
        except:
            result = string.split(f"val {metric}")[1].strip()
        result = result.split(" [")[0]
        result = result.split("\n")[0]
        all_floats = find_all_floats(result)

        if len(all_floats) == 1:
            return float(all_floats[0])
        elif (len(all_floats) == 0) and ("nan" in result):
            return nan
        else:
            print(string)
            print(result)
            print(all_floats)
            raise ValueError

    def get_nfold(chunk):
        split_chunk = chunk.split(f"fold ")
        if len(split_chunk) > 1:
            fold = split_chunk[1].strip()  # remove chance of "None\n"
            if not re.search(
                "[a-zA-Z]", fold
            ):  # if "fold" only has numbers, convert it to an int
                return int(fold)
            else:
                return None
        else:
            return None

    def process_r2(val):
        if val > 1:
            return nan
        else:
            return val

    # load the training output
    text = ""
    text_lines = []
    for path_ in train_out_path:
        with open(path_, "r") as f:
            text += f.read()
        with open(path_, "r") as f:
            text_lines += f.readlines()

    # compute the number of folds
    fold_n = [get_nfold(line) for line in text_lines]
    fold_n = [x for x in fold_n if x != None]
    try:
        fold_n = max(fold_n) + 1  # + 1 since we are zero-indexed
        n_fold = fold_n  # overwrite whatever n_fold value was passed in
    except ValueError:
        # The above failed, which means that n_fold cannot be inferred
        # from "train_out_path". So let us try and use a passed in value
        # instead
        if n_fold != None:
            pass
        else:
            raise (
                ValueError,
                "n_fold could not be computed from train_out_path nor was a valid value passed in as a keyword argument.",
            )

    # load the group map
    with open(group_map_path, "rb") as f:
        group_map = pickle.load(f)

    # initialize containers
    props = []
    groups = []  # should be same length as prop
    cv_rmse = [[] for _ in range(n_fold)]
    cv_r2 = [[] for _ in range(n_fold)]

    test_rmse = []
    test_r2 = []

    GroupChunks = group_chunks(text)  # one chunk per group. Each group
    # can have either one or multiple tasks.
    for _group, GroupChunk in zip(list(group_map.keys()), GroupChunks):
        EnsembleChunk = ensemble_chunk(GroupChunk)
        SubmodelChunks = submodel_chunks(EnsembleChunk)  # one chunk per submodel
        if split_test_on:
            TestSetChunk = test_chunk(EnsembleChunk)
        for n_submodel, SubmodelChunk in zip(list(range(n_fold)), SubmodelChunks):
            EpochChunks = epoch_chunk(SubmodelChunk)  # one chunk per epoch
            CleanEpochChunks = epoch_no_grad_chunk(EpochChunks)  # remove gradient info
            BestEpochChunks = best_epoch_chunks(
                CleanEpochChunks
            )  # keep only best-so-far epochs
            BestEpochChunk = BestEpochChunks[-1]  # keep only the last best epoch
            for task in group_map[_group]:
                r2_val = get_float(BestEpochChunk, task, "r2")
                r2_val = process_r2(r2_val)
                rmse_val = get_float(BestEpochChunk, task, "rmse")
                cv_rmse[n_submodel].append(rmse_val)
                cv_r2[n_submodel].append(r2_val)
                # only need to do below once
                if n_submodel == 0:
                    props.append(task)
                    groups.append(_group)
                    if split_test_on:
                        r2_val = get_float(TestSetChunk, task, "r2")
                        r2_val = process_r2(r2_val)
                        rmse_val = get_float(TestSetChunk, task, "rmse")
                        test_rmse.append(rmse_val)
                        test_r2.append(r2_val)

    # ###############
    # error handling
    # ###############
    n_props = 0
    for _, val in group_map.items():
        n_props += len(val)

    assert all([len(x) == n_props for x in cv_r2])
    assert all([len(x) == n_props for x in cv_rmse])
    assert len(test_rmse) == n_props
    assert len(test_r2) == n_props
    assert len(props) == n_props
    assert len(groups) == n_props

    # ###############################
    # assemble results into DataFrame
    # ###############################
    error_df = pd.DataFrame(
        {
            "group": groups,
            "property": props,
        }
    )
    for i in range(n_fold):
        error_df[f"fold {i+1} rmse"] = cv_rmse[i]
        error_df[f"fold {i+1} r2"] = cv_r2[i]

    error_df["avg_rmse_skipna"] = (
        error_df[[key for key in error_df.columns if "rmse" in key]]
        .mean(axis=1, skipna=True)
        .tolist()
    )
    error_df["avg_r2_skipna"] = (
        error_df[[key for key in error_df.columns if "r2" in key]]
        .mean(axis=1, skipna=True)
        .tolist()
    )

    error_df["test_r2"] = test_r2
    error_df["test_rmse"] = test_rmse

    # ##############
    # save error_df
    # ##############
    error_df.to_csv(save_path)
