import os
import pickle

# Specify path to pickled_data directory
PICKLED_DATA = 'pickled_data'


def get_data(
        argument_1,
        argument_2,
        use_pickled: bool = True
):
    file_path = os.path.join(
        PICKLED_DATA,
        # Use name of method as first part of pickle file name, then follow
        # with argument values.
        f"get_data_{argument_1}_{argument_2}.pickle"
    )
    if use_pickled and os.path.exists(file_path):
        # Load pickled data if use_pickled is True and the file exists
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
    else:
        # Generate the data if it doesn't exist or if use_pickled is False
        data = "Mr. Poopy Butthole"

        # Pickle for the next time you call this method with the same arguments
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    return data
