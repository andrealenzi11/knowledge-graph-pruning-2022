import pickle


def write_kg_pickle(graph, out_path: str):
    with open(out_path, 'wb') as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_kg_pickle(in_path: str):
    with open(in_path, 'rb') as handle:
        return pickle.load(handle)
