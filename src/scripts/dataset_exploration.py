from src.dao.dataset_loading import PykeenDatasetLoader

if __name__ == '__main__':

    # "FB15k237" | "WN18RR" | "YAGO310" | "UMLS" | "Countries" | "CoDExSmall" | "Nations"
    dataset_name = "Nations"
    print(f"\n {'#' * 10} {dataset_name} {'#' * 10} \n")

    dataset_loader_obj = PykeenDatasetLoader(dataset_name=dataset_name)
    dataset = dataset_loader_obj.get_pykeen_dataset()

    print(type(dataset))
    print(dataset.training.triples.shape)

    print("\n- Whole DataSet:")
    print(dataset)

    print("\n- Training:")
    print(dataset.training)

    print("\n- Validation:")
    print(dataset.validation)

    print("\n- Testing:")
    print(dataset.testing)

    print("\n\n Mapping:")
    print(list(dataset.entity_to_id.keys())[:10])
    print(list(dataset.entity_to_id.values())[:10])
    print(dataset.factory_dict)

    print(f"\n\n{'-' * 80}\n >>>> Summarization...")
    print(dataset.summarize(title=None, show_examples=20, file=None))
    # print(pykeen_dataset.summary_str(title=None, show_examples=5, end='\\n'))
    print(f"{'-' * 80}\n")

    # allintitle: Similar to “intitle,” but only results containing
    #             all of the specified words in the title tag will be returned

    # allintext: Similar to “intext,” but only results containing all of the specified words
    #            somewhere on the page will be returned.
