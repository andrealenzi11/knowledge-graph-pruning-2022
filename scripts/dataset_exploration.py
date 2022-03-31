from pykeen.datasets import FB15k237, WN18RR, YAGO310, UMLS, Countries, CoDExSmall

if __name__ == '__main__':

    # "FB15k237" | "WN18RR" | "YAGO310" | "UMLS" | "Countries" | "CoDExSmall"
    dataset_name = "CoDExSmall"
    print(f"\n {'#'* 10} {dataset_name} {'#'* 10} \n")

    if dataset_name == "FB15k237":
        dataset = FB15k237(create_inverse_triples=False)
    elif dataset_name == "WN18RR":
        dataset = WN18RR(create_inverse_triples=False)
    elif dataset_name == "YAGO310":
        dataset = YAGO310(create_inverse_triples=False)
    elif dataset_name == "UMLS":
        dataset = UMLS(create_inverse_triples=False)
    elif dataset_name == "Countries":
        dataset = Countries(create_inverse_triples=False)
    elif dataset_name == "CoDExSmall":
        dataset = CoDExSmall(create_inverse_triples=False)
    else:
        raise ValueError("Invalid pykeen_dataset name!")

    print(type(dataset))
    print(dataset.training.triples.shape)

    print("- Whole DataSet:")
    print(dataset)
    print("- Training:")
    print(dataset.training)
    print(type(dataset.training))
    print("- Validation:")
    print(dataset.validation)
    print(type(dataset.validation))
    print("- Testing:")
    print(dataset.testing)
    print(type(dataset.testing))

    print("\n Mapping:")
    print(list(dataset.entity_to_id.keys())[:10])
    print(list(dataset.entity_to_id.values())[:10])
    print(dataset.factory_dict)

    print(f"\n{'-'*80}\n >>>> Summarization...")
    print(dataset.summarize(title=None, show_examples=50, file=None))
    # print(pykeen_dataset.summary_str(title=None, show_examples=5, end='\\n'))
    print(f"{'-'*80}\n")

    # allintitle: Similar to “intitle,” but only results containing
    #             all of the specified words in the title tag will be returned

    # allintext: Similar to “intext,” but only results containing all of the specified words
    #            somewhere on the page will be returned.
