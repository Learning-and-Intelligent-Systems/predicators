import dill as pkl

for num in [10, 11, 12, 15]:
    f = open(f"results/behavior__nsrt_learning__{num}________None.pkl","rb")
    results = pkl.load(f)
    print(results['config'].seed)
    print(results['config'].approach)
    print("results:", results['results']['num_solved'], "/", results['results']['num_total'])
    print()