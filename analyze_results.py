import dill as pkl

for num in [21, 22, 23, 24, 25, 26]:
    f = open(f"results/behavior__nsrt_learning__{num}________None.pkl","rb")
    results = pkl.load(f)
    print(results['config'].seed)
    print(results['config'].approach)
    print("results:", results['results']['num_solved'], "/", results['results']['num_total'])
    print()