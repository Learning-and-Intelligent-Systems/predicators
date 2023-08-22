import pickle

results_ebm_full = []
results_ebm_local = []
results_gauss_local = []
results_gauss_full = []

for i in range(10):
    with open('results_ebm_full/bookshelf__sampler_learning__{}________0.pkl'.format(i), 'rb') as f:
        r = pickle.load(f)
    results_ebm_full.append(r['results']['num_solved'])
    
    with open('results_ebm_local/bookshelf__sampler_learning__{}________0.pkl'.format(i), 'rb') as f:
        r = pickle.load(f)
    results_ebm_local.append(r['results']['num_solved'])

    with open('results_gauss_full/bookshelf__sampler_learning__{}________0.pkl'.format(i), 'rb') as f:
        r = pickle.load(f)
    results_gauss_full.append(r['results']['num_solved'])

    with open('results_gauss_local/bookshelf__sampler_learning__{}________0.pkl'.format(i), 'rb') as f:
        r = pickle.load(f)
    results_gauss_local.append(r['results']['num_solved'])

print('results_ebm_full', sum(results_ebm_full) / 10)
print('results_ebm_local', sum(results_ebm_local) / 10)
print('full > local', [f > l for f, l in zip(results_ebm_full, results_ebm_local)])
print('results_gauss_full', sum(results_gauss_full) / 10)
print('results_gauss_local', sum(results_gauss_local) / 10)
print('full > local', [f > l for f, l in zip(results_gauss_full, results_gauss_local)])
