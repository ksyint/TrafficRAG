RECIPE = {'version': 1,
 'pipeline': {'proposal': {'threshold': 0.5, 'sigma': 1.0, 'window': 2.0, 'stride': 1.0},
              'verification': {'semantic_weight': 0.4, 'topk': 3},
              'refinement': {'base_padding': 0.5, 'adaptive_padding': 1.0}},
 'motion': {'seed': 42, 'epochs': 50, 'batch_size': 1, 'lr': 1e-06, 'weight_decay': 0.01}}
