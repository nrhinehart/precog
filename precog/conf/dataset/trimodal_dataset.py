dataset:
  class: dataset.dill_dataset.DillDataset
  params:
    root_path: /home/nrhinehart/data/datasets/nuscenes_full_preproc_A5
    # root_path: /home/rowan/nuscenes_full_preproc_shuffle_A5/
    # root_path: /home/nrhinehart/tmp/
    max_A: 5
    B: 1
    load_bev: True
    perturb: True
    perturb_epsilon: 1e-2
