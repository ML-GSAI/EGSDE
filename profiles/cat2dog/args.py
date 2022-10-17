import argparse
argsall = argparse.Namespace(
    testdata_path = 'data/afhq/val/cat',
    ckpt = 'pretrained_model/afhq_dog_4m.pt',
    dsepath = 'pretrained_model/cat2dog_dse.pt',
    config_path = 'profiles/cat2dog/cat2dog.yml',
    t = 500,
    ls =  500,
    li = 2,
    s1 = 'cosine',
    s2 = 'neg_l2',
    phase = 'test',
    root = 'runs/',
    sample_step= 1,
    batch_size = 20,
    diffusionmodel = 'ADM',
    down_N = 32,
    seed=1234)