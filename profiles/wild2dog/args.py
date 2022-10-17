import argparse
argsall = argparse.Namespace(
    testdata_path = 'data/afhq/val/wild',
    ckpt = 'pretrained_model/afhq_dog_4m.pt',
    dsepath = 'pretrained_model/wild2dog_dse.pt',
    config_path = 'profiles/wild2dog/wild2dog.yml',
    t = 500,
    ls =  500.0,
    li = 2.0,
    s1 = 'cosine',
    s2 = 'neg_l2',
    phase = 'test',
    root = 'runs/',
    sample_step= 1,
    batch_size = 20,
    diffusionmodel = 'ADM',
    down_N = 32,
    seed=1234)