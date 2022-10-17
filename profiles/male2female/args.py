import argparse
argsall = argparse.Namespace(
    testdata_path='data/celeba_hq/val/male',
    ckpt = 'pretrained_model/celebahq_female_ddpm.pth',
    dsepath = 'pretrained_model/male2female_dse.pt',
    config_path = 'profiles/male2female/male2female.yml',
    t = 500,
    ls =  500.0,
    li = 2.0,
    s1 = 'cosine',
    s2 = 'neg_l2',
    phase = 'test',
    root = 'runs/',
    sample_step= 1,
    batch_size = 20,
    diffusionmodel = 'DDPM',
    down_N = 32,
    seed=1234)