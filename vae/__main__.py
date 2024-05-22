import tqdm
import torch
import pandas as pd
from vae import vae_models
from base_classes import schedulers
from vae.trainer import VAETrainer, VAEArgParser
from data.molecule_iterator import SmileBucketIterator

if __name__ == '__main__':

    """ 0. Args Input/Setting """
    args = VAEArgParser().parse_args()
    args.logdir = '../log/'
    args.latent_size = 256
    args.smi_load_batch_size = 256
    args.max_len = 100

    # load restored models(optional)
    # args.restore = "../log/anion/best"
    # args.restore = "../log/cation/best"

    seed = 1919
    args.num_epochs = 30

    """ 1.Selecting Mode """
    args.training = True  # Default: False
    args.fine_tune = False
    args.validity_test = False
    args.reconstruction_test = False
    args.generate_samples = False  # Default:store_true
    args.vuns = False # Validity, Uniqueness, Novelty & SAScore


    """ 1.5 Setting Smiles&Vocab file path"""
    vocab_file = '../data/Smiles&Vocab/cation_vocab'
    # vocab_file = '../data/Smiles&Vocab/anion_vocab'

    # For pretraining:
    # train_smi_file = '../data/Smiles&Vocab/anion_list.txt'
    # train_smi_file = '../data/Smiles&Vocab/cation_list.txt'

    # For finetuning:
    train_smi_file = '../data/Smiles&Vocab/Finetune/213cationcan.txt'
    test_smi_file = '../data/Smiles&Vocab/Finetune/213cationcan.txt'
    # train_smi_file = '../data/Smiles&Vocab/Finetune/213anioncan.txt'
    # test_smi_file = '../data/Smiles&Vocab/Finetune/213anioncan.txt'
    
    # You may want to load SMILES from excel but not file:
    file_read = True # Setting to False when loading SMILES from excel
    # data = pd.read_excel('../data/CO2ModelData.xlsx', sheet_name=1)
    # train_smi_file = data['cation SMILES'].unique()
    # test_smi_file = data['cation SMILES'].unique()

    """ 2.Initializing Smi_iterator """
    smi_iterator = SmileBucketIterator(vocab_file=vocab_file, batch_size=args.smi_load_batch_size,
                                       train_data_file=train_smi_file, test_data_file=test_smi_file, valid_data_file=None,
                                       seed_random=0, random_test=False, file_read=file_read, load_vocab=True, training=args.training)
    train_bucket_iter = smi_iterator.train_bucket_iter() if args.training is True else None
    test_bucket_iter = smi_iterator.test_bucket_iter()
    # Attribute Properties for creating vae
    vocab_size = smi_iterator.vocab_size
    padding_idx = smi_iterator.padding_idx
    sos_idx = smi_iterator.sos_idx
    eos_idx = smi_iterator.eos_idx
    unk_idx = smi_iterator.unk_idx
    vocab = smi_iterator.get_vocab()

    """ 3.Define Vae model & Optimizer, scheduler, device, etc """
    vae = vae_models.Vae(vocab=vocab, vocab_size=vocab_size, embedding_size=args.embedding_size, dropout=args.vae_dropout,
                         padding_idx=padding_idx, sos_idx=sos_idx, unk_idx=unk_idx,
                         max_len=args.vae_max_len, n_layers=args.vae_n_layers, hidden_size=args.hidden_size,
                         bidirectional=args.enc_bidir, latent_size=args.latent_size,
                         partialsmiles=args.partialsmiles).cuda()
    enc_optimizer = torch.optim.Adam(vae.encoder_params, lr=3e-4)
    dec_optimizer = torch.optim.Adam(vae.decoder_params, lr=1e-4)
    scheduler = schedulers.StepScheduler(enc_optimizer, dec_optimizer, epoch_anchors=[200, 250, 275])
    trainer = VAETrainer(args, vocab, vae, enc_optimizer, dec_optimizer, scheduler, train_bucket_iter,
                                 test_bucket_iter)

    """ 4.Mode Processing """
    if args.training:
        print("Start Training.")
        if args.fine_tune:
            trainer.load_raw(args.restore)
        trainer.train()
        print('recon_acc:', trainer.validate(epsilon_std=1e-6))
    else:
        trainer.load_raw(args.restore)

    if args.validity_test:
        print('recon_acc:', trainer.validate(epsilon_std=1e-6))

    if args.reconstruction_test:
        trainer.validate_rec(epsilon_std=1e-6)

    if args.vuns:
        trainer.vuns_test(test_smi_file)

    if args.generate_samples:
        # random sampling
        samples = []
        for _ in tqdm.tqdm(range(10)):
            samples.append(trainer.sample_prior(1000).cpu())
        samples = torch.cat(samples, 0)
        torch.save(samples, 'prior_samples.pkl')


