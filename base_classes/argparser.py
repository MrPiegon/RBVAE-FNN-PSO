import argparse


class JoinArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(JoinArgParser, self).__init__()
        self.add_argument('--logdir', type=str, default='./log/PSO')
        self.add_argument('--device', type=str, default="cuda:0")
        # Vae args
        self.add_argument('--smi_load_batch_size', type=int, default=50)
        self.add_argument('--smi_test_size', type=int, default=10000)
        self.add_argument('--vocab_file', type=str, default="../data/Smiles&Vocab/gz/9000kvocab_rdcan")
        self.add_argument('--vocab_load_batch_size', type=int, default=50)
        self.add_argument('--vae_dropout', type=float, default=0.5)
        self.add_argument('--vae_train', type=bool, default=False)
        self.add_argument('--embedding_size', type=int, default=48)
        self.add_argument('--vae_n_layers', type=int, default=2)
        self.add_argument('--hidden_size', type=int, default=512)
        self.add_argument('--latent_size', type=int, default=256)
        self.add_argument('--vae_max_len', type=int, default=120)
        self.add_argument('--cation_vae_dir', type=str, default='../log/ion_VAE/cation/checkpoints/60')
        self.add_argument('--anion_vae_dir', type=str, default='../log/ion_VAE/anionbest/checkpoints/90')
        self.add_argument('--enc_bidir', action='store_false')
        self.add_argument('--partialsmiles', action='store_true')
        # mlp args
        self.add_argument('--mlp_batch_size', type=int, default=512)
        self.add_argument('--num_epochs', type=int, default=50)
        # mlp load args
        self.add_argument('--mlp_cat_index', type=int, default=6)
        self.add_argument('--mlp_network', type=list, default=[256, 128, 64, 32, 16, 8, 4, 8, 8, 8])


