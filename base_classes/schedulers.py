class StepScheduler:
    def __init__(self, enc_optimizer, dec_optimizer, lr_step=0.5, epoch_anchors=None):
        if epoch_anchors is None:
            epoch_anchors = [200, 250, 275]
        self.enc_optimizer = enc_optimizer
        self.dec_optimizer = dec_optimizer
        self._lr_step = lr_step
        self._epoch_anchors = epoch_anchors

    @property
    def lr(self):
        for pg in self.enc_optimizer.param_groups:
            lr = pg['lr']
        return lr

    @lr.setter
    def lr(self, new_lr):
        for pg in self.enc_optimizer.param_groups:
            pg['lr'] = new_lr
        for pg in self.dec_optimizer.param_groups:
            pg['lr'] = new_lr