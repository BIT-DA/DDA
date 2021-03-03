

def inv_lr_scheduler(optimizer,
                     iter_num,
                     gamma,
                     power,
                     lr=0.001,
                     weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr_mult = (1 + gamma * iter_num / 200000)**(-power)
    lr = lr * lr_mult
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay
        i += 1

    return optimizer


schedule_dict = {"inv": inv_lr_scheduler}
