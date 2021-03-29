import torch
import os

def latent_transversal(model, input_img, lower, upper, n_imgs, latent_index):
    x = input_img.view(-1, 64*64)
    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)

    ran = upper - lower
    step_size = ran / n_imgs

    imgs = []
    start = lower
    while start < upper:
        z[0][latent_index] = start
        imgs.append(model.decode(z).cpu().view(64,64))
        start += step_size
    return imgs

def save_checkpoint(model, optimizer, filename, losslogger, epoch):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'losslogger': losslogger, }
    torch.save(state, filename)

def load_checkpoint(model, optimizer, losslogger, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger