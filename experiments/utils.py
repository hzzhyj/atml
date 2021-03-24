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