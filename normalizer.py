#TO TRAIN


def get_norm_factors(train_loader, n_examples):
    means = []
    stds = []
    count_imgs = 0
    for samples, target, road_image, extra  in trainloader:
        for sample in samples:
            means.append(torch.mean(sample,axis = [1,2]).numpy())
            stds.append(torch.std(sample,axis = [1,2]).numpy())
            count_imgs = count_imgs+1
        if count_imgs > n_examples:
            break
    means = np.array(means).mean(axis = 0)
    stds = np.array(stds).mean(axis = 0)
    return means,stds
        

#means = [0.6394939, 0.6755114, 0.7049375]
#stds = [0.31936955, 0.3117349 , 0.2953726 ]
