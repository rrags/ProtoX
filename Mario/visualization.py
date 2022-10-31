def color_ball(net, oidx,min_sim, n_samples=1000,alpha=.2,Af=True):        
    fig, axes = plt.subplots(1,2)
    axes[0].imshow(color_observations[oidx].astype(int))
    
    if Af:
        enc1, _ = net.push_forward(torch.FloatTensor(expert_observations[oidx]).unsqueeze(0).to(device)) 
    else:
        enc1 = net.encoder(torch.FloatTensor(expert_observations[oidx]).unsqueeze(0).to(device))
        
    best_list = []
    for i in range(n_samples):
        if Af:
            enc2, _ = net.push_forward(torch.FloatTensor(expert_observations[i]).unsqueeze(0).to(device)) 
        else:
            enc2 = net.encoder(torch.FloatTensor(expert_observations[i]).unsqueeze(0).to(device))
            
        d = torch.norm(enc1 - enc2)
        sim = torch.exp(-.05 * d)
        
        if sim > min_sim:
            best_list.append(i)
            
    print('Found ', len(best_list))
    for i, ix in enumerate(best_list):
    	axes[1].imshow(color_observations[ix].astype(int),alpha=alpha)#min(.2 + .2*i, 1))
    for i in [0,1]:        
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].axis('off')
    axes[0].set_title('Input')
    axes[1].set_title('Top 30\n Most Similar States')

def explain_color(net,
                 oidx,
                 push_loader,
                 action_filter=False,
                 k=3,
                 fname=None,
                 score_sort=False,
                 sim_method=0):
    net.eval()
    acts = {0:'NOOP',1:'RIGHT',2:'RIGHT+A',3:'RIGHT+B',4:'RIGHT+A+B',5:'A',6:'LEFT'}

    num_prototypes_per_action = net.num_prototypes // net.num_actions
    
    fig, axes = plt.subplots(1,1+k,figsize=(30,30))
    
    action = expert_actions[oidx].item()
    
    out, _ = net(torch.FloatTensor(expert_observations[oidx]).unsqueeze(0).to(device))
    #print(out.shape)
    logit = out[0][int(action)].item()
    # show input
    axes[0].imshow(color_observations[oidx].astype(int))
    if k > 1:
        axes[0].set_title('Input w/ action: ' + acts[action],size=25)# + '\nTotal points: '+str(logit),size=25)
    else:
        axes[0].set_title('Input w/ action: ' + acts[action] + '\nat t='+str(oidx),size=40)# + '\nTotal points: '+str(logit),size=25)
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].axis('off')
    
    # show prots
    top_k_ix, sims, fcs = top_k_prots(net, oidx, 
                                      action_filter=action_filter,
                                      k=k,
                                      sim_method=sim_method,
                                      score_sort=score_sort)
    for j in range(1,1+k):
        im, _ = prot_rep3(net, top_k_ix[j-1], push_loader)
        axes[j].imshow(im.astype(np.uint8))
        axes[j].set_xticklabels([])
        axes[j].set_yticklabels([])
        axes[j].set_xticks([])
        axes[j].set_yticks([])
        axes[j].axis('off')
        if k > 1:
            axes[j].set_title('Sim score: {:.2f} \n'+acts[action]+' score: {:.2f}\nPoints: {:.2f}'.format(sims[j-1], fcs[j-1],sims[j-1]*fcs[j-1]),size=25)
        else:
             axes[j].set_title('Most Similar Prototype'.format(sims[j-1], fcs[j-1],sims[j-1]*fcs[j-1]),size=40)
