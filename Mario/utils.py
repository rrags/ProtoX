import gc
import numpy as np
from skimage.transform import resize
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset, random_split

class ExpertDataset(Dataset):
    def __init__(self, expert_observations, expert_actions):
        self.observations = expert_observations
        self.actions = expert_actions
        
    def __getitem__(self, index):
        return (self.observations[index].astype(np.float32), self.actions[index].astype(np.long))

    def __len__(self):
        return len(self.observations)

def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {
        '__getitem__': __getitem__,
    })

def crop_pong(obs):
    if obs.shape[0] == 1:
        return obs[:,10:84,:,:]
    elif obs.shape[0] == 4:
        return obs[:,10:84,:]


def gen_color_data(world=1,
             stage=1,
             action_type='simple',
             saved_path='trained_models',
             output_path='output',
             num_interactions=int(3e4),
             crop=True,
             bad=False):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = create_train_env(world, stage, actions,output_path=output_path)
    model = PPO(env.observation_space.shape[0], len(actions))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(saved_path, world, stage)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/ppo_super_mario_bros_{}_{}".format(saved_path, world, stage),
                                         map_location=lambda storage, loc: storage))
    model.eval()

    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    
    expert_observations = np.empty((num_interactions,4,84,84))
    color_observations = np.empty((num_interactions,84,84,3))
    expert_actions = np.empty((num_interactions,) + env.action_space.shape)
    episode_schedule = np.empty((num_interactions,2))
    
    ep_number = 0
    step_number = 0
    while step_number < num_interactions:
        print(step_number, num_interactions)
        state = torch.from_numpy(env.reset())
        while step_number < num_interactions:#True:
            if torch.cuda.is_available():
                state = state.cuda()
            logits, value = model(state)
            policy = F.softmax(logits, dim=1)
            
            # move save state to before step
            if crop:
                state = torchvision.transforms.functional.crop(state, 15,0,84-15,84)
                state = torchvision.transforms.functional.resize(state,(84,84))
            
            expert_observations[step_number] = state.detach().cpu().numpy()#.transpose(0,3,1,2) 
            frame = env.render(mode='rgb_array')#.astype(int)
            im = Image.fromarray(frame)
            im = im.resize(size=(84,84), resample=Image.BICUBIC, reducing_gap=3.0)
            
            if crop:
                im = im.crop((0,15,84,84))
                im = im.resize(size=(84,84), resample=Image.BICUBIC, reducing_gap=3.0)
            
            color_observations[step_number] = np.array(im)
            
            if random.random() < .995:
                action = torch.argmax(policy).item()
            else:
                action = np.random.choice(7,size=1,p=policy.detach().cpu().numpy()[0])[0]
                
            print('org act: ', action)
            if bad:
                #acts = {0:'NOOP',1:'RIGHT',2:'RIGHT+A',3:'RIGHT+B',4:'RIGHT+A+B',5:'A',6:'LEFT'}
                if action in [2,4,5]:
                    action = 3
                
                
            state, reward, done, info = env.step(action)
            
            # save data
            #save state WAS here
            expert_actions[step_number] = action
            episode_schedule[step_number] = np.array([ep_number, step_number])
            
            state = torch.from_numpy(state)
            env.render()
            
            step_number += 1
            
            if info["flag_get"]:
                print("World {} stage {} completed".format(world, stage))
                ep_number += 1
                break
                
        

    env.close()
    return expert_observations, color_observations, expert_actions, episode_schedule

def prot_equivs(net):
    eq = {}
    seen = set([])
    skipped = set([])
    for i in range(net.prototype_vectors.shape[0]):
        for j in range(net.prototype_vectors.shape[0]):
            if j in seen:
                skipped.add(j)
                continue
                
            if (net.prototype_vectors[i] == net.prototype_vectors[j]).all():# and i not in seen and j not in seen:
                if i not in list(eq):
                    eq[i] = [j]
                else:
                    eq[i].append(j)
                seen.add(i)
                seen.add(j)
            if j == net.prototype_vectors.shape[0] - 1 and i not in list(eq) and i not in seen:
                eq[i] = []
                
    return eq

def merge_weights(net,device,eq, unique_ix,action_filter=False):
    fc_in = torch.unique(net.prototype_vectors,dim=0).shape[0]
    fc_out = 7
    
    unique_prots = net.prototype_vectors[unique_ix]
    net.prototype_vectors = nn.Parameter(unique_prots)
    
    fc_old = net.last_layer.weight.data
    
    net.last_layer = nn.Linear(fc_in, fc_out)
    net.last_layer.weight.data = torch.zeros(net.last_layer.weight.data.shape).to(device)
    
    for ix, i in enumerate(unique_ix):
        for j in eq[i]:
            for k in range(7):
                if fc_old.T[ix][k] < 0 and action_filter:
                    net.last_layer.weight.data.T[ix][k] = -.5#fc_old.T[j][k]
                #print(ix, j, k)
                else:
                    net.last_layer.weight.data.T[ix][k] += fc_old.T[j][k]
    return net

# p
def save_proto(net, push_loader, device, project = True, action_restriction=False):
    update_proto = net.prototype_vectors.data.clone().detach().cpu().numpy()#torch.randn(net.prototype_vectors.data.shape).to(device)
    min_dists = 9999999 * np.ones(net.num_prototypes)
    protos = {}
    net = net.to(device)
    for i, data in enumerate(push_loader):
        try:
            obs, actions = data
        except Exception as e:
            obs, actions, _ = data
            
        #obs = obs.permute(0, 3, 1, 2)
        obs = obs.to(device)
        encodings, distances = net.push_forward(obs.float())
        distances = distances.squeeze(1)
        if len(list(distances.size())) > 2:
            distances = distances.squeeze(0)
        #batch_best_idx = torch.argmin(distances, dim=0) #WAS argmax!!!

        distances = distances.detach().cpu().numpy()
        encodings = encodings.detach().cpu().numpy()
        obs = obs.detach().cpu().numpy()
        
        for k in range(distances.shape[0]):
            for j in range(net.num_prototypes):
                target_class = torch.argmax(net.prototype_action_identity[j]).item()

                # only set prototypes to be images of same action class
                if distances[k][j] < min_dists[j]:# and actions[k] == target_class:
                    if (action_restriction and actions[k] == target_class) or not action_restriction: 
                        protos[j] = obs[k]
                        min_dists[j] = distances[k][j]
                        update_proto[j] = encodings[k]
        del obs
        del actions
        del encodings
        del distances
        #del batch_best_idx
        torch.cuda.empty_cache()
        gc.collect()
    if project:
        print('Copying new prototypes into network')
        net.prototype_vectors.data.copy_(torch.FloatTensor(update_proto).to(device))

        del update_proto
        torch.cuda.empty_cache()
    
    return protos
