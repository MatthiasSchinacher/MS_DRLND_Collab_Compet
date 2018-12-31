# * ---------------- *
#
#   ** Deep Reinforcement Learning Nano Degree **
#   project: Collaboration and Competition
#   author:  Matthias Schinacher
#
#   the script implements DDPG and is derived from the one
#   I wrote for the "continuous control" project
#
# * ---------------- *
#    importing the packages we need
# * ---------------- *
import os.path
import sys
import re
import configparser
import pickle
from unityagents import UnityEnvironment
import numpy as np
import torch
import torch.nn.functional as fct

# * ---------------- *
#   command line arguments:
#    we expect exactly 2, the actual script name and the command-file-name
# * ---------------- *
if len(sys.argv) != 2:
    print('usage:')
    print('   python {} command-file-name'.format(sys.argv[0]))
    quit()

if not os.path.isfile(sys.argv[1]):
    print('usage:')
    print('   python {} command-file-name'.format(sys.argv[0]))
    print('[error] "{}" file not found or not a file!'.format(sys.argv[1]))
    quit()

# * ---------------- *
#   constants:
#    this code is only for the Reacher- scenario, no generalization (yet)
# * ---------------- *
STATE_SIZE = int(24)
ACTION_SIZE = int(2)
NUM_AGENTS = int(2)
fzero = float(0)
fone = float(1)

# * ---------------- *
#   the command-file uses the ConfigParser module, thus must be structured that way
#    => loading the config and setting the respective script values
# * ---------------- *
booleanpattern = re.compile('^\\s*(true|yes|1|on)\\s*$', re.IGNORECASE)

config = configparser.ConfigParser()
config.read(sys.argv[1])

# start the logfile
rlfn = 'run.log' # run-log-file-name
if 'global' in config and 'runlog' in config['global']:
    rlfn = config['global']['runlog']
print('!! using logfile "{}"\n'.format(rlfn))
rl = open(rlfn,'w')
rl.write('# ## configuration from "{}"\n'.format(sys.argv[1]))

if 'rand' in config and 'seed' in config['rand']:
    seed = int(config['rand']['seed'])
    rl.write('# [debug] using random seed: {}\n'.format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)

TRAIN = True   # default to training mode
if 'mode' in config and 'train' in config['mode']:
    train = config['mode']['train']
    TRAIN = True if booleanpattern.match(train) else False
    rl.write('# [debug] using mode.train: {} from "{}"\n'.format(TRAIN,train))
SHOW  = not TRAIN  # default for "show"- mode
if 'mode' in config and 'show' in config['mode']:
    show = config['mode']['show']
    SHOW = True if booleanpattern.match(show) else False
    rl.write('# [debug] using mode.show: {} from "{}"\n'.format(SHOW,show))

# * ---------------- *
#   hyper- parameters
# * ---------------- *
# defaults
EPISODES = int(1000)             # number of episodes (including warm-up)
WARMUP_EPISODES = int(50)        # number of warm-up episodes (training only)
WARMUP_EPISODES_F = float(0.4)   # scale factor for actions randomly sampled
REPLAY_BUFFERSIZE = int(10000)   # replay buffer/memory- size (training only)
REPLAY_BATCHSIZE = int(512)      # batch size for replay (training only)
REPLAY_STEPS = int(1)            # replay transisitions- batch each x steps (training only)
GAMMA = float(0.99)              # gamma- parameter (training only)
LEARNING_RATE = float(0.0001)    # optimizing actor model (training only)
LEARNING_RATE_C = float(0.001)   # optimizing critic model (training only)
OPTIMIZER_STEPS = 1              # optimizer steps (training only)
TAU = float(0.01)                # soft update target networks (training only)
MAX_STEPS = int(500)             # we restrict this to as many steps

# noise- parameters
EPSILON_START = float(2.5)       # start-value for epsilon (training and show- mode!)
EPSILON_DELTA = float(0.001)     # value to substract from delta (training only)
EPSILON_MIN   = float(0.02)      # min value for epsilon
NOISE_THETA = float(0.15)
NOISE_SIGMA = float(0.2)

PRIO_REPLAY = True
PRIO_OFFSET = float(0.2)
GRAD_NORM_CLIP = -float(1.0)

# overwrite defaults
if 'hyperparameters' in config:
    hp = config['hyperparameters']
    EPISODES          = int(hp['episodes'])          if 'episodes'          in hp else EPISODES
    MAX_STEPS         = int(hp['max_steps'])         if 'max_steps'         in hp else MAX_STEPS
    WARMUP_EPISODES   = int(hp['warmup_episodes'])   if 'warmup_episodes'   in hp else WARMUP_EPISODES
    WARMUP_EPISODES_F = float(hp['warmup_episodes_f']) if 'warmup_episodes_f' in hp else WARMUP_EPISODES_F
    REPLAY_BUFFERSIZE = int(hp['replay_buffersize']) if 'replay_buffersize' in hp else REPLAY_BUFFERSIZE
    REPLAY_BATCHSIZE  = int(hp['replay_batchsize'])  if 'replay_batchsize'  in hp else REPLAY_BATCHSIZE
    REPLAY_STEPS      = int(hp['replay_steps'])      if 'replay_steps'      in hp else REPLAY_STEPS
    GAMMA             = float(hp['gamma'])           if 'gamma'             in hp else GAMMA
    LEARNING_RATE     = float(hp['learning_rate'])   if 'learning_rate'     in hp else LEARNING_RATE
    LEARNING_RATE_C   = float(hp['learning_rate_c']) if 'learning_rate_c'   in hp else LEARNING_RATE_C
    OPTIMIZER_STEPS   = int(hp['optimizer_steps'])   if 'optimizer_steps'   in hp else OPTIMIZER_STEPS
    TAU               = float(hp['tau'])             if 'tau'               in hp else TAU

    EPSILON_START     = float(hp['epsilon_start'])   if 'epsilon_start'     in hp else EPSILON_START
    EPSILON_DELTA     = float(hp['epsilon_delta'])   if 'epsilon_delta'     in hp else EPSILON_DELTA
    EPSILON_MIN       = float(hp['epsilon_min'])     if 'epsilon_min'       in hp else EPSILON_MIN
    NOISE_THETA       = float(hp['noise_theta'])     if 'noise_theta'       in hp else NOISE_THETA
    NOISE_SIGMA       = float(hp['noise_sigma'])     if 'noise_sigma'       in hp else NOISE_SIGMA

    PRIO_REPLAY       = (True if booleanpattern.match(hp['prio_replay']) else False) if 'prio_replay' in hp else PRIO_REPLAY
    PRIO_OFFSET       = float(hp['prio_offset'])     if 'prio_offset'       in hp else PRIO_OFFSET
    GRAD_NORM_CLIP    = float(hp['grad_norm_clip'])  if 'grad_norm_clip'    in hp else GRAD_NORM_CLIP

# model- defaults (only if model is not loaded from file)
MODEL_H1 = int(311)     # hidden layer size 1
MODEL_H2 = int(177)     # hidden layer size 2
MODEL_C_H1 = int(309)   # hidden layer size 1, critic
MODEL_C_H2 = int(179)   # hidden layer size 2, critic
BATCH_NORM = False

# filenames for loading the models etc.
load_file = 'DDPG' if not TRAIN else None # only default when not training
# filenames for saving the models etc.
save_file = 'DDPG-out' if TRAIN else None # only default when training

# overwrite defaults
if 'model' in config:
    m = config['model']
    MODEL_H1   = int(m['h1'])    if 'h1'        in m else MODEL_H1
    MODEL_H2   = int(m['h2'])    if 'h2'        in m else MODEL_H2
    MODEL_C_H1 = int(m['c_h1'])  if 'c_h1'      in m else MODEL_C_H1
    MODEL_C_H2 = int(m['c_h2'])  if 'c_h2'      in m else MODEL_C_H2
    BATCH_NORM = (True if booleanpattern.match(m['batch_norm']) else False)  if 'batch_norm' in m else BATCH_NORM
    load_file = m['load_file']   if 'load_file' in m else load_file
    save_file = m['save_file']   if 'save_file' in m else save_file

# * ---------------- *
#   writing the used config to the logfile
# * ---------------- *
rl.write('# TRAIN (mode):      {}\n'.format(TRAIN))
rl.write('# SHOW (mode):       {}\n\n'.format(SHOW))
rl.write('# EPISODES:          {}\n'.format(EPISODES))
rl.write('# MAX_STEPS:         {}\n'.format(MAX_STEPS))
rl.write('# WARMUP_EPISODES:   {}\n'.format(WARMUP_EPISODES))
rl.write('# WARMUP_EPISODES_F: {}\n'.format(WARMUP_EPISODES_F))
rl.write('# REPLAY_BUFFERSIZE: {}\n'.format(REPLAY_BUFFERSIZE))
rl.write('# REPLAY_BATCHSIZE:  {}\n'.format(REPLAY_BATCHSIZE))
rl.write('# REPLAY_STEPS:      {}\n'.format(REPLAY_STEPS))
rl.write('# GAMMA:             {}\n'.format(GAMMA))
rl.write('# LEARNING_RATE:     {}\n'.format(LEARNING_RATE))
rl.write('# LEARNING_RATE_C:   {}\n'.format(LEARNING_RATE_C))
rl.write('# OPTIMIZER_STEPS:   {}\n'.format(OPTIMIZER_STEPS))
rl.write('# TAU:               {}\n#\n'.format(TAU))
rl.write('# EPSILON_START:     {}\n'.format(EPSILON_START))
rl.write('# EPSILON_DELTA:     {}\n'.format(EPSILON_DELTA))
rl.write('# EPSILON_MIN:       {}\n'.format(EPSILON_MIN))
rl.write('# NOISE_THETA:       {}\n'.format(NOISE_THETA))
rl.write('# NOISE_SIGMA:       {}\n#\n'.format(NOISE_SIGMA))
rl.write('# PRIO_REPLAY:       {}\n'.format(PRIO_REPLAY))
rl.write('# PRIO_OFFSET::      {}\n'.format(PRIO_OFFSET))
rl.write('# GRAD_NORM_CLIP:    {}\n#\n'.format(GRAD_NORM_CLIP))
rl.write('#   -- model\n')
rl.write('# H1:          {}\n'.format(MODEL_H1))
rl.write('# H2:          {}\n'.format(MODEL_H2))
rl.write('# H1 (critic): {}\n'.format(MODEL_C_H1))
rl.write('# H2 (critic): {}\n'.format(MODEL_C_H2))
rl.write('# batch_norm:  {}\n'.format(BATCH_NORM))
rl.write('# load_file:   {}\n'.format(load_file))
rl.write('# save_file:   {}\n'.format(save_file))
rl.flush()

dtype = torch.float64
torch.set_default_dtype(dtype)
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

class MSA(torch.nn.Module):
    def __init__(self, action_size, state_size, size1=111, size2=87, flag_batch_norm=False):
        super(MSA, self).__init__()
        self.ll1 = torch.nn.Linear(state_size, size1)
        self.r1  = torch.nn.ReLU()
        self.ll2 = torch.nn.Linear(size1, size2)
        self.r2  = torch.nn.ReLU()
        self.ll3 = torch.nn.Linear(size2, action_size)
        self.th  = torch.nn.Tanh()
        
        self.flag_batch_norm = flag_batch_norm
        if flag_batch_norm:
            self.batch = torch.nn.BatchNorm1d(state_size)

        torch.nn.init.uniform_(self.ll1.weight,-0.1,0.1)
        torch.nn.init.constant_(self.ll1.bias,0.1)
        torch.nn.init.uniform_(self.ll2.weight,-0.1,0.1)
        torch.nn.init.constant_(self.ll2.bias,0.1)
        torch.nn.init.uniform_(self.ll3.weight,-0.001,0.001)
        torch.nn.init.constant_(self.ll3.bias,0.1)

    def forward(self, state):
        if self.flag_batch_norm:
            return self.th(self.ll3(self.r2(self.ll2(self.r1(self.ll1(self.batch(state)))))))
        else:
            return self.th(self.ll3(self.r2(self.ll2(self.r1(self.ll1(state))))))

class MSC(torch.nn.Module):
    def __init__(self, action_size, state_size, size1=111, size2=87, flag_batch_norm=False):
        super(MSC, self).__init__()
        self.ll1 = torch.nn.Linear(state_size, size1)
        self.r1  = torch.nn.ReLU()
        self.ll2 = torch.nn.Linear(size1+action_size, size2)
        self.r2  = torch.nn.ReLU()
        self.ll3 = torch.nn.Linear(size2, action_size)
        
        self.flag_batch_norm = flag_batch_norm
        if flag_batch_norm:
            self.batch = torch.nn.BatchNorm1d(state_size)

        torch.nn.init.uniform_(self.ll1.weight,-0.1,0.1)
        torch.nn.init.constant_(self.ll1.bias,0.1)
        torch.nn.init.uniform_(self.ll2.weight,-0.1,0.1)
        torch.nn.init.constant_(self.ll2.bias,0.1)
        torch.nn.init.uniform_(self.ll3.weight,-0.001,0.001)
        torch.nn.init.constant_(self.ll3.bias,0.1)

    def forward(self, state, action):
        x = state
        if self.flag_batch_norm:
            x = self.r1(self.ll1(self.batch(x)))
            return self.ll3(self.r2(self.ll2(torch.cat((x, action), dim=1))))
        else:
            x = self.r1(self.ll1(x))
            return self.ll3(self.r2(self.ll2(torch.cat((x, action), dim=1))))

# * ---------------- *
#   buildung and initializing the torch- models
# * ---------------- *
if load_file:
    lf = 'actor_{}.model'.format(load_file)
    rl.write('# .. loading actor from "{}"\n'.format(lf))
    modelA = torch.load(lf)
    lf = 'target_actor_{}.model'.format(load_file)
    rl.write('# .. loading target-actor from "{}"\n'.format(lf))
    modelAt = torch.load(lf)
    lf = 'critic_{}.model'.format(load_file)
    rl.write('# .. loading critic from "{}"\n'.format(lf))
    modelC = torch.load(lf)
    lf = 'target_critic_{}.model'.format(load_file)
    rl.write('# .. loading target-critic from "{}"\n'.format(lf))
    modelCt = torch.load(lf)
else:
# actor; function of state to action values
    modelA = MSA(ACTION_SIZE,STATE_SIZE,MODEL_H1,MODEL_H2,flag_batch_norm=BATCH_NORM)
# actor target; initialize with the same weights
    modelAt = MSA(ACTION_SIZE,STATE_SIZE,MODEL_H1,MODEL_H2,flag_batch_norm=BATCH_NORM)
    for tp, p in zip(modelAt.parameters(), modelA.parameters()):
        #tp.data.copy_(p.data)
        tp.detach_()
        tp.copy_(p)
# critic
    modelC = MSC(ACTION_SIZE,STATE_SIZE,MODEL_C_H1,MODEL_C_H2,flag_batch_norm=BATCH_NORM)
# critic target
    modelCt = MSC(ACTION_SIZE,STATE_SIZE,MODEL_C_H1,MODEL_C_H2,flag_batch_norm=BATCH_NORM)
    for tp, p in zip(modelCt.parameters(), modelC.parameters()):
        #tp.data.copy_(p.data)
        tp.detach_()
        tp.copy_(p)

# * ---------------- *
#   loading the Tennis environment, loading the default brain (external)
# * ---------------- *
env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# * ---------------- *
#   the actual algorithm
# * ---------------- *

# a very simple replay memory, a list (of tuples)
#   - is assumed to never shrink
#   - only inserts at given index, if next index would be > size, start at 0
#         => list entries 0..size-1 are occupied
replay_memory = []   # actual replay memory
replay_prio = [] # priority used to compute sampling probability
priomax = 1.0
priosum = 0.0
rm_size = 0          # number of entries in replay memory
rm_next = 0          # next index to use for insert

if load_file:
    lf = 'transitions_{}.pickle'.format(load_file)
    if os.path.isfile(lf):
        with open(lf, 'rb') as f:
            tmp = pickle.load(f)
            if len(tmp) < 3:
                rl.write('# .. loading replay-buffer/ reward-buffer from "{}"\n'.format(lf))
                ( tmpm, tmpr ) = tmp
                replay_memory = tmpm if REPLAY_BUFFERSIZE >= len(tmpm) else tmpm[0:REPLAY_BUFFERSIZE]
                replay_prio = tmpr if REPLAY_BUFFERSIZE >= len(tmpr) else tmpr[0:REPLAY_BUFFERSIZE]
                rm_size = len(replay_memory)
                rm_next = rm_size if rm_size < REPLAY_BUFFERSIZE else 0
            else:
                rl.write('# .. [new] loading replay-buffer/ reward_buffer and rm_next from "{}"\n'.format(lf))
                ( tmpm, tmpr, rm_next ) = tmp
                replay_memory = tmpm if REPLAY_BUFFERSIZE >= len(tmpm) else tmpm[0:REPLAY_BUFFERSIZE]
                replay_prio = tmpr if REPLAY_BUFFERSIZE >= len(tmpr) else tmpr[0:REPLAY_BUFFERSIZE]
                rm_size = len(replay_memory)
                if rm_next >= rm_size:
                    rm_next = 0
        if PRIO_REPLAY and rm_size > 0:
            priomax = float(max(replay_prio))
            priosum = float(sum(replay_prio))

score_buffer = []
noise = np.zeros( (NUM_AGENTS,ACTION_SIZE) )
epsilon = EPSILON_START
index_array = [i for i in range(REPLAY_BUFFERSIZE)]
r_steps = REPLAY_STEPS # initialize big, so we replay the first chance we have

rl.write('#\n# Episode Score average(last-100-Scores) Steps RMSize Epsilon\n')
train_mode = not SHOW

optimizer = torch.optim.Adam(modelC.parameters(),lr=LEARNING_RATE_C)#,weight_decay=0.0001)
optimizerA = torch.optim.Adam(modelA.parameters(),lr=LEARNING_RATE)

solved = False
maxscore = 0.0
ep_maxscore = 0

for episode in range(1,EPISODES+1):
    env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
    state  = env_info.vector_observations               # get the start state
    score  = np.zeros(NUM_AGENTS)                       # initialize the episode- score
    step   = 0                                          # step within episodes
    while True:
        step += 1
        if TRAIN and episode <= WARMUP_EPISODES:
            action = WARMUP_EPISODES_F* np.random.randn(NUM_AGENTS,ACTION_SIZE) # select random actions
            action = np.clip(action, -1, 1)
        else:
            action = np.zeros( (NUM_AGENTS,ACTION_SIZE) )
            modelA.eval()
            with torch.no_grad():
                tmp = torch.unsqueeze(torch.tensor(state),0)
                action = np.resize(modelA(tmp).detach().numpy(),(NUM_AGENTS,ACTION_SIZE))
            modelA.train()

            noise += -NOISE_THETA * noise + NOISE_SIGMA * np.random.rand(NUM_AGENTS,ACTION_SIZE)
            action += epsilon * noise
            action = np.clip(action, -1, 1)

        env_info = env.step(action)[brain_name]     # send the action to the environment
        next_state = env_info.vector_observations   # get the next state
        reward = env_info.rewards          # get the reward
        done = env_info.local_done         # see if episode has finished

        score += reward # update the score
        
        # store transition in replay memory
        for z in range(NUM_AGENTS):
            fr = float(reward[z])
            fd = 1.0 if done[z] else 0.0
            transition = (state[z,:],action[z,:],fr,next_state[z,:],fd)
            if rm_size < REPLAY_BUFFERSIZE:
                replay_memory.append(transition)
                if PRIO_REPLAY:
                    replay_prio.append(priomax)
                    priosum += priomax
                rm_size += 1
            else:
                replay_memory[rm_next] = transition
                if PRIO_REPLAY:
                    priosum += priomax - replay_prio[rm_next]
                    replay_prio[rm_next] = priomax
            
            rm_next += 1
            if rm_next >= REPLAY_BUFFERSIZE:
                rm_next = 0
            
        if TRAIN:
            modelA.train()

            if rm_size >= REPLAY_BATCHSIZE and episode > WARMUP_EPISODES:
                r_steps += 1
                if r_steps >= REPLAY_STEPS:
                    r_steps = 0
                    for _ in range(OPTIMIZER_STEPS):
                        if PRIO_REPLAY:
                            tmp = float(1)/priosum
                            P = np.array(replay_prio,dtype=np.float64) * tmp
                            if rm_size < REPLAY_BUFFERSIZE:
                                batch_idx = np.random.choice(index_array[0:rm_size],size=REPLAY_BATCHSIZE,p=P)
                            else:
                                batch_idx = np.random.choice(index_array,size=REPLAY_BATCHSIZE,p=P)
                        else:
                            if rm_size < REPLAY_BUFFERSIZE:
                                batch_idx = np.random.choice(index_array[0:rm_size],size=REPLAY_BATCHSIZE)
                            else:
                                batch_idx = np.random.randint(rm_size, size=REPLAY_BATCHSIZE)
                        
                        listt  = [replay_memory[idx] for idx in batch_idx]
                        lists = torch.tensor([s for s,_,_,_,_ in listt],dtype=torch.float64)
                        lista = torch.tensor([a for _,a,_,_,_ in listt],dtype=torch.float64)
                        listns = torch.tensor([ns for _,_,_,ns,_ in listt],dtype=torch.float64)
                        listna = modelAt(listns)
                        listr = torch.tensor([[r] for _,_,r,_,_ in listt],dtype=torch.float64)
                        listd = torch.tensor([[d] for _,_,_,_,d in listt],dtype=torch.float64)

                        y = listr + ((1.0 - listd) * (GAMMA * modelCt(listns,listna)))
                        y.detach()

                        # update critic, by minimizing the loss (plus update replay- prio)
                        y_ = modelC(lists,lista)

                        if PRIO_REPLAY:
                            delta = PRIO_OFFSET + np.resize(torch.abs(y - y_).detach().numpy(),(len(batch_idx),1))
                            #print('[DEBUG] delta:',type(delta),delta)
                            
                            for idx,i in enumerate(batch_idx):
                                tmpp = float(delta[idx])
                                tmpp = tmpp*tmpp
                                priosum -= replay_prio[i] - tmpp
                                replay_prio[i] = tmpp
                            priomax = float(max(replay_prio))
                        
                        loss = fct.mse_loss(y_,y)#(y - y_).pow(2).mul(0.5).sum(-1).mean()

                        optimizer.zero_grad()
                        loss.backward()
                        if GRAD_NORM_CLIP > 0.0:
                            torch.nn.utils.clip_grad_norm_(modelC.parameters(), GRAD_NORM_CLIP)
                        optimizer.step()

                        # update actor by maximizing J => minimizing -J
                        lista  = modelA(lists)
                        loss = -modelC(lists.detach(),lista)
                        loss = loss.mean()

                        optimizerA.zero_grad()
                        loss.backward()#retain_graph=True)
                        #torch.nn.utils.clip_grad_norm_(modelA.parameters(), 1.0)
                        optimizerA.step()

                    for tp, p in zip(modelAt.parameters(), modelA.parameters()):
                        #tp.data.copy_(TAU* p.data + (1.0 - TAU) * tp.data)
                        tp.detach_()
                        tp.copy_(TAU* p + (1.0 - TAU) * tp)
                    for tp, p in zip(modelCt.parameters(), modelC.parameters()):
                        #tp.data.copy_(TAU* p.data + (1.0 - TAU) * tp.data)
                        tp.detach_()
                        tp.copy_(TAU* p + (1.0 - TAU) * tp)

        state = np.copy(next_state)                    # roll over the state to next time step

        if step >= MAX_STEPS or any(done):             # exit loop if episode finished or if we reached max. no of steps we allow
            break

    if TRAIN:
        if epsilon - EPSILON_DELTA >= EPSILON_MIN:
            epsilon -= EPSILON_DELTA

    fscore = float(score.max())
    score_buffer.append(fscore)
    while len(score_buffer) > 100:
        score_buffer.pop(0)
    l100_score = float(sum(score_buffer))/float(len(score_buffer)) if len(score_buffer) >= 100 else float(0)
    
    if l100_score > 0.5:
        solved = True
    if solved:
        if fscore > maxscore:
            maxscore = fscore
            ep_maxscore = episode

            if save_file:
                rl.write('# maximum solving score until now; score: {}"\n'.format(fscore))
                sf = 'actor_{}_max.model'.format(save_file)
                rl.write('# .. writing actor to "{}"\n'.format(sf))
                torch.save(modelA,sf)
                sf = 'target_actor_{}_max.model'.format(save_file)
                rl.write('# .. writing target-actor to "{}"\n'.format(sf))
                torch.save(modelAt,sf)
                sf = 'critic_{}_max.model'.format(save_file)
                rl.write('# .. writing critic to "{}"\n'.format(sf))
                torch.save(modelC,sf)
                sf = 'target_critic_{}_max.model'.format(save_file)
                rl.write('# .. writing target-critic to "{}"\n'.format(sf))
                torch.save(modelCt,sf)
        
                sf = 'transitions_{}_max.pickle'.format(save_file)
                rl.write('# .. saving transisitions to "{}"\n'.format(sf))
                with open(sf, 'wb') as f:
                    pickledata = ( replay_memory, replay_prio, rm_next )
                    pickle.dump(pickledata, f, pickle.HIGHEST_PROTOCOL)

    rl.write('{} {} {} {} {} {}\n'.format(episode,fscore,l100_score,step,rm_size,(epsilon if episode > WARMUP_EPISODES else float(0))))
    rl.flush()
    print("Episode: {}; Score: {} ({}); Step: {}; RMSize: {}; Epsilon: {}".format(episode,fscore,l100_score,step,rm_size,(epsilon if episode > WARMUP_EPISODES else '-')))

    if TRAIN and save_file and episode%50 == 0:
        sf = 'actor_{}_bak.model'.format(save_file)
        rl.write('# .. writing actor to "{}"\n'.format(sf))
        torch.save(modelA,sf)
        sf = 'target_actor_{}_bak.model'.format(save_file)
        rl.write('# .. writing target-actor to "{}"\n'.format(sf))
        torch.save(modelAt,sf)
        sf = 'critic_{}_bak.model'.format(save_file)
        rl.write('# .. writing critic to "{}"\n'.format(sf))
        torch.save(modelC,sf)
        sf = 'target_critic_{}_bak.model'.format(save_file)
        rl.write('# .. writing target-critic to "{}"\n'.format(sf))
        torch.save(modelCt,sf)

        sf = 'transitions_{}_bak.pickle'.format(save_file)
        rl.write('# .. saving transisitions to "{}"\n'.format(sf))
        with open(sf, 'wb') as f:
            pickledata = ( replay_memory, replay_prio, rm_next )
            pickle.dump(pickledata, f, pickle.HIGHEST_PROTOCOL)

env.close()
rl.write('# *******\n# Solved: {}; maxscore (only if solved): {} at episode {}\n# *******\n'.format(solved,maxscore,ep_maxscore))
print('*******\nSolved: {}; maxscore (only if solved): {} at episode {}\n*******\n'.format(solved,maxscore,ep_maxscore))

if TRAIN:
    if save_file:
        sf = 'actor_{}.model'.format(save_file)
        rl.write('# .. writing final actor to "{}"\n'.format(sf))
        torch.save(modelA,sf)
        sf = 'target_actor_{}.model'.format(save_file)
        rl.write('# .. writing final target-actor to "{}"\n'.format(sf))
        torch.save(modelAt,sf)
        sf = 'critic_{}.model'.format(save_file)
        rl.write('# .. writing final critic to "{}"\n'.format(sf))
        torch.save(modelC,sf)
        sf = 'target_critic_{}.model'.format(save_file)
        rl.write('# .. writing final target-critic to "{}"\n'.format(sf))
        torch.save(modelCt,sf)

        sf = 'transitions_{}.pickle'.format(save_file)
        rl.write('# .. saving transisitions to "{}"\n'.format(sf))
        with open(sf, 'wb') as f:
            pickledata = ( replay_memory, replay_prio, rm_next )
            pickle.dump(pickledata, f, pickle.HIGHEST_PROTOCOL)

rl.close()
