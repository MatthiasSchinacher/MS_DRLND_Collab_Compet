# * ---------------- *
#
#   ** Deep Reinforcement Learning Nano Degree **
#   project: Collaboration and Competition
#   author:  Matthias Schinacher
#
#    derived partially from earlier project "Continuous Control"
#
#   the script implements DDPG
#
# * ---------------- *
#    importing the packages we need
# * ---------------- *
import os.path
import sys
import re
import configparser
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
else:
    np.random.seed(123456)
    torch.manual_seed(76543)

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
load_file = 'DDPG_SEP' if not TRAIN else None # only default when not training
# filenames for saving the models etc.
save_file = 'DDPG_SEP-out' if TRAIN else None # only default when training

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

        torch.nn.init.uniform_(self.ll1.weight,-1.1,1.1)
        torch.nn.init.constant_(self.ll1.bias,0.0)
        torch.nn.init.uniform_(self.ll2.weight,-1.1,1.1)
        torch.nn.init.constant_(self.ll2.bias,0.0)
        torch.nn.init.uniform_(self.ll3.weight,-1.1,1.1)
        torch.nn.init.constant_(self.ll3.bias,0.0)

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
        self.ll3 = torch.nn.Linear(size2, 1)
        
        self.flag_batch_norm = flag_batch_norm
        if flag_batch_norm:
            self.batch = torch.nn.BatchNorm1d(state_size)

        torch.nn.init.uniform_(self.ll1.weight,-1.1,1.1)
        torch.nn.init.constant_(self.ll1.bias,0.0)
        torch.nn.init.uniform_(self.ll2.weight,-1.1,1.1)
        torch.nn.init.constant_(self.ll2.bias,0.0)
        torch.nn.init.uniform_(self.ll3.weight,-0.1,0.1)
        torch.nn.init.constant_(self.ll3.bias,0.0)

    def forward(self, state, action):
        x = state
        if self.flag_batch_norm:
            x = self.r1(self.ll1(self.batch(x)))
            return self.ll3(self.r2(self.ll2(torch.cat((x, action), dim=1))))
        else:
            x = self.r1(self.ll1(x))
            return self.ll3(self.r2(self.ll2(torch.cat((x, action), dim=1))))

class MSReplayBuffer:
    def __init__(self,buffer_size):
        self.buffer_size = buffer_size
        self.rm_size = 0
        self.rm_next = 0
        self.buffer  = []
        self.batch   = None
        self.batch_idx = None
        self.index_array = [i for i in range(buffer_size)]
        self.prio_buffer = []
        self.priosum = float(0)
        self.priomax = 1.0

    def put_sample(self,state,action,reward,next_state,done,reward_factor=1.0,prio_replay=True):
#        if abs(float(reward)) <= 0.001: # we can not learn from this episode! the ball never entered our side!
#            return
#        if prio_replay:
#            prio = 0.03 + abs(float(reward))
#        else:
#            prio = float(1)
        prio = self.priomax
        t = (state,action,reward,next_state,done)
        if self.rm_size < self.buffer_size:
            self.rm_size += 1
            self.rm_next = self.rm_size
            self.buffer.append(t)
            self.prio_buffer.append(prio)
            self.priosum += prio
        else:
            if self.rm_next >= self.rm_size:
                self.rm_next = 0
            self.buffer[self.rm_next] = t
            self.priosum += prio - self.prio_buffer[self.rm_next]
            self.prio_buffer[self.rm_next] = prio
            self.rm_next += 1
            
    def sample_batch(self,batch_size,prio_replay=True):
        if prio_replay:
            tmp = float(1)/self.priosum
            P = np.array(self.prio_buffer,dtype=np.float64) * tmp
            if self.rm_size < self.buffer_size:
                batch_idx = np.random.choice(self.index_array[0:self.rm_size],size=batch_size,p=P)
            else:
                batch_idx = np.random.randint(self.rm_size,size=batch_size,p=P)
        else:
            batch_idx = np.random.randint(self.buffer_size, size=batch_size)
        if self.rm_size < self.buffer_size:
            batch_idx = [int(idx) % self.rm_size for idx in batch_idx]
        self.batch = [self.buffer[idx] for idx in batch_idx]
        
        self.batch_idx = batch_idx
        
    def redo_prio(self,prios):
        if not self.batch_idx is None:
            for idx,i in enumerate(self.batch_idx):
                self.priosum -= float(self.prio_buffer[i] - prios[idx])
                self.prio_buffer[i] = float(prios[idx])
            self.priomax = float(max(self.prio_buffer))
    
    def get_batch_states_as_tensor(self):
        if self.batch is None:
            return None
        
        return torch.tensor([s for s,_,_,_,_ in self.batch],dtype=torch.float64)

    def get_batch_next_states_as_tensor(self):
        if self.batch is None:
            return None
        
        return torch.tensor([ns for _,_,_,ns,_ in self.batch],dtype=torch.float64)

    def get_batch_actions_as_tensor(self):
        if self.batch is None:
            return None
        
        return torch.tensor([a for _,a,_,_,_ in self.batch],dtype=torch.float64)

    def get_batch_rewards_as_tensor(self):
        if self.batch is None:
            return None
        
        return torch.tensor([r for _,_,r,_,_ in self.batch],dtype=torch.float64)

    def get_batch_dones_as_tensor(self):
        if self.batch is None:
            return None
        
        return torch.tensor([d for _,_,_,_,d in self.batch],dtype=torch.float64)

class MSPlayer:
    def __init__(self, action_size, state_size, actor_size1=311, actor_size2=187, critic_size1=299, critic_size2=155, flag_batch_norm=False):
        self.action_size = action_size
        self.state_size = state_size
        
        self.actor = MSA(action_size, state_size, size1=actor_size1, size2=actor_size2, flag_batch_norm=flag_batch_norm)
        self.critic = MSC(action_size, state_size, size1=critic_size1, size2=critic_size2, flag_batch_norm=flag_batch_norm)
        self.t_actor = MSA(action_size, state_size, size1=actor_size1, size2=actor_size2, flag_batch_norm=flag_batch_norm)
        self.t_critic = MSC(action_size, state_size, size1=critic_size1, size2=critic_size2, flag_batch_norm=flag_batch_norm)
        
        for tp, p in zip(self.t_actor.parameters(), self.actor.parameters()):
            tp.detach_()
            tp.copy_(p)
        for tp, p in zip(self.t_critic.parameters(), self.critic.parameters()):
            tp.detach_()
            tp.copy_(p)
            
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),lr=LEARNING_RATE)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(),lr=LEARNING_RATE_C)#,weight_decay=0.0001)

        self.rb = MSReplayBuffer(50000) # TODO
        self.noise = np.zeros( (action_size,) )
        
    def next_warmup_action(self):
        action = WARMUP_EPISODES_F* np.random.randn(1,self.action_size) # TODO: F
        action = np.clip(action, -1, 1)
        return action
    
    def next_action(self,state,noise_epsilon=1.0):
        self.actor.eval()
        with torch.no_grad():
            tmp = torch.unsqueeze(torch.tensor(state),0)
            action = np.resize(self.actor(tmp).detach().numpy(),(1,self.action_size))
        self.actor.train()

        if noise_epsilon > 0.0:
        #    self.noise += -NOISE_THETA * self.noise + NOISE_SIGMA * np.random.rand(self.action_size)
        #    action += noise_epsilon * self.noise
            action += noise_epsilon * np.random.randn(1,self.action_size)
        action = np.clip(action, -1, 1)
        
        return action

    def learn(self,batch_size,prio_replay=True):
        if self.rb.rm_size < batch_size:
            return False
        
        self.rb.sample_batch(batch_size)
        lr = self.rb.get_batch_rewards_as_tensor()
        ld = self.rb.get_batch_dones_as_tensor()
        ls = self.rb.get_batch_states_as_tensor()
        lns = self.rb.get_batch_next_states_as_tensor()
        la = self.rb.get_batch_actions_as_tensor()
        lna = self.actor(lns)

        #print('[DEBUG] lr:',type(lr),lr.shape)
        #print('[DEBUG] ld:',type(ld),ld.shape)
        #print('[DEBUG] lns:',type(lns),lns.shape)
        #print('[DEBUG] lna:',type(lna),lna.shape)
        tmp = self.t_critic(lns,lna)
        #print('[DEBUG] tmp:',type(tmp),tmp.shape)

        y = lr + ((1.0 - ld) * (GAMMA * tmp))
        y.detach()
        y_ = self.critic(ls,la)

        if prio_replay:
            delta = PRIO_OFFSET + np.resize(torch.abs(y - y_).detach().numpy(),(batch_size,1))
            prios = [float(x)**2 for x in delta]
            #print('[DEBUG] delta:',type(delta),delta)
            self.rb.redo_prio(prios)

        # update critic by minimizing loss
        loss = fct.mse_loss(y_,y)
        self.optimizer_critic.zero_grad()
        loss.backward()
        if GRAD_NORM_CLIP > 0.0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_NORM_CLIP)
        self.optimizer_critic.step()
        
        # update actor by maximizing J => minimizing -J
        la  = self.actor(ls)
        loss = -self.critic(ls.detach(),la).mean()
        self.optimizer_actor.zero_grad()
        loss.backward()
        self.optimizer_actor.step()
        return True

    def soft_update(self):
        for tp, p in zip(self.t_actor.parameters(), self.actor.parameters()):
            tp.detach_()
            tp.copy_(TAU* p + (1.0 - TAU) * tp)
        for tp, p in zip(self.t_critic.parameters(), self.critic.parameters()):
            tp.detach_()
            tp.copy_(TAU* p + (1.0 - TAU) * tp)

# * ---------------- *
#   loading the Reacher environment, loading the default brain (external)
# * ---------------- *
env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# * ---------------- *
#   the actual algorithm
# * ---------------- *

score_buffer = []

rl.write('#\n# Episode Score1 Score2\n')
train_mode = not SHOW

player1 = MSPlayer(ACTION_SIZE, STATE_SIZE, MODEL_H1, MODEL_H2, MODEL_C_H1, MODEL_C_H1, flag_batch_norm=BATCH_NORM)
player2 = MSPlayer(ACTION_SIZE, STATE_SIZE, MODEL_H1, MODEL_H2, MODEL_C_H1, MODEL_C_H1, flag_batch_norm=BATCH_NORM)

epsilon = EPSILON_START

for episode in range(1,EPISODES+1):
    step = 0
    env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
    state  = env_info.vector_observations                   # get the start state
    
    score1 = 0.0
    score2 = 0.0
    
    while True:
        step += 1

        if episode > WARMUP_EPISODES:
            action1 = player1.next_action(state[0],noise_epsilon=epsilon)
            action2 = player2.next_action(state[1],noise_epsilon=epsilon)
        else:
            action1 = player1.next_warmup_action()
            action2 = player2.next_warmup_action()
            
        action  = np.resize(np.concatenate( (action1,action2) ), (2,ACTION_SIZE) )
        env_info = env.step(action)[brain_name]     # send the action to the environment
        next_state = env_info.vector_observations   # get the next state
#        if step==10:
#            print('[DEBUG]: state1:',next_state[0])
#            print('[DEBUG]: state2:',next_state[1])
#            quit()
        reward = env_info.rewards                   # get the reward
        done = env_info.local_done                  # see if episode has finished    
        fr1 = float(reward[0])
        fr2 = float(reward[1])
        d1  = 1.0 if done[0] else 0.0
        d2  = 1.0 if done[1] else 0.0
        
        player1.rb.put_sample(state[0],action[0],fr1,next_state[0],d1)
        player2.rb.put_sample(state[1],action[1],fr2,next_state[1],d2)
        #print('[DEBUG] state: ',state,type(state),state.shape)
        #print('[DEBUG] next_state: ',next_state,type(next_state),next_state.shape)
        #print('[DEBUG] reward: ',reward,type(reward))
        #print('[DEBUG] done: ',done,type(done))
        #quit()
        score1 += fr1
        score2 += fr2

        if episode > WARMUP_EPISODES:
            if step % REPLAY_STEPS == 0:
                #print('[DEBUG] step:',step,'; <= learning!')
                learned = player1.learn(REPLAY_BATCHSIZE,prio_replay=PRIO_REPLAY)
                if learned:
                    player1.soft_update()
                learned = player2.learn(REPLAY_BATCHSIZE,prio_replay=PRIO_REPLAY)
                if learned:
                    player2.soft_update()

        state = np.copy(next_state)                    # roll over the state to next time step
        if step >= MAX_STEPS or any(done):             # exit loop if episode finished
            break

    rl.write('{} {} {} {}\n'.format(episode,score1,score2,step))
    rl.flush()
    if episode % 10 == 0:
        print("Episode: {}; Score1: {}; Score2: {}; Step-count: {}".format(episode,score1,score2,step))

    if TRAIN:
        if epsilon - EPSILON_DELTA >= EPSILON_MIN:
            epsilon -= EPSILON_DELTA

rl.close()
