import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import os

if torch.cuda.is_available():
    print("true")
    device="cuda"
else:
    device="cpu"    

env = gymnasium.make("FlappyBird-v0", render_mode="human")
class Agent:
    def __init__(self,params_set):
        self.params_set = params_set
        with open("parameters.yaml","r") as f:
            all_param_set=yaml.safe_load(f)
            params=all_param_set[params_set]
        self.alpha=params['alpha'] 
        self.epsilon_min=params['epsilon_min'] 
        self.epsilon_init=params['epsilon_init'] 
        self.epsilon_decay=params['epsilon_decay'] 
        self.replay_memory_size=params['replay_memory_size'] 
        self.mini_batch_size=params['mini_batch_size'] 
        self.network_sync_rate=params['network_sync_rate'] 
        self.reward_threshold=params['reward_threshold'] 
        self.gamma=params['gamma'] 

        self.loss_fn=nn.MSELoss()
        self.optimizer=None

        RUNS_DIR='runs'
        os.makedirs(RUNS_DIR,exist_ok=True)
        self.LOG_FILE=os.path.join(RUNS_DIR,f'{self.params_set}.log')
        self.MODEL_FILE=os.path.join(RUNS_DIR,f'{self.params_set}.pt')

    


    def run(self,is_training=True,render=False):
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None)
        num_states=env.observation_space.shape[0]
        num_actions=env.action_space.n
        policy_dqn=DQN(num_states,num_actions).to(device)


        if is_training:
            memory=ReplayMemory(self.replay_memory_size)
            epsilon=self.epsilon_init
            target_dqn=DQN(num_states,num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            steps=0
            self.optimizer=optim.Adam(policy_dqn.parameters(),lr=self.alpha)
            best_reward=float("-inf")
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()    

        for e in itertools.count():
            state, _ = env.reset()
            state=torch.as_tensor(state,dtype=torch.float,device=device)
            episode_rewards=0
            terminated=False


            while (not terminated and episode_rewards<self.reward_threshold):
                if is_training and random.random()<epsilon:
                    action=env.action_space.sample()
                    action=torch.as_tensor(action,dtype=torch.long,device=device)
                else:
                    with torch.no_grad():
                        action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()    
               

                next_state,reward,terminated, _,_ = env.step(action.item())

                next_state=torch.as_tensor(next_state,dtype=torch.float,device=device)
                reward=torch.as_tensor(reward,dtype=torch.float,device=device)

                if is_training:
                    memory.append((state,action,next_state,reward,terminated))
                    steps+=1

                state=next_state
                episode_rewards+=reward.item()
            if is_training:
                if episode_rewards>best_reward:
                    best_reward=episode_rewards
                    log_msg=f"reward={episode_rewards} for episode ={e+1}"

                    with open(self.LOG_FILE,'a') as f:
                        f.write(log_msg +"\n")
                    torch.save(policy_dqn.state_dict(),self.MODEL_FILE)  


                epsilon=max(epsilon*self.epsilon_decay,self.epsilon_min)
            if is_training and len(memory)>self.mini_batch_size:
                mini_batch=memory.sample(self.mini_batch_size)

                self.optimize(mini_batch,policy_dqn,target_dqn)
                if steps>self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps=0        
            if is_training:
                print(f"Episode :{e+1} reward :{episode_rewards} and Epsilon {epsilon}")
            else:
             print(f"Episode :{e+1} reward :{episode_rewards}")
 


    def optimize(self,mini_batch,policy_dqn,target_dqn):
        states,actions,next_states,rewards,terminations=zip(*mini_batch)
        states=torch.stack(states)
        actions=torch.stack(actions)
        next_states=torch.stack(next_states)
        rewards=torch.stack(rewards)
        terminations=torch.tensor(terminations).float().to(device)
        with torch.no_grad():
            target_q=rewards+(1-terminations)*self.gamma*target_dqn(next_states).max(dim=1)[0]

        current_q=policy_dqn(states).gather(dim=1,index=actions.unsqueeze(dim=1)).squeeze()

        loss=self.loss_fn(current_q,target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()    

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Train Or Test Model")
    parser.add_argument('hyperparameters',help='')
    parser.add_argument("--train",help='Training mode',action='store_true')    
    args=parser.parse_args()
    dql=Agent(params_set=args.hyperparameters)
    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False,render=True)        

            


