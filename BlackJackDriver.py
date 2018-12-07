import gym
import sys
import operator
import pickle
sys.path.append('/mnt/c/Users/Nevaeh//Desktop/CS425-HW5/')
import QLearningAgents
from DeepRLAgent import DictQLearningAgent

# testing
env = gym.make('Blackjack-v0')
#See details about the game rules here, but not necessary for your agent -- it will learn the rules by experimentation!
#Environment definition: https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
#actions, observations described in detail above
#so your policy network needs to learn to predict one of these actions based on the observation.


def train():
    agent = DictQLearningAgent(action_space=env.action_space, exploration_rate=0.5, learning_rate=0.1, discount=0.2,
                               exploration_decay_rate=0.95)
    observation = env.reset()
    average_rewards = []
    num_rounds = 1000
    i_episode = 100

    for i in range(1, i_episode+1):
        round = 1
        eachEpisodeReward = 0
        while round <= num_rounds:
            total_rewards = 0
            while True:
                action = agent.act(observation)
                next_observation, reward, done, info = env.step(action)
                agent.update(observation, action, next_observation, reward)
                total_rewards += reward
                observation = next_observation
                if done:
                    observation = env.reset()
                    round += 1
                    eachEpisodeReward += total_rewards
                    break
            eachEpisodeReward += total_rewards
        print (
            "After {} episodes, Average payout after {} rounds after training for episodes is {}".format(i, num_rounds,
                                                                                                         eachEpisodeReward / num_rounds))
        agent.reset()

    agent.save("QTable.pickle")
    env.close()

def test():
    agent = DictQLearningAgent(action_space=env.action_space, exploration_rate=0.5, learning_rate=0.1, discount=0.2,
                               exploration_decay_rate=0.95)
    agent.load("QTable.pickle")
    total = 0
    for i_episode in range(1000):
        total_rewards = 0
        observation = env.reset()
        while True:
            t=0
            #env.render()  #comment out for faster training!
            # print(observation)
            action = agent.act(observation) #random action, use your own action policy here
            observation, reward, done, info = env.step(action)
            total_rewards+=reward
            t+=1
            if done:
                print("Episode finished after {} timesteps %d with reward %d ", t, total_rewards)
                break
        total += total_rewards
    print (total)

    env.close()


program_name = sys.argv[0]
command = sys.argv[1]

if command == "train":
    train()
elif command == "test":
    test()
else:
    print ("Please use command line: -train / -test")