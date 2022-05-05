#os.environ["SDL_VIDEODRIVER"] = "dummy"

from collections import namedtuple

import numpy as np

from snake import CustomSnake as Snake
from snake_environment import SnakeMode

from DQN import DQN
from plotter import plot


# Are we rendering or not
RENDER = False

# Number of steps to train
NUM_STEPS = 50000

# The step limit per episode, since we do not want infinite loops inside episodes
MAX_STEPS_PER_EPISODE = 200

# Here, we are creating a data structure to store our transitions
# This is just a convenient way to store
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])

gamemode = SnakeMode.TRON
# Here, we are creating the environment with our predefined observation space
env = Snake(mode=gamemode, render=RENDER)

# Observation and action space
obs_space = env.observation_space
number_of_states = env.observation_space.shape[0]

action_space = env.action_space
number_of_actions = env.action_space.n
print("The observation space: {}".format(obs_space))
# Output: The observation space: Box(n,)
print("The action space: {}".format(action_space))
# Output: The action space: Discrete(m)
MAX_CAPACITY = 15000
BATCH_SIZE = 256
LEARNING_RATE = 0.5*1e-3
HYPERPARAMETERS = {
    "number_of_states": number_of_states,
    "number_of_actions": number_of_actions,
    "number_of_steps": NUM_STEPS,
    "replay_buffer_capacity": MAX_CAPACITY,
    "learning_rate":LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "gamma": 0.995,
    "epsilon_annealing": "linear"
}


def main():

    agent = DQN(HYPERPARAMETERS)

    episodes = 0
    steps = 0
    episode_rewards = []
    episode_losses = []
    
    while steps < NUM_STEPS:
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        while True:
            steps += 1
            # The select action method inside DQN will select based on policy or random, depending on the epsilon value
            action = agent.select_action(state)

            # Here, we will step the environment with the action
            # Next_state: the state after the action is taken
            # Reward: The reward agent will get. It is generally
            # 1 if the agent wins the game, -1 if the agent loses, 0 otherwise
            # You can add intermediate rewards other than win-lose conditions
            # Done: is the game finished
            # Info: Further info you can get from the environment, you can ignore this part
            next_state, reward, done, info = env.step(action)

            # Render each frame?
            if RENDER:
                env.render()

            transition = Transition(state, action, reward, next_state)
            agent.store_transition(transition)

            episode_reward += reward

            if done or env.elapsed_steps >= MAX_STEPS_PER_EPISODE:
                episodes += 1


                # We do not want the env to run indefinitely
                # When a done condition is met, we finish
                # We want to update the DQN
                episode_loss = agent.mean_loos_for_batch
                episode_losses.append(episode_loss)
                episode_rewards.append(episode_reward)
                '''
                Uncomment for the plot operations.
                if steps > MAX_CAPACITY:
                    agent.writer.add_scalar('loss/value_loss', episode_loss, episodes)
                    agent.writer.add_scalar('reward/value_reward', episode_reward, episodes)
                '''
                agent.update()
                break

            if steps % 100 == 0:
                print("step=", steps)
                if episodes > 10:
                    print("Step: {}, Epsilon: {}, Mean Reward for last 10: {} , Mean Loss for last 10: {} , ".format(steps, agent.e, np.average(episode_rewards[:-10],),np.average(episode_losses[:-10],)))
                    '''
                    Uncomment for the plot operations.
                    if steps > MAX_CAPACITY:
                        agent.writer.add_scalar('loss/mean_value_loss', np.average(episode_losses[:-10], ), episodes)
                        agent.writer.add_scalar('reward/mean_value_reward', np.average(episode_rewards[:-10], ), episodes)
                    '''
            state = next_state

    agent.save(str(gamemode))

    # Delete the current pygame instance
    env.quit()

    print(episode_rewards)

    # Create a new pygame instance with render enabled
    render_env = Snake(mode=gamemode, render=True)
    state = render_env.reset()
    while True:
        action = agent.select_action(state)
        next_state, reward, done, info = render_env.step(action)
        render_env.render()
        if done:
            state = render_env.reset()
        else:
            state = next_state


if __name__ == '__main__':
    main()


