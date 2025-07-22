import logging
import torch
from torch.utils.tensorboard import SummaryWriter
import functools
import json
import os
import time
import warnings
import random
from collections import deque
from tqdm import tqdm
import wandb

from rdkit.Chem import Draw
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED


from mol_dqn.chemgraph.dqn import environment as molecules_mdp
from mol_dqn.chemgraph.dqn.py import molecules
from mol_dqn.utils.schedular import PiecewiseSchedule, LinearSchedule
from mol_dqn.utils.buffer import ReplayBuffer, PrioritizedReplayBuffer
from mol_dqn.utils.utils import get_fingerprint


def run_training(hparams, environment, dqn, model_dir,use_wandb=False):

    """Runs the training procedure.
  
    Args:
        hparams: The hyper parameters of the model.
        environment: molecules.Molecule. The environment to run on.
        dqn: An instance of the DeepQNetwork class.
        model_dir: String. Directory to save model and logs.

    Returns:
        None
    """

    summary_writer = SummaryWriter(model_dir)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dqn.to(device)

    exploration = PiecewiseSchedule(
        [(0, 1.0), (int(hparams.num_episodes / 2), 0.1),
         (hparams.num_episodes, 0.01)],
        outside_value=0.01)
    
    if hparams.prioritized:
        memory = PrioritizedReplayBuffer(hparams.replay_buffer_size, hparams.prioritized_alpha)
        beta_schedule = LinearSchedule(
            hparams.num_episodes, initial_p=hparams.prioritized_beta, final_p=1.0)
    else:
        memory = ReplayBuffer(hparams.replay_buffer_size)
        beta_schedule = None
    

    global_step = 0

    #print(f"sampling之前跑通")

    #assert False,"检查点"

    for episode in tqdm(range(hparams.num_episodes), desc="Episodes"):
        #print(f"Episode {episode + 1}/{hparams.num_episodes}")
        global_step = _episode(
            environment=environment,
            dqn=dqn,
            memory=memory,
            episode=episode,
            global_step=global_step,
            hparams=hparams,
            summary_writer=summary_writer,
            exploration=exploration,
            beta_schedule=beta_schedule,
            model_dir=model_dir,
            save_video = False,
            use_wandb=use_wandb)
        
        if (episode + 1) % hparams.update_frequency == 0:
            dqn.update_target_network() 

            if use_wandb:
                wandb.log({
                    "target_network_updates": episode + 1,
                    "episode": episode
               })
        
        if (episode + 1) % hparams.save_frequency == 0:
            """
            checkpoint = {
                'episode': episode + 1,
                'model_state_dict': dqn.state_dict(),
                'global_step': global_step
            }
            checkpoint_path = os.path.join(model_dir, f'ckpt_episode_{episode+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            logging.info(f'Model saved to {checkpoint_path}')
            """
            checkpoint_path = os.path.join(model_dir, f'ckpt_episode_{episode+1}.pt')
            dqn.save_checkpoint(checkpoint_path)
    
    # 关闭 summary writer
    summary_writer.close()

def _episode(environment, dqn, memory, episode, global_step, hparams,
             summary_writer, exploration, beta_schedule, model_dir,save_video = False,use_wandb=False):
    """Runs a single episode.

    Args:
        environment: molecules.Molecule; the environment to run on.
        dqn: DeepQNetwork used for estimating rewards.
        memory: ReplayBuffer used to store observations and rewards.
        episode: Integer episode number.
        global_step: Integer global step.
        hparams: HParams.
        summary_writer: SummaryWriter used for writing logs.
        exploration: Schedule used for exploration in the environment.
        beta_schedule: Schedule used for prioritized replay buffers.
        model_dir: String. Directory for saving visualizations.

    Returns:
        Updated global_step.
    """
    
    episode_start_time = time.time()
    environment.initialize()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if save_video:
        if environment._state:
            try:
                mol = Chem.MolFromSmiles(environment._state)
                if mol is not None:
                    output_dir = os.path.join(model_dir, f'episode_{episode}_visualization')
                    os.makedirs(output_dir, exist_ok=True)
                    
                    img = Draw.MolToImage(mol, size=(400, 300))
                    img_path = os.path.join(output_dir, f'step_000.png')
                    img.save(img_path)
                    print(f"Episode {episode}, Step 0: {environment._state} (Initial state)")
            except Exception as e:
                print(f"Error saving initial molecule image: {e}")
    
    if hparams.num_bootstrap_heads:
        head = np.random.randint(hparams.num_bootstrap_heads)
    
    else:
        head = 0
    
    for step in range(hparams.max_steps_per_episode):
        result = _step(
            environment=environment,
            dqn=dqn,
            memory=memory,
            episode=episode,
            hparams=hparams,
            exploration=exploration,
            head=head,
            model_dir=model_dir,
            save_video = save_video)

        #print("跑通了_step")
        #print(f"Episode {episode}, Step {step}: {result.state}, Reward: {result.reward:.4f}")
        
        #assert False,"检查点"
    
        if step == hparams.max_steps_per_episode - 1:
            reward_value = result.reward if isinstance(result.reward, (int, float)) else float(result.reward)
            summary_writer.add_scalar('episode/reward', reward_value, global_step)
            summary_writer.add_text('episode/smiles', str(result.state), global_step)

            if use_wandb:
                wandb.log({
                    "episode/reward": reward_value,
                    "episode/smiles": str(result.state),
                    "episode/steps": step + 1,
                    "episode": episode,
                    "global_step": global_step
                })


            logging.info('Episode %d/%d took %gs', episode + 1, hparams.num_episodes,
                        time.time() - episode_start_time)
            #print(f"Episode {episode + 1}/{hparams.num_episodes} took {time.time() - episode_start_time:.2f} seconds")
            logging.info('SMILES: %s\n', result.state)
            logging.info('The reward is: %s', str(result.reward))

            if save_video:
                _create_video_from_images(episode, model_dir)
        
        if (episode > min(50, hparams.num_episodes / 10)) and (
            global_step % hparams.learning_frequency == 0):
            #print(f"Episode {episode}, Step {step}: 收集到足够的经验，开始训练 DQN")
            #print(f"hparams.learning_frequency: {hparams.learning_frequency}, global_step: {global_step},hparams.num_episodes: {hparams.num_episodes}")


            if len(memory) < hparams.batch_size:
                global_step += 1
                continue


            if hparams.prioritized:
                (state_t, _, reward_t, state_tp1, done_mask, weight, indices) = memory.sample(
                    hparams.batch_size, beta=beta_schedule.value(episode))
            else:
                (state_t, _, reward_t, state_tp1, done_mask) = memory.sample(hparams.batch_size)
                weight = np.ones([reward_t.shape[0]])
                indices = None
            

            if reward_t.ndim == 1:
                reward_t = np.expand_dims(reward_t, axis=1)
            

            state_t_tensor = torch.FloatTensor(state_t).to(device)
            reward_t_tensor = torch.FloatTensor(reward_t).to(device)
            #state_tp1_tensor = torch.FloatTensor(state_tp1).to(device)
            done_tensor = torch.FloatTensor(np.expand_dims(done_mask, axis=1)).to(device)
            weight_tensor = torch.FloatTensor(np.expand_dims(weight, axis=1)).to(device)


            #print(f"训练前跑通,检查数据")
            #assert False,"检查点"


            loss, td_error = dqn.train(
                states=state_t_tensor,
                rewards=reward_t_tensor,
                next_states=state_tp1,
                done=done_tensor,
                weight=weight_tensor)

            if use_wandb:
                wandb.log({
                    "training/loss": loss,
                    "training/td_error_mean": td_error.mean().item(),
                    "training/learning_rate": dqn.optimizer.param_groups[0]['lr'],
                    "training/epsilon": dqn.epsilon,
                    "episode": episode,
                    "global_step": global_step
                })
            

            summary_writer.add_scalar('training/loss', loss, global_step)
            logging.info('Current Loss: %.4f', loss)

            #print(f"Episode {episode}, Step {step}: Loss: {loss:.4f}, TD Error: {td_error.mean().item():.4f}")
            #print(f"完成一次更新")

            #assert False,"检查点"


            if hparams.prioritized and indices is not None:
                # 将 PyTorch tensor 转回 numpy
                td_error_np = td_error.detach().cpu().numpy()
                memory.update_priorities(
                    indices,
                    np.abs(np.squeeze(td_error_np) + hparams.prioritized_epsilon).tolist())
                
        global_step += 1
    
    return global_step
            

def _step(environment, dqn, memory, episode, hparams, exploration, head, model_dir,save_video = False):
    """Runs a single step within an episode.

    Args:
        environment: molecules.Molecule; the environment to run on.
        dqn: DeepQNetwork used for estimating rewards.
        memory: ReplayBuffer used to store observations and rewards.
        episode: Integer episode number.
        hparams: HParams.
        exploration: Schedule used for exploration in the environment.
        head: Integer index of the DeepQNetwork head to use.
        model_dir: String. Directory for saving visualizations.

    Returns:
        molecules.Result object containing the result of the step.
    """

    steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
    valid_actions = list(environment.get_valid_actions()) #此时是在上一个state下可能的所有valid_actions


    observations = np.vstack([
        np.append(get_fingerprint(act, hparams), steps_left)
        for act in valid_actions
    ])


    action = valid_actions[dqn.get_action(
        observations, head=head, update_epsilon=exploration.value(episode))] #在当前步里DQN产生的action，也是next_state,最终计算当前Q(s,a)的时候用的就是这个action
    
    action_t_fingerprint = np.append(
        get_fingerprint(action, hparams), steps_left) #current_state
    
    result = environment.step(action) #在这一步之后环境里的state就更改了，编程action__fingerprints，这里的reward是已经变成action_fingerprints之后的reward

    if save_video and result.state:
        try:
            mol = Chem.MolFromSmiles(result.state)
            if mol is not None:
                output_dir = os.path.join(model_dir, f'episode_{episode}_visualization')
                os.makedirs(output_dir, exist_ok=True)
                
                img = Draw.MolToImage(mol, size=(400, 300))
                step_num = hparams.max_steps_per_episode - steps_left + 1
                img_path = os.path.join(output_dir, f'step_{step_num:03d}.png')
                img.save(img_path)
                
                print(f"Episode {episode}, Step {step_num}: {result.state}, Reward: {result.reward:.4f}")
        except Exception as e:
            print(f"Error saving molecule image: {e}")
    
    steps_left = hparams.max_steps_per_episode - environment.num_steps_taken


    action_fingerprints = np.vstack([
      np.append(get_fingerprint(act, hparams), steps_left)
      for act in environment.get_valid_actions()
    ]) #这里的actions是在当前action_t_fingerprins下的所有可能动作，其实也就是Q更新的时候的s‘下所有可能的action

    memory.add(
      obs_t=action_t_fingerprint,
      action=0,
      reward=result.reward,
      obs_tp1=action_fingerprints,
      done=float(result.terminated))
    
    return result




    








def _create_video_from_images(episode, model_dir):
    """从图片创建视频 - 替代原来使用 FLAGS.model_dir"""
    try:
        import imageio
        
        output_dir = os.path.join(model_dir, f'episode_{episode}_visualization')
        if not os.path.exists(output_dir):
            return
            
        # 获取所有图片文件
        image_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
        if len(image_files) < 2:
            return
            
        # 读取图片并创建视频
        images = []
        for img_file in image_files:
            img_path = os.path.join(output_dir, img_file)
            img = imageio.imread(img_path)
            # 每张图片显示2秒
            for _ in range(2):
                images.append(img)
        
        # 生成视频文件
        video_path = os.path.join(output_dir, f'episode_{episode}_evolution.mp4')
        imageio.mimsave(video_path, images, fps=1)
        print(f"Video saved: {video_path}")
        
        # 也生成GIF
        gif_path = os.path.join(output_dir, f'episode_{episode}_evolution.gif')
        imageio.mimsave(gif_path, images[::2], fps=1)  # 降低帧率
        print(f"GIF saved: {gif_path}")
        
    except ImportError:
        print("imageio not installed. Install with: pip install imageio[ffmpeg]")
    except Exception as e:
        print(f"Error creating video: {e}")