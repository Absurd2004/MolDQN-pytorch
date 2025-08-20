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


import io
from PIL import Image, ImageDraw, ImageFont


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

    best_reward = float('-inf')
    run_training.best_qed = 0.0
    best_model_path = os.path.join(model_dir, 'best_model.pt')
    eval_frequency = max(1, hparams.eval_frequency)

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
        if ((episode + 1) % eval_frequency == 0 and episode > min(50, hparams.num_episodes / 10)) or episode == hparams.num_episodes - 1:
            avg_final_reward = _evaluate_model(environment, dqn, hparams, num_eval_episodes=1)

            summary_writer.add_scalar('evaluation/average_final_reward', avg_final_reward, global_step)
            logging.info(f'Evaluation after episode {episode + 1}: Average Final Reward (QED): {avg_final_reward:.4f}')

            if avg_final_reward > best_reward:
                best_reward = avg_final_reward
                dqn.save_checkpoint(best_model_path)
                logging.info(f'New best model saved with final average reward(QED): {best_reward:.4f}')



            if use_wandb:
                wandb.log({
                    "evaluation/average_final_reward": avg_final_reward,
                    "evaluation/best_reward": best_reward,
                    "episode": episode,
                    "global_step": global_step
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

            from rdkit import Chem
            from rdkit.Chem import QED

            current_qed = 0.0
            if result.state:
                mol = Chem.MolFromSmiles(result.state)
                if mol is not None:
                    current_qed = QED.qed(mol)
            
            if current_qed > run_training.best_qed:
                old_best = run_training.best_qed
                run_training.best_qed = current_qed
                logging.info(f'Single Agent: New best QED! {old_best:.4f} -> {run_training.best_qed:.4f} (Episode {episode})')
            
            summary_writer.add_scalar('episode/reward', reward_value, global_step)
            summary_writer.add_text('episode/smiles', str(result.state), global_step)
            # === 新增：记录QED指标 ===
            summary_writer.add_scalar('episode/qed', current_qed, global_step)
            summary_writer.add_scalar('episode/best_qed', run_training.best_qed, global_step)

            if use_wandb:
                wandb.log({
                    "episode/reward": reward_value,
                    "episode/smiles": str(result.state),
                    "episode/steps": step + 1,
                    "episode": episode,
                    "global_step": global_step,
                    # === 新增：wandb记录QED ===
                    "episode/qed": current_qed,
                    "episode/best_qed": run_training.best_qed,
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


            loss, td_error = dqn.train_step(
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
    
    result = environment.step(action) #在这一步之后环境里的state就更改了，变成action__fingerprints，这里的reward是已经变成action_fingerprint之后的reward

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




def _evaluate_model(environment, dqn, hparams, num_eval_episodes=3):
    """评估模型性能（不使用探索策略）
    
    Args:
        environment: 环境实例
        dqn: DQN智能体
        hparams: 超参数
        num_eval_episodes: 评估的episode数量
    
    Returns:
        float: 平均奖励
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dqn.eval()

    final_rewards = []

    for eval_episode in range(num_eval_episodes):
        environment.initialize()
        episode_reward = 0.0

        for step in range(hparams.max_steps_per_episode):
            valid_actions = list(environment.get_valid_actions())
            if not valid_actions:
                logging.warning(f"Evaluation Episode {eval_episode}, Step {step}: No valid actions available")
                break
            steps_left = hparams.max_steps_per_episode - environment.num_steps_taken

            observations = np.vstack([
                np.append(get_fingerprint(act, hparams), steps_left)
                for act in valid_actions
            ])

            head = 0 if not hparams.num_bootstrap_heads else np.random.randint(hparams.num_bootstrap_heads)
            action_idx = dqn.get_action(observations, head=head, stochastic=False)
            action = valid_actions[action_idx]

            result = environment.step(action)
            #episode_reward += result.reward

            final_reward = result.reward


            if result.terminated:
                break
        
        final_rewards.append(final_reward)
    dqn.train() #恢复训练模式
    return np.mean(final_rewards)








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


def draw_molecule_with_qed(smiles, step, qed_value, reward_value, img_size=(500, 400)):
    """绘制带有QED标注的分子结构图"""
    if not smiles or smiles == "":
        # 空分子的情况
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        text = f"Step {step}: Empty Molecule\nQED: 0.0\nReward: {reward_value:.4f}"
        draw.text((10, 10), text, fill='black', font=font)
        return img
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 无效分子的情况
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        text = f"Step {step}: Invalid Molecule\nSMILES: {smiles}\nQED: 0.0\nReward: {reward_value:.4f}"
        draw.text((10, 10), text, fill='red', font=font)
        return img
    
    # 绘制分子结构
    mol_img = Draw.MolToImage(mol, size=(img_size[0], img_size[1]-80))
    
    # 创建最终图像
    final_img = Image.new('RGB', img_size, 'white')
    final_img.paste(mol_img, (0, 0))
    
    # 添加文字标注
    draw = ImageDraw.Draw(final_img)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 文字信息
    text_y = img_size[1] - 75
    draw.text((10, text_y), f"Step {step}", fill='black', font=font_large)
    draw.text((10, text_y + 25), f"QED: {qed_value:.4f}", fill='blue', font=font_small)
    #draw.text((10, text_y + 45), f"Reward: {reward_value:.4f}", fill='green', font=font_small)
    
    # 在右边显示SMILES（如果不太长）
    #if len(smiles) < 30:
        #draw.text((200, text_y + 35), f"SMILES: {smiles}", fill='black', font=font_small)
    
    return final_img


def run_display(hparams, environment, dqn, model_dir, checkpoint_path, 
                num_episodes=5, reward_name = "QED",save_video=True,protect_initial = True):
    """显示模式：加载训练好的模型，运行几个episode并保存视频"""
    
    # 加载训练好的模型
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    dqn.load_checkpoint(checkpoint_path)
    #dqn.epsilon = 1.0 
    dqn.eval()  # 设置为评估模式

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dqn.to(device)
    
    # 创建保存目录
    display_dir = os.path.join(model_dir, "display_results")
    #把现在时间也加到display_dir中
    display_dir += f"{reward_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    if not os.path.exists(display_dir):
        os.makedirs(display_dir, exist_ok=True)
    
    for episode in tqdm(range(num_episodes), desc="Display Episodes"):
        episode_dir = os.path.join(display_dir, f"episode_{episode}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # 运行单个episode
        _display_episode(environment, dqn, hparams, episode, episode_dir, reward_name,save_video,protect_initial)
        
        # 创建视频
        if save_video:
            _create_video_from_images_with_qed(episode_dir)
    
    logging.info("Display completed!")


def _display_episode(environment, dqn, hparams, episode, episode_dir, reward_name = "QED",save_video=True,protect_initial=True):
    """运行单个显示episode"""
    # 初始化环境
    environment.initialize()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 保存QED数据
    reward_data = []
    step_data = []

    if environment.state_mol:
        from mol_dqn.chemgraph.dqn.environment import get_protection_info
        protection_info = get_protection_info(environment.state_mol)
        if reward_name == "QED":
            initial_reward = QED.qed(environment.state_mol)
        elif reward_name == "LogP":
            initial_reward = molecules.penalized_logp(environment.state_mol)
        #initial_qed = QED.qed(environment.state_mol)
        
        print(f" 初始状态保护信息: {protection_info}")
        
        if save_video:
            if protect_initial:
                img = draw_molecule_with_protection(environment.state_mol, 0, initial_reward,reward_name=reward_name)
            else:
                img = draw_molecule_with_qed(environment.state_mol, 0, initial_reward, reward_name=reward_name)
            img.save(os.path.join(episode_dir, f"step_000.png"))
    


    for step in range(hparams.max_steps_per_episode):
        # 获取有效动作
        valid_actions = list(environment.get_valid_actions())
        if not valid_actions:
            logging.warning(f"Episode {episode}, Step {step}: No valid actions available")
            break

        steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
        observations = np.vstack([
            np.append(get_fingerprint(act, hparams), steps_left)
            for act in valid_actions
        ])


        head = 0 if not hparams.num_bootstrap_heads else np.random.randint(hparams.num_bootstrap_heads)
        #action_idx = dqn.get_action(observations, head=head, stochastic=False)
        action_idx = dqn.get_action(observations, head=head)
        action = valid_actions[action_idx]

        result = environment.step(action)

        if reward_name == "QED":
            current_reward = QED.qed(Chem.MolFromSmiles(result.state)) if result.state else 0.0
        elif reward_name == "LogP":
            current_reward = molecules.penalized_logp(Chem.MolFromSmiles(result.state)) if result.state else 0.0


        reward_data.append(current_reward)

        if environment.state_mol and protect_initial:
            from mol_dqn.chemgraph.dqn.environment import get_protection_info
            protection_info = get_protection_info(environment.state_mol)
            print(f" Step {step+1} 保护信息: {protection_info}")

        step_info = {
            'step': step + 1,
            'smiles': result.state,
            'reward': current_reward,
            'action': str(action),
            'terminated': result.terminated,
            'protection_info': protection_info if (environment.state_mol and protect_initial) else None
        }
        step_data.append(step_info)


        if save_video:
            if protect_initial:
                img = draw_molecule_with_protection(environment.state_mol, step + 1, current_reward,reward_name=reward_name)
            else:

                img = draw_molecule_with_qed(environment.state_mol, step + 1, current_reward, result.reward)
            img.save(os.path.join(episode_dir, f"step_{step+1:03d}.png"))
        
        if reward_name == "QED":
            logging.info(f"Episode {episode}, Step {step+1}: SMILES: {result.state}, "
                        f"QED: {current_reward:.4f}")
        elif reward_name == "LogP":
            logging.info(f"Episode {episode}, Step {step+1}: SMILES: {result.state}, "
                        f"LogP: {current_reward:.4f}")
        
        if result.terminated:
            logging.info(f"Episode {episode} terminated at step {step+1}")
            break
    if reward_name == "QED":
        qed_file = os.path.join(episode_dir, "qed_data.json")
    elif reward_name == "LogP":
        qed_file = os.path.join(episode_dir, "logp_data.json")
    
    with open(qed_file, 'w') as f:
        json.dump({
            'episode': episode,
            'steps': step_data,
            'reward_name': reward_name,
            'final_reward': reward_data[-1],
            'max_reward': max(reward_data),
            'reward_improvement': reward_data[-1] - reward_data[0],
            'protection_info': protection_info if (environment.state_mol and protect_initial) else None
        }, f, indent=2)
    
    if reward_name == "QED":
        logging.info(f"Episode {episode} completed. Final QED: {reward_data[-1]:.4f}, "
                    f"QED improvement: {reward_data[-1] - reward_data[0]:.4f}")
    elif reward_name == "LogP":
        logging.info(f"Episode {episode} completed. Final LogP: {reward_data[-1]:.4f}, "
                    f"LogP improvement: {reward_data[-1] - reward_data[0]:.4f}")
        
def _create_video_from_images_with_qed(episode_dir):
    """从图片创建视频（显示模式专用）"""
    try:
        import imageio
        
        # 获取所有图片文件
        image_files = sorted([f for f in os.listdir(episode_dir) if f.endswith('.png')])
        if len(image_files) < 2:
            return
        
        # 读取图片并创建视频
        images = []
        for img_file in image_files:
            img_path = os.path.join(episode_dir, img_file)
            img = imageio.imread(img_path)
            # 每张图片显示1.5秒
            for _ in range(3):
                images.append(img)
        
        # 生成MP4视频
        video_path = os.path.join(episode_dir, 'molecule_evolution.mp4')
        imageio.mimsave(video_path, images, fps=2)
        logging.info(f"Video saved: {video_path}")
        
        # 生成GIF
        gif_path = os.path.join(episode_dir, 'molecule_evolution.gif')
        imageio.mimsave(gif_path, images[::2], fps=1, duration=1.5)
        logging.info(f"GIF saved: {gif_path}")
        
    except ImportError:
        logging.warning("imageio not installed. Install with: pip install imageio[ffmpeg]")
    except Exception as e:
        logging.error(f"Error creating video: {e}")

def draw_molecule_with_protection(mol_obj, step, reward_value, reward_name = "QED",img_size=(600, 500)):
    """绘制带有保护原子高亮的分子结构图 - 直接使用Mol对象"""
    if mol_obj is None:
        # 空分子的情况
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        if reward_name == "QED":
            text = f"Step {step}: Empty Molecule\nQED: 0.0\n"
        elif reward_name == "LogP":
            text = f"Step {step}: Empty Molecule\nLogP: 0.0\n"
        
        
        draw.text((10, 10), text, fill='black', font=font)
        return img
    
    try:
        # 获取保护的原子和键
        protected_atoms = []
        protected_bonds = []
        
        for atom in mol_obj.GetAtoms():
            if atom.HasProp('_protected') and atom.GetBoolProp('_protected'):
                protected_atoms.append(atom.GetIdx())
        
        for bond in mol_obj.GetBonds():
            if bond.HasProp('_protected') and bond.GetBoolProp('_protected'):
                protected_bonds.append(bond.GetIdx())
        
        # 设置高亮颜色
        atom_colors = {}
        bond_colors = {}
        
        # 红色高亮受保护的原子
        for atom_idx in protected_atoms:
            atom_colors[atom_idx] = (1.0, 0.0, 0.0)  # 红色
        
        # 红色高亮受保护的键
        for bond_idx in protected_bonds:
            bond_colors[bond_idx] = (1.0, 0.0, 0.0)  # 红色
        

        from rdkit.Chem.Draw import rdMolDraw2D
        
        drawer = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1] - 100)
        
        # 根据是否有保护元素决定绘制方式
        if protected_atoms or protected_bonds:
            drawer.DrawMolecule(
                mol_obj,
                highlightAtoms=protected_atoms,
                highlightAtomColors=atom_colors,
                highlightBonds=protected_bonds,
                highlightBondColors=bond_colors
            )
        else:
            drawer.DrawMolecule(mol_obj)
        
        drawer.FinishDrawing()
        
        # 获取绘制的图像
        mol_img_data = drawer.GetDrawingText()
        mol_img = Image.open(io.BytesIO(mol_img_data))
        
    except Exception as e:
        # 如果高亮绘制失败，回退到普通绘制
        print(f"Warning: Failed to draw with protection highlighting, using fallback: {e}")
        mol_img = Draw.MolToImage(mol_obj, size=(img_size[0], img_size[1] - 100))
        protected_atoms = []
        protected_bonds = []
    
    # 创建最终图像
    final_img = Image.new('RGB', img_size, 'white')
    final_img.paste(mol_img, (0, 0))
    
    # 添加文字标注
    draw = ImageDraw.Draw(final_img)
    
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 文字信息
    text_y = img_size[1] - 95
    draw.text((10, text_y), f"Step {step}", fill='black', font=font_large)
    if reward_name == "QED":
        draw.text((10, text_y + 25), f"QED: {reward_value:.4f}", fill='blue', font=font_small)
    elif reward_name == "LogP":
        draw.text((10, text_y + 25), f"LogP: {reward_value:.4f}", fill='blue', font=font_small)
    draw.text((10, text_y + 45), f"Protected Atoms: {len(protected_atoms)}", fill='red', font=font_small)
    draw.text((10, text_y + 65), f"Protected Bonds: {len(protected_bonds)}", fill='red', font=font_small)
    
    # 在右边显示保护状态
    if protected_atoms or protected_bonds:
        draw.text((300, text_y + 25), f"red_atoms = Protected", fill='red', font=font_small)
    else:
        draw.text((300, text_y + 25), f"No Protection", fill='gray', font=font_small)
    
    return final_img