import os
import time
import logging
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
import pickle

from mol_dqn.chemgraph.sampler.opp_sampler import OpponentHistory, OpponentSampler
from mol_dqn.chemgraph.sampler.mol_sampler import MolecularSampler
from mol_dqn.chemgraph.dqn.multi_agent_env import DualMoleculeEnv
from mol_dqn.utils.buffer import ReplayBuffer, PrioritizedReplayBuffer
from mol_dqn.utils.schedular import PiecewiseSchedule, LinearSchedule
from mol_dqn.utils.utils import get_fingerprint

class AdversarialTrainer:
    """对抗训练器 - 管理两个智能体的竞争训练"""
    def __init__(self, hparams, env_class, env_kwargs, agent_A, agent_B, model_dir):
        """
        Args:
            hparams: 超参数配置（已通过 extend_hparams_for_adversarial 扩展）
            env_class: 环境类（如 QEDRewardMolecule）
            env_kwargs: 环境初始化参数
            agent_A, agent_B: 两个 DoubleDQNAgent 实例
            model_dir: 模型保存目录
        """

        self.hparams = hparams
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.model_dir = model_dir

        self.model_dir_A = os.path.join(model_dir, 'agent_A')
        self.model_dir_B = os.path.join(model_dir, 'agent_B')

        os.makedirs(self.model_dir_A, exist_ok=True)
        os.makedirs(self.model_dir_B, exist_ok=True)
        
        logging.info(f"Agent A models will be saved to: {self.model_dir_A}")
        logging.info(f"Agent B models will be saved to: {self.model_dir_B}")

        self.best_qed_A = 0.0
        self.best_qed_B = 0.0



        self.debug = False

        # 创建两个对抗环境
        self.dual_env_A = DualMoleculeEnv(env_class, env_kwargs, hparams)  # A为主，B为对手
        self.dual_env_B = DualMoleculeEnv(env_class, env_kwargs, hparams)  # B为主，A为对手


        self.agent_A = agent_A
        self.agent_B = agent_B


        self.history_A = OpponentHistory()
        self.history_B = OpponentHistory()

        self.win_stats_A = {'win': 0, 'loss': 0, 'draw': 0, 'total': 0}
        self.win_stats_B = {'win': 0, 'loss': 0, 'draw': 0, 'total': 0}

        if hparams.prioritized:
            self.buffer_A = PrioritizedReplayBuffer(hparams.replay_buffer_size, hparams.prioritized_alpha)
            self.buffer_B = PrioritizedReplayBuffer(hparams.replay_buffer_size, hparams.prioritized_alpha)
            self.beta_schedule = LinearSchedule(
                hparams.num_epoch, initial_p=hparams.prioritized_beta, final_p=1.0)  # 改为 num_epoch
        else:
            self.buffer_A = ReplayBuffer(hparams.replay_buffer_size)
            self.buffer_B = ReplayBuffer(hparams.replay_buffer_size)
            self.beta_schedule = None
        
        self.exploration = PiecewiseSchedule(
            [(0, 1.0), (int(hparams.num_epoch / 2), 0.2), (hparams.num_epoch, 0.05)],  # 改为 num_epoch
            outside_value=0.05
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.global_step_A = 0
        self.global_step_B = 0

        self.summary_writer = SummaryWriter(model_dir)

        self.best_reward_A = float('-inf')
        self.best_reward_B = float('-inf')

        #assert False,"check init"
    
    def _save_checkpoint(self, agent, epoch, agent_name):
        """保存智能体 checkpoint 到对应的子目录"""
        if agent_name == 'A':
            ckpt_path = os.path.join(self.model_dir_A, f'epoch_{epoch}.pt')
        else:  # agent_name == 'B'
            ckpt_path = os.path.join(self.model_dir_B, f'epoch_{epoch}.pt')
        
        agent.save_checkpoint(ckpt_path)
        logging.info(f'Saved Agent {agent_name} checkpoint: {ckpt_path}')
        return ckpt_path

    def run_training(self, use_wandb=False):
        """运行对抗训练主循环"""
        logging.info("Starting adversarial training...")

        self.agent_A.to(self.device)
        self.agent_B.to(self.device)

        self.best_win_rate_A = 0.0
        self.best_win_rate_B = 0.0

        for epoch in tqdm(range(self.hparams.num_epoch), desc="Adversarial Epochs"):

            self.epoch = epoch
            logging.info(f'=== Epoch {epoch + 1}/{self.hparams.num_epoch} ===')
            logging.info(f"Training Agent A (epoch {epoch + 1})")

            self._train_agent_epoch(
                epoch=epoch,
                main_agent=self.agent_A,
                main_buffer=self.buffer_A,
                opponent_history=self.history_B,
                opponent_template=self.agent_B,
                dual_env=self.dual_env_A,
                agent_name='A',
                global_step_attr='global_step_A',
                use_wandb=use_wandb
            )

            ckpt_path_A = self._save_checkpoint(self.agent_A, epoch, 'A')
            self.history_A.register(epoch, ckpt_path_A)

            logging.info(f"Training Agent B (epoch {epoch + 1})")

            self._train_agent_epoch(
                epoch=epoch,
                main_agent=self.agent_B,
                main_buffer=self.buffer_B,
                opponent_history=self.history_A,
                opponent_template=self.agent_A,
                dual_env=self.dual_env_B,
                agent_name='B',
                global_step_attr='global_step_B',
                use_wandb=use_wandb
            )

            ckpt_path_B = self._save_checkpoint(self.agent_B, epoch, 'B')
            self.history_B.register(epoch, ckpt_path_B)




            if (epoch + 1) % self.hparams.update_frequency == 0:
                self.agent_A.update_target_network()
                self.agent_B.update_target_network()
                logging.info("Updated target networks for both agents")

                if use_wandb:
                    wandb.log({
                        "target_network_updates": epoch + 1,
                        "epoch": epoch
                    })
                
            #print(f"eval_frequency: {getattr(self.hparams, 'eval_frequency', 10)}")
            
            if ((epoch + 1) % getattr(self.hparams, 'eval_frequency', 10) == 0 and 
                epoch > min(10, self.hparams.num_epoch / 10)) or epoch == self.hparams.num_epoch - 1:  # 改为 num_epoch
                self._evaluate_agents(epoch, use_wandb)
                #assert False,"Evaluation finished, check result"
            print(f"=== Epoch {epoch + 1} completed ===")
            
            #assert False,"train first epoch finished"

        self.summary_writer.close()
        logging.info("Adversarial training completed!")


    def _train_agent_epoch(self, epoch, main_agent, main_buffer, opponent_history, 
                          opponent_template, dual_env, agent_name, global_step_attr, use_wandb=False):
        """训练单个智能体一个 epoch"""

        warmup_epochs = getattr(self.hparams, 'sparse_reward_warmup_epochs', 0)
    
        if epoch < warmup_epochs:
            sparse_weight = 0.0  # warm-up阶段
        elif self.hparams.use_sparse_reward and self.hparams.sparse_reward_annealing:
            termination_epoch = getattr(self.hparams, 'sparse_reward_termination_epoch', 1000)
            adjusted_epoch = epoch - warmup_epochs
            adjusted_total_epochs = termination_epoch - warmup_epochs
            
            if adjusted_total_epochs <= 0:
                sparse_weight = 1.0
            else:
                alpha = max((adjusted_total_epochs - adjusted_epoch) / adjusted_total_epochs, 0)
                sparse_weight = 1 - alpha
        else:
            sparse_weight = 1.0 if self.hparams.use_sparse_reward else 0.0
            
        self.summary_writer.add_scalar(f'training/{agent_name}_sparse_weight', sparse_weight, epoch)

        epoch_stats = {'win': 0, 'loss': 0, 'draw': 0, 'total': 0}


        opponent_sampler = OpponentSampler(opponent_history, self.hparams.adversarial_delta)
        if epoch == 0:
            # epoch=0时，没有历史ckpt，使用当前初始化的对手
            opponent_ckpt_path = None
            logging.info(f"Agent {agent_name}: Epoch 0, using randomly initialized opponent")
        else:
            # epoch>0时，正常采样历史ckpt
            opponent_ckpt_path = opponent_sampler.sample(epoch)
            #print(f"Agent {agent_name}: Sampled opponent checkpoint for epoch {epoch}: {opponent_ckpt_path}")
            logging.info(f"Agent {agent_name}: Sampled opponent checkpoint for epoch {epoch}: {opponent_ckpt_path}")

        opponent_agent = MolecularSampler(opponent_template, self.device)
        if opponent_ckpt_path:
            try:
                opponent_agent.load_checkpoint(opponent_ckpt_path)
                logging.info(f"Agent {agent_name}: Loaded opponent from {opponent_ckpt_path}")
            except (FileNotFoundError, pickle.UnpicklingError, Exception) as e:
                logging.warning(f"Agent {agent_name}: Failed to load opponent ckpt {opponent_ckpt_path}: {e}")
                logging.info(f"Agent {agent_name}: Using randomly initialized opponent")
        else:
            logging.info(f"Agent {agent_name}: Using randomly initialized opponent (no ckpt specified)")
        
        #assert False,"check initialization"
        

        for episode_in_epoch in range(self.hparams.episodes_per_epoch):
            global_episode = epoch * self.hparams.episodes_per_epoch + episode_in_epoch

            #print(f"Agent {agent_name}: Epoch {epoch + 1}, Episode {episode_in_epoch + 1}/{self.hparams.episodes_per_epoch}, Global Episode {global_episode + 1}")


            current_global_step = self._run_adversarial_episode(
                dual_env=dual_env,
                main_agent=main_agent,
                opponent_agent=opponent_agent,
                main_buffer=main_buffer,
                episode=global_episode,
                global_step=getattr(self, global_step_attr),
                agent_name=agent_name,
                epoch_stats=epoch_stats,  # 传入当前epoch的统计
                use_wandb=use_wandb
            )


            setattr(self, global_step_attr, current_global_step)
        
        #assert False,"train first epoch finished"
        
        epoch_win_rate = epoch_stats['win'] / epoch_stats['total'] if epoch_stats['total'] > 0 else 0.0


        self.summary_writer.add_scalar(f'training/{agent_name}_win_rate', epoch_win_rate, epoch)


        if use_wandb:
            wandb.log({
                f"training/{agent_name}_win_rate": epoch_win_rate,
                f"training/{agent_name}_wins": epoch_stats['win'],
                f"training/{agent_name}_losses": epoch_stats['loss'],
                f"training/{agent_name}_draws": epoch_stats['draw'],
                f"training/{agent_name}_total": epoch_stats['total'],
                f"training/{agent_name}_sparse_weight": sparse_weight,
                f"training/{agent_name}_best_qed": self.best_qed_A if agent_name == 'A' else self.best_qed_B,  # 新增
                "epoch": epoch
            })
        
        logging.info(f'Agent {agent_name} Epoch {epoch}: Win Rate={epoch_win_rate:.3f} '
                f'({epoch_stats["win"]}/{epoch_stats["total"]}), '
                f'Wins={epoch_stats["win"]}, Losses={epoch_stats["loss"]}, Draws={epoch_stats["draw"]}, '
                f'Best QED={self.best_qed_A if agent_name == "A" else self.best_qed_B:.4f}, '  # 新增
                f'Sparse Weight={sparse_weight:.3f}')
        #self.summary_writer.add_scalar(f'training/{agent_name}_win_rate', epoch_win_rate, epoch)

    def _run_adversarial_episode(self, dual_env, main_agent, opponent_agent, main_buffer, 
                                episode, global_step, agent_name, epoch_stats, use_wandb=False):
        """运行单个对抗训练 episode"""

        episode_start_time = time.time()
        
        # 重置环境
        dual_env.reset()

        current_epoch = self.epoch

        if self.debug:
            from rdkit import Chem
            from rdkit.Chem import Draw
            import os

            # 根据agent选择对应的子目录保存图片
            if agent_name == 'A':
                image_dir = os.path.join(self.model_dir_A, f'episode_{episode}_molecules')
            else:
                image_dir = os.path.join(self.model_dir_B, f'episode_{episode}_molecules')
            
            os.makedirs(image_dir, exist_ok=True)

            if dual_env.env_main.state:
                main_mol = Chem.MolFromSmiles(dual_env.env_main.state)
                if main_mol:
                    img = Draw.MolToImage(main_mol, size=(300, 300))
                    img.save(os.path.join(image_dir, f'step_0_main_initial.png'))
            
            if dual_env.env_opp.state:
                opp_mol = Chem.MolFromSmiles(dual_env.env_opp.state)
                if opp_mol:
                    img = Draw.MolToImage(opp_mol, size=(300, 300))
                    img.save(os.path.join(image_dir, f'step_0_opp_initial.png'))
        


        

        if self.hparams.num_bootstrap_heads:
            head = np.random.randint(self.hparams.num_bootstrap_heads)
        else:
            head = 0
        

        main_result = None
        opp_result = None

        #current_epoch = episode // self.hparams.episodes_per_epoch
        
        #print(f"Agent {agent_name}: Starting Episode {episode + 1}, Current Epoch {current_epoch + 1}")
        

        for step in range(self.hparams.max_steps_per_episode):
            # 检查终止条件
            #if dual_env.is_terminated():
                #break

            main_obs, main_valid_actions = dual_env.get_main_observations()
            if len(main_valid_actions) == 0:
                logging.warning(f"Agent {agent_name}, Episode {episode}, Step {step}: No valid actions")
                break

            #assert False,"check main_obs shape"



            main_action_idx = main_agent.get_action(
                main_obs, head=head, 
                update_epsilon=self.exploration.value(current_epoch)
            )
            #print(f"update_epsilon: {self.exploration.value(current_epoch)} ")

            #print(f"Main action index: {main_action_idx}, valid actions: {len(main_valid_actions)}")
            #assert False,"check main_action_idx"

            main_action = main_valid_actions[main_action_idx]

            main_obs_t = main_obs[main_action_idx]

            opp_obs, opp_valid_actions = dual_env.get_opp_observations()
            if len(opp_valid_actions) == 0:
                logging.warning(f"Opponent of Agent {agent_name}, Episode {episode}, Step {step}: No valid actions")
                # 如果对手无法行动，只执行主智能体动作
                opp_action = None
            
            else:
                opp_action_idx = opponent_agent.get_action(opp_obs, head=head, update_epsilon=self.exploration.value(current_epoch))
                opp_action = opp_valid_actions[opp_action_idx]

            #print(f"Opponent action index: {opp_action_idx}, valid actions: {len(opp_valid_actions)}")
            #assert False,"check opp_action_idx"
            

            if opp_action is not None:
                step_result = dual_env.step_both(main_action, opp_action)
            
            main_result = step_result['main_result']
            opp_result = step_result['opp_result']

            if self.debug:
                
                step_num = step + 1

                if main_result and main_result.state:
                    try:
                        main_mol = Chem.MolFromSmiles(main_result.state)
                        if main_mol:
                            img = Draw.MolToImage(main_mol, size=(300, 300))
                            filename = f'step_{step_num}_main_{main_result.state.replace("/", "_")}.png'
                            img.save(os.path.join(image_dir, filename))
                            #print(f"Saved main molecule: {filename}")
                    except Exception as e:
                        print(f"Failed to save main molecule at step {step_num}: {e}")
                
                # 保存对手智能体的分子
                if opp_result and opp_result.state:
                    try:
                        opp_mol = Chem.MolFromSmiles(opp_result.state)
                        if opp_mol:
                            img = Draw.MolToImage(opp_mol, size=(300, 300))
                            filename = f'step_{step_num}_opp_{opp_result.state.replace("/", "_")}.png'
                            img.save(os.path.join(image_dir, filename))
                            #print(f"Saved opponent molecule: {filename}")
                    except Exception as e:
                        print(f"Failed to save opponent molecule at step {step_num}: {e}")


            #if not step_result['terminated']:
            next_main_obs, _ = dual_env.get_main_observations()
            #print(f"Next main observation shape: {next_main_obs.shape}")
            
            #else:
                # 终止状态：空的下一状态观测
                #next_main_obs = np.zeros((1, main_obs.shape[1]), dtype=np.float32)
            
            dense_reward = main_result.reward
            #print(f"Agent {agent_name}, Episode {episode}, Step {step}: Dense Reward={dense_reward:.4f}")

            #assert False,"check main_obs_t shape"

            if step_result['terminated']:
                win_result = self._calculate_win_result(main_result, opp_result, agent_name, epoch_stats)  # 传入epoch_stats
                sparse_reward = self._compute_sparse_reward(win_result, current_epoch)
                logging.info(f"Agent {agent_name}, Episode {episode}, Step {step}: Win Result={win_result}, Sparse Reward={sparse_reward:.4f}")
            else:
                sparse_reward = 0.0
                #print(f"Agent {agent_name}, Episode {episode}, Step {step}: Ongoing episode, Sparse Reward=0.0")
            
            total_reward = dense_reward + sparse_reward

            """
            
            main_buffer.add(
                obs_t=main_obs_t,
                action=0,  # 占位符
                reward=main_result.reward,
                obs_tp1=next_main_obs,
                done=float(step_result['terminated'])
            )
            """
            main_buffer.add(
                obs_t=main_obs_t,
                action=0,
                reward=total_reward,  # 使用总奖励（密集+稀疏）
                obs_tp1=next_main_obs,
                done=float(step_result['terminated'])
            )

            if (episode > min(50, self.hparams.num_epoch * self.hparams.episodes_per_epoch / 10) and 
                global_step % self.hparams.learning_frequency == 0):  # 改为 num_epoch
                #print(f"Agent {agent_name}, Episode {episode}, Step {step}: Training DQN step, Buffer size: {len(main_buffer)}")
                if len(main_buffer) >= self.hparams.batch_size:
                    self._train_dqn_step(main_agent, main_buffer, episode, global_step, agent_name, use_wandb)
                    #print(f"finished dqn step training, buffer size: {len(main_buffer)}")
                    #assert False,"check main_buffer"
                
            global_step += 1

            #if step_result['terminated']:
                #break
        #print(f"global_step: {global_step}, agent_name: {agent_name}, episode: {episode}, step: {step + 1}")
        #print(f"buffer size: {len(main_buffer)}")
        #assert False,"check main_result and opp_result"
        
        #win_result = self._calculate_win_result(main_result, opp_result, agent_name)

        #current_epoch = episode // self.hparams.episodes_per_epoch

        #sparse_reward = self._compute_sparse_reward(win_result, current_epoch)
    
        episode_time = time.time() - episode_start_time
        final_dense_reward = main_result.reward if main_result else 0.0

        if agent_name == 'A':
            if final_dense_reward > self.best_qed_A:
                old_best = self.best_qed_A
                self.best_qed_A = final_dense_reward
                logging.info(f'Agent A: New best QED! {old_best:.4f} -> {self.best_qed_A:.4f} (Episode {episode})')
        else:  # agent_name == 'B'
            if final_dense_reward > self.best_qed_B:
                old_best = self.best_qed_B
                self.best_qed_B = final_dense_reward
                logging.info(f'Agent B: New best QED! {old_best:.4f} -> {self.best_qed_B:.4f} (Episode {episode})')

        final_sparse_reward = sparse_reward if 'sparse_reward' in locals() else 0.0
        final_total_reward = final_dense_reward + final_sparse_reward





        self.summary_writer.add_scalar(f'episode/{agent_name}_dense_reward', final_dense_reward, global_step)
        self.summary_writer.add_scalar(f'episode/{agent_name}_sparse_reward', final_sparse_reward, global_step)
        self.summary_writer.add_scalar(f'episode/{agent_name}_total_reward', final_total_reward, global_step)
        self.summary_writer.add_scalar(f'episode/{agent_name}_best_qed', self.best_qed_A if agent_name == 'A' else self.best_qed_B, global_step)  # 新增
        self.summary_writer.add_text(f'episode/{agent_name}_smiles', str(main_result.state) if main_result else "None", global_step)


        if use_wandb:
            wandb.log({
                f"episode/{agent_name}_dense_reward": final_dense_reward,
                f"episode/{agent_name}_sparse_reward": final_sparse_reward,
                f"episode/{agent_name}_total_reward": final_total_reward,
                f"episode/{agent_name}_best_qed": self.best_qed_A if agent_name == 'A' else self.best_qed_B,  # 新增
                f"episode/{agent_name}_smiles": str(main_result.state) if main_result else "None",
                f"episode/{agent_name}_steps": step + 1,
                f"episode/{agent_name}_win_result": win_result if 'win_result' in locals() else "unknown",
                "episode": episode,
                "global_step": global_step
            })
        
        logging.info(f'Agent {agent_name} Episode {episode}: DenseReward={final_dense_reward:.4f}, '
                f'SparseReward={final_sparse_reward:.4f}, TotalReward={final_total_reward:.4f}, '
                f'BestQED={self.best_qed_A if agent_name == "A" else self.best_qed_B:.4f}, '  # 新增
                f'Steps={step+1}, Time={episode_time:.2f}s, Result={win_result if "win_result" in locals() else "unknown"}')
        
        
        return global_step

    def _calculate_win_result(self, main_result, opp_result, agent_name,epoch_stats):
        """计算单个episode的胜负结果"""
        from rdkit import Chem
        from rdkit.Chem import QED
        
        # 获取QED分数
        if main_result and main_result.state:
            main_mol = Chem.MolFromSmiles(main_result.state)
            main_qed = QED.qed(main_mol) if main_mol else 0.0
        else:
            main_qed = 0.0
            
        if opp_result and opp_result.state:
            opp_mol = Chem.MolFromSmiles(opp_result.state)
            opp_qed = QED.qed(opp_mol) if opp_mol else 0.0
        else:
            opp_qed = 0.0
        
        #print(f"Agent {agent_name} Final QED - Main: {main_qed:.4f}, Opponent: {opp_qed:.4f}")

        # 计算胜负（使用小的容差避免浮点误差）
        tolerance = 1e-6
        if abs(main_qed - opp_qed) < tolerance:
            result = "draw"
        elif main_qed > opp_qed:
            result = "win"
        else:
            result = "loss"
        #print(f"Agent {agent_name} Episode Result: {result}")

        # 更新统计
        epoch_stats[result] += 1
        epoch_stats['total'] += 1
        #print(f"Agent {agent_name} Current Epoch Stats: {epoch_stats}")

        return result

    def _train_dqn_step(self, agent, buffer, episode, global_step, agent_name, use_wandb=False):
        """执行一次 DQN 训练步骤"""
        if self.hparams.prioritized:
            (state_t, _, reward_t, state_tp1, done_mask, weight, indices) = buffer.sample(
                self.hparams.batch_size, beta=self.beta_schedule.value(episode))
        else:
            (state_t, _, reward_t, state_tp1, done_mask) = buffer.sample(self.hparams.batch_size)
            weight = np.ones([reward_t.shape[0]])
            indices = None

        if reward_t.ndim == 1:
            reward_t = np.expand_dims(reward_t, axis=1)
        
        state_t_tensor = torch.FloatTensor(state_t).to(self.device)
        reward_t_tensor = torch.FloatTensor(reward_t).to(self.device)
        done_tensor = torch.FloatTensor(np.expand_dims(done_mask, axis=1)).to(self.device)
        weight_tensor = torch.FloatTensor(np.expand_dims(weight, axis=1)).to(self.device)


        loss, td_error = agent.train_step(
            states=state_t_tensor,
            rewards=reward_t_tensor,
            next_states=state_tp1,
            done=done_tensor,
            weight=weight_tensor
        )

        self.summary_writer.add_scalar(f'training/{agent_name}_loss', loss, global_step)

        if use_wandb:
            wandb.log({
                f"training/{agent_name}_loss": loss,
                f"training/{agent_name}_td_error_mean": td_error.mean().item(),
                "episode": episode,
                "global_step": global_step
            })
        
        logging.info(f'Agent {agent_name} Training: Loss={loss:.4f}, TD_Error={td_error.mean().item():.4f}')


        if self.hparams.prioritized and indices is not None:
            td_error_np = td_error.detach().cpu().numpy()
            buffer.update_priorities(
                indices,
                np.abs(np.squeeze(td_error_np) + self.hparams.prioritized_epsilon).tolist()
            )

    def _evaluate_agents(self, epoch, use_wandb=False):
        """评估两个智能体的性能"""
        logging.info("Evaluating agents...")
        


        # *** 新增：对抗评估 ***
        eval_win_rate_A, eval_win_rate_B = self._evaluate_adversarial(num_episodes=1)


        
        self.summary_writer.add_scalar('evaluation/agent_A_win_rate', eval_win_rate_A, epoch)
        self.summary_writer.add_scalar('evaluation/agent_B_win_rate', eval_win_rate_B, epoch)
        
        if use_wandb:
            wandb.log({
                "evaluation/agent_A_win_rate": eval_win_rate_A,
                "evaluation/agent_B_win_rate": eval_win_rate_B,
                "epoch": epoch
            })
        
        # 保存最佳模型（可以基于奖励或胜率）
        if eval_win_rate_A > getattr(self, 'best_win_rate_A', 0.0):
            self.best_win_rate_A = eval_win_rate_A
            best_path_A = os.path.join(self.model_dir_A, 'best_model.pt')
            self.agent_A.save_checkpoint(best_path_A)
            logging.info(f'New best Agent A win rate: {eval_win_rate_A:.3f}, saved to: {best_path_A}')
        
        if eval_win_rate_B > getattr(self, 'best_win_rate_B', 0.0):
            self.best_win_rate_B = eval_win_rate_B
            best_path_B = os.path.join(self.model_dir_B, 'best_model.pt')
            self.agent_B.save_checkpoint(best_path_B)
            logging.info(f'New best Agent B win rate: {eval_win_rate_B:.3f}, saved to: {best_path_B}')
        
        logging.info(f'Evaluation Win Rates - Agent A: {eval_win_rate_A:.3f}, Agent B: {eval_win_rate_B:.3f}')
        



    def _evaluate_adversarial(self, num_episodes=1):
        """对抗评估：当前epoch训练的A vs B 真实对战（类似具身项目的评估）"""
        from rdkit import Chem
        from rdkit.Chem import QED
        
        # 设置为评估模式（贪心策略）
        self.agent_A.eval()
        self.agent_B.eval()
        
        wins_A = 0
        wins_B = 0
        draws = 0

        eval_dual_env = DualMoleculeEnv(self.env_class, self.env_kwargs, self.hparams)

        for eval_ep in range(num_episodes):
            eval_dual_env.reset()
            
            # 用于存储最终结果
            final_main_result = None
            final_opp_result = None


            for step in range(self.hparams.max_steps_per_episode):
                if eval_dual_env.is_terminated():
                    break
                    
                # === Agent A (主) 选择动作 ===
                main_obs, main_valid_actions = eval_dual_env.get_main_observations()
                if not main_valid_actions:
                    break

                head_A = 0 if not self.hparams.num_bootstrap_heads else np.random.randint(self.hparams.num_bootstrap_heads)
                action_idx_A = self.agent_A.get_action(main_obs, head=head_A, stochastic=False)
                action_A = main_valid_actions[action_idx_A]


                opp_obs, opp_valid_actions = eval_dual_env.get_opp_observations()
                if not opp_valid_actions:
                    break
                    
                head_B = 0 if not self.hparams.num_bootstrap_heads else np.random.randint(self.hparams.num_bootstrap_heads)
                action_idx_B = self.agent_B.get_action(opp_obs, head=head_B, stochastic=False)
                action_B = opp_valid_actions[action_idx_B]


                step_result = eval_dual_env.step_both(action_A, action_B)

                final_main_result = step_result.get('main_result')
                final_opp_result = step_result.get('opp_result')
                
                if step_result['terminated']:
                    break
            
            # === 计算本局胜负 ===
            qed_A = 0.0
            qed_B = 0.0

            if final_main_result and final_main_result.state:
                mol_A = Chem.MolFromSmiles(final_main_result.state)
                qed_A = QED.qed(mol_A) if mol_A else 0.0
                
            if final_opp_result and final_opp_result.state:
                mol_B = Chem.MolFromSmiles(final_opp_result.state)
                qed_B = QED.qed(mol_B) if mol_B else 0.0
            

            tolerance = 1e-4
            if abs(qed_A - qed_B) < tolerance:
                draws += 1
                result_str = "draw"
            elif qed_A > qed_B:
                wins_A += 1
                result_str = "A wins"
            else:
                wins_B += 1
                result_str = "B wins"
            
            logging.info(f'Eval Episode {eval_ep+1}: A_QED={qed_A:.4f}, B_QED={qed_B:.4f}, Result={result_str}')
        

        self.agent_A.train()
        self.agent_B.train()

        total_games = num_episodes
        win_rate_A = wins_A / total_games if total_games > 0 else 0.0
        win_rate_B = wins_B / total_games if total_games > 0 else 0.0


        logging.info(f'Adversarial Evaluation Results: A wins {wins_A}, B wins {wins_B}, draws {draws}')
        logging.info(f'Win Rates: A={win_rate_A:.3f}, B={win_rate_B:.3f}')

        return win_rate_A, win_rate_B
    
    def _compute_sparse_reward(self, win_result, epoch):
        """
        计算稀疏奖励，仿照具身项目的逐步增大策略
        
        Args:
            win_result: str, "win", "loss", "draw"
            epoch: int, 当前epoch
            
        Returns:
            float: 稀疏奖励值
        """
        if not self.hparams.use_sparse_reward:
            return 0.0
        
        warmup_epochs = getattr(self.hparams, 'sparse_reward_warmup_epochs', 0)


        if win_result == "win":
            base_sparse_reward = self.hparams.sparse_win_reward
        elif win_result == "loss":
            base_sparse_reward = self.hparams.sparse_lose_reward
        else:  # draw
            base_sparse_reward = self.hparams.sparse_draw_reward
        
        if not self.hparams.sparse_reward_annealing:
            return base_sparse_reward
        #print(f"Sparse reward annealing is enabled, using epoch {epoch} for computation")
        #print(f"Base sparse reward: {base_sparse_reward}")
        if epoch < warmup_epochs:
            sparse_weight = 0.0  # warm-up阶段
            #print(f"Sparse weight for epoch {epoch} (warmup): {sparse_weight}")
        elif self.hparams.use_sparse_reward and self.hparams.sparse_reward_annealing:
            termination_epoch = getattr(self.hparams, 'sparse_reward_termination_epoch', 1000)
            adjusted_epoch = epoch - warmup_epochs
            adjusted_total_epochs = termination_epoch - warmup_epochs
            
            if adjusted_total_epochs <= 0:
                sparse_weight = 1.0
            else:
                alpha = max((adjusted_total_epochs - adjusted_epoch) / adjusted_total_epochs, 0)
                sparse_weight = 1 - alpha
                #print(f"Sparse weight for epoch {epoch}: {sparse_weight} (adjusted_epoch: {adjusted_epoch}, adjusted_total_epochs: {adjusted_total_epochs})")
        else:
            sparse_weight = 1.0 if self.hparams.use_sparse_reward else 0.0

        #print(f"Sparse weight for epoch {epoch}: {sparse_weight}")

        return base_sparse_reward * sparse_weight
    
    
        
        

            









