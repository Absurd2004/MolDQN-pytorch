import argparse
import json
import logging
import os
import time
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import QED, Draw
from rdkit.Chem.Draw import rdMolDraw2D

from mol_dqn.chemgraph.dqn.agent import DoubleDQNAgent
from mol_dqn.chemgraph.dqn.multi_agent_env import DualMoleculeEnv
from mol_dqn.chemgraph.optimize_qed import QEDRewardMolecule
from mol_dqn.utils.hparams import get_hparams


def draw_molecule_with_info(mol_obj, agent_name, step, qed_value, is_winner=False, img_size=(400, 300)):
    """绘制带有信息标注的分子结构图"""
    if mol_obj is None:
        # 空分子的情况
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # 绘制边框
        draw.rectangle([(0, 0), (img_size[0]-1, img_size[1]-1)], outline='black', width=2)
        
        # 添加文字
        draw.text((10, 10), f"{agent_name}", fill='black', font=font_large)
        draw.text((10, 40), f"Step {step}", fill='black', font=font_small)
        draw.text((10, 60), "Empty Molecule", fill='gray', font=font_small)
        draw.text((10, 80), f"QED: {qed_value:.4f}", fill='blue', font=font_small)
        if is_winner:
            draw.text((10, 100), "🏆 WIN!", fill='red', font=font_large)
        
        return img
    
    try:
        # 绘制分子结构 - 增加底部空间
        mol_img = Draw.MolToImage(mol_obj, size=(img_size[0], img_size[1] - 120))
        
        # 创建最终图像
        final_img = Image.new('RGB', img_size, 'white')
        final_img.paste(mol_img, (0, 0))
        
        # 绘制边框
        draw = ImageDraw.Draw(final_img)
        draw.rectangle([(0, 0), (img_size[0]-1, img_size[1]-1)], outline='black', width=2)
        
        # 添加文字标注
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # 文字信息区域（底部）- 增加空间
        text_y = img_size[1] - 115
        
        # 半透明背景
        overlay = Image.new('RGBA', img_size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([(0, text_y-5), (img_size[0], img_size[1])], 
                              fill=(255, 255, 255, 200))
        final_img = Image.alpha_composite(final_img.convert('RGBA'), overlay).convert('RGB')
        
        draw = ImageDraw.Draw(final_img)
        
        # Agent名称 - 用不同颜色区分
        agent_color = 'red' if agent_name == 'Agent A' else 'blue'
        draw.text((10, text_y), f"{agent_name}", fill=agent_color, font=font_large)
        draw.text((10, text_y + 25), f"Step {step}", fill='black', font=font_small)
        draw.text((10, text_y + 45), f"QED: {qed_value:.4f}", fill='darkgreen', font=font_small)
        
        # 显示完整SMILES
        smiles = Chem.MolToSmiles(mol_obj)
        # 分行显示长SMILES
        if len(smiles) > 25:
            smiles_line1 = smiles[:25]
            smiles_line2 = smiles[25:50] if len(smiles) > 50 else smiles[25:]
            draw.text((10, text_y + 65), f"SMILES: {smiles_line1}", fill='gray', font=font_small)
            draw.text((10, text_y + 80), f"        {smiles_line2}", fill='gray', font=font_small)
        else:
            draw.text((10, text_y + 65), f"SMILES: {smiles}", fill='gray', font=font_small)
        
        # 胜利标记
        if is_winner:
            draw.text((300, text_y), " WIN!", fill='red', font=font_large)
        
        return final_img
        
    except Exception as e:
        logging.error(f"Error drawing molecule for {agent_name}: {e}")
        # 回退到简单绘制
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (img_size[0]-1, img_size[1]-1)], outline='black', width=2)
        draw.text((10, 10), f"{agent_name} - Error", fill='red')
        return img


def combine_agent_images(img_A, img_B, step, img_size=(850, 350)):
    """将两个agent的图像水平拼接"""
    # 创建组合图像
    combined_img = Image.new('RGB', img_size, 'white')
    
    # 粘贴两个agent的图像
    combined_img.paste(img_A, (10, 10))  # Agent A在左侧
    combined_img.paste(img_B, (430, 10))  # Agent B在右侧
    
    # 添加分割线
    draw = ImageDraw.Draw(combined_img)
    
    try:
        step_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        step_font = ImageFont.load_default()
    
    # 步数信息
    step_text = f"Step {step}"
    step_bbox = draw.textbbox((0, 0), step_text, font=step_font)
    step_width = step_bbox[2] - step_bbox[0]
    step_x = (img_size[0] - step_width) // 2
    draw.text((step_x, img_size[1] - 25), step_text, fill='darkblue', font=step_font)
    
    # 中央分割线
    center_x = img_size[0] // 2
    draw.line([(center_x, 10), (center_x, img_size[1] - 30)], fill='gray', width=2)
    
    return combined_img


def run_adversarial_display_episode(dual_env, agent_A, agent_B, hparams, episode_dir, episode_num):
    """运行单个对抗显示episode"""
    logging.info(f"Running adversarial display episode {episode_num}")
    
    # 重置环境
    dual_env.reset()
    
    # 保存数据
    episode_data = {
        'episode': episode_num,
        'steps': [],
        'final_results': {}
    }
    
    # 初始状态保存
    if dual_env.env_main.state_mol:
        qed_A_initial = QED.qed(dual_env.env_main.state_mol)
    else:
        qed_A_initial = 0.0
        
    if dual_env.env_opp.state_mol:
        qed_B_initial = QED.qed(dual_env.env_opp.state_mol)
    else:
        qed_B_initial = 0.0
    
    # 保存初始状态图像
    img_A = draw_molecule_with_info(dual_env.env_main.state_mol, "Agent A", 0, qed_A_initial)
    img_B = draw_molecule_with_info(dual_env.env_opp.state_mol, "Agent B", 0, qed_B_initial)
    combined_img = combine_agent_images(img_A, img_B, 0)
    combined_img.save(os.path.join(episode_dir, "step_000.png"))
    
    # 记录初始状态
    episode_data['steps'].append({
        'step': 0,
        'agent_A_smiles': dual_env.env_main.state if dual_env.env_main.state else "",
        'agent_B_smiles': dual_env.env_opp.state if dual_env.env_opp.state else "",
        'agent_A_qed': qed_A_initial,
        'agent_B_qed': qed_B_initial,
        'terminated': False
    })
    
    # 运行episode步骤
    winner = None  # 添加winner变量
    for step in range(hparams.max_steps_per_episode):
        # 检查是否终止
        if dual_env.is_terminated():
            break
            
        # === Agent A (主) 选择动作 ===
        main_obs, main_valid_actions = dual_env.get_main_observations()
        if not main_valid_actions:
            logging.warning(f"No valid actions for Agent A at step {step}")
            break
            
        head_A = 0 if not hparams.num_bootstrap_heads else np.random.randint(hparams.num_bootstrap_heads)
        action_idx_A = agent_A.get_action(main_obs, head=head_A, stochastic=False)
        action_A = main_valid_actions[action_idx_A]
        
        # === Agent B (对手) 选择动作 ===
        opp_obs, opp_valid_actions = dual_env.get_opp_observations()
        if not opp_valid_actions:
            logging.warning(f"No valid actions for Agent B at step {step}")
            break
            
        head_B = 0 if not hparams.num_bootstrap_heads else np.random.randint(hparams.num_bootstrap_heads)
        action_idx_B = agent_B.get_action(opp_obs, head=head_B, stochastic=False)
        action_B = opp_valid_actions[action_idx_B]
        
        # === 执行动作 ===
        step_result = dual_env.step_both(action_A, action_B)
        main_result = step_result['main_result']
        opp_result = step_result['opp_result']
        
        # === 计算QED ===
        if main_result and main_result.state:
            mol_A = Chem.MolFromSmiles(main_result.state)
            qed_A = QED.qed(mol_A) if mol_A else 0.0
        else:
            mol_A = None
            qed_A = 0.0
            
        if opp_result and opp_result.state:
            mol_B = Chem.MolFromSmiles(opp_result.state)
            qed_B = QED.qed(mol_B) if mol_B else 0.0
        else:
            mol_B = None
            qed_B = 0.0
        
        # 判断是否是最后一步，如果是则确定胜者
        is_last_step = (step == hparams.max_steps_per_episode - 1) or step_result['terminated']
        if is_last_step and winner is None:
            tolerance = 1e-6
            if abs(qed_A - qed_B) < tolerance:
                winner = "Draw"
            elif qed_A > qed_B:
                winner = "Agent A"
            else:
                winner = "Agent B"
        
        # === 保存图像 ===
        is_winner_A = (winner == "Agent A") if is_last_step else False
        is_winner_B = (winner == "Agent B") if is_last_step else False
        
        img_A = draw_molecule_with_info(mol_A, "Agent A", step + 1, qed_A, is_winner_A)
        img_B = draw_molecule_with_info(mol_B, "Agent B", step + 1, qed_B, is_winner_B)
        combined_img = combine_agent_images(img_A, img_B, step + 1)
        combined_img.save(os.path.join(episode_dir, f"step_{step+1:03d}.png"))
        
        # === 记录数据 ===
        step_data = {
            'step': step + 1,
            'agent_A_smiles': main_result.state if main_result else "",
            'agent_B_smiles': opp_result.state if opp_result else "",
            'agent_A_qed': qed_A,
            'agent_B_qed': qed_B,
            'agent_A_action': str(action_A),
            'agent_B_action': str(action_B),
            'terminated': step_result['terminated']
        }
        episode_data['steps'].append(step_data)
        
        logging.info(f"Step {step+1}: Agent A QED={qed_A:.4f}, Agent B QED={qed_B:.4f}")
        
        if step_result['terminated']:
            logging.info(f"Episode terminated at step {step+1}")
            break
    
    # === 计算最终结果 ===
    final_step = episode_data['steps'][-1]
    final_qed_A = final_step['agent_A_qed']
    final_qed_B = final_step['agent_B_qed']
    
    # 如果还没确定胜者，现在确定
    if winner is None:
        tolerance = 1e-6
        if abs(final_qed_A - final_qed_B) < tolerance:
            winner = "Draw"
        elif final_qed_A > final_qed_B:
            winner = "Agent A"
        else:
            winner = "Agent B"
    
    episode_data['final_results'] = {
        'agent_A_final_qed': final_qed_A,
        'agent_B_final_qed': final_qed_B,
        'agent_A_improvement': final_qed_A - qed_A_initial,
        'agent_B_improvement': final_qed_B - qed_B_initial,
        'winner': winner,
        'qed_difference': abs(final_qed_A - final_qed_B)
    }
    
    # 保存episode数据
    import json
    with open(os.path.join(episode_dir, "episode_data.json"), 'w') as f:
        json.dump(episode_data, f, indent=2)
    
    logging.info(f"Episode {episode_num} completed:")
    logging.info(f"  Agent A: {qed_A_initial:.4f} → {final_qed_A:.4f} (Δ{final_qed_A - qed_A_initial:+.4f})")
    logging.info(f"  Agent B: {qed_B_initial:.4f} → {final_qed_B:.4f} (Δ{final_qed_B - qed_B_initial:+.4f})")
    logging.info(f"  Winner: {winner}")
    
    return episode_data



def create_video_from_images(episode_dir, fps=1):
    """从图片创建视频"""
    try:
        import imageio
        
        # 获取所有图片文件
        image_files = sorted([f for f in os.listdir(episode_dir) if f.endswith('.png')])
        if len(image_files) < 2:
            logging.warning(f"Not enough images found in {episode_dir}")
            return
        
        # 读取图片并创建视频
        images = []
        for img_file in image_files:
            img_path = os.path.join(episode_dir, img_file)
            img = imageio.imread(img_path)
            # 每张图片显示2秒
            for _ in range(2):
                images.append(img)
        
        # 生成MP4视频
        video_path = os.path.join(episode_dir, 'adversarial_competition.mp4')
        imageio.mimsave(video_path, images, fps=fps)
        logging.info(f"Video saved: {video_path}")
        
        # 生成GIF（较低帧率）
        gif_path = os.path.join(episode_dir, 'adversarial_competition.gif')
        imageio.mimsave(gif_path, images[::2], fps=fps//2, duration=2.0)
        logging.info(f"GIF saved: {gif_path}")
        
    except ImportError:
        logging.warning("imageio not installed. Install with: pip install imageio[ffmpeg]")
    except Exception as e:
        logging.error(f"Error creating video: {e}")


def run_adversarial_display(hparams, env_class, env_kwargs, agent_A_path, agent_B_path, 
                           model_dir, num_episodes=3, save_video=True):
    """运行对抗显示主函数"""
    logging.info("Starting adversarial display...")
    
    # 创建保存目录
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    display_dir = os.path.join(model_dir, f"adversarial_display_{timestamp}")
    os.makedirs(display_dir, exist_ok=True)
    
    # 创建智能体
    agent_A = DoubleDQNAgent(input_dim=hparams.input_dim, hparams=hparams)
    agent_B = DoubleDQNAgent(input_dim=hparams.input_dim, hparams=hparams)
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent_A.to(device)
    agent_B.to(device)
    
    try:
        agent_A.load_checkpoint(agent_A_path)
        logging.info(f"Loaded Agent A from: {agent_A_path}")
    except Exception as e:
        logging.error(f"Failed to load Agent A: {e}")
        return
    
    try:
        agent_B.load_checkpoint(agent_B_path)
        logging.info(f"Loaded Agent B from: {agent_B_path}")
    except Exception as e:
        logging.error(f"Failed to load Agent B: {e}")
        return
    
    # 设置为评估模式
    agent_A.eval()
    agent_B.eval()
    
    # 创建环境
    dual_env = DualMoleculeEnv(env_class, env_kwargs, hparams)
    
    # 运行多个episode
    all_episode_data = []
    
    for episode in range(num_episodes):
        episode_dir = os.path.join(display_dir, f"episode_{episode}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # 运行episode
        episode_data = run_adversarial_display_episode(
            dual_env, agent_A, agent_B, hparams, episode_dir, episode
        )
        all_episode_data.append(episode_data)
        
        # 创建视频
        if save_video:
            create_video_from_images(episode_dir)
    
    # 保存总结数据
    summary_data = {
        'total_episodes': num_episodes,
        'agent_A_path': agent_A_path,
        'agent_B_path': agent_B_path,
        'episodes': all_episode_data,
        'summary': {
            'agent_A_wins': sum(1 for ep in all_episode_data if ep['final_results']['winner'] == 'Agent A'),
            'agent_B_wins': sum(1 for ep in all_episode_data if ep['final_results']['winner'] == 'Agent B'),
            'draws': sum(1 for ep in all_episode_data if ep['final_results']['winner'] == 'Draw'),
        }
    }
    
    with open(os.path.join(display_dir, "summary.json"), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logging.info("Adversarial display completed!")
    logging.info(f"Results saved to: {display_dir}")
    logging.info(f"Summary: Agent A wins: {summary_data['summary']['agent_A_wins']}, "
                f"Agent B wins: {summary_data['summary']['agent_B_wins']}, "
                f"Draws: {summary_data['summary']['draws']}")


def main():
    parser = argparse.ArgumentParser(description='Display Adversarial Molecular DQN Competition')
    parser.add_argument('--config', type=str, default="./mol_dqn/chemgraph/configs/multi_agent_dqn.json",
                       help='Path to hyperparameters JSON file')
    parser.add_argument('--agent_A_checkpoint', type=str, required=True,
                       help='Path to Agent A checkpoint')
    parser.add_argument('--agent_B_checkpoint', type=str, required=True,
                       help='Path to Agent B checkpoint')
    parser.add_argument('--model_dir', type=str, default="./display/multi_agent",
                       help='Directory to save display results')
    parser.add_argument('--start_molecule', type=str, default=None,
                       help='Starting molecule SMILES string')
    parser.add_argument('--num_episodes', type=int, default=1,
                       help='Number of episodes to display')
    parser.add_argument('--no_video', action='store_true',
                       help='Disable video generation')
    
    args = parser.parse_args()
    
    # 设置日志
    os.makedirs(args.model_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.model_dir, 'adversarial_display.log'))
        ]
    )
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = json.load(f)
    # 移除注释项
    config = {k: v for k, v in config.items() if not k.startswith('_comment')}
    hparams = get_hparams(**config)
    
    # 创建环境参数
    env_kwargs = {
        'atom_types': hparams.atom_types,
        'init_mol': args.start_molecule,
        'allow_removal': hparams.allow_removal,
        'allow_no_modification': hparams.allow_no_modification,
        'allow_bonds_between_rings': hparams.allow_bonds_between_rings,
        'allowed_ring_sizes': hparams.allowed_ring_sizes,
        'max_steps': hparams.max_steps_per_episode,
        'discount_factor': hparams.discount_factor
    }
    
    # 运行对抗显示
    run_adversarial_display(
        hparams=hparams,
        env_class=QEDRewardMolecule,
        env_kwargs=env_kwargs,
        agent_A_path=args.agent_A_checkpoint,
        agent_B_path=args.agent_B_checkpoint,
        model_dir=args.model_dir,
        num_episodes=args.num_episodes,
        save_video=not args.no_video
    )


if __name__ == '__main__':
    main()