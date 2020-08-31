import sac
import torch
from torch.utils.tensorboard import SummaryWriter

# TODO: Implement training loop with logging support

def train(iter_count = 100):

    writer = SummaryWriter()
    rl = sac.SoftActorCritic()

    for n in range(100):
        native_reward, adj_reward = rl.env_iter()
        writer.add_scalar('Eval/Original Reward', native_reward, n)
        writer.add_scalar('Eval/Additional Reward', adj_reward, n)
        writer.add_scalar('Eval/Improvement Percentage', adj_reward/native_reward, n)



        x,y,z,w = rl.grad_iter()

        writer.add_scalar('Loss/Q_1', x, n)
        writer.add_scalar('Loss/Q_2', y, n)
        writer.add_scalar('Loss/Policy', z, n)
        writer.add_scalar('Loss/Temperature', w, n)

        writer.flush()

    writer.close()
    pass

train()