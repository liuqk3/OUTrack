
import matplotlib.pyplot as plt
import os
import math

def get_data(log_file, key='loss', phase='train'):
    """This function get the data from log file

    Args:
        log_file: str, the path of log file
        key: str, the key word to get the loss
    """
    itr2value = {}
    with open(log_file) as log_file:
        for line in log_file:
            if phase + ':' not in line:
                continue
            line = line.split('|')
            # get itr
            itr_state = line[1] #  train: [1][141/4459]
            itr_state = itr_state.replace(phase+':', '').strip() # [1][141/4459]
            itr_state = itr_state.split('][')
            epoch = int(itr_state[0].replace('[', '').split('/')[0])
            itr_state = itr_state[1].replace(']', '').strip().split('/')
            itr = int(itr_state[0])
            itr_per_epoch = int(itr_state[1])
            itr_cur = (epoch - 1) * itr_per_epoch + itr
            # itrs.append(itr_cur)

            # get the value
            for one_state in line:
                if key not in one_state or '/' in one_state:
                    continue
                else: # loss 33.3963 
                    value = float(one_state.split(' ')[1].strip())
                    # values.append(value)
                    itr2value[itr_cur] = value
                    break

    return itr2value


def plot_curv(itr2value, name='', save_path=None):
    x = []
    y = []
    for itr in itr2value.keys():
        x.append(itr)
        y.append(itr2value[itr])
    plt.figure()
    plt.plot(x, y)
    plt.title(name)
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
        print('figure saved in {}'.format(save_path))



def uncertainty(x_min=-4, x_max=10, alpha=1, save=False):

    num_samples = 100
    x = list(range(x_min * num_samples, x_max * num_samples))
    x = [i/num_samples for i in x]

    iter2value = {}
    for i in x:
        v = math.exp(-i) * alpha + i
        iter2value[i] = v
    if save:
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'exp', 'uncertainty', '{}.jpg'.format(alpha)))
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
    else:
        save_path = None
    plot_curv(iter2value, name=str(alpha), save_path=save_path)
    


if __name__ == '__main__':


    # uncertainty(x_min=-4, x_max=100, alpha=2.5, save=True)
    # plt.show()


    key = 'sim_max'
    log_file='/home/liuqk/Program/python/OUTrack/exp/mot/mot17_half_bs8_dla34_cycle2ReIDSup_1_Pzero_0.5M_lr1e-4_2/logs/train_log.txt'
    itr2value = get_data(log_file=log_file, key=key)

    # save path
    save_path = os.path.join(os.path.dirname(log_file), 'loss_curve', key+'.png')
    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))

    plot_curv(itr2value, name=key, save_path=save_path)
    plt.show()


    

