import numpy as np
import matplotlib.pyplot as plt


def plotma(#max_tot_power, min_tot_power,
           agent_t1, #agent_t2, agent_t3,
           agent_c1,# agent_c2,#, agent_c3,# agent_c4, #agent_c5,
           agent_p,
           #agent_p1,# agent_p2,# agent_p3,# agent_p4, #agent_p5,
           agent_l1):#, agent_l2):#, agent_l3):#, agent_l4):#, agent_l5):


    plt.subplot(2, 2, 1)
    # plt.plot(np.arange(len(agent_t1)), agent_t1)
    rt1mv = movingaverage(agent_t1, 200)
    plt.plot(np.arange(len(rt1mv)), rt1mv)
    # plt.subplot(4, 3, 2)
    # plt.plot(np.arange(len(agent_t2)), agent_t2)
    # rt2mv = movingaverage(agent_t2, 100)
    # plt.plot(np.arange(len(rt2mv)), rt2mv)
    # plt.subplot(4, 3, 3)
    # plt.plot(np.arange(len(agent_t3)), agent_t3)
    # rt3mv = movingaverage(agent_t3, 100)
    # plt.plot(np.arange(len(rt3mv)), rt3mv)

    plt.subplot(2, 2, 2)
    # plt.plot(np.arange(len(agent_c1)), agent_c1)
    rc1mv = movingaverage(agent_c1, 200)
    plt.plot(np.arange(len(rc1mv)), rc1mv)
    # plt.subplot(4, 3, 2)
    # plt.plot(np.arange(len(agent_c2)), agent_c2)
    # rc2mv = movingaverage(agent_c2, 100)
    # plt.plot(np.arange(len(rc2mv)), rc2mv)
    # plt.subplot(4, 3, 3)
    # plt.plot(np.arange(len(agent_c3)), agent_c3)
    # rc3mv = movingaverage(agent_c3, 100)
    # plt.plot(np.arange(len(rc3mv)), rc3mv)
    # plt.subplot(3, 5, 4)
    # plt.plot(np.arange(len(agent_c4)), agent_c4)
    # rc4mv = movingaverage(agent_c4, 100)
    # plt.plot(np.arange(len(rc4mv)), rc4mv)
    # plt.subplot(3, 5, 5)
    # plt.plot(np.arange(len(agent_c5)), agent_c5)
    # rc5mv = movingaverage(agent_c5, 100)
    # plt.plot(np.arange(len(rc5mv)), rc5mv)


    plt.subplot(2, 2, 3)
    # plt.plot(np.arange(len(agent_p)), agent_p)
    rpmv = movingaverage(agent_p, 200)
    plt.plot(np.arange(len(rpmv)), rpmv)
    # plt.subplot(4, 3, 4)
    # plt.plot(np.arange(len(agent_p1)), agent_p1)
    # rp1mv = movingaverage(agent_p1, 100)
    # plt.plot(np.arange(len(rp1mv)), rp1mv)
    # plt.subplot(4, 3, 5)
    # plt.plot(np.arange(len(agent_p2)), agent_p2)
    # rp2mv = movingaverage(agent_p2, 100)
    # plt.plot(np.arange(len(rp2mv)), rp2mv)
    # plt.subplot(4, 3, 6)
    # plt.plot(np.arange(len(agent_p3)), agent_p3)
    # rp3mv = movingaverage(agent_p3, 100)
    # plt.plot(np.arange(len(rp3mv)), rp3mv)
    # plt.subplot(3, 5, 8)
    # plt.plot(np.arange(len(agent_p4)), agent_p4)
    # rp4mv = movingaverage(agent_p4, 100)
    # plt.plot(np.arange(len(rp4mv)), rp4mv)

    plt.subplot(2, 2, 4)
    plt.plot(np.arange(len(agent_l1)), agent_l1)
    rl1mv = movingaverage(agent_l1, 1500)
    plt.plot(np.arange(len(rl1mv)), rl1mv)
    # plt.subplot(4, 3, 8)
    # plt.plot(np.arange(len(agent_l2)), agent_l2)
    # rl2mv = movingaverage(agent_l2, 1500)
    # plt.plot(np.arange(len(rl2mv)), rl2mv)
    # plt.subplot(4, 3, 9)
    # plt.plot(np.arange(len(agent_l3)), agent_l3)
    # rl3mv = movingaverage(agent_l3, 1500)
    # plt.plot(np.arange(len(rl3mv)), rl3mv)
    # plt.subplot(3, 5, 12)
    # plt.plot(np.arange(len(agent_l4)), agent_l4)
    # rl4mv = movingaverage(agent_l4, 1500)
    # plt.plot(np.arange(len(rl4mv)), rl4mv)
    # plt.subplot(3, 5, 15)
    # plt.plot(np.arange(len(agent_l5)), agent_l5)
    # rl5mv = movingaverage(agent_l5, 1500)
    # plt.plot(np.arange(len(rl5mv)), rl5mv)
    # plt.subplot(4, 3, 11)
    # plt.plot(np.arange(len(max_tot_power)), max_tot_power)
    # ptmaxmv = movingaverage(max_tot_power, 10)
    # plt.plot(np.arange(len(ptmaxmv)), ptmaxmv)
    # plt.subplot(4, 3, 12)
    # plt.plot(np.arange(len(min_tot_power)), min_tot_power)
    # ptminmv = movingaverage(min_tot_power, 10)
    # plt.plot(np.arange(len(ptminmv)), ptminmv)

    plt.show()


def movingaverage(data, window_size):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return ma_vec
