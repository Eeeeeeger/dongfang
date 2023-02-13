'''
Hull-White Model
dr = (theta(t)-alpha*r)*dt + sigma*dw
Calibration for `theta(t)` by Trident Tree

dt = T/(1+steps)
dR = sigma*sqrt(s*dt)
j_max = ceil(0.184/alpha/dt)
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm


def calibration_alpha_sigma(ts):
    res = sm.OLS(endog=ts.diff().dropna().values, exog=sm.add_constant(ts.values[:-1])).fit()
    return np.array(list(res.params)+[res.resid.std()])


def calculate_init_zero_bonds(init_zero_rates, T, steps):
    """ 线性插值零息利率，计算出树形每层对应时刻P(0,ti)的值，
        在计算利率调整量alpha时需要用到。
        树形的末端时间T应该在初始零息利率数组时间范围内。
    """
    rates = list(init_zero_rates)
    dt = T/(steps+1)
    Pti = []
    p = 0
    for i in range(steps+2):
        ti = dt*i
        while ti > rates[p][0]:
            p += 1
        if p == 0:
            ri = rates[0][1]
        else:
            ri = rates[p-1][1]+(rates[p][1]-rates[p-1][1])/(rates[p][0]-rates[p-1][0])*(ti-rates[p-1][0])
        Pti.append(np.exp(-ri*ti))

    return Pti


def build_rates_tree(a, sigma, T, steps, init_zero_rates):
    """ 和股票价格树形不同的是，这里每个节点的利率为从该层时刻开始
        到下一层的时刻之间的利率R，是瞬时无风险利率r在t至t+delta t之间的平均值。
    """
    dt = T / (steps + 1.0)  # 分叉steps次，需要考虑（steps+1）个时间区间。
    dR = sigma * np.sqrt(3 * dt)
    j_max = int(np.ceil(0.184 / a / dt))

    # Pti =  [np.exp(-init_zero_rates[ti][1]*init_zero_rates[ti][0]) for ti in range(len(init_zero_rates))]
    Pti = calculate_init_zero_bonds(init_zero_rates, T ,steps)
    # 初始化树，Probs为分叉概率，prob为到该节点概率。
    tree = [[{"Probs": [None, None, None], "prob": 1, "Q": 1, "R": None, 'alpha': None}]]

    # 先生成和处理出现非标准分叉之前的树形。
    # 流程为：
    # 1. 产生下一层树形空节点。
    # 2. 计算当前层每个节点处 R，需要先计算出该层的利率调整alpha。
    # 3. 计算当前层每个节点每个方向的分叉概率。
    # 4. 把到节点的概率和“Q”的值传递到下一层初始化好的节点。
    for lvl in range(min(steps, j_max)):
        next_lvl_nodes = []
        for j in range(2 * lvl + 3):
            next_lvl_nodes.append({"Probs": [None, None, None], "prob": 0, "Q": 0, "R": None, 'alpha': None})
        tree.append(next_lvl_nodes)

        nodes = tree[lvl]
        S = 0
        for j in range(-lvl, lvl + 1, 1):
            S += nodes[j]["Q"] * np.exp(-j * dR * dt)
        alpha = (np.log(S) - np.log(Pti[lvl + 1])) / dt
        for each in nodes:
            each['alpha'] = alpha
        # Python里list[-num]会自动从后往前数取值。
        for j in range(-lvl, lvl + 1, 1):
            nodes[j]["R"] = j * dR + alpha

            nodes[j]["Probs"][0] = 1.0 / 6.0 + 0.5 * (a * a * j * j * dt * dt + a * j * dt)
            nodes[j]["Probs"][1] = 2.0 / 3.0 - a * a * j * j * dt * dt
            nodes[j]["Probs"][2] = 1.0 / 6.0 + 0.5 * (a * a * j * j * dt * dt - a * j * dt)

            for k in range(3):
                # 0 1 2 下降 不变 上升
                tree[lvl + 1][j + k - 1]["prob"] += nodes[j]["prob"] * nodes[j]["Probs"][k]
                tree[lvl + 1][j + k - 1]["Q"] += nodes[j]["Q"] * nodes[j]["Probs"][k] * np.exp(-nodes[j]["R"] * dt)

    # 生成树形开始出现非标准分叉的层。
    # 过程类似上面。但先处理每层中间标准分叉的节点，再处理非标准分叉节点。
    if steps > j_max:
        for lvl in range(j_max, steps, 1):
            next_lvl_nodes = []
            for j in range(2 * j_max + 1):
                next_lvl_nodes.append({"Probs": [None, None, None], "prob": 0, "Q": 0, "R": None})
            tree.append(next_lvl_nodes)

            nodes = tree[lvl]
            S = 0
            for j in range(-j_max, j_max + 1, 1):
                S += nodes[j]["Q"] * np.exp(-j * dR * dt)
            alpha = (np.log(S) - np.log(Pti[lvl + 1])) / dt
            for each in nodes:
                each['alpha'] = alpha
            # 处理标准分叉节点。
            for j in range(-j_max + 1, j_max, 1):
                nodes[j]["R"] = j * dR + alpha

                nodes[j]["Probs"][0] = 1.0 / 6.0 + 0.5 * (a * a * j * j * dt * dt + a * j * dt)
                nodes[j]["Probs"][1] = 2.0 / 3.0 - a * a * j * j * dt * dt
                nodes[j]["Probs"][2] = 1.0 / 6.0 + 0.5 * (a * a * j * j * dt * dt - a * j * dt)

                for k in range(3):
                    tree[lvl + 1][j + k - 1]["prob"] += nodes[j]["prob"] * nodes[j]["Probs"][k]
                    tree[lvl + 1][j + k - 1]["Q"] += nodes[j]["Q"] * nodes[j]["Probs"][k] * np.exp(-nodes[j]["R"] * dt)

            # 处理上下边界非标准分叉节点。
            j = -j_max
            nodes[j]["R"] = j * dR + alpha
            nodes[j]["Probs"][0] = 7.0 / 6.0 + 0.5 * (a * a * j * j * dt * dt + 3 * a * j * dt)
            nodes[j]["Probs"][1] = -1.0 / 3.0 - a * a * j * j * dt * dt - 2.0 * a * j * dt
            nodes[j]["Probs"][2] = 1.0 / 6.0 + 0.5 * (a * a * j * j * dt * dt + a * j * dt)
            for k in range(3):
                tree[lvl + 1][j + k]["prob"] += nodes[j]["prob"] * nodes[j]["Probs"][k]
                tree[lvl + 1][j + k]["Q"] += nodes[j]["Q"] * nodes[j]["Probs"][k] * np.exp(-nodes[j]["R"] * dt)
            j = j_max
            nodes[j]["R"] = j * dR + alpha
            nodes[j]["Probs"][0] = 1.0 / 6.0 + 0.5 * (a * a * j * j * dt * dt - a * j * dt)
            nodes[j]["Probs"][1] = -1.0 / 3.0 - a * a * j * j * dt * dt + 2.0 * a * j * dt
            nodes[j]["Probs"][2] = 7.0 / 6.0 + 0.5 * (a * a * j * j * dt * dt - 3 * a * j * dt)
            for k in range(3):
                tree[lvl + 1][j + k - 2]["prob"] += nodes[j]["prob"] * nodes[j]["Probs"][k]
                tree[lvl + 1][j + k - 2]["Q"] += nodes[j]["Q"] * nodes[j]["Probs"][k] * np.exp(-nodes[j]["R"] * dt)

    # 树形最后一层上面并没有计算。
    # 这里把该层的利率计算出并填入。
    S = 0
    radius = min(steps, j_max)
    for j in range(-radius, radius + 1, 1):
        S += tree[steps][j]["Q"] * np.exp(-j * dR * dt)
    alpha = (np.log(S) - np.log(Pti[steps + 1])) / dt
    for each in tree[steps]:
        each['alpha'] = alpha
    for j in range(-radius, radius + 1, 1):
        tree[steps][j]["R"] = j * dR + alpha

    return tree


def PtT_explicit(t, T, R, dt, a, sigma, init_zero_rates):
    """ 使用Hull-White单因子模型P(t,T)解析表达式计算价格。
        R为从t开始到t+delta t的利率。
    """
    rates = list(init_zero_rates)

    def P(t, rates):
        p = 0
        while t > rates[p][0]:
            p += 1
        if p == 0:
            r_t = rates[0][1]
        else:
            r_t = rates[p - 1][1] + (rates[p][1] - rates[p - 1][1]) / (rates[p][0] - rates[p - 1][0]) * (
                        t - rates[p - 1][0])
        return np.exp(-r_t * t)

    def B(t, T):
        return (1.0 - np.exp(-a * (T - t))) / a

    B_hat = B(t, T) / B(t, t + dt) * dt
    A_hat = P(T, rates) / P(t, rates) / np.power(P(t + dt, rates) / P(t, rates), B(t, T) / B(t, t + dt))
    A_hat /= np.exp(sigma * sigma / 4 / a * (1 - np.exp(-2 * a * t)) * B(t, T) * (B(t, T) - B(t, t + dt)))

    return A_hat * np.exp(-B_hat * R)


def bond_option(K, t, T, steps, a, sigma, init_zero_rates):
    """ 计算零息债券上期权价格。
        先生成t=0到t=T+delta t的利率树形，这里多一个时间间隔是为了和解析表达式相符合。
        然后在树形末端用P(t,T)解析表达式计算债券价格。
        最后直接用树形末端的"Q"值对期权价格加权。
    """
    dt = t / steps
    tree = build_rates_tree(a, sigma, t + dt, steps, init_zero_rates)

    call_price = 0
    put_price = 0
    for j in range(len(tree[-1])):
        P = 100.0 * PtT_explicit(t, T, tree[-1][j]["R"], dt, a, sigma, init_zero_rates)
        if P > K:
            call_price += (P - K) * tree[-1][j]["Q"]
        else:
            put_price += (K - P) * tree[-1][j]["Q"]

    return call_price, put_price


if __name__ == "__main__":
    init_rates = pd.read_csv('../input/rate.csv', index_col=0)/100
    params = calibration_alpha_sigma(init_rates.iloc[:,-1])
    alpha, sigma = np.abs(params[1]*250), params[2]*np.sqrt(250)
    rates = init_rates.iloc[0].tolist()
    rates = [[ii, rates[ii]] for ii in range(len(rates))]
    # trees = build_rates_tree(alpha, sigma, 10, 9, rates)
    # print(trees)
    # for lvl in range(len(trees)):
    #     nodes = len(trees[lvl])
    #     print(lvl)
    #     for ii in range(-int(nodes/2), int(nodes/2) + 1, 1):
    #         print(trees[lvl][ii]['R'])
    call_price, put_price = bond_option(63, 3, 9, 99, alpha, sigma, rates)
    print("债券期权，看涨和看跌期权价格分别为: {0:.5f}  {1:.5f}".format(call_price, put_price))

    # tree = build_rates_tree(0.1, 0.01, 3, 50, init_rates)
