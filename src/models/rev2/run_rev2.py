import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from src.models.rev2.rev2_arg_parser import arg_parser


def preprocessing(args):
    data_name = args.data_name
    source_col = args.source_col
    target_col = args.target_col
    weight_col = args.weight_col

    # 生の評価ネットワークを読み込む
    df = pd.read_csv(f'./data/raw/{data_name}/network.csv')
    assert ~df[source_col].str.startswith('u').any() or \
           ~df[target_col].str.startswith('p').any(), 'bad prefix'
    assert set(df[weight_col]) == set([1, 2, 3, 4, 5]), \
        '未対応のレーティング'
    df['source'] = 'u'+df[source_col].astype(str)
    df['target'] = 'p'+df[target_col].astype(str)
    df['weight'] = df[weight_col].map({1: -1.0,
                                       2: -0.5,
                                       3: 0,
                                       4: 0.5,
                                       5: 1.0, })
    X = df[['source', 'target', 'weight']]
    # nx形式に変換
    G = nx.from_pandas_edgelist(X, edge_attr=True,
                                create_using=nx.DiGraph())
    return G


def iterate(args):

    NETWORKNAME = args.data_name

    alpha1 = int(args.alpha1)
    alpha2 = 0

    beta1 = int(args.beta1)
    beta2 = 0

    gamma1 = float(args.gamma1)+0.01
    gamma2 = float(args.gamma2)+0.01
    gamma3 = 0

    G = preprocessing(args)

    nodes = G.nodes()
    edges = G.edges(data=True)

    print(f"{NETWORKNAME} network has {len(nodes)} nodes and {len(edges)} edges")

    df = nx.to_pandas_edgelist(G)

    user_names = [node for node in nodes if node.startswith('u')]
    product_names = [node for node in nodes if node.startswith('p')]
    assert set(user_names) == set(df.source.unique()
                                  ), "user_names doesn't match the original user set"
    assert set(product_names) == set(df.target.unique()
                                     ), "product_names doesn't match the original product set"
    num_users = len(user_names)
    num_products = len(product_names)
    user_map = dict(zip(user_names, range(len(user_names))))
    product_map = dict(zip(product_names, range(len(product_names))))

    for node in nodes:
        if node.startswith("u"):
            G.node[node]["fairness"] = 1
        else:
            G.node[node]["goodness"] = 1

    for edge in edges:
        G[edge[0]][edge[1]]["fairness"] = 1

    def Updating_goodness_of_product():
        print('Updating goodness of product')

        def calc_Gp(node):
            inedges = G.in_edges(node, data=True)

            ftotal = len(inedges)  # |In(p)|何レビューあるか
            gtotal = np.sum([edge[2]["fairness"] * edge[2]["weight"]
                             for edge in inedges])  # sum(R(u,p)) fairnessで重み付けしたweight

            if ftotal > 0.0:
                mean_rating_fairness = (
                    gtotal + beta1*median_gvals) / (ftotal + beta1)
            else:
                mean_rating_fairness = 0.0

            mean_rating_fairness = np.clip(
                mean_rating_fairness, -1, 1)  # -1~1にclipping

            return mean_rating_fairness

        # 現状の商品のgoodness
        currentgvals = [G.node[node]["goodness"]
                        for node in nodes if 'p' == node[0]]
        # goodnessの中央値(μg)
        median_gvals = np.median(currentgvals)

        dp = 0
        for node in nodes:
            if "p" not in node[0]:
                continue

            x = calc_Gp(node)
            dp += abs(G.node[node]["goodness"] - x)
            G.node[node]["goodness"] = x

        return dp

    def Updating_fairness_of_ratings():
        def calc_FR(edge):
            user_fairness = G.node[edge[0]]["fairness"]  # F(u)

            rating_distance = 1 - \
                (abs(edge[2]["weight"] - G.node[edge[1]]["goodness"]) / 2.0)

            x = (gamma1 * user_fairness + gamma2 *
                 rating_distance) / (gamma1 + gamma2)
            x = np.clip(x, 0, 1)
            return x

        print("Updating fairness of ratings")

        dr = 0

        for edge in edges:
            x = calc_FR(edge)
            dr += abs(edge[2]["fairness"] - x)
            G.adj[edge[0]][edge[1]]["fairness"] = x
        return dr

    def Updating_fairness_of_users():
        def calc_FU(node):
            outedges = G.out_edges(node, data=True)
            rating_fairness_sum = np.sum(
                [edge[2]["fairness"] for edge in outedges])

            x = (rating_fairness_sum + alpha1*median_fvals) / \
                (len(outedges) + alpha1)
            x = np.clip(x, 0, 1)
            return x

        print('updating fairness of users')

        currentfvals = [G.node[node]["fairness"]
                        for node in nodes if 'u' == node[0]]
        # Alternatively, we can use mean here, intead of median
        median_fvals = np.median(currentfvals)

        du = 0

        for node in nodes:
            if "u" not in node[0]:
                continue

            x = calc_FU(node)
            du += abs(G.node[node]["fairness"] - x)
            G.node[node]["fairness"] = x

        return du

    dp = 0
    du = 0
    dr = 0
    max_epochs = 100

    for epoch in range(max_epochs):
        print('-----------------')
        print("Epoch number %d with du = %f, dp = %f, dr = %f, for (%d,%d,%d,%d,%d,%d,%d)" % (
            epoch, du, dp, dr, alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3))
        if np.isnan(du) or np.isnan(dp) or np.isnan(dr):
            break

        dp = Updating_goodness_of_product()
        dr = Updating_fairness_of_ratings()
        du = Updating_fairness_of_users()

        if du < 0.01 and dp < 0.01 and dr < 0.01:
            break

    # result
    result_dir = Path(
        "./data/intermediate/rev2_results/fairness/") / NETWORKNAME
    if not result_dir.exists():
        result_dir.mkdir()

    df = nx.to_pandas_edgelist(G)

    user_fairness_df = pd.DataFrame([(x[0], x[1]['fairness']) for x in G.nodes(
        data=True) if x[0].startswith('u')], columns=['source', 'fairness'])
    product_goodness_df = pd.DataFrame([(x[0], x[1]['goodness']) for x in G.nodes(data=True) if x[0].startswith('p')],
                                       columns=['target', 'goodness'])
    edge_fairness_df = nx.to_pandas_edgelist(G)

    # Fairness of User
    currentfvals = [G.node[node]["fairness"]
                    for node in nodes if 'u' == node[0]]
    # Alternatively, we can use mean here, intead of median
    median_fvals = np.median(currentfvals)
    print(len(currentfvals), median_fvals)

    all_node_vals = []
    fair_node_vals = []

    for node in nodes:
        if "u" not in node[0]:
            continue
        f = G.node[node]["fairness"]
        all_node_vals.append(
            [node, (f - median_fvals) * np.log(G.out_degree(node) + 1), f, G.out_degree(node)])
        fair_node_vals.append(
            [node[1:], (f - median_fvals) * np.log(G.out_degree(node) + 1)])

    # sort users based on their scores
    all_node_vals_sorted = sorted(all_node_vals, key=lambda x: (
        float(x[1]), float(x[2]), -1 * float(x[3])))[::-1]
    fair_node_vals_sorted = sorted(
        fair_node_vals, key=lambda x: (float(x[1])))[::-1]

    fw = open(result_dir /
              f"fng-sorted-users-{alpha1}-{alpha2}-{beta1}-{beta2}-{gamma1}-{gamma2}-{gamma3}.csv", "w")

    for i, sl in enumerate(all_node_vals_sorted):
        # if sl[3] in badusers or sl[3] in goodusers:  # dont store users for which we dont have ground truth
        fw.write(f"{str(sl[0])},{str(sl[1])},{str(sl[2])},{str(sl[3])}\n")
    fw.close()

    fw = open(result_dir /
              f"only_fairnes-sorted-users-{alpha1}-{alpha2}-{beta1}-{beta2}-{gamma1}-{gamma2}-{gamma3}.csv", "w")

    for i, sl in enumerate(fair_node_vals_sorted):
        # if sl[3] in badusers or sl[3] in goodusers:  # dont store users for which we dont have ground truth
        fw.write(f"{str(sl[0])},{str(sl[1])}\n")
    fw.close()

    user_count = 0
    product_count = 0
    for node in nodes:
        if "u" in node[0]:
            user_count += 1

        if "p" in node[0]:
            product_count += 1

    print("total nodes %d" % len(list(nodes)))
    print("user nodes %d" % user_count)
    print("product nodes %d" % product_count)

    # 現状の商品のgoodness
    currentgvals = [G.node[node]["goodness"]
                    for node in nodes if 'p' == node[0]]
    # goodnessの中央値(μg)
    median_gvals = np.median(currentgvals)
    print(len(currentgvals), median_gvals)

    all_node_vals_good = []
    for node in nodes:
        if "p" not in node[0]:
            continue
        f = G.node[node]["goodness"]
        all_node_vals_good.append(
            [node, (f - median_gvals) * np.log(G.in_degree(node) + 1), f, G.in_degree(node)])

    # sort users based on their scores
    all_node_vals_good_sorted = sorted(all_node_vals_good, key=lambda x: (
        float(x[1]), float(x[2]), -1 * float(x[3])))[::-1]

    fw = open(f"./data/output/rev2_results/goodness/{NETWORKNAME}-" +
              f"good-sorted-users-{alpha1}-{alpha2}-{beta1}-{beta2}-{gamma1}-{gamma2}-{gamma3}.csv", "w")

    for i, sl in enumerate(all_node_vals_good_sorted):
        # if sl[3] in badusers or sl[3] in goodusers:  # dont store users for which we dont have ground truth
        fw.write(f"{str(sl[0])},{str(sl[1])},{str(sl[2])},{str(sl[3])}\n")
    fw.close()

    # 現在のリンクのriliability
    nx.to_pandas_edgelist(G).to_csv(result_dir /
                                    f'reliability-{alpha1}-{alpha2}-{beta1}-{beta2}-{gamma1}-{gamma2}-{gamma3}.csv')


if __name__ == '__main__':
    args = arg_parser()
    iterate(args)
