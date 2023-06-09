import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neural_exploration import MultimodalContextualBandit, NeuralUCB, Model
import argparse
import pickle
from sklearn.model_selection import train_test_split
import os
import logging
from datetime import datetime

sns.set()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--num_items", type=int, default=2647)
    parser.add_argument("--num_pca", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--confidence_scaling_factor", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


def train(args, train_df, bandit, model, optimizer, logger):
    regrets = np.empty((len(train_df["user_id"]), args.T))
    for i, uid in enumerate(train_df["user_id"]):
        logger.info(f"Simulation {i} (User {uid}) Starts")
        logger.info("Bandit reset...")
        bandit.reset(uid)
        logger.info("Run UCB")
        neural_ucb = NeuralUCB(
            bandit,
            model=model,
            optimizer=optimizer,
            hidden_size=args.hidden_size,
            reg_factor=1.0,
            delta=0.1,
            confidence_scaling_factor=args.confidence_scaling_factor,
            training_window=100,
            p=args.dropout,
            learning_rate=0.01,
            epochs=args.epochs,
            train_every=args.train_every,
            use_cuda=torch.cuda.is_available(),
            logger=logger,
        )
        neural_ucb.run()
        regrets[i] = np.cumsum(neural_ucb.regrets)
        logger.info(f"regret: {regrets[i][-1]}")
    return (model, regrets)


def draw_figure(T, regrets, filepath):
    fig, ax = plt.subplots(figsize=(11, 4), nrows=1, ncols=1)
    t = np.arange(T)
    mean_regrets = np.mean(regrets, axis=0)
    std_regrets = np.std(regrets, axis=0) / np.sqrt(regrets.shape[0])
    ax.plot(t, mean_regrets)
    ax.fill_between(t, mean_regrets - 2 * std_regrets, mean_regrets + 2 * std_regrets, alpha=0.15)
    ax.set_title("Cumulative regret")
    plt.tight_layout()
    plt.show()
    fig.savefig(filepath)


def split_dataset(T, test_size):
    with open("/data/projects/contextual_bandit/ml-25m/ml-25m_inter.pkl", "rb") as pf:
        inter = pickle.load(pf)
    selected = [user_id for user_id, items in zip(inter["user_id"], inter["items"]) if len(items) >= T]
    inter = inter[inter["user_id"].isin(selected)]
    train_df, test_df = train_test_split(inter, test_size=test_size)
    return (train_df, test_df)


def get_logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    streaming_handler = logging.StreamHandler()
    streaming_handler.setFormatter(formatter)
    filename = "ml-25m.log"
    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(streaming_handler)
    logger.addHandler(file_handler)
    return logger


def evaluate(args, test_df, bandit, model, optimizer, logger):
    model.eval()
    metrics = {}
    metrics["regrets"] = np.empty((len(test_df["user_id"]), args.T))

    for i, uid in enumerate(test_df["user_id"]):
        logger.info(f"Simulation {i} (User {uid}) Starts")
        logger.info("Bandit reset...")
        bandit.reset(uid)
        neural_ucb = NeuralUCB(
            bandit,
            model=model,
            optimizer=optimizer,
            hidden_size=args.hidden_size,
            reg_factor=1.0,
            delta=0.1,
            confidence_scaling_factor=args.confidence_scaling_factor,
            training_window=100,
            p=args.dropout,
            learning_rate=0.01,
            epochs=args.epochs,
            train_every=args.T,
            use_cuda=torch.cuda.is_available(),
            logger=logger,
        )

        neural_ucb.run()
        metrics["regrets"][i] = np.cumsum(neural_ucb.regrets)

        ##### TODO: evaluation #####
        # neural_ucb.upper_confidence_bounds[t] -> score
        # bandit.rewards or test_df["items"][t] -> gt
        metrics["NDCG"] = None
        metrics["Recall"] = None
    return metrics


if __name__ == "__main__":
    args = parse_arguments()
    log_time = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    logger = get_logger(os.path.join("log", log_time))
    logger.info(args)

    train_df, test_df = split_dataset(args.T, args.test_size)
    logger.info(f"train size: {len(train_df)}, test size: {len(test_df)}")

    logger.info("Bandit setting...")
    bandit = MultimodalContextualBandit(
        T=args.T,
        num_items=args.num_items,
        data_path="/data/projects/contextual_bandit/ml-25m",
        simulator_path="/data/projects/contextual_bandit/simulator",
        seed=args.seed,
        num_pca=args.num_pca,
        topk=args.topk,
    )
    model = Model(
        input_size=bandit.n_features,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        p=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    model, regrets = train(args, train_df, bandit, model, optimizer, logger)
    draw_figure(args.T, regrets, "figures/neural_ucb.pdf")

    model.eval()
    metrics = evaluate(
        args, test_df, bandit, model, optimizer, logger
    )
