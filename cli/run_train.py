import yaml
import os
import argparse
import random
import logging
import numpy as np
import torch
from torch.utils.data import random_split
from e3nn import o3

from minimace.data.read import read
from minimace.graph.atomic_data import AtomicData
from minimace.graph.dataset import AtomicDataset
from minimace.graph.utils import AtomicNumberTable
from minimace.graph.dataloader import DataLoader
from minimace.modules.models import MACE, MSMACE
from minimace.modules.blocks import RealAgnosticResidualInteractionBlock
from minimace.modules.utils import compute_avg_num_neighbors


# ====================== 日志初始化 ======================
def init_logger(log_save_dir: str):
    os.makedirs(log_save_dir, exist_ok=True)
    log_path = os.path.join(log_save_dir, "train.log")

    # 日志格式
    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 基础配置
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


# ==========================================================
# 配置加载
# ==========================================================
def load_config(config_path: str) -> dict:
    """加载YAML配置文件，自动转换原子序数键为int类型"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 原子能量表的键强制转int（兼容YAML数字键解析）
    config["atomic_energies"] = {
        int(k): float(v) for k, v in config["atomic_energies"].items()
    }
    return config


# ==========================================================
# 随机种子固定
# ==========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# 读取轨迹帧
# ==========================================================
def read_frames(xyz_path: str, max_frames: int = None):
    frames = read(xyz_path)
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


# ==========================================================
# 构建原子序数表
# ==========================================================
def build_z_table(frames):
    unique_z = sorted({
        z
        for frame in frames
        for z in frame.atomic_numbers
    })
    z_table = AtomicNumberTable(unique_z)
    return z_table, unique_z


# ==========================================================
# 构建数据集
# ==========================================================
def build_dataset(frames, z_table, cutoff: float):
    atomic_datas = []
    for frame in frames:
        atomic_datas.append(
            AtomicData.from_frame(
                frame,
                z_table=z_table,
                cutoff=cutoff,
            )
        )
    return AtomicDataset(atomic_datas)


# ==========================================================
# 构建数据加载器
# ==========================================================
def build_dataloader(
        dataset,
        batch_size: int = 2,
        train_ratio: float = 0.8,
        seed: int = 42,
):
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_valid = n_total - n_train
    generator = torch.Generator().manual_seed(seed)

    train_dataset, valid_dataset = random_split(
        dataset,
        [n_train, n_valid],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, valid_loader


# ==========================================================
# 原子孤立能（E0s）
# ==========================================================
def build_atomic_energies(unique_z, E0s: dict):
    atomic_energies = np.array(
        [E0s[z] for z in unique_z],
        dtype=np.float64,
    )
    return atomic_energies


# ==========================================================
# 构建MACE模型
# ==========================================================
def build_model(
        unique_z,
        atomic_energies,
        avg_num_neighbors: float,
        device,
        model_config: dict,
):
    hidden_irreps = o3.Irreps(model_config["hidden_irreps"])
    MLP_irreps = o3.Irreps(model_config["MLP_irreps"])

    model = MACE(
        r_max=model_config["r_max"],
        num_bessel=model_config["num_bessel"],
        num_polynomial_cutoff=model_config["num_polynomial_cutoff"],
        max_ell=model_config["max_ell"],
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        num_interactions=model_config["num_interactions"],
        num_elements=len(unique_z),
        hidden_irreps=hidden_irreps,
        MLP_irreps=MLP_irreps,
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=unique_z,
        correlation=model_config["correlation"],
        gate=torch.nn.functional.silu,
    )
    return model.to(device)


def build_msmace(
        unique_z,
        atomic_energies: np.ndarray,
        short_avg_num_neighbors: float,
        long_avg_num_neighbors: float,
        device: torch.device,
        model_config: dict,
):
    """
    构建多尺度 MS-MACE 模型，与原版单尺度 MACE 构建函数保持一致的编码风格
    """
    # 解析不可约表示配置
    short_hidden_irreps = o3.Irreps(model_config["short_hidden_irreps"])
    long_hidden_irreps = o3.Irreps(model_config["long_hidden_irreps"])
    short_MLP_irreps = o3.Irreps(model_config["short_MLP_irreps"])
    long_MLP_irreps = o3.Irreps(model_config["long_MLP_irreps"])

    # 可选参数读取，默认值与原版 MACE 保持对齐
    radial_type = model_config.get("radial_type", "bessel")
    distance_transform = model_config.get("distance_transform", "None")
    apply_cutoff = model_config.get("apply_cutoff", True)
    use_reduced_cg = model_config.get("use_reduced_cg", True)
    use_so3 = model_config.get("use_so3", True)
    use_agnostic_product = model_config.get("use_agnostic_product", False)
    use_last_readout_only = model_config.get("use_last_readout_only", False)
    radial_MLP = model_config.get("radial_MLP", None)
    keep_last_layer_irreps = model_config.get("keep_last_layer_irreps", False)
    edge_irreps = model_config.get("edge_irreps", None)
    use_edge_irreps_first = model_config.get("use_edge_irreps_first", False)

    if edge_irreps is not None:
        edge_irreps = o3.Irreps(edge_irreps)

    model = MSMACE(
        # ========== 多尺度几何核心参数 ==========
        r_max=model_config["r_max"],
        r_short=model_config["r_short"],
        width=model_config["width"],
        # ========== 径向嵌入参数 ==========
        num_bessel=model_config["num_bessel"],
        num_polynomial_cutoff=model_config["num_polynomial_cutoff"],
        radial_type=radial_type,
        distance_transform=distance_transform,
        apply_cutoff=apply_cutoff,
        # ========== 等变基础配置 ==========
        max_ell=model_config["max_ell"],
        use_so3=use_so3,
        use_reduced_cg=use_reduced_cg,
        use_agnostic_product=use_agnostic_product,
        # ========== 交互块类（与单尺度完全一致） ==========
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        # ========== 层数配置 ==========
        short_num_interactions=model_config["short_num_interactions"],
        long_num_interactions=model_config["long_num_interactions"],
        # ========== 原子基础属性 ==========
        num_elements=len(unique_z),
        atomic_numbers=unique_z,
        atomic_energies=atomic_energies,
        # ========== 短程支路超参数 ==========
        short_hidden_irreps=short_hidden_irreps,
        short_MLP_irreps=short_MLP_irreps,
        short_correlation=model_config["short_correlation"],
        short_avg_num_neighbors=short_avg_num_neighbors,
        # ========== 长程支路超参数 ==========
        long_hidden_irreps=long_hidden_irreps,
        long_MLP_irreps=long_MLP_irreps,
        long_correlation=model_config["long_correlation"],
        long_avg_num_neighbors=long_avg_num_neighbors,
        # ========== 读出与激活配置 ==========
        gate=torch.nn.functional.silu,
        use_last_readout_only=use_last_readout_only,
        edge_irreps=edge_irreps,
        use_edge_irreps_first=use_edge_irreps_first,
        radial_MLP=radial_MLP,
        keep_last_layer_irreps=keep_last_layer_irreps,
    )
    return model.to(device)


# ==========================================================
# 单轮训练（梯度裁剪）
# ==========================================================
def train_one_epoch(
        model,
        loader,
        optimizer,
        device,
        grad_clip_norm: float,
        force_weight: float = 100.0
):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        output = model(batch, training=True, compute_force=True)

        pred_energy = output["energy"]
        pred_force = output["forces"]
        ref_energy = batch.energy
        ref_force = batch.forces

        energy_loss = torch.mean((pred_energy - ref_energy) ** 2)
        force_loss = torch.mean((pred_force - ref_force) ** 2)
        loss = energy_loss + force_weight * force_loss

        optimizer.zero_grad()
        loss.backward()
        # 梯度全局裁剪，防止数值爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ==========================================================
# 单轮验证
# ==========================================================
def validate_one_epoch(
        model,
        loader,
        device,
        force_weight: float = 100.0,
):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        output = model(batch, training=False, compute_force=True)

        energy_loss = torch.mean((output["energy"] - batch.energy) ** 2)
        force_loss = torch.mean((output["forces"] - batch.forces) ** 2)
        loss = energy_loss + force_weight * force_loss
        total_loss += loss.item()

    return total_loss / len(loader)


# ==========================================================
# 指标评估
# ==========================================================
def evaluate(model, loader, device):
    model.eval()
    energy_preds, energy_refs = [], []
    force_preds, force_refs = [], []

    for batch in loader:
        batch = batch.to(device)
        output = model(batch, training=False, compute_force=True)

        energy_preds.append(output["energy"].detach().cpu())
        energy_refs.append(batch.energy.detach().cpu())
        force_preds.append(output["forces"].detach().cpu())
        force_refs.append(batch.forces.detach().cpu())

    energy_preds = torch.cat(energy_preds)
    energy_refs = torch.cat(energy_refs)
    force_preds = torch.cat(force_preds)
    force_refs = torch.cat(force_refs)

    energy_mae = torch.mean(torch.abs(energy_preds - energy_refs))
    energy_rmse = torch.sqrt(torch.mean((energy_preds - energy_refs) ** 2))
    force_mae = torch.mean(torch.abs(force_preds - force_refs))
    force_rmse = torch.sqrt(torch.mean((force_preds - force_refs) ** 2))

    return {
        "energy_mae": energy_mae.item(),
        "energy_rmse": energy_rmse.item(),
        "force_mae": force_mae.item(),
        "force_rmse": force_rmse.item(),
    }


# ==========================================================
# 每轮结束快速预测少量样本输出误差
# ==========================================================
def quick_predict_epoch_sample(model, loader, device, sample_num=3, logger=None):
    model.eval()
    pred_err_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch, training=False, compute_force=False)
            pred_e = out["energy"].cpu()
            ref_e = batch.energy.cpu()
            for p, r in zip(pred_e, ref_e):
                pred_err_list.append((p.item(), r.item(), abs(p - r).item()))
                if len(pred_err_list) >= sample_num:
                    break
            if len(pred_err_list) >= sample_num:
                break
    logger.info(f"  Quick Predict Sample (top {sample_num}):")
    for idx, (p, r, err) in enumerate(pred_err_list):
        logger.info(f"    Frame{idx}: Pred={p:.6f} Ref={r:.6f} | AbsErr={err:.6f} eV")


# ==========================================================
# 最终能量预测样例输出
# ==========================================================
def predict_samples(model, loader, device, logger=None):
    model.eval()
    logger.info("\nPrediction Examples (Full Final)")
    logger.info("-" * 60)

    for batch in loader:
        batch = batch.to(device)
        output = model(batch, training=False, compute_force=False)

        pred = output["energy"].cpu()
        ref = batch.energy.cpu()

        for i in range(len(pred)):
            error = pred[i] - ref[i]
            logger.info(
                f"Pred={pred[i].item():.6f} "
                f"Ref={ref[i].item():.6f} "
                f"Err={error.item():.6f}"
            )
        break


# ==========================================================
# 受力预测样例
# ==========================================================
def predict_force_samples(model, loader, device, logger=None, n_atoms: int = 5):
    model.eval()
    logger.info("\nForce Comparison")
    logger.info("-" * 60)

    batch = next(iter(loader))
    batch = batch.to(device)

    output = model(batch, training=False, compute_force=True)
    pred_force = output["forces"].detach().cpu()
    ref_force = batch.forces.detach().cpu()

    n_atoms = min(n_atoms, pred_force.shape[0])
    for i in range(n_atoms):
        err = pred_force[i] - ref_force[i]
        logger.info(f"\nAtom {i}")
        logger.info(f"Pred: {pred_force[i].numpy()}")
        logger.info(f"Ref : {ref_force[i].numpy()}")
        logger.info(f"Err : {err.numpy()}")


# ==========================================================
# 主流程
# ==========================================================
def main():
    # 命令行参数：指定配置文件路径
    parser = argparse.ArgumentParser(description="MiniMACE Training with YAML config")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)"
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    general_cfg = config["general"]
    paths_cfg = config["paths"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    E0s = config["atomic_energies"]

    # 路径定义
    best_model_path = paths_cfg["best_model_path"]
    ckpt_save_dir = paths_cfg["ckpt_save_dir"]
    log_dir = ckpt_save_dir  # 日志和ckpt放同一目录

    # 初始化日志
    logger = init_logger(log_dir)

    # 设备与种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(general_cfg["seed"])
    logger.info(f"Using device: {device}")

    # 训练超参读取
    num_epochs = int(train_cfg["num_epochs"])
    force_weight = float(train_cfg["force_weight"])
    lr = float(train_cfg["lr"])
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 10.0))
    save_interval = int(train_cfg.get("save_interval", 5))
    sample_num_per_epoch = int(train_cfg.get("sample_num_per_epoch", 3))

    # 创建目录
    best_dir = os.path.dirname(best_model_path)
    if best_dir:
        os.makedirs(best_dir, exist_ok=True)
    os.makedirs(ckpt_save_dir, exist_ok=True)

    # 1. 读取数据
    frames = read_frames(
        paths_cfg["xyz_path"],
        max_frames=data_cfg.get("max_frames", None),
    )
    logger.info(f"Loaded {len(frames)} frames")

    # 2. 构建原子序数表
    z_table, unique_z = build_z_table(frames)
    logger.info(f"Unique atomic numbers: {unique_z}")

    # 3. 构建数据集
    dataset = build_dataset(frames, z_table, cutoff=data_cfg["cutoff"])

    # 4. 划分数据集与加载器
    train_loader, valid_loader = build_dataloader(
        dataset,
        batch_size=data_cfg["batch_size"],
        train_ratio=data_cfg["train_ratio"],
        seed=general_cfg["seed"],
    )

    # 5. 计算平均邻居数
    avg_num_neighbors = compute_avg_num_neighbors(train_loader)
    logger.info(f"Average number of neighbors: {avg_num_neighbors:.4f}")

    # 6. 构建原子孤立能
    atomic_energies = build_atomic_energies(unique_z, E0s)

    # 7. 构建模型
    model = build_model(
        unique_z,
        atomic_energies,
        avg_num_neighbors,
        device,
        model_cfg,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")

    # 8. 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 9. 训练循环
    best_valid_loss = float("inf")

    logger.info("\nStart Training")
    logger.info("=" * 60)
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            grad_clip_norm=grad_clip_norm, force_weight=force_weight
        )
        valid_loss = validate_one_epoch(
            model, valid_loader, device, force_weight=force_weight
        )

        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.6f} | "
            f"Valid Loss {valid_loss:.6f}"
        )

        # 每轮结束快速预测输出误差
        quick_predict_epoch_sample(model, valid_loader, device, sample_num=sample_num_per_epoch, logger=logger)

        # 保存最优模型
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  -> New best model saved (Valid={valid_loss:.6f})")

        # 每save_interval轮保存一次断点模型到独立文件夹
        if (epoch + 1) % save_interval == 0:
            ckpt_name = f"model_epoch_{epoch+1:03d}.pth"
            ckpt_path = os.path.join(ckpt_save_dir, ckpt_name)
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  -> Interval checkpoint saved: {ckpt_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Training Finished")

    # 10. 加载最优模型评估
    model.load_state_dict(
        torch.load(best_model_path, map_location=device)
    )
    metrics = evaluate(model, valid_loader, device)

    logger.info("\nFinal Evaluation (Validation Set)")
    logger.info("-" * 60)
    logger.info(f"Energy MAE  : {metrics['energy_mae']:.6f} eV")
    logger.info(f"Energy RMSE : {metrics['energy_rmse']:.6f} eV")
    logger.info(f"Force MAE   : {metrics['force_mae']:.6f} eV/Å")
    logger.info(f"Force RMSE  : {metrics['force_rmse']:.6f} eV/Å")
    logger.info("")
    logger.info(f"Energy MAE  : {metrics['energy_mae'] * 1000:.2f} meV")
    logger.info(f"Force MAE   : {metrics['force_mae'] * 1000:.2f} meV/Å")

    # 11. 完整预测样例输出
    predict_samples(model, valid_loader, device, logger=logger)
    predict_force_samples(model, valid_loader, device, logger=logger)


if __name__ == "__main__":
    main()