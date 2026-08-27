import yaml
import os
import argparse
import random
import logging
import numpy as np
import torch
from e3nn import o3
from ase.io import write

from minimace.data.read import read
from minimace.graph.atomic_data import AtomicData
from minimace.graph.dataset import AtomicDataset
from minimace.graph.utils import AtomicNumberTable
from minimace.graph.dataloader import DataLoader
from minimace.modules.models import MACE
from minimace.modules.blocks import RealAgnosticResidualInteractionBlock
from minimace.modules.utils import compute_avg_num_neighbors


# ====================== 日志初始化 ======================
def init_logger(log_save_dir: str):
    os.makedirs(log_save_dir, exist_ok=True)
    log_path = os.path.join(log_save_dir, "eval.log")
    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
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


# ====================== 配置加载 ======================
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["atomic_energies"] = {
        int(k): float(v) for k, v in config["atomic_energies"].items()
    }
    return config


# ====================== 随机种子固定 ======================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ====================== 读取测试帧 ======================
def read_frames(xyz_path: str, max_frames: int = None):
    frames = read(xyz_path)
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


# ====================== 构建原子序数表 ======================
def build_z_table(frames):
    unique_z = sorted({
        z for frame in frames for z in frame.atomic_numbers
    })
    z_table = AtomicNumberTable(unique_z)
    return z_table, unique_z


# ====================== 构建测试数据集 ======================
def build_dataset(frames, z_table, cutoff: float):
    atomic_datas = []
    for frame in frames:
        atomic_datas.append(
            AtomicData.from_frame(frame, z_table=z_table, cutoff=cutoff)
        )
    return AtomicDataset(atomic_datas)


# ====================== 原子孤立能 ======================
def build_atomic_energies(unique_z, E0s: dict):
    return np.array([E0s[z] for z in unique_z], dtype=np.float64)


# ====================== 构建MACE模型结构 ======================
def build_model(unique_z, atomic_energies, avg_num_neighbors, device, model_config: dict):
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


# ====================== 计算评估指标（修复版：移除no_grad）======================
def evaluate(model, loader, device):
    model.eval()
    energy_preds, energy_refs = [], []
    force_preds, force_refs = [], []

    # 【删除了 torch.no_grad()，计算受力必须保留梯度通路】
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

    return {
        "energy_mae": torch.mean(torch.abs(energy_preds - energy_refs)).item(),
        "energy_rmse": torch.sqrt(torch.mean((energy_preds - energy_refs) ** 2)).item(),
        "force_mae": torch.mean(torch.abs(force_preds - force_refs)).item(),
        "force_rmse": torch.sqrt(torch.mean((force_preds - force_refs) ** 2)).item(),
    }


# ====================== 全量预测能量+受力（修复版：移除no_grad）======================
def predict_all(model, loader, device):
    model.eval()
    all_energies = []
    all_forces = []

    # 【删除了 torch.no_grad()，计算受力必须保留梯度通路】
    for batch in loader:
        batch = batch.to(device)
        output = model(batch, training=False, compute_force=True)

        batch_energies = output["energy"].detach().cpu().numpy()
        batch_forces = output["forces"].detach().cpu().numpy()

        # 按结构拆分原子受力
        ptr = batch.ptr.cpu().numpy()
        for i in range(len(ptr) - 1):
            all_forces.append(batch_forces[ptr[i]:ptr[i + 1]])
            all_energies.append(batch_energies[i])

    return all_energies, all_forces


# ====================== 导出extxyz结果文件 ======================
def write_pred_xyz(frames, pred_energies, pred_forces, output_path):
    for i, frame in enumerate(frames):
        frame.info["MACE_energy"] = pred_energies[i]
        frame.new_array("MACE_forces", pred_forces[i])
    write(output_path, frames, format="extxyz")


# ====================== 主流程 ======================
def main():
    parser = argparse.ArgumentParser(description="MiniMACE 测试集评估脚本")
    parser.add_argument("--config", required=True, help="训练时使用的config.yaml")
    parser.add_argument("--model", required=True, help="训练好的模型权重 .pth 文件")
    parser.add_argument("--input", required=True, help="测试集 extxyz 文件路径")
    parser.add_argument("--output", default="test_pred.xyz", help="预测结果输出路径")
    parser.add_argument("--batch_size", type=int, default=None, help="评估用batch size，不指定则使用config中的值")
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    general_cfg = config["general"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    E0s = config["atomic_energies"]

    # 确定batch size
    batch_size = args.batch_size if args.batch_size is not None else data_cfg["batch_size"]

    # 日志
    log_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    logger = init_logger(log_dir)

    # 设备与种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(general_cfg["seed"])
    logger.info(f"Device: {device}")
    logger.info(f"Test file: {args.input}")
    logger.info(f"Model weight: {args.model}")
    logger.info(f"Batch size: {batch_size}")

    # 1. 读取测试数据
    frames = read_frames(args.input, max_frames=args.max_frames)
    logger.info(f"Loaded {len(frames)} test frames")

    # 2. 构建原子序数表
    z_table, unique_z = build_z_table(frames)
    logger.info(f"Atomic numbers: {unique_z}")

    # 3. 构建数据集与加载器
    dataset = build_dataset(frames, z_table, cutoff=data_cfg["cutoff"])
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 4. 平均邻居数
    if "avg_num_neighbors" in model_cfg:
        avg_num_neighbors = model_cfg["avg_num_neighbors"]
    else:
        avg_num_neighbors = compute_avg_num_neighbors(test_loader)
        logger.warning("使用测试集自动计算的avg_num_neighbors，建议写入config与训练集保持一致")
    logger.info(f"Avg num neighbors: {avg_num_neighbors:.4f}")

    # 5. 构建模型 + 加载权重 + 冻结参数
    atomic_energies = build_atomic_energies(unique_z, E0s)
    model = build_model(unique_z, atomic_energies, avg_num_neighbors, device, model_cfg)
    model.load_state_dict(torch.load(args.model, map_location=device))

    # 【新增】冻结模型所有参数，省显存，不影响算受力
    for param in model.parameters():
        param.requires_grad_(False)

    logger.info("Model weight loaded successfully, parameters frozen")

    # 6. 计算评估指标
    logger.info("\n" + "=" * 50)
    logger.info("Test Set Evaluation")
    logger.info("-" * 50)
    metrics = evaluate(model, test_loader, device)
    logger.info(f"Energy MAE  : {metrics['energy_mae']:.6f} eV")
    logger.info(f"Energy RMSE : {metrics['energy_rmse']:.6f} eV")
    logger.info(f"Force MAE   : {metrics['force_mae']:.6f} eV/Å")
    logger.info(f"Force RMSE  : {metrics['force_rmse']:.6f} eV/Å")
    logger.info("")
    logger.info(f"Energy MAE  : {metrics['energy_mae'] * 1000:.2f} meV")
    logger.info(f"Force MAE   : {metrics['force_mae'] * 1000:.2f} meV/Å")

    # 7. 全量预测并导出extxyz
    logger.info("\nExporting prediction results...")
    pred_energies, pred_forces = predict_all(model, test_loader, device)
    write_pred_xyz(frames, pred_energies, pred_forces, args.output)
    logger.info(f"Result saved to: {args.output}")
    logger.info("Evaluation finished!")


if __name__ == "__main__":
    main()