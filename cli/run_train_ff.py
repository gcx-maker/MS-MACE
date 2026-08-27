import sys
from pathlib import Path

# 当前脚本：minimace/cli/run_train.py
# __file__ → cli/run_train.py
# .parent.parent → minimace/ 包目录
# .parent.parent.parent → 外层 test/minimace/
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))


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
from minimace.graph.atomic_data import AtomicData, MSAtomicData
from minimace.graph.dataset import AtomicDataset
from minimace.graph.utils import AtomicNumberTable
from minimace.graph.dataloader import DataLoader
from minimace.modules.models import MACE, MSMACE
from minimace.modules.blocks import RealAgnosticResidualInteractionBlock, NonLinearReadoutBlock


class EMA:
    """
    模型参数指数移动平均（Exponential Moving Average）
    训练时维护影子参数，验证/推理时使用，提升模型泛化性与稳定性
    """
    def __init__(self, model, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}   # 影子参数（EMA平滑后的参数）
        self.backup = {}   # 原始参数备份，用于验证后恢复训练
        self._register()

    def _register(self):
        """初始化影子参数为模型当前参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """每个训练步后更新影子参数（in-place 高效实现）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        """将模型参数替换为影子参数，用于验证/推理"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """恢复模型原始参数，继续训练流程"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()


def tensor_to_voigt(stress):
    return torch.stack(
        [
            stress[:, 0, 0],  # xx
            stress[:, 1, 1],  # yy
            stress[:, 2, 2],  # zz
            stress[:, 1, 2],  # yz
            stress[:, 0, 2],  # xz
            stress[:, 0, 1],  # xy
        ],
        dim=-1,
    )


def init_logger(log_save_dir: str):
    os.makedirs(log_save_dir, exist_ok=True)
    log_path = os.path.join(log_save_dir, "train.log")

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


def load_config(config_path: str) -> dict:
    """加载YAML配置文件，自动转换原子序数键为int类型"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["atomic_energies"] = {
        int(k): float(v) for k, v in config["atomic_energies"].items()
    }
    return config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_frames(
    xyz_path: str,
    max_frames: int = None,
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    compute_stress: bool = False,
):
    frames = read(
        xyz_path,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        compute_stress=compute_stress,
    )

    if max_frames is not None:
        frames = frames[:max_frames]

    return frames


def build_z_table(frames):
    unique_z = sorted({
        z
        for frame in frames
        for z in frame.atomic_numbers
    })
    z_table = AtomicNumberTable(unique_z)
    return z_table, unique_z


def build_dataset(frames, z_table, model_type: str, model_config: dict):
    """
    - MACE 模式：调用 AtomicData.from_frame 构建单尺度图
    - MSMACE 模式：调用 MSAtomicData.from_frame 构建双尺度图
    """
    atomic_datas = []

    if model_type == "MACE":
        cutoff = model_config["r_max"]
        for frame in frames:
            atomic_datas.append(
                AtomicData.from_frame(frame, z_table=z_table, cutoff=cutoff)
            )
    elif model_type == "MSMACE":
        r_short = model_config["r_short"]
        r_long = model_config["r_max"]
        width = model_config["width"]
        for frame in frames:
            atomic_datas.append(
                MSAtomicData.from_frame(
                    frame,
                    z_table=z_table,
                    r_short=r_short,
                    r_long=r_long,
                    width=width
                )
            )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，可选 MACE / MSMACE")

    return AtomicDataset(atomic_datas)


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


def build_atomic_energies(unique_z, E0s: dict):
    atomic_energies = np.array(
        [E0s[z] for z in unique_z],
        dtype=np.float64,
    )
    return atomic_energies


def compute_avg_neighbors_by_edge(loader, edge_key: str = "edge_index"):
    """直接从指定边索引字段统计平均邻居数，兼容单/多尺度"""
    total_edges = 0
    total_nodes = 0
    for batch in loader:
        edge_index = getattr(batch, edge_key)
        total_edges += edge_index.shape[1]
        total_nodes += batch.num_nodes
    return total_edges / total_nodes


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
    short_hidden_irreps = o3.Irreps(model_config["short_hidden_irreps"])
    long_hidden_irreps = o3.Irreps(model_config["long_hidden_irreps"])
    MLP_irreps = o3.Irreps(model_config["MLP_irreps"])

    # 可选参数默认值对齐原版MACE
    radial_type = model_config.get("radial_type", "bessel")
    distance_transform = model_config.get("distance_transform", "None")
    apply_cutoff = model_config.get("apply_cutoff", True)
    use_reduced_cg = model_config.get("use_reduced_cg", True)
    use_so3 = model_config.get("use_so3", True)
    use_agnostic_product = model_config.get("use_agnostic_product", False)
    use_last_readout_only = model_config.get("use_last_readout_only", False)
    short_radial_MLP = model_config.get("short_radial_MLP", None)
    long_radial_MLP = model_config.get("long_radial_MLP", None)
    keep_last_layer_irreps = model_config.get("keep_last_layer_irreps", False)
    edge_irreps = model_config.get("edge_irreps", None)
    use_edge_irreps_first = model_config.get("use_edge_irreps_first", False)

    if edge_irreps is not None:
        edge_irreps = o3.Irreps(edge_irreps)

    model = MSMACE(
        # 多尺度几何核心参数
        r_max=model_config["r_max"],
        r_short=model_config["r_short"],
        width=model_config["width"],
        # 径向嵌入参数
        short_num_bessel=model_config["short_num_bessel"],
        long_num_bessel=model_config["long_num_bessel"],
        num_polynomial_cutoff=model_config["num_polynomial_cutoff"],
        radial_type=radial_type,
        distance_transform=distance_transform,
        apply_cutoff=apply_cutoff,
        # 等变基础配置
        max_ell=model_config["max_ell"],
        use_so3=use_so3,
        use_reduced_cg=use_reduced_cg,
        use_agnostic_product=use_agnostic_product,
        # 交互块类
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        # 层数配置
        short_num_interactions=model_config["short_num_interactions"],
        long_num_interactions=model_config["long_num_interactions"],
        # 原子基础属性
        num_elements=len(unique_z),
        atomic_numbers=unique_z,
        atomic_energies=atomic_energies,
        # 短程支路超参数
        short_hidden_irreps=short_hidden_irreps,
        MLP_irreps=MLP_irreps,
        short_correlation=model_config["short_correlation"],
        short_avg_num_neighbors=short_avg_num_neighbors,
        # 长程支路超参数
        long_hidden_irreps=long_hidden_irreps,
        long_correlation=model_config["long_correlation"],
        long_avg_num_neighbors=long_avg_num_neighbors,
        # 读出与激活配置
        gate=torch.nn.functional.silu,
        use_last_readout_only=use_last_readout_only,
        edge_irreps=edge_irreps,
        use_edge_irreps_first=use_edge_irreps_first,
        short_radial_MLP=short_radial_MLP,
        long_radial_MLP=long_radial_MLP,
        readout_cls=NonLinearReadoutBlock,
        keep_last_layer_irreps=keep_last_layer_irreps,
    )
    return model.to(device)


def train_one_epoch(
        model,
        loader,
        optimizer,
        device,
        grad_clip_norm: float,
        energy_weight: float = 1.0,
        force_weight: float = 100.0,
        stress_weight: float = 100.0,
        compute_stress: bool = False,
        ema=None,
):
    model.train()

    total_loss = 0.0
    total_energy_loss = 0.0
    total_force_loss = 0.0
    total_stress_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        output = model(batch, training=True, compute_force=True, compute_stress=compute_stress)
        num_atoms = batch.ptr[1:] - batch.ptr[:-1]

        pred_energy = output["energy"]
        pred_force = output["forces"]

        ref_energy = batch.energy
        ref_force = batch.forces

        energy_loss = torch.mean(((pred_energy - ref_energy) / num_atoms) ** 2)
        force_loss = torch.mean((pred_force - ref_force) ** 2)
        stress_loss = torch.tensor(
            0.0,
            device=device,
            dtype=energy_loss.dtype
        )
        if compute_stress:
            pred_stress = tensor_to_voigt(output["stress"])
            ref_stress = batch.stress
            stress_loss = torch.mean((pred_stress - ref_stress) ** 2)
        # 双权重加权损失
        loss = (
            energy_weight * energy_loss
            + force_weight * force_loss
            + stress_weight * stress_loss
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        # 每个训练步更新EMA影子参数
        if ema is not None:
            ema.update()

        total_loss += loss.item()
        total_energy_loss += energy_weight * energy_loss.item()
        total_force_loss += force_weight * force_loss.item()
        total_stress_loss += stress_weight * stress_loss.item()

    return {
        "loss": total_loss / len(loader),
        "energy_loss": total_energy_loss / len(loader),
        "force_loss": total_force_loss / len(loader),
        "stress_loss": total_stress_loss / len(loader),
    }


def validate_one_epoch(
        model,
        loader,
        device,
        energy_weight: float = 1.0,
        force_weight: float = 100.0,
        stress_weight: float = 100.0,
        compute_stress: bool = False,
):
    model.eval()

    total_loss = 0.0
    total_energy_loss = 0.0
    total_force_loss = 0.0
    total_stress_loss = 0.0

    for batch in loader:

        batch = batch.to(device)

        output = model(
            batch,
            training=False,
            compute_force=True,
            compute_stress=compute_stress,
        )

        num_atoms = batch.ptr[1:] - batch.ptr[:-1]

        energy_loss = torch.mean(
            ((output["energy"] - batch.energy) / num_atoms) ** 2
        )

        force_loss = torch.mean(
            (output["forces"] - batch.forces) ** 2
        )

        stress_loss = 0.0

        if compute_stress:
            stress_loss = torch.mean(
                (tensor_to_voigt(output["stress"]) - batch.stress) ** 2
            )

        loss = (
            energy_weight * energy_loss
            + force_weight * force_loss
            + stress_weight * stress_loss
        )

        total_loss += loss.item()
        total_energy_loss += energy_weight * energy_loss.item()
        total_force_loss += force_weight * force_loss.item()
        total_stress_loss += stress_weight * stress_loss.item()

    return {
        "loss": total_loss / len(loader),
        "energy_loss": total_energy_loss / len(loader),
        "force_loss": total_force_loss / len(loader),
        "stress_loss": total_stress_loss / len(loader),
    }


def evaluate(
        model,
        loader,
        device,
        compute_stress: bool = False,
):
    model.eval()

    energy_preds = []
    energy_refs = []

    force_preds = []
    force_refs = []

    stress_preds = []
    stress_refs = []

    for batch in loader:

        batch = batch.to(device)

        output = model(
            batch,
            training=False,
            compute_force=True,
            compute_stress=compute_stress,
        )

        energy_preds.append(
            output["energy"].detach().cpu()
        )
        energy_refs.append(
            batch.energy.detach().cpu()
        )

        force_preds.append(
            output["forces"].detach().cpu()
        )
        force_refs.append(
            batch.forces.detach().cpu()
        )

        if compute_stress:

            stress_preds.append(
                tensor_to_voigt(output["stress"]).detach().cpu()
            )

            stress_refs.append(
                batch.stress.detach().cpu()
            )

    energy_preds = torch.cat(energy_preds)
    energy_refs = torch.cat(energy_refs)

    force_preds = torch.cat(force_preds)
    force_refs = torch.cat(force_refs)

    energy_mae = torch.mean(
        torch.abs(energy_preds - energy_refs)
    )

    energy_rmse = torch.sqrt(
        torch.mean(
            (energy_preds - energy_refs) ** 2
        )
    )

    force_mae = torch.mean(
        torch.abs(force_preds - force_refs)
    )

    force_rmse = torch.sqrt(
        torch.mean(
            (force_preds - force_refs) ** 2
        )
    )

    metrics = {
        "energy_mae": energy_mae.item(),
        "energy_rmse": energy_rmse.item(),
        "force_mae": force_mae.item(),
        "force_rmse": force_rmse.item(),
    }

    if compute_stress:

        stress_preds = torch.cat(stress_preds)
        stress_refs = torch.cat(stress_refs)

        stress_mae = torch.mean(
            torch.abs(stress_preds - stress_refs)
        )

        stress_rmse = torch.sqrt(
            torch.mean(
                (stress_preds - stress_refs) ** 2
            )
        )

        metrics["stress_mae"] = stress_mae.item()
        metrics["stress_rmse"] = stress_rmse.item()

    return metrics


def quick_predict_epoch_sample(
        model,
        loader,
        device,
        sample_num=3,
        compute_stress=False,
        logger=None,
):
    model.eval()

    pred_err_list = []

    for batch in loader:

        batch = batch.to(device)

        out = model(
            batch,
            training=False,
            compute_force=True,
            compute_stress=compute_stress,
        )

        pred_e = out["energy"].detach().cpu()
        ref_e = batch.energy.detach().cpu()

        pred_f = out["forces"].detach().cpu()
        ref_f = batch.forces.detach().cpu()

        if compute_stress:
            pred_s = tensor_to_voigt(out["stress"]).detach().cpu()
            ref_s = batch.stress.detach().cpu()

        for i in range(pred_e.shape[0]):

            energy_err = torch.abs(
                pred_e[i] - ref_e[i]
            ).item()

            ptr = batch.ptr.cpu()
            atom_start = ptr[i].item()
            atom_end = ptr[i + 1].item()

            force_mae = torch.mean(
                torch.abs(
                    pred_f[atom_start:atom_end]
                    - ref_f[atom_start:atom_end]
                )
            ).item()

            if compute_stress:

                stress_mae = torch.mean(
                    torch.abs(
                        pred_s[i] - ref_s[i]
                    )
                ).item()

                pred_pressure = (
                                        pred_s[i, 0]
                                        + pred_s[i, 1]
                                        + pred_s[i, 2]
                                ) / 3.0

                ref_pressure = (
                                       ref_s[i, 0]
                                       + ref_s[i, 1]
                                       + ref_s[i, 2]
                               ) / 3.0

                pressure_mae = torch.abs(
                    pred_pressure - ref_pressure
                ).item()

            else:
                stress_mae = None
                pressure_mae = None

            pred_err_list.append(
                (
                    pred_e[i].item(),
                    ref_e[i].item(),
                    energy_err,
                    force_mae,
                    stress_mae,
                    pressure_mae,
                )
            )

            if len(pred_err_list) >= sample_num:
                break

        if len(pred_err_list) >= sample_num:
            break

    logger.info(
        f"  Quick Predict Sample (top {sample_num}):"
    )

    for idx, item in enumerate(pred_err_list):

        (p,r,e_err,f_err,s_err,p_err,) = item

        msg = (
            f"    Frame{idx}: "
            f"Pred={p:.6f} "
            f"Ref={r:.6f} "
            f"| EAbsErr={e_err:.6f} eV "
            f"| FMAE={f_err:.6f} eV/A"
        )

        if compute_stress:
            msg += (
                f" | SMAE={s_err:.6e}"
                f" | PMAE={p_err:.6e}"
            )

        logger.info(msg)


def predict_samples(
        model,
        loader,
        device,
        compute_stress=False,
        logger=None,
):
    model.eval()

    logger.info("\nPrediction Examples (Full Final)")
    logger.info("-" * 80)

    for batch in loader:

        batch = batch.to(device)

        output = model(
            batch,
            training=False,
            compute_force=True,
            compute_stress=compute_stress,
        )

        pred_e = output["energy"].detach().cpu()
        ref_e = batch.energy.detach().cpu()

        pred_f = output["forces"].detach().cpu()
        ref_f = batch.forces.detach().cpu()

        if compute_stress:
            pred_s = tensor_to_voigt(output["stress"]).detach().cpu()
            ref_s = batch.stress.detach().cpu()

        for i in range(len(pred_e)):

            energy_err = (
                pred_e[i] - ref_e[i]
            ).item()

            atom_start = batch.ptr[i].item()
            atom_end = batch.ptr[i + 1].item()

            force_mae = torch.mean(
                torch.abs(
                    pred_f[atom_start:atom_end]
                    - ref_f[atom_start:atom_end]
                )
            ).item()

            msg = (
                f"Pred={pred_e[i].item():.6f} "
                f"Ref={ref_e[i].item():.6f} "
                f"Err={energy_err:.6f} eV "
                f"| FMAE={force_mae:.6f} eV/A"
            )

        if compute_stress:
            stress_mae = torch.mean(
                torch.abs(
                    pred_s[i] - ref_s[i]
                )
            ).item()

            # Voigt stress:
            # [xx, yy, zz, yz, xz, xy]

            pred_pressure = (
                                    pred_s[i][0]
                                    + pred_s[i][1]
                                    + pred_s[i][2]
                            ) / 3.0

            ref_pressure = (
                                   ref_s[i][0]
                                   + ref_s[i][1]
                                   + ref_s[i][2]
                           ) / 3.0

            pressure_err = (
                    pred_pressure - ref_pressure
            ).item()

            msg += (
                f" | SMAE={stress_mae:.6e}"
                f" | PErr={pressure_err:.6e}"
            )
            logger.info(msg)

        break


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


def run_training_stage(
    stage_name: str,
    stage_cfg: dict,
    model,
    train_loader,
    valid_loader,
    device,
    ema,
    global_epoch: int,
    best_valid_loss: float,
    best_model_path: str,
    ckpt_save_dir: str,
    save_interval: int,
    eval_interval: int,
    sample_num_per_epoch: int,
    logger,
):

    stage_epochs = int(stage_cfg["num_epochs"])

    if stage_epochs <= 0:
        logger.info(
            f"Stage {stage_name} skipped (epochs <= 0)"
        )
        return global_epoch, best_valid_loss

    logger.info("\n" + "=" * 60)
    logger.info(f"Stage {stage_name} Training Start")
    logger.info(f"  Total epochs   : {stage_epochs}")
    logger.info(f"  Energy weight  : {stage_cfg['energy_weight']}")
    logger.info(f"  Force weight   : {stage_cfg['force_weight']}")
    logger.info(f"  Stress weight   : {stage_cfg['stress_weight']}")
    logger.info(f"  Learning rate  : {stage_cfg['lr']}")
    logger.info("=" * 60)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(stage_cfg["lr"]),
        weight_decay=float(
            stage_cfg.get("weight_decay", 0.0)
        ),
    )

    logger.info(
        f"Using AdamW optimizer, "
        f"weight_decay={stage_cfg.get('weight_decay',0.0)}"
    )

    lr_scheduler_type = stage_cfg.get(
        "lr_scheduler",
        "exponential",
    )

    scheduler = None

    if lr_scheduler_type == "exponential":

        lr_gamma = stage_cfg.get(
            "lr_gamma",
            0.9995,
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=lr_gamma,
        )

        logger.info(
            f"Using ExponentialLR, gamma={lr_gamma}"
        )

    elif lr_scheduler_type == "plateau":

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=stage_cfg.get(
                "lr_factor",
                0.5,
            ),
            patience=stage_cfg.get(
                "lr_patience",
                5,
            ),
        )

        logger.info(
            "Using ReduceLROnPlateau"
        )

    grad_clip_norm = stage_cfg.get(
        "grad_clip_norm",
        10.0,
    )

    energy_weight = float(
        stage_cfg["energy_weight"]
    )

    force_weight = float(
        stage_cfg["force_weight"]
    )

    stress_weight = float(
        stage_cfg.get(
            "stress_weight",
            0.0,
        )
    )

    compute_stress = bool(
        stage_cfg.get(
            "compute_stress",
            False,
        )
    )

    for epoch in range(stage_epochs):

        current_epoch = global_epoch + epoch + 1

        # =====================================
        # Train
        # =====================================
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=grad_clip_norm,
            energy_weight=energy_weight,
            force_weight=force_weight,
            stress_weight=stress_weight,
            compute_stress=compute_stress,
            ema=ema,
        )

        if ema is not None:
            ema.apply_shadow()

        # =====================================
        # Valid
        # =====================================
        valid_metrics = validate_one_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            energy_weight=energy_weight,
            force_weight=force_weight,
            stress_weight=stress_weight,
            compute_stress=compute_stress,
        )

        logger.info(
            f"[{stage_name}] "
            f"Epoch {current_epoch:03d} | "
            f"Train Loss {train_metrics['loss']:.6f} | "
            f"Valid Loss {valid_metrics['loss']:.6f}"
        )

        logger.info(
            f"    Train(E/F/S)=("
            f"{train_metrics['energy_loss']:.6e}, "
            f"{train_metrics['force_loss']:.6e}, "
            f"{train_metrics['stress_loss']:.6e})"
        )

        logger.info(
            f"    Valid(E/F/S)=("
            f"{valid_metrics['energy_loss']:.6e}, "
            f"{valid_metrics['force_loss']:.6e}, "
            f"{valid_metrics['stress_loss']:.6e})"
        )

        if current_epoch % eval_interval == 0:

            metrics = evaluate(
                model,
                valid_loader,
                device,
                compute_stress=compute_stress,
            )

            logger.info(
                f"  --- Detailed Metrics "
                f"(Epoch {current_epoch:03d}) ---"
            )

            logger.info(
                f"    Energy MAE : "
                f"{metrics['energy_mae']*1000:.6f} meV "
                f"| RMSE : {metrics['energy_rmse']*1000:.6f} meV"
            )

            logger.info(
                f"    Force  MAE : "
                f"{metrics['force_mae']*1000:.6f} meV/Å "
                f"| RMSE : {metrics['force_rmse']*1000:.6f} meV/Å"
            )

            if compute_stress:

                logger.info(
                    f"    Stress MAE : "
                    f"{metrics['stress_mae']:.6e} ev/Å3"
                )

                logger.info(
                    f"    Stress RMSE : "
                    f"{metrics['stress_rmse']:.6e} ev/Å3"
                )

            quick_predict_epoch_sample(
                model=model,
                loader=valid_loader,
                device=device,
                sample_num=sample_num_per_epoch,
                compute_stress=compute_stress,
                logger=logger,
            )

        current_valid_loss = valid_metrics["loss"]

        if current_valid_loss < best_valid_loss:

            best_valid_loss = current_valid_loss

            torch.save(
                model,
                best_model_path,
            )

            logger.info(
                f"  -> New best model saved "
                f"(Valid={current_valid_loss:.6f})"
            )

        if ema is not None:
            ema.restore()

        if current_epoch % save_interval == 0:

            ckpt_name = (
                f"model_epoch_"
                f"{current_epoch:03d}_"
                f"{stage_name.lower()}.pth"
            )

            ckpt_path = os.path.join(
                ckpt_save_dir,
                ckpt_name,
            )

            torch.save(
                model,
                ckpt_path,
            )

            logger.info(
                f"  -> Interval checkpoint saved: "
                f"{ckpt_path}"
            )

        if scheduler is not None:

            if lr_scheduler_type == "plateau":
                scheduler.step(current_valid_loss)
            else:
                scheduler.step()

    global_epoch += stage_epochs

    logger.info(
        f"\nStage {stage_name} Training Finished"
    )

    return global_epoch, best_valid_loss


def main():
    parser = argparse.ArgumentParser(description="MiniMACE Two-Stage Training with YAML config")
    parser.add_argument(
        "--config",
        default="ms_config.yaml",
        help="Path to YAML config file (default: ms_config.yaml)"
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

    # 读取模型类型
    model_type = model_cfg.get("model_type", "MACE")
    if model_type not in ["MACE", "MSMACE"]:
        raise ValueError(f"model_type 仅支持 MACE / MSMACE，当前为 {model_type}")

    # 路径统一使用 Path 对象，与脚本开头风格一致
    best_model_path = Path(paths_cfg["best_model_path"])
    ckpt_save_dir = Path(paths_cfg["ckpt_save_dir"])
    log_dir = ckpt_save_dir

    # 创建目录
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)

    # 初始化日志
    logger = init_logger(str(log_dir))
    logger.info(f"配置文件路径: {args.config}")
    logger.info(f"当前训练模型类型: {model_type}")
    logger.info("训练模式：双阶段训练，每阶段独立控制能量与力的损失权重")

    # 设备与种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(general_cfg["seed"])
    logger.info(f"Using device: {device}")

    # 全局训练通用配置
    use_ema = bool(train_cfg.get("use_ema", False))
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    save_interval = int(train_cfg.get("save_interval", 5))
    eval_interval = int(train_cfg.get("eval_interval", 1))
    sample_num_per_epoch = int(train_cfg.get("sample_num_per_epoch", 3))

    # 双阶段独立配置
    stage1_cfg = train_cfg["stage1"]
    stage2_cfg = train_cfg["stage2"]

    # 全局应力计算开关，同步到两个训练阶段，避免重复配置
    compute_stress = bool(data_cfg.get("compute_stress", False))
    stage1_cfg["compute_stress"] = compute_stress
    stage2_cfg["compute_stress"] = compute_stress

    # 1. 读取数据
    energy_key = data_cfg.get("energy_key", "REF_energy")
    forces_key = data_cfg.get("forces_key", "REF_forces")
    stress_key = data_cfg.get("stress_key", "REF_stress")
    frames = read_frames(
        paths_cfg["xyz_path"],
        max_frames=data_cfg["max_frames"],
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        compute_stress=compute_stress,
    )
    logger.info(f"Loaded {len(frames)} frames")

    # 2. 构建原子序数表
    z_table, unique_z = build_z_table(frames)
    logger.info(f"Unique atomic numbers: {unique_z}")

    # 3. 构建数据集（按模型类型自动选用对应数据类）
    dataset = build_dataset(
        frames,
        z_table=z_table,
        model_type=model_type,
        model_config=model_cfg
    )
    logger.info(f"Total dataset size: {len(dataset)}")

    # 4. 划分数据集与加载器
    train_loader, valid_loader = build_dataloader(
        dataset,
        batch_size=data_cfg["batch_size"],
        train_ratio=data_cfg["train_ratio"],
        seed=general_cfg["seed"],
    )
    logger.info(f"Train set size: {len(train_loader.dataset)} | Valid set size: {len(valid_loader.dataset)}")

    # 5. 计算平均邻居数
    if model_type == "MACE":
        avg_num_neighbors = compute_avg_neighbors_by_edge(train_loader, edge_key="edge_index")
        logger.info(f"Average number of neighbors: {avg_num_neighbors:.4f}")
    else:
        short_avg_num_neighbors = compute_avg_neighbors_by_edge(train_loader, edge_key="short_edge_index")
        long_avg_num_neighbors = compute_avg_neighbors_by_edge(train_loader, edge_key="long_edge_index")
        logger.info(f"Short-range avg neighbors: {short_avg_num_neighbors:.4f}")
        logger.info(f"Long-range avg neighbors : {long_avg_num_neighbors:.4f}")

    # 6. 构建原子孤立能
    atomic_energies = build_atomic_energies(unique_z, E0s)

    # 7. 构建模型
    if model_type == "MACE":
        model = build_model(
            unique_z=unique_z,
            atomic_energies=atomic_energies,
            avg_num_neighbors=avg_num_neighbors,
            device=device,
            model_config=model_cfg,
        )
    else:
        model = build_msmace(
            unique_z=unique_z,
            atomic_energies=atomic_energies,
            short_avg_num_neighbors=short_avg_num_neighbors,
            long_avg_num_neighbors=long_avg_num_neighbors,
            device=device,
            model_config=model_cfg,
        )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")

    # 初始化EMA（跨阶段连续维护）
    ema = None
    if use_ema:
        ema = EMA(model, decay=ema_decay)
        logger.info(f"Enabled EMA with decay={ema_decay}")

    # 8. 双阶段训练循环
    best_valid_loss = float("inf")
    global_epoch = 0

    # 第一阶段训练
    global_epoch, best_valid_loss = run_training_stage(
        stage_name="Stage1",
        stage_cfg=stage1_cfg,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        ema=ema,
        global_epoch=global_epoch,
        best_valid_loss=best_valid_loss,
        best_model_path=str(best_model_path),
        ckpt_save_dir=str(ckpt_save_dir),
        save_interval=save_interval,
        eval_interval=eval_interval,
        sample_num_per_epoch=sample_num_per_epoch,
        logger=logger,
    )

    # 第二阶段训练
    global_epoch, best_valid_loss = run_training_stage(
        stage_name="Stage2",
        stage_cfg=stage2_cfg,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        ema=ema,
        global_epoch=global_epoch,
        best_valid_loss=best_valid_loss,
        best_model_path=str(best_model_path),
        ckpt_save_dir=str(ckpt_save_dir),
        save_interval=save_interval,
        eval_interval=eval_interval,
        sample_num_per_epoch=sample_num_per_epoch,
        logger=logger,
    )

    logger.info("\n" + "=" * 60)
    logger.info("All Training Stages Finished")

    # 9. 加载最优完整模型进行最终评估
    best_model = torch.load(str(best_model_path), map_location=device)
    best_model.eval()
    metrics = evaluate(
        best_model,
        valid_loader,
        device,
        compute_stress=compute_stress
    )

    logger.info("\nFinal Evaluation (Best Model on Validation Set)")
    logger.info("-" * 60)
    logger.info(f"Energy MAE  : {metrics['energy_mae']:.6f} eV")
    logger.info(f"Energy RMSE : {metrics['energy_rmse']:.6f} eV")
    logger.info(f"Force MAE   : {metrics['force_mae']:.6f} eV/Å")
    logger.info(f"Force RMSE  : {metrics['force_rmse']:.6f} eV/Å")
    logger.info("")
    logger.info(f"Energy MAE  : {metrics['energy_mae'] * 1000:.2f} meV")
    logger.info(f"Force MAE   : {metrics['force_mae'] * 1000:.2f} meV/Å")

    # 开启应力时补充输出应力指标
    if compute_stress:
        logger.info("")
        logger.info(f"Stress MAE  : {metrics['stress_mae']:.6e}")
        logger.info(f"Stress RMSE : {metrics['stress_rmse']:.6e}")

    # 10. 完整预测样例输出
    predict_samples(
        best_model,
        valid_loader,
        device,
        compute_stress=compute_stress,
        logger=logger
    )
    predict_force_samples(best_model, valid_loader, device, logger=logger)


if __name__ == "__main__":
    main()