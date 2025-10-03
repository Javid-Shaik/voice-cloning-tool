"""
Training script for emotion-aware voice cloning model
"""
import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm

from src.voice_clone import VoiceCloneModel
from src.processor import AudioProcessor
from src.mixer import AudioMixer
from src.utils.logging import setup_logger
from src.dataset import VoiceCloneDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train emotion-aware voice cloning model")
    parser.add_argument("--config", type=str, default="configs/training.yaml",
                       help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_environment(config):
    # Set random seed
    torch.manual_seed(config['environment']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['environment']['seed'])
        torch.backends.cudnn.enabled = config['environment']['cudnn_enabled']
        torch.backends.cudnn.benchmark = config['environment']['benchmark']
        torch.backends.cudnn.deterministic = config['environment']['deterministic']
    
    # Create directories
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)

def setup_wandb(config, disabled=False):
    if not disabled and config['logging']['wandb']['enabled']:
        wandb.init(
            project=config['logging']['wandb']['project'],
            config=config
        )

def get_optimizer(model, config):
    if config['training']['optimizer']['type'] == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=(config['training']['optimizer']['beta1'],
                  config['training']['optimizer']['beta2']),
            weight_decay=config['training']['optimizer']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']['type']}")

def get_scheduler(optimizer, config):
    if config['training']['lr_scheduler']['type'] == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['lr_scheduler']['T_max'],
            eta_min=config['training']['lr_scheduler']['eta_min']
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config['training']['lr_scheduler']['type']}")

def train_epoch(model, train_loader, optimizer, scheduler, device, config):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move data to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch)
        
        # Calculate losses
        losses = {}
        for loss_name, weight in config['training']['loss_weights'].items():
            if loss_name in outputs:
                losses[loss_name] = outputs[loss_name] * weight
        
        total_batch_loss = sum(losses.values())
        
        # Backward pass
        total_batch_loss.backward()
        
        # Gradient clipping
        if config['training']['grad_clip'] > 0:
            nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])
        
        optimizer.step()
        
        total_loss += total_batch_loss.item()
        
        # Log batch metrics
        if wandb.run is not None:
            wandb.log({f"train/{k}_loss": v.item() for k, v in losses.items()})
    
    # Update learning rate
    scheduler.step()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device, config):
    model.eval()
    total_loss = 0
    metrics = {metric: 0.0 for metric in config['evaluation']['metrics']}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Calculate losses
            losses = {}
            for loss_name, weight in config['training']['loss_weights'].items():
                if loss_name in outputs:
                    losses[loss_name] = outputs[loss_name] * weight
            
            total_batch_loss = sum(losses.values())
            total_loss += total_batch_loss.item()
            
            # Calculate metrics
            for metric in config['evaluation']['metrics']:
                if metric in outputs:
                    metrics[metric] += outputs[metric].item()
    
    # Average metrics
    for metric in metrics:
        metrics[metric] /= len(val_loader)
    
    return total_loss / len(val_loader), metrics

def save_checkpoint(model, optimizer, scheduler, epoch, config, metrics, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config,
        'metrics': metrics
    }
    torch.save(checkpoint, path)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Setup
    setup_environment(config)
    logger = setup_logger('train', config['logging']['log_dir'])
    setup_wandb(config, disabled=args.no_wandb)
    writer = SummaryWriter(config['logging']['log_dir']) if config['logging']['tensorboard'] else None
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = VoiceCloneModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        emotion_dim=config['model']['emotion_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional']
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Data loading
    # Create datasets
    train_dataset = VoiceCloneDataset(
        data_dir=config['data']['train_data'],
        config=config,
        split="train"
    )
    
    val_dataset = VoiceCloneDataset(
        data_dir=config['data']['val_data'],
        config=config,
        split="val"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        prefetch_factor=config['data']['prefetch_factor'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        shuffle=False
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(start_epoch, config['training']['num_epochs']):
        logger.info(f"Starting epoch {epoch}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, config)
        logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}")
        
        # Validate
        val_loss, metrics = validate(model, val_loader, device, config)
        logger.info(f"Epoch {epoch} - Validation loss: {val_loss:.4f}")
        for metric, value in metrics.items():
            logger.info(f"Epoch {epoch} - {metric}: {value:.4f}")
        
        # Save checkpoint
        if epoch % config['checkpoint']['save_frequency'] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, config, metrics,
                os.path.join(config['checkpoint']['save_dir'], f'checkpoint_epoch_{epoch}.pt')
            )
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, config, metrics,
                os.path.join(config['checkpoint']['save_dir'], 'best_model.pt')
            )
        
        # Log metrics
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            for metric, value in metrics.items():
                writer.add_scalar(f'Metrics/{metric}', value, epoch)
        
        if wandb.run is not None:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
                **{f'metrics/{k}': v for k, v in metrics.items()}
            })
    
    logger.info("Training completed")
    if writer is not None:
        writer.close()
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()