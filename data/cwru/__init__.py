from torch.utils.data import DataLoader
from .cwru import build_transfer_task
from .cwru import __cwru_class__


def transfer_task_time_fft(args):
    args.fft = False
    source_train_set, source_val_set, taregt_train_set, target_val_set = (
        build_transfer_task(args)
    )

    source_train_loader = DataLoader(
        dataset=source_train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    source_val_loader = DataLoader(
        dataset=source_val_set,
        batch_size=len(source_val_set),
        shuffle=False,
        drop_last=True,
    )
    target_train_loader = DataLoader(
        dataset=taregt_train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    target_val_loader = DataLoader(
        dataset=target_val_set,
        batch_size=len(target_val_set),
        shuffle=False,
        drop_last=True,
    )

    args.fft = True
    (
        source_train_set_fft,
        source_val_set_fft,
        taregt_train_set_fft,
        target_val_set_fft,
    ) = build_transfer_task(args)

    source_train_loader_fft = DataLoader(
        dataset=source_train_set_fft,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    source_val_loader_fft = DataLoader(
        dataset=source_val_set_fft,
        batch_size=len(source_val_set_fft),
        shuffle=False,
        drop_last=True,
    )
    target_train_loader_fft = DataLoader(
        dataset=taregt_train_set_fft,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    target_val_loader_fft = DataLoader(
        dataset=target_val_set_fft,
        batch_size=len(target_val_set_fft),
        shuffle=False,
        drop_last=True,
    )
    return (
        source_train_loader,
        source_val_loader,
        target_train_loader,
        target_val_loader,
        source_train_loader_fft,
        source_val_loader_fft,
        target_train_loader_fft,
        target_val_loader_fft,
    )
