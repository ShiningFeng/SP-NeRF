from .spnerf import *


def load_model(args):
    if args.model == "sp-nerf":
        model = SPNeRF(num_sem_classes=args.num_sem_classes,
                       s_embedding_factor=args.s_embedding_factor,
                       layers=args.fc_layers,
                       feat=args.fc_units,
                       mapping=args.mapping,
                       t_embedding_dims=args.t_embbeding_tau,
                       beta=args.beta,
                       sem=args.sem)
    else:
        raise ValueError(f'model {args.model} is not valid')
    return model
