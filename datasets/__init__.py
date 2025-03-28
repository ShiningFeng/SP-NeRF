from .satellite_scene import SatelliteSceneDataset


def load_dataset(args, split):
    outputs = []
    if args.model == 'sp-nerf':
        outputs.append(SatelliteSceneDataset(depth_dir=args.depth_dir,
                                             json_dir=args.json_dir,
                                             img_dir=args.img_dir,
                                             sem_dir=args.sem_dir,
                                             aoi_id=args.aoi_id,
                                             cache_dir=args.cache_dir,
                                             split=split,
                                             img_downscale=args.img_downscale,
                                             stdscale=args.stdscale,
                                             margin=args.margin,
                                             dense_ss=args.dense_ss,
                                             sem_downscale=args.sem_downscale,
                                             sem=args.sem,
                                             num_sem_classes=args.num_sem_classes
                                             ))

    return outputs
