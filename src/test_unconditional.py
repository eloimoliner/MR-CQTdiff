import os
import re
import hydra
import torch


def _main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
        raise Exception(f"Model directory {args.model_dir} does not exist")

    args.exp.model_dir = args.model_dir

    #################
    ## diff params ##
    #################

    diff_params = hydra.utils.instantiate(args.diff_params)

    #############
    ## Network ##
    #############

    # it prints some logs.
    network = hydra.utils.instantiate(args.network)
    network = network.to(device)


    #############
    ## Tester  ##
    #############

    if args.tester.tester._target_ == "testing.tester.Tester":
        from testing.tester import Tester
    else:
        raise ValueError(f"tester target {args.tester.tester._target_} not recognized")

    tester = Tester(args=args, network=network, diff_params=diff_params, inference_train_set=None,
                    inference_test_set=None,
                    device=device)  # this will be used for making demos during training

    # Print options
    audio_len = args.exp.audio_len if not "audio_len" in args.tester.unconditional.keys() else args.tester.unconditional.audio_len

    if args.tester.checkpoint != 'None':
        ckpt_path = os.path.join(dirname, args.tester.checkpoint)

        try:
            # relative path
            ckpt_path = os.path.join(dirname, args.tester.checkpoint)
            tester.load_checkpoint(ckpt_path)
        except:
            # absolute path
            tester.load_checkpoint(os.path.join(args.model_dir, args.tester.checkpoint))
    else:
        print("trying to load latest checkpoint")
        tester.load_latest_checkpoint()

    tester.do_test()


@hydra.main(config_path="conf", config_name="conf", version_base=str(hydra.__version__))
def main(args):
    try:
        torch.cuda.set_device(args.gpu)
    except:
        pass
    _main(args)


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
