from datetime import date
from functools import partial
import re
import torch
import os
import wandb
import copy
from glob import glob
import omegaconf
import hydra
import utils.log as utils_logging
import utils.training_utils as tr_utils



class Tester():
    def __init__(
            self, args, network, diff_params, inference_train_set=None, inference_test_set=None, device=None,
            in_training=False,
            training_wandb_run=None
    ):
        self.args = args
        self.network = network
        self.diff_params = copy.copy(diff_params)
        self.device = device
        self.inference_train_set = inference_train_set
        self.inference_test_set = inference_test_set
        self.use_wandb = False  # hardcoded for now
        self.in_training = in_training
        self.sampler = hydra.utils.instantiate(args.tester.sampler, self.network, self.diff_params, self.args)

        if in_training:
            self.use_wandb = True
            # Will inherit wandb_run from Trainer
        else:  # If we use the tester in training, we will log in WandB in the Trainer() class, no need to create all these paths
            torch.backends.cudnn.benchmark = True
            if self.device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_wandb(self):
        """
        Configure wandb, open a new run and log the configuration.
        """
        config = omegaconf.OmegaConf.to_container(
            self.args, resolve=True, throw_on_missing=True
        )
        self.wandb_run = wandb.init(project="testing" + self.args.exp.wandb.project, entity=self.args.exp.wandb.entity,
                                    config=config)
        wandb.watch(self.network,
                    log_freq=self.args.logging.heavy_log_interval)
        blind_estimation = ("BLIND" + self.args.tester.blind_distortion.op_type + "_" +
                            self.args.tester.blind_distortion.op_hp.NLD + "_" +
                            self.args.tester.blind_distortion.op_hp.filters)
        self.wandb_run.name = self.args.exp.exp_name + "_" + blind_estimation
        self.use_wandb = True

    def setup_wandb_run(self, run):
        # get the wandb run object from outside (in trainer.py or somewhere else)
        self.wandb_run = run
        self.use_wandb = True

    def load_latest_checkpoint(self):
        # load the latest checkpoint from self.args.model_dir
        try:
            # find latest checkpoint_id
            save_basename = f"{self.args.exp.exp_name}-*.pt"
            save_name = f"{self.args.model_dir}/{save_basename}"
            list_weights = glob(save_name)
            id_regex = re.compile(f"{self.args.exp.exp_name}-(\d*)\.pt")
            list_ids = [int(id_regex.search(weight_path).groups()[0])
                        for weight_path in list_weights]
            checkpoint_id = max(list_ids)

            state_dict = torch.load(
                f"{self.args.model_dir}/{self.args.exp.exp_name}-{checkpoint_id}.pt", map_location=self.device)
            try:
                self.network.load_state_dict(state_dict['ema'])
            except Exception as e:
                print(e)
                print("Failed to load in strict mode, trying again without strict mode")
                self.network.load_state_dict(state_dict['model'], strict=False)

            print(f"Loaded checkpoint {checkpoint_id}")
            return True
        except (FileNotFoundError, ValueError):
            raise ValueError("No checkpoint found")

    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.it = state_dict['it']
        except:
            self.it = 0

        print(f"loading checkpoint {self.it}")
        return tr_utils.load_state_dict(state_dict, ema=self.network)

    def log_audio(self, pred, name: str):
        if self.use_wandb:
            print(pred.shape)
            pred = pred.view(-1)
            maxim = torch.max(torch.abs(pred)).detach().cpu().numpy()
            if maxim < 1:
                maxim = 1
            self.wandb_run.log(
                {name: wandb.Audio(pred.detach().cpu().numpy() / maxim, sample_rate=self.args.exp.sample_rate)},
                step=self.it)

            if self.args.logging.log_spectrograms:
                raise NotImplementedError

    # ------------- UNCONDITIONAL SAMPLING ---------------#

    ##############################
    ### UNCONDITIONAL SAMPLING ###
    ##############################


    def sample_unconditional(self, mode):
        # the audio length is specified in the args.exp, doesnt depend on the tester --> well should probably change that
        audio_len = self.args.exp.audio_len if not "audio_len" in self.args.tester.unconditional.keys() else self.args.tester.unconditional.audio_len

        shape = [self.args.tester.unconditional.num_samples, audio_len]

        preds, noise_init = self.sampler.predict_unconditional(shape, self.device)
        sigma_data = self.sampler.diff_params.sigma_data

        if self.use_wandb:
            # preds=preds/torch.max(torch.abs(preds))
            self.log_audio(preds[0], f"unconditional+{self.sampler.T}")  # Just log first sample
            # self.log_unconditional_metrics(preds) #But compute metrics on several
        else:
                    for i in range(len(preds)):
                        path_generated = utils_logging.write_audio_file(preds[i] ,
                                                                        self.args.exp.sample_rate,
                                                                        f"unconditional_{i}",
                                                                        path=self.paths["unconditional"])

        return preds

            


    def prepare_directories(self, mode, unconditional=False, blind=False, string=None):

        today = date.today()
        self.paths = {}
        if "overriden_name" in self.args.tester.keys() and self.args.tester.overriden_name is not None:
            self.path_sampling = os.path.join(self.args.model_dir, self.args.tester.overriden_name)
        else:
            self.path_sampling = os.path.join(self.args.model_dir, 'test' + today.strftime("%d_%m_%Y"))
        if not os.path.exists(self.path_sampling):
            os.makedirs(self.path_sampling)

        self.paths[mode] = os.path.join(self.path_sampling, mode, self.args.exp.exp_name)

        if not os.path.exists(self.paths[mode]):
            os.makedirs(self.paths[mode])
        if string is None:
            string = ""


    def save_experiment_args(self, mode):
        with open(os.path.join(self.paths[mode], ".argv"),
                  'w') as f:  # Keep track of the arguments we used for this experiment
            omegaconf.OmegaConf.save(config=self.args, f=f.name)

    def do_test(self, it=0):

        self.it = it
        print(self.args.tester.modes)
        for m in self.args.tester.modes:

            if m == "unconditional":
                print("testing unconditional")
                if not self.in_training:
                    self.prepare_directories(m, unconditional=True)
                    self.save_experiment_args(m)
                self.sample_unconditional(m)
            else:
                print("Warning: unknown mode: ", m)
