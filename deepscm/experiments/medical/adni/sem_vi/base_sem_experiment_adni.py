import pyro

from typing import Mapping

from pyro.infer import SVI, TraceGraph_ELBO
from pyro.nn import pyro_method
from pyro.optim import Adam
from torch.distributions import Independent

import torch
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.transforms import (
    ComposeTransform, AffineTransform, ExpTransform, Spline
)
from pyro.distributions import LowRankMultivariateNormal, MultivariateNormal, Normal, TransformedDistribution
from deepscm.arch.medical import Decoder, Encoder
from deepscm.distributions.transforms.reshape import ReshapeTransform
from deepscm.distributions.transforms.affine import LowerCholeskyAffine

from deepscm.distributions.deep import DeepMultivariateNormal, DeepIndepNormal, Conv2dIndepNormal, DeepLowRankMultivariateNormal

import numpy as np

from deepscm.experiments.medical.base_experiment_adni import BaseCovariateExperiment, BaseSEM, EXPERIMENT_REGISTRY, MODEL_REGISTRY 

class CustomELBO(TraceGraph_ELBO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trace_storage = {'model': None, 'guide': None}

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)

        self.trace_storage['model'] = model_trace
        self.trace_storage['guide'] = guide_trace

        return model_trace, guide_trace


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class BaseVISEM(BaseSEM):
    context_dim = 0 # number of context dimensions for decoder 

    def __init__(self, latent_dim: int, logstd_init: float = -5, enc_filters: str = '16,32,64,128', dec_filters: str = '128,64,32,16',
                 num_convolutions: int = 2, use_upconv: bool = False, decoder_type: str = 'fixed_var', decoder_cov_rank: int = 10, eps=0.01, **kwargs):
        super().__init__(**kwargs)

        self.img_shape = (1, 192 // self.downsample, 192 // self.downsample) if self.downsample > 0 else (1, 192, 192)

        self.latent_dim = latent_dim
        self.logstd_init = logstd_init

        self.enc_filters = tuple(int(f.strip()) for f in enc_filters.split(','))
        self.dec_filters = tuple(int(f.strip()) for f in dec_filters.split(','))
        self.num_convolutions = num_convolutions
        self.use_upconv = use_upconv
        self.decoder_type = decoder_type
        self.decoder_cov_rank = decoder_cov_rank

        # decoder parts 
        decoder = Decoder(
            num_convolutions=self.num_convolutions, filters=self.dec_filters,
            latent_dim=self.latent_dim + self.context_dim, upconv=self.use_upconv,
            output_size=self.img_shape
            )

        if self.decoder_type == 'fixed_var':
            self.decoder = Conv2dIndepNormal(decoder, 1, 1)

            torch.nn.init.zeros_(self.decoder.logvar_head.weight)
            self.decoder.logvar_head.weight.requires_grad = False

            torch.nn.init.constant_(self.decoder.logvar_head.bias, self.logstd_init)
            self.decoder.logvar_head.bias.requires_grad = False
        elif self.decoder_type == 'learned_var':
            self.decoder = Conv2dIndepNormal(decoder, 1, 1)

            torch.nn.init.zeros_(self.decoder.logvar_head.weight)
            self.decoder.logvar_head.weight.requires_grad = False

            torch.nn.init.constant_(self.decoder.logvar_head.bias, self.logstd_init)
            self.decoder.logvar_head.bias.requires_grad = True
        elif self.decoder_type == 'independent_gaussian':
            self.decoder = Conv2dIndepNormal(decoder, 1, 1)

            torch.nn.init.zeros_(self.decoder.logvar_head.weight)
            self.decoder.logvar_head.weight.requires_grad = True

            torch.nn.init.normal_(self.decoder.logvar_head.bias, self.logstd_init, 1e-1)
            self.decoder.logvar_head.bias.requires_grad = True
        elif self.decoder_type == 'multivariate_gaussian':
            seq = torch.nn.Sequential(decoder, Lambda(lambda x: x.view(x.shape[0], -1)))
            self.decoder = DeepMultivariateNormal(seq, np.prod(self.img_shape), np.prod(self.img_shape))
        elif self.decoder_type == 'sharedvar_multivariate_gaussian':
            seq = torch.nn.Sequential(decoder, Lambda(lambda x: x.view(x.shape[0], -1)))
            self.decoder = DeepMultivariateNormal(seq, np.prod(self.img_shape), np.prod(self.img_shape))

            torch.nn.init.zeros_(self.decoder.logdiag_head.weight)
            self.decoder.logdiag_head.weight.requires_grad = False

            torch.nn.init.zeros_(self.decoder.lower_head.weight)
            self.decoder.lower_head.weight.requires_grad = False

            torch.nn.init.normal_(self.decoder.logdiag_head.bias, self.logstd_init, 1e-1)
            self.decoder.logdiag_head.bias.requires_grad = True
        elif self.decoder_type == 'lowrank_multivariate_gaussian':
            seq = torch.nn.Sequential(decoder, Lambda(lambda x: x.view(x.shape[0], -1)))
            self.decoder = DeepLowRankMultivariateNormal(seq, np.prod(self.img_shape), np.prod(self.img_shape), decoder_cov_rank)
        elif self.decoder_type == 'sharedvar_lowrank_multivariate_gaussian':
            seq = torch.nn.Sequential(decoder, Lambda(lambda x: x.view(x.shape[0], -1)))
            self.decoder = DeepLowRankMultivariateNormal(seq, np.prod(self.img_shape), np.prod(self.img_shape), decoder_cov_rank)

            torch.nn.init.zeros_(self.decoder.logdiag_head.weight)
            self.decoder.logdiag_head.weight.requires_grad = False

            torch.nn.init.zeros_(self.decoder.factor_head.weight)
            self.decoder.factor_head.weight.requires_grad = False

            torch.nn.init.normal_(self.decoder.logdiag_head.bias, self.logstd_init, 1e-1)
            self.decoder.logdiag_head.bias.requires_grad = True
        else:
            raise ValueError(f'unknown decoder type {self.decoder_type}.')
        
        # encoder parts 
        self.encoder = Encoder(
            num_convolutions=self.num_convolutions, 
            filters=self.enc_filters, 
            latent_dim=self.latent_dim, 
            input_size=self.img_shape
            )
        
        latent_layers = torch.nn.Sequential(torch.nn.Linear(self.latent_dim + self.context_dim, self.latent_dim), torch.nn.ReLU())
        self.latent_encoder = DeepIndepNormal(latent_layers, self.latent_dim, self.latent_dim)

        # Priors
        # priors for discrete variables 
        self.sex_logits = torch.nn.Parameter(torch.zeros([1, ])) 

        self.register_buffer('slice_number_min', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('slice_number_max', 12.*torch.ones([1, ], requires_grad=False)+1.)

        self.register_buffer('apoE_min', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('apoE_max', 3.*torch.ones([1, ], requires_grad=False)+1.)

        # priors for continuous variables - note base and flow priors 
        for k in self.required_data - {'sex', 'x', 'slice_number', 'APOE4'}:  
            self.register_buffer(f'{k}_base_loc', torch.zeros([1, ], requires_grad=False))
            self.register_buffer(f'{k}_base_scale', torch.ones([1, ], requires_grad=False)) 
        
        for k in self.required_data - {'sex', 'x', 'slice_number', 'APOE4'}:
            self.register_buffer(f'{k}_flow_lognorm_loc', torch.zeros([], requires_grad=False))
            self.register_buffer(f'{k}_flow_lognorm_scale', torch.ones([], requires_grad=False))

        # image prior 
        self.register_buffer('x_base_loc', torch.zeros(self.img_shape, requires_grad=False))
        self.register_buffer('x_base_scale', torch.ones(self.img_shape, requires_grad=False))

        # latent space prior 
        self.register_buffer('z_loc', torch.zeros([latent_dim, ], requires_grad=False))
        self.register_buffer('z_scale', torch.ones([latent_dim, ], requires_grad=False))
        
        # flows 
        # First flows with no parents which are just modelled with a spline 
        # age flow 
        self.age_flow_components = ComposeTransformModule([Spline(1)])
        self.age_flow_lognorm = AffineTransform(loc=self.age_flow_lognorm_loc.item(), scale=self.age_flow_lognorm_scale.item())
        self.age_flow_constraint_transforms = ComposeTransform([self.age_flow_lognorm, ExpTransform()])
        self.age_flow_transforms = ComposeTransform([self.age_flow_components, self.age_flow_constraint_transforms])

        # tau flow 
        self.tau_flow_components = ComposeTransformModule([Spline(1)])
        self.tau_flow_lognorm = AffineTransform(loc=self.tau_flow_lognorm_loc.item(), scale=self.tau_flow_lognorm_scale.item())
        self.tau_flow_constraint_transforms = ComposeTransform([self.tau_flow_lognorm, ExpTransform()])
        self.tau_flow_transforms = ComposeTransform([self.tau_flow_components, self.tau_flow_constraint_transforms])

        # education flow 
        self.education_flow_components = ComposeTransformModule([Spline(1)])
        self.education_flow_lognorm = AffineTransform(loc=self.education_flow_lognorm_loc.item(), scale=self.education_flow_lognorm_scale.item())
        self.education_flow_constraint_transforms = ComposeTransform([self.education_flow_lognorm, ExpTransform()])
        self.education_flow_transforms = ComposeTransform([self.education_flow_components, self.education_flow_constraint_transforms])

        # other flows' shared components 
        # all flows require an affine normalisation and exponentiation transformation 
        self.ventricle_volume_flow_lognorm = AffineTransform(loc=self.ventricle_volume_flow_lognorm_loc.item(), scale=self.ventricle_volume_flow_lognorm_scale.item())  # noqa: E501
        self.ventricle_volume_flow_constraint_transforms = ComposeTransform([self.ventricle_volume_flow_lognorm, ExpTransform()])

        self.brain_volume_flow_lognorm = AffineTransform(loc=self.brain_volume_flow_lognorm_loc.item(), scale=self.brain_volume_flow_lognorm_scale.item())
        self.brain_volume_flow_constraint_transforms = ComposeTransform([self.brain_volume_flow_lognorm, ExpTransform()])    

        self.av45_flow_lognorm = AffineTransform(loc=self.av45_flow_lognorm_loc.item(), scale=self.av45_flow_lognorm_scale.item())
        self.av45_flow_constraint_transforms = ComposeTransform([self.av45_flow_lognorm, ExpTransform()]) 

        self.moca_flow_lognorm = AffineTransform(loc=self.moca_flow_lognorm_loc.item(), scale=self.moca_flow_lognorm_scale.item())
        self.moca_flow_constraint_transforms = ComposeTransform([self.moca_flow_lognorm, ExpTransform()])
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if 'flow_lognorm_loc' in name:
            name_ = name.replace('flow_lognorm_loc', '')
            getattr(self, name_ + 'flow_lognorm').loc = value.item()
        elif 'flow_lognorm_scale' in name:
            name_ = name.replace('flow_lognorm_scale', '')
            getattr(self, name_ + 'flow_lognorm').scale = value.item()
    
    def _get_preprocess_transforms(self):
        return super()._get_preprocess_transforms().inv
    
    def _get_transformed_x_dist(self, latent):
        x_pred_dist = self.decoder.predict(latent) # returns a normal distribution with mean of the predicted image 
        x_base_dist = Normal(self.x_base_loc, self.x_base_scale).to_event(3) # 3 dimensions starting from right dep. 

        preprocess_transform = self._get_preprocess_transforms()

        if isinstance(x_pred_dist, MultivariateNormal) or isinstance(x_pred_dist, LowRankMultivariateNormal):
            chol_transform = LowerCholeskyAffine(x_pred_dist.loc, x_pred_dist.scale_tril)
            reshape_transform = ReshapeTransform(self.img_shape, (np.prod(self.img_shape), ))
            x_reparam_transform = ComposeTransform([reshape_transform, chol_transform, reshape_transform.inv])
        elif isinstance(x_pred_dist, Independent):
            x_pred_dist = x_pred_dist.base_dist
            x_reparam_transform = AffineTransform(x_pred_dist.loc, x_pred_dist.scale, 3)
        else:
            raise ValueError(f'{x_pred_dist} not valid.')

        return TransformedDistribution(x_base_dist, ComposeTransform([x_reparam_transform, preprocess_transform]))

    @pyro_method
    def guide(self, obs):
        raise NotImplementedError()
    
    @pyro_method
    def svi_guide(self, obs):
        self._check_observation(obs)
        self.guide(obs)
    
    @pyro_method 
    def svi_model(self, obs):
        self._check_observation(obs)
        batch_size = obs['x'].shape[0]
        with pyro.plate('observations', batch_size):
            pyro.condition(self.model, data=obs)()
    
    @pyro_method
    def infer_z(self, *args, **kwargs):
        return self.guide(*args, **kwargs)
    
    @property
    def required_data(self):
        return {'x', 'sex', 'age', 'ventricle_volume', 'brain_volume', 'tau', 'av45', 
        'moca', 'education', 'APOE4', 'slice_number'}

    def _check_observation(self, obs):
        keys = obs.keys()
        assert self.required_data == set(keys), f'Incompatible observation: {tuple(keys)}' 
    
    @pyro_method 
    def infer(self, obs): 
        self._check_observation(obs)
        
        obs_ = obs.copy()
        
        z = self.infer_z(obs_)
        obs_.update(dict(z=z))
        
        exogenous = self.infer_exogeneous(obs_)
        exogenous['z'] = z

        return exogenous
    
    @pyro_method
    def reconstruct(self, obs, num_particles:int=1): 
        self._check_observation(obs)

        z_dist = pyro.poutine.trace(self.guide).get_trace(obs).nodes['z']['fn']

        batch_size = obs['x'].shape[0]
        obs_ = {k: v for k, v in obs.items() if k != 'x'}

        recons = []
        for _ in range(num_particles): 
            z = pyro.sample('z', z_dist) 
            obs_.update({'z': z})
            recon  = pyro.poutine.condition(
                self.sample, data=obs_)(batch_size)
            
            recons += [recon['x']]
        return torch.stack(recons).mean(0)

    def _cf_dict(self, counterfactuals):
        out = {k: [] for k in self.required_data}
        for cf in counterfactuals:
            for k in self.required_data:
                out[k].append(cf[k])
        out = {k: torch.mean(torch.tensor(torch.stack(v), dtype=float), dim=0) for k, v in out.items()}
        return out

    @pyro_method
    def counterfactual(self, obs, condition: Mapping = None, num_particles: int = 1): 
        self._check_observation(obs)
        obs_ = obs.copy()

        z_dist = pyro.poutine.trace(self.guide).get_trace(obs_).nodes['z']['fn'] # variational posterior
        n = obs_['x'].shape[0]

        counterfactuals = []
        for _ in range(num_particles): 
            z = pyro.sample('z', z_dist)
            obs_.update(dict(z=z))
            exogenous = self.infer_exogeneous(obs_)
            exogenous['z'] = z
            # condition on these variables if they aren't included in 'do' as they are root nodes 
            # and we don't have the exogenous noise for them yet 
            if 'sex' not in condition.keys(): 
                exogenous['sex'] = obs_['sex']
            if 'slice_number' not in condition.keys(): 
                exogenous['slice_number'] = obs_['slice_number']
            if 'APOE4' not in condition.keys(): 
                exogenous['APOE4'] = obs_['APOE4']
            
            counter = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data=exogenous), data=condition)(n)
            counterfactuals.append(counter)
            
        
        return self._cf_dict(counterfactuals)
    
    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument('--latent_dim', default=100, type=int, help="latent dimension of model (default: %(default)s)")
        parser.add_argument('--logstd_init', default=-5, type=float, help="init of logstd (default: %(default)s)")
        parser.add_argument('--enc_filters', default='16,24,32,64,128', type=str, help="number of filters to use (default: %(default)s)")
        parser.add_argument('--dec_filters', default='128,64,32,24,16', type=str, help="number of filters to use (default: %(default)s)")
        parser.add_argument('--num_convolutions', default=3, type=int, help="number of convolutions to build model (default: %(default)s)")
        parser.add_argument('--use_upconv', default=False, action='store_true', help="toogle upconv (default: %(default)s)")
        parser.add_argument(
            '--decoder_type', default='fixed_var', help="var type (default: %(default)s)",
            choices=['fixed_var', 'learned_var', 'independent_gaussian', 'sharedvar_multivariate_gaussian', 'multivariate_gaussian',
                     'sharedvar_lowrank_multivariate_gaussian', 'lowrank_multivariate_gaussian'])
        parser.add_argument('--decoder_cov_rank', default=10, type=int, help="rank for lowrank cov approximation (requires lowrank decoder) (default: %(default)s)")  # noqa: E501

        return parser
    
class SVIExperiment(BaseCovariateExperiment):
    def __init__(self, hparams, pyro_model: BaseSEM):
        super().__init__(hparams, pyro_model)

        self.svi_loss = CustomELBO(num_particles=hparams.num_svi_particles)

        self._build_svi()
    
    def _build_svi(self, loss=None):
        def per_param_callable(module_name, param_name):
            params = {'eps': 1e-5, 'amsgrad': self.hparams.use_amsgrad, 'weight_decay': self.hparams.l2}
            if 'flow_components' in module_name or 'sex_logits' in param_name:
                params['lr'] = self.hparams.pgm_lr
            else:
                params['lr'] = self.hparams.lr

            print(f'building opt for {module_name} - {param_name} with p: {params}')
            return params

        if loss is None:
            loss = self.svi_loss

        if self.hparams.use_cf_guide:
            def guide(*args, **kwargs):
                return self.pyro_model.counterfactual_guide(*args, **kwargs, counterfactual_type=self.hparams.cf_elbo_type)
            self.svi = SVI(self.pyro_model.svi_model, guide, Adam(per_param_callable), loss)
        else:
            self.svi = SVI(self.pyro_model.svi_model, self.pyro_model.svi_guide, Adam(per_param_callable), loss)
        self.svi.loss_class = loss
    
    def backward(self, *args, **kwargs):
        pass  # No loss to backpropagate since we're using Pyro's optimisation machinery

    def print_trace_updates(self, batch):
        with torch.no_grad():
            print('Traces:\n' + ('#' * 10))

            guide_trace = pyro.poutine.trace(self.pyro_model.svi_guide).get_trace(batch)
            model_trace = pyro.poutine.trace(pyro.poutine.replay(self.pyro_model.svi_model, trace=guide_trace)).get_trace(batch)

            guide_trace = pyro.poutine.util.prune_subsample_sites(guide_trace)
            model_trace = pyro.poutine.util.prune_subsample_sites(model_trace)

            model_trace.compute_log_prob()
            guide_trace.compute_score_parts()

            print(f'model: {model_trace.nodes.keys()}')
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    fn = site['fn']
                    if isinstance(fn, Independent):
                        fn = fn.base_dist
                    print(f'{name}: {fn} - {fn.support}')
                    log_prob_sum = site["log_prob_sum"]
                    is_obs = site["is_observed"]
                    print(f'model - log p({name}) = {log_prob_sum} | obs={is_obs}')
                    if torch.isnan(log_prob_sum):
                        value = site['value'][0]
                        conc0 = fn.concentration0
                        conc1 = fn.concentration1

                        print(f'got:\n{value}\n{conc0}\n{conc1}')

                        raise Exception()

            print(f'guide: {guide_trace.nodes.keys()}')

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    fn = site['fn']
                    if isinstance(fn, Independent):
                        fn = fn.base_dist
                    print(f'{name}: {fn} - {fn.support}')
                    entropy = site["score_parts"].entropy_term.sum()
                    is_obs = site["is_observed"]
                    print(f'guide - log q({name}) = {entropy} | obs={is_obs}')
    
    def get_trace_metrics(self, batch): 
        metrics = {}

        model = self.svi.loss_class.trace_storage['model']
        guide = self.svi.loss_class.trace_storage['guide']

        for k in self.required_data: 
            # print(f'the nodes are: {model.nodes[k]["name"]}')
            metrics[f'log p({k})'] = model.nodes[k]['log_prob'].mean()
        
        metrics['p(z)'] = model.nodes['z']['log_prob'].mean()
        metrics['q(z)'] = guide.nodes['z']['log_prob'].mean()
        metrics['log p(z) - log q(z)'] = metrics['p(z)'] - metrics['q(z)']

        return metrics 
    
    # can come back and add noise per Their 2016 - check MS implementation for this. 
    
    def prep_batch(self, batch):
        x = 255. * batch['x'].float()  # multiply by 255 b/c preprocess tfms
        out = dict(x=x)
        for k in self.required_data:
            if k in batch and k != 'x':
                out[k] = batch[k].unsqueeze(1).float()

        if self.training: 
            out['x'] += (torch.rand_like(out['x']) - 0.5)
        
        return out 
    
    def training_step(self, batch, batch_idx): 
        batch = self.prep_batch(batch)

        if self.hparams.validate: 
            print('Validation:')
            self.print_trace_updates(batch)
        
        loss = self.svi.step(batch)

        metrics = self.get_trace_metrics(batch)

        if np.isnan(loss):
            self.logger.experiment.add_text('nan', f'nand at {self.current_epoch}:\n{metrics}')
            raise ValueError('loss went to nan with metrics:\n{}'.format(metrics))
        
        tensorboard_logs = {('train/' + k): v for k, v in metrics.items()}
        tensorboard_logs['train/loss'] = loss

        self.log_dict(tensorboard_logs)

        return torch.Tensor([loss])
    
    def validation_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        loss = self.svi.evaluate_loss(batch)

        metrics = self.get_trace_metrics(batch)

        return {'loss': loss, **metrics}
    
    def test_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)
        print(f' the batch type is: {type(batch)}')

        loss = self.svi.evaluate_loss(batch)

        metrics = self.get_trace_metrics(batch)

        samples = self.build_test_samples(batch)

        return {'loss': loss, **metrics, 'samples': samples}
    
    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument('--num_svi_particles', default=4, type=int, help="number of particles to use for ELBO (default: %(default)s)")
        parser.add_argument('--num_sample_particles', default=32, type=int, help="number of particles to use for MC sampling (default: %(default)s)")
        parser.add_argument('--use_cf_guide', default=False, action='store_true', help="whether to use counterfactual guide (default: %(default)s)")
        parser.add_argument(
            '--cf_elbo_type', default=-1, choices=[-1, 0, 1, 2],
            help="-1: randomly select per batch, 0: shuffle thickness, 1: shuffle intensity, 2: shuffle both (default: %(default)s)")

        return parser
    
EXPERIMENT_REGISTRY[SVIExperiment.__name__] = SVIExperiment