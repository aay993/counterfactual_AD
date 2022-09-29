import torch
import pyro

from pyro.nn import pyro_method, DenseNN
from pyro.distributions import Normal, Bernoulli, Uniform, TransformedDistribution
from pyro.distributions.conditional import ConditionalTransformedDistribution
from deepscm.distributions.transforms.affine import ConditionalAffineTransform

from deepscm.experiments.medical.adni.sem_vi.base_sem_experiment_adni import BaseVISEM, MODEL_REGISTRY


class ConditionalVISEM(BaseVISEM): 
    # number of context dimensions for decoder (4 b/c brain vol, ventricle vol, MOCA, slice num)
    context_dim = 4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        nonlinearity = torch.nn.LeakyReLU(.1)

        # now we're adding the conditional affine transformations to the flows  
        # (note the shared affine normalisations and exponentiation transforms are in base_sem_experiment) 

        # ventricle_volume flow 
        ventricle_volume_net = DenseNN(2, [8, 16], param_dims=[1,1], nonlinearity=nonlinearity)
        self.ventricle_volume_flow_components = ConditionalAffineTransform(context_nn=ventricle_volume_net, event_dim=0) 
        self.ventricle_volume_flow_transforms = [
            self.ventricle_volume_flow_components, self.ventricle_volume_flow_constraint_transforms
        ]

        # brain_volume flow 
        brain_volume_net = DenseNN(4, [16, 24], param_dims=[1,1], nonlinearity=nonlinearity) 
        self.brain_volume_flow_components = ConditionalAffineTransform(context_nn=brain_volume_net, event_dim=0)
        self.brain_volume_flow_transforms = [
            self.brain_volume_flow_components, self.brain_volume_flow_constraint_transforms
        ]

        # MOCA flow 
        moca_net = DenseNN(2, [8, 16], param_dims=[1,1], nonlinearity=nonlinearity)
        self.moca_flow_components = ConditionalAffineTransform(context_nn=moca_net, event_dim=0)
        self.moca_flow_transforms = [
            self.moca_flow_components, self.moca_flow_constraint_transforms
        ]

        # av45 flow 
        av45_net = DenseNN(1, [8, 16], param_dims=[1,1], nonlinearity=nonlinearity)
        self.av45_flow_components = ConditionalAffineTransform(context_nn=av45_net, event_dim=0)
        self.av45_flow_transforms = [
            self.av45_flow_components, self.av45_flow_constraint_transforms
        ]
    
    @pyro_method
    def pgm_model(self): 
        sex_dist = Bernoulli(logits=self.sex_logits).to_event(1)
        # pseudo call to register with pyro 
        _ = self.sex_logits
        sex = pyro.sample('sex', sex_dist)

        slice_number_dist = Uniform(self.slice_number_min, self.slice_number_max).to_event(1)
        slice_number = pyro.sample('slice_number', slice_number_dist)

        apoE_dist = Uniform(self.apoE_min, self.apoE_max).to_event(1)
        apoE = pyro.sample('APOE4', apoE_dist) #might need a pseudo call here 

        age_base_dist = Normal(self.age_base_loc, self.age_base_scale).to_event(1)
        age_dist = TransformedDistribution(age_base_dist, self.age_flow_transforms)
        age = pyro.sample('age', age_dist)
        _ = self.age_flow_components
        age_ = self.age_flow_constraint_transforms.inv(age)

        tau_base_dist = Normal(self.tau_base_loc, self.tau_base_scale).to_event(1)
        tau_dist = TransformedDistribution(tau_base_dist, self.tau_flow_transforms)
        tau = pyro.sample('tau', tau_dist)
        _ = self.tau_flow_components
        tau_ = self.tau_flow_constraint_transforms.inv(tau)

        education_base_dist = Normal(self.education_base_loc, self.education_base_scale).to_event(1)
        education_dist = TransformedDistribution(education_base_dist, self.education_flow_transforms)
        education = pyro.sample('education', education_dist)
        _ = self.education_flow_components
        education_ = self.education_flow_constraint_transforms.inv(education)

        av45_context = torch.cat([apoE], 1)
        av45_base_dist = Normal(self.av45_base_loc, self.av45_base_scale).to_event(1)
        av45_dist = ConditionalTransformedDistribution(av45_base_dist, self.av45_flow_transforms).condition(av45_context)
        av45 = pyro.sample('av45', av45_dist)
        _ = self.av45_flow_components
        av45_ = self.av45_flow_constraint_transforms.inv(av45)

        brain_context = torch.cat([sex, av45_, tau_, age_], 1)
        brain_volume_base_dist = Normal(self.brain_volume_base_loc, self.brain_volume_base_scale).to_event(1)
        brain_volume_dist = ConditionalTransformedDistribution(brain_volume_base_dist, self.brain_volume_flow_transforms).condition(brain_context)
        brain_volume = pyro.sample('brain_volume', brain_volume_dist)
        _ = self.brain_volume_flow_components
        brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(brain_volume)

        ventricle_context = torch.cat([age_, brain_volume_], 1)
        ventricle_volume_base_dist = Normal(self.ventricle_volume_base_loc, self.ventricle_volume_base_scale).to_event(1)
        ventricle_volume_dist = ConditionalTransformedDistribution(ventricle_volume_base_dist, self.ventricle_volume_flow_transforms).condition(ventricle_context)
        ventricle_volume = pyro.sample('ventricle_volume', ventricle_volume_dist)
        _ = self.ventricle_volume_flow_components
        
        moca_context = torch.cat([brain_volume_, education_], 1)
        moca_base_dist = Normal(self.moca_base_loc, self.moca_base_scale).to_event(1)
        moca_dist = ConditionalTransformedDistribution(moca_base_dist, self.moca_flow_transforms).condition(moca_context)
        moca = pyro.sample('moca', moca_dist)
        _ = self.moca_flow_components

        return dict(sex=sex, slice_number=slice_number, APOE4=apoE, age=age, tau=tau, education=education,
        av45=av45, brain_volume=brain_volume, ventricle_volume=ventricle_volume, moca=moca)
    
    @pyro_method
    def model(self): 
        obs = self.pgm_model()

        ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(obs['ventricle_volume'])
        brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(obs['brain_volume'])
        moca_ = self.moca_flow_constraint_transforms.inv(obs['moca'])
        slice_number = obs['slice_number']
        context = torch.cat([ventricle_volume_, brain_volume_, moca_, slice_number], 1)

        z_base_dist = Normal(self.z_loc, self.z_scale).to_event(1)
        z = pyro.sample('z', z_base_dist)

        latent = torch.cat([z, context], 1)

        x_dist = self._get_transformed_x_dist(latent) # run decoder 
        x = pyro.sample('x', x_dist) 

        obs.update(dict(x=x, z=z))
        return obs 
    
    @pyro_method 
    def guide(self, obs):
        batch_size = obs['x'].shape[0]
        with pyro.plate('observations', batch_size): 
            
            hidden = self.encoder(obs['x']) 

            

            ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(obs['ventricle_volume'])
            brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(obs['brain_volume'])
            
            moca_ = self.moca_flow_constraint_transforms.inv(obs['moca'])
           
            slice_number = obs['slice_number']
            context = torch.cat([ventricle_volume_, brain_volume_, slice_number, moca_], 1)

            
            hidden = torch.cat([hidden, context], 1)

            

            z_base_dist = self.latent_encoder.predict(hidden)
            z = pyro.sample('z', z_base_dist)
        return z

MODEL_REGISTRY[ConditionalVISEM.__name__] = ConditionalVISEM