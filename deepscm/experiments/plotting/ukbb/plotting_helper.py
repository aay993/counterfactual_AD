ROOT_PATH = '../../../../'
UKBB_DATA_PATH = ROOT_PATH + 'assets/data/ukbb/'
BASE_LOG_PATH = ROOT_PATH + 'assets/models/ukbb/SVIExperiment'

import sys
import os

sys.path.append(ROOT_PATH)

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
import inspect
from collections import OrderedDict
from functools import partial
import torch
from PIL import Image

from tqdm import tqdm, trange

import traceback
import warnings
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore", category=UserWarning)

torch.autograd.set_grad_enabled(False);

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

from matplotlib.colors import ListedColormap
from matplotlib import cm
cmaps = [cm.Reds, cm.Blues, cm.Greens]
img_cm = 'Greys_r'
diff_cm = 'seismic'

from deepscm.datasets.medical.adni import ADNIDataset

# data_dir = '/home/aay993/full_imputed_clinical_covariates.csv' # standard data path 
base_path = '/home/aay993/bias_corrected_registered_slices/' # standard img path 

data_dir = '/home/aay993/val_patients_clinical_covariates.csv' # validation data path 
# base_path = '/home/aay993/validation_brains/slices' # validation img path 
downsample = 3
ukbb_test = ADNIDataset(data_dir, base_path=base_path, crop_type='center', downsample=downsample)

from deepscm.experiments.medical import ukbb  # noqa: F401
from deepscm.experiments.medical.base_experiment_adni import EXPERIMENT_REGISTRY, MODEL_REGISTRY


var_name = {'ventricle_volume': 'v', 
'brain_volume': 'b', 
'sex': 's', 'age': 'a',
'tau': 't', 'education': 'e', 'moca': 'm', 'av45': 'av'}
value_fmt = {
    'ventricle_volume': lambda s: rf'{float(s)/1000:.4g}\,\mathrm{{ml}}',
    'brain_volume': lambda s: rf'{float(s)/1000:.4g}\,\mathrm{{ml}}',
    'age': lambda s: rf'{int(s):d}\,\mathrm{{y}}',
    'sex': lambda s: '{}'.format(['\mathrm{male}', '\mathrm{female}'][int(s)]),
    'tau': lambda s: rf'{float(s):.4}\,\mathrm{{pg/ml}}',
    'education': lambda s: rf'{int(s):d}\,\mathrm{{u}}',
    'moca': lambda s: rf'{int(s):d}\,\mathrm{{score}}',
    'av45': lambda s: rf'{float(s):.4}\,\mathrm{{mSUVR}}',
}

def fmt_intervention(intervention):
    if isinstance(intervention, str):
        var, value = intervention[3:-1].split('=')
        return f"$do({var_name[var]}={value_fmt[var](value)})$"
    else:
        all_interventions = ',\n'.join([f'${var_name[k]}={value_fmt[k](v)}$' for k, v in intervention.items()])
        return f"do({all_interventions})"

def prep_data(batch):
    x = 255. * batch['x'].float().unsqueeze(0)
    age = batch['age'].unsqueeze(0).unsqueeze(0).float()
    sex = batch['sex'].unsqueeze(0).unsqueeze(0).float()
    ventricle_volume = batch['ventricle_volume'].unsqueeze(0).unsqueeze(0).float()
    brain_volume = batch['brain_volume'].unsqueeze(0).unsqueeze(0).float()
    moca = batch['moca'].unsqueeze(0).unsqueeze(0).float()
    education = batch['education'].unsqueeze(0).unsqueeze(0).float()
    tau = batch['tau'].unsqueeze(0).unsqueeze(0).float()
    av45 = batch['av45'].unsqueeze(0).unsqueeze(0).float()
    apoe = batch['APOE4'].unsqueeze(0).unsqueeze(0).float()
    slice_number = batch['slice_number'].unsqueeze(0).unsqueeze(0).float()
    return {'x': x, 'age': age, 'sex': sex, 'ventricle_volume': ventricle_volume,
            'brain_volume': brain_volume, 'education': education,
            'tau': tau, 'moca': moca, 'av45': av45, 'APOE4': apoe, 'slice_number': slice_number}
experiments = ['ConditionalVISEM']
models = {}
loaded_models = {}

for exp in experiments:
    try:
        # checkpoint_path = f'{BASE_LOG_PATH}/{exp}/version_0/'
        checkpoint_path = '/home/aay993/dscm/DSCM_implementation/SVIExperiment/ConditionalVISEM/version_117'

        base_path = os.path.join(checkpoint_path, 'checkpoints')
        checkpoint_path = os.path.join(base_path, os.listdir(base_path)[0])

        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        # hparams = ckpt['hparams'] # previous function (for checkpoints already trained)
        hparams = ckpt['hyper_parameters'] # for newly trained model checkpoints 
        
        model_class = MODEL_REGISTRY[hparams['model']]

        model_params = {
            k: v for k, v in hparams.items() if (k in inspect.signature(model_class.__init__).parameters
                                                 or k in k in inspect.signature(model_class.__bases__[0].__init__).parameters
                                                 or k in k in inspect.signature(model_class.__bases__[0].__bases__[0].__init__).parameters)
        }
        
        new_state_dict = OrderedDict()

        for key, value in ckpt['state_dict'].items():
            new_key = key.replace('pyro_model.', '')
            new_state_dict[new_key] = value
            
        loaded_model = model_class(**model_params)
        loaded_model.load_state_dict(new_state_dict)
        
        for p in loaded_model._buffers.keys():
            if 'norm' in p:
                setattr(loaded_model, p, getattr(loaded_model, p))
                
        loaded_model.eval()
                
                
        loaded_models[exp] = loaded_model
        
        def sample_pgm(num_samples, model):
            with pyro.plate('observations', num_samples):
                return model.pgm_model()
        
        models[exp] = partial(sample_pgm, model=loaded_model)
    except Exception as e:
        print(e)
        traceback.print_exc()
        
def plot_gen_intervention_range(model_name, interventions, idx, normalise_all=True, num_samples=32, save=False):
    fig, ax = plt.subplots(3, len(interventions), figsize=(1.6 * len(interventions), 5), gridspec_kw=dict(wspace=0, hspace=0))
    lim = 0
    
    orig_data = prep_data(ukbb_test[idx])
    
    imgs = []
    for intervention in interventions:
        pyro.clear_param_store()
        cond = {k: torch.tensor([[v]]) for k, v in intervention.items()}
        counterfactual = loaded_models[model_name].counterfactual(orig_data, cond, num_samples)
       
        imgs += [counterfactual['x']] 

        if save: 
            original_image = Image.fromarray(np.array(orig_data['x']).squeeze()).convert("L")
            original_image.save(f"original_{idx}.png")
            counterfactual_image = Image.fromarray(np.array(imgs[-1]).squeeze()).convert("L")
            counterfactual_image.save(f"counterfactual_{idx}.png")
        
        diff = (orig_data['x'] - imgs[-1]).squeeze()

        lim = np.maximum(lim, diff.abs().max())

    for i, intervention in enumerate(interventions):
        x = imgs[i]
        x_test = orig_data['x']
        diff = (x_test - x).squeeze()
        if not normalise_all:
            lim = diff.abs().max()

        ax[0, i].imshow(x_test.squeeze(), 'Greys_r', vmin=0, vmax=255)
        
        ax[0, i].set_title(fmt_intervention(intervention))
        ax[1, i].imshow(x.squeeze(), 'Greys_r', vmin=0, vmax=255)

        ax[2, i].imshow(diff, 'seismic', clim=[-lim, lim])

        for axi in ax[:, i]:
            axi.axis('off')
            axi.xaxis.set_major_locator(plt.NullLocator())
            axi.yaxis.set_major_locator(plt.NullLocator())
    
    suptitle = '$s={sex}; a={age}; b={brain_volume}; v={ventricle_volume}; t={tau}; av45={av45}$'.format(
        **{att: value_fmt[att](orig_data[att].item()) for att in ('sex', 'age', 'brain_volume', 'ventricle_volume', 'tau', 'av45')}
    )
    fig.suptitle(suptitle, fontsize=20, y=1.02)
    
    fig.tight_layout()
    fig.savefig('counterfactual.png', dpi=900, bbox_inches='tight') # Note you're always saving the results here. 
    plt.show()
    
    
def interactive_plot(model_name):
    def plot_intervention(intervention, idx, num_samples=32):
        fig, ax = plt.subplots(1, 4, figsize=(10, 2.5), gridspec_kw=dict(wspace=0, hspace=0))
        lim = 0

        orig_data = prep_data(ukbb_test[idx])
        x_test = orig_data['x']

        pyro.clear_param_store()
        cond = {k: torch.tensor([[v]]) for k, v in intervention.items()}
        counterfactual = loaded_models[model_name].counterfactual(orig_data, cond, num_samples)

        x = counterfactual['x'] 

        diff = (x_test - x).squeeze()

        lim = diff.abs().max()

        ax[1].set_title('Original')
        ax[1].imshow(x_test.squeeze(), 'Greys_r', vmin=0, vmax=255)

        ax[2].set_title(fmt_intervention(intervention))
        ax[2].imshow(x.squeeze(), 'Greys_r', vmin=0, vmax=255)
        print(f'image dimensions are: {x.shape}')

        ax[3].set_title('Difference')
        ax[3].imshow(diff, 'seismic', clim=[-lim, lim])

        print(f'the MSE is:{np.mean(np.array(diff)**2)}')

        for axi in ax:
            axi.axis('off')
            axi.xaxis.set_major_locator(plt.NullLocator())
            axi.yaxis.set_major_locator(plt.NullLocator())

        att_str = '$s={sex}$\n$a={age}$\n$b={brain_volume}$\n$v={ventricle_volume}$\n$av45={av45}$\n$tau={tau}$\n$moca={moca}$\n$education={education}$'.format(
            **{att: value_fmt[att](orig_data[att].item()) for att in ('sex', 'age', 'brain_volume', 'ventricle_volume', 'av45', 'tau', 'moca', 'education')}
        )

        ax[0].text(0.5, 0.5, att_str, horizontalalignment='center',
                      verticalalignment='center', transform=ax[0].transAxes,
                      fontsize=mpl.rcParams['axes.titlesize'])

        plt.show()
    
    from ipywidgets import interactive, IntSlider, FloatSlider, HBox, VBox, Checkbox, Dropdown

    def plot(image, age, sex, brain_volume, ventricle_volume, tau, av45, education, moca, do_age, do_sex, do_brain_volume, do_ventricle_volume,  do_tau, do_av45, do_education, do_moca):
        intervention = {}
        if do_age:
            intervention['age'] = age
        if do_sex:
            intervention['sex'] = sex
        if do_brain_volume:
            intervention['brain_volume'] = brain_volume * 1000.
        if do_ventricle_volume:
            intervention['ventricle_volume'] = ventricle_volume * 1000.
        if do_tau:
            intervention['tau'] = tau
        if do_av45: 
            intervention['av45'] = av45
        if do_education: 
            intervention['education'] = education 
        if do_moca: 
            intervention['moca'] = moca 

        plot_intervention(intervention, image)

    w = interactive(plot, image=IntSlider(min=0, max=400, description='Image #'), age=FloatSlider(min=30., max=120., step=1., continuous_update=False, description='Age'),
                    do_age=Checkbox(description='do(age)'),
              sex=Dropdown(options=[('female', 0.), ('male', 1.)], description='Sex'),
                    do_sex=Checkbox(description='do(sex)'),
              brain_volume=FloatSlider(min=800., max=1600., step=10., continuous_update=False, description='Brain Volume (ml):', style={'description_width': 'initial'}),
              do_brain_volume=Checkbox(description='do(brain_volume)'),
              ventricle_volume=FloatSlider(min=3., max=330., step=1., continuous_update=False, description='Ventricle Volume (ml):', style={'description_width': 'initial'}),
              do_ventricle_volume=Checkbox(description='do(ventricle_volume)'),
              tau=FloatSlider(min=5., max=200., step=1., continuous_update=False, description='Phosphorylated tau (unit):', style={'description_width': 'initial'}),
              do_tau=Checkbox(description='do(tau)'),
              av45=FloatSlider(min=.5, max=150., step=1., continuous_update=False, description='AV45 (unit):', style={'description_width': 'initial'}),
              do_av45=Checkbox(description='do(av45)'),
              education=FloatSlider(min=5., max=50., step=1., continuous_update=False, description='Education (unit):', style={'description_width': 'initial'}),
              do_education=Checkbox(description='do(education)'),
              moca=FloatSlider(min=1., max=50., step=1., continuous_update=False, description='MOCA score (unit):', style={'description_width': 'initial'}),
              do_moca=Checkbox(description='do(moca)'))

    ui = VBox([w.children[0], VBox([HBox([w.children[i + 1], w.children[i + 9]]) for i in range(8)]), w.children[-1]])


    display(ui)

    w.update()
