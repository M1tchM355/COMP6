"""
    Example for Evaluating COMP6 outputs for a single model. 
    
    Outputs dictionary as JSON file.
"""
import os
import json 
from copy import deepcopy
import itertools

# TODO -> Update to your directories. 
import sys
#sys.path.append("/vast/home/smatin/Code_Repo_22/Hipnn_Fit-Zn/readers/") 
sys.path.append("/usr/projects/ml4chem/internal_datasets/CCSD_DATR/CCSD_gdb11-10-13.h5")
import pyanitools

# Ase for units
import ase.units
import torch
import numpy as np
import hippynn
hippynn.settings.PROGRESS=None
hippynn.settings.WARN_LOW_DISTANCES=False

# Disable Numba warnings. 
from numba.core.errors import NumbaPerformanceWarning
import warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# TODO -> Worth check which version of COMP6 db to work on. 
# Subset of comp6 to use
comp6_subsets={
#'ani_md':['ani_md_bench.h5'],
#'drugbank':['drugbank_testset.h5'],
#'gdb7-9':['gdb11_07_test500.h5','gdb11_08_test500.h5','gdb11_09_test500.h5'],
'gdb10-13':['gdb11_10_test500.h5','gdb11_11_test500.h5','gdb13_12_test1000.h5','gdb13_13_test1000.h5'],
#'gdb7':['gdb11_07_test500.h5'],
#'gdb8':['gdb11_08_test500.h5'],
#'gdb9':['gdb11_09_test500.h5'],
#'gdb10':['gdb11_10_test500.h5'],
#'gdb11':['gdb11_11_test500.h5'],
#'gdb12':['gdb13_12_test1000.h5'],
#'gdb13':['gdb13_13_test1000.h5'],
#'s66x8':['s66x8_wb97x6-31gd.h5'],
#'tripeptide':['tripeptide_full.h5'],
}
comp6_subsets['all']=[x for v in comp6_subsets.values() for x in v]


def get_model_predictor(model_name):
    print(model_name)

    with hippynn.tools.active_directory(model_name,create=False):
        check  = hippynn.experiment.serialization.load_checkpoint_from_cwd(restore_db=False,map_location='cpu')
    model = check['training_modules'].model
    #evaluator = check['training_modules'].evaluator
    
    db_info = check['training_modules'].evaluator.db_info
    
    if torch.cuda.is_available():
        model_device = torch.device('cuda')
    else:
        model_device= torch.device('cpu')
    predictor = hippynn.graphs.Predictor.from_graph(model,model_device=model_device)

    return predictor, db_info


def report_errors(preds, data, db_name=None):
    data = data.splits['all']

    # TODO -> update the key's in the dictionary to match your model and data numbers.
    en_true = data['T']
    en_pred = preds['T'].squeeze(1)
    #print(en_pred.shape,en_true.shape)
    where_atoms = data['Z']!=0
    force_true = data['Grad'][where_atoms]
    force_pred = preds['Grad'][where_atoms]

    en_diff = (en_pred-en_true)
    en_mae = en_diff.abs().mean().item()
    en_rmse = torch.pow(torch.pow(en_diff,2).mean(),0.5).item()

    f_diff = (force_pred-force_true)
    f_mae = f_diff.abs().mean().item()
    f_rmse = torch.pow(torch.pow(f_diff,2).mean(),0.5).item()

    sys_num=data['sys_number']
    unique_sys = np.sort(np.unique(sys_num))
    #print(np.equal(unique_sys,np.arange(len(unique_sys))).all())

    en_diffcomps = []
    for molnum in unique_sys:
        where = sys_num==molnum
        true_mol_en = en_true[where]
        pred_mol_en = en_pred[where]
        true_diff_en = (true_mol_en-true_mol_en.unsqueeze(1))
        pred_diff_en = (pred_mol_en-pred_mol_en.unsqueeze(1))
        n_comf = len(true_mol_en)
        ind = torch.triu_indices(n_comf,n_comf,offset=1).unbind(0)

        #print("n_comf",n_comf,ind)
        #print(ind[0].shape,n_comf*(n_comf-1)/2)
        true_diff_en=true_diff_en[ind]
        pred_diff_en=pred_diff_en[ind]
        differr = pred_diff_en - true_diff_en
        #print("err diff shape",differr.shape)
        en_diffcomps.append(differr)
    en_diffcomps = torch.cat(en_diffcomps)
    endiff_mae = en_diffcomps.abs().mean().item()
    endiff_rmse = torch.pow(torch.pow(en_diffcomps,2).mean(),0.5).item()
    print(f"{en_mae=} {en_rmse=}")
    print(f"{endiff_mae=} {endiff_rmse=}")
    print(f"{f_mae=} {f_rmse=}")

    error_dict = {
        db_name + "_en_mae" : en_mae,
        db_name + "_en_rmse" : en_rmse,
        db_name + "_en_diff_mae" : endiff_mae, 
        db_name + "_en_diff_rmse" : endiff_rmse, 
        db_name + "_f_mae" : f_mae,
        db_name + "_f_rmse" : f_rmse,
    }
    
    return error_dict



def get_dataset(file_list, reduce, data_folder="/vast/home/smatin/Data/COMP6/COMP6v1/"):
    # TODO -> Update data_folder to your variation of COMP6
    
    # TODO -> Self energy for training and comp6 should be similar?
    SELF_ENERGY_APPROX = {'C': -37.830234, 'H': -0.500608, 'N': -54.568004, 'O': -75.036223}
    SELF_ENERGY_APPROX = {k: SELF_ENERGY_APPROX[v] for k, v in zip([6, 1, 7, 8], 'CHNO')}
    SELF_ENERGY_APPROX[0] = 0
    
    from hippynn.databases.h5_pyanitools import PyAniDirectoryDB
    
    torch.set_default_dtype(torch.float64)

    # Manually setting input and target labels. 
    # TODO -> Ensure correct db names. 
    dataset = PyAniDirectoryDB(directory=data_folder, files=file_list, seed=0, allow_unfound=True, inputs=["species", "coordinates"], targets=["energies", "forces"], quiet=False)
    # dataset = PyAniDirectoryDB(directory=data_folder,   files=file_list, seed=0, allow_unfound=True, inputs=None, targets=None, quiet=False)
    
    from hippynn.experiment.routines import test_model
    # TODO -> name map should be updated to match names. 
    name_map={"coordinates":'R','species':'Z','forces':'Grad','energies':'T'}

    del dataset.arr_dict['lot']
    for k,v in name_map.items():
        dataset.arr_dict[v]=dataset.arr_dict[k]
    en_name='T'
    db=dataset
    self_energy = np.vectorize(SELF_ENERGY_APPROX.__getitem__)(db.arr_dict['Z'])

    self_energy = self_energy.sum(axis=1)  # Add up over atoms in system.
    db.arr_dict[en_name] = db.arr_dict[en_name] - self_energy
    torch.set_default_dtype(torch.float32)

    # Unit conversion
    # TODO -> verify that your model units match the COMP6 units. 
    en_keys='Grad','T'
    kcpm = (ase.units.kcal/ase.units.mol)/ase.units.Ha
    kcpm = kcpm**-1
    print("conversion",kcpm)
    for k in en_keys:
        dataset.arr_dict[k]=dataset.arr_dict[k]*kcpm
        pass
    #dataset.arr_dict['Grad']=-dataset.arr_dict['Grad'] 

    if reduce:
        dataset.make_random_split('throw away',.98)
        del dataset.splits['throw away']
    dataset.split_the_rest('all')

    return dataset


def singleModelEval(model_dir, reduce, tag):
    from hippynn.experiment.routines import test_model
    
    comp6_subset_data = {subname: get_dataset(file_list, reduce=reduce) for subname,file_list in comp6_subsets.items()}
    
    ****pred, db_info = get_model_predictor(model_dir)
    
    # db_info 
    all_preds =[] ; error_summary = {}
    for subname,data in comp6_subset_data.items():
        print("Evaluating", subname)
        data.inputs = db_info['inputs']
        data.targets = db_info['targets']
        ****prediction = pred.apply_to_database(data,batch_size=64)['all']
        
        all_preds.append(prediction)
        ed = report_errors(prediction, data, db_name=subname)
        error_summary.update(ed)
    
    with open(f'err_summary_{tag}.json', 'w') as fp:
        json.dump(error_summary, fp)
        
    return error_summary


if __name__=='__main__':
    # Location of model
    model_loc = ""
    model_tag = "" # Name for output data file.
    
    # NOTE : This just uses a very small subset of comp6. Only use when debugging. 
    reduce_arg = False
    if reduce_arg:
        print("reducing data size for debugging")
        comp6_subsets ={'ani_md':['ani_md_bench.h5'],}

    singleModelEval(model_dir=model_loc, reduce=reduce_arg, tag=model_tag)  
