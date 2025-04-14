import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import sys
import time
from PAOFLOW.defs.perturb_split import perturb_split
from PAOFLOW.defs.constants import ELECTRONVOLT_SI,ANGSTROM_AU,H_OVER_TPI,LL
from PAOFLOW.defs.communication import gather_full
from PAOFLOW.defs.smearing import intgaussian, intmetpax, gaussian

# bohr_to_m = 5.29177249e-11
bohr_to_cm = 5.29177249e-9

def linear_response_eqn1(data_controller):  
    arry,attr = data_controller.data_dicts()    
    if 'ree' in arry['prop'] or 'shc' in arry['prop']:
        if attr['dftSO'] == False:
            if rank == 0:
                print('Relativistic calculation with SO required')
                comm.Abort()
            comm.Barrier()   
    if attr['twoD']:
        av0,av1 = arry['a_vectors'][0,:],arry['a_vectors'][1,:]
        attr['cgs_conv'] = 1./(np.linalg.norm(np.cross(av0,av1))*attr['alat']**2)
    else: 
        attr['cgs_conv'] = 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/(H_OVER_TPI*attr['omega']) ##only for Bulk
    
    nk,nawf,nspin = arry['E_k'].shape
    '''compute denominator now'''
    E_band = arry['E_k'][:,arry['selected_bands'],:]
    ene = np.linspace(attr['eminH'], attr['emaxH'],attr['esize'])
    denominator = np.zeros((len(arry['gamma']),len(ene),nk,len(arry['selected_bands']),len(arry['selected_bands']), nspin), dtype=float)
    
    for igam,gam in enumerate(arry['gamma']):
        for ispin in range(nspin):
            for ie in range(len(ene)):
                for n in range(len(arry['selected_bands'])):
                    for m in range(len(arry['selected_bands'])):  
                        denominator[igam,ie,:,n,m,ispin] = ((ene[ie]-E_band[:,n,ispin])**2 +gam**2)*((ene[ie]-E_band[:,m,ispin])**2 +gam**2)
                 
    
    for properties in arry['prop']:
        if properties == 'shc':
            my_tensor = arry['s_tensor']
        if properties == 'ree' or properties == 'cond':
            my_tensor = arry['ree_tensor']  
        for tensor in my_tensor:
            if rank == 0:
                start_time = time.time()
            calc_chi1(data_controller = data_controller,denominator=denominator,tensor=tensor, prop=properties)
            if rank == 0:
                end_time = time.time()
                total_time = (end_time-start_time)/60
                if properties == 'shc':
                    print(f'ODD SHC [{tensor[0]}{tensor[1]}{tensor[2]}] completed in {total_time:.4f} mins.')
                if properties == 'ree':
                    print(f'EVEN REE [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')
                if properties == 'cond':
                    print(f'Conductivity [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')
    denominator=E_band=None
                
  
def calc_chi1(data_controller = None,denominator=None,tensor = None,prop = None):
    arry,attr = data_controller.data_dicts()
    nk,nbnd,nspin = arry['E_k'].shape
    gamma = arry['gamma']
    attr['emaxH'] = np.amin(np.array([attr['shift'],attr['emaxH']]))
    ene = np.linspace(attr['eminH'], attr['emaxH'],attr['esize'])
    
    
    oper_matrix1 = np.empty((nk,nbnd,nbnd,nspin), dtype=complex)
    oper_matrix2 = np.empty((nk,nbnd,nbnd,nspin), dtype=complex)
    if prop == 'shc':
        '''spin current operator and corresponding matrix'''
        spol,jpol,ipol = tensor
        jksp_op = spin_current(data_controller = data_controller,tensor= tensor)
        for ispin in range(nspin):
            for ik in range(nk):
                oper_matrix1[ik,:,:,ispin],oper_matrix2[ik,:,:,ispin] = perturb_split(jksp_op[ik,:,:,ispin], arry['dHksp'][ik,ipol,:,:,ispin], arry['v_k'][ik,:,:,ispin], arry['degen'][ispin][ik]) 
        jksp_op = None             
    if prop == 'ree':
        '''compute spin expectation value'''
        spol,ipol = tensor
        for ispin in range(nspin):
            for ik in range(nk):
                oper_matrix1[ik,:,:,ispin],oper_matrix2[ik,:,:,ispin] = perturb_split(arry['Sj'][spol,:,:], arry['dHksp'][ik,ipol,:,:,ispin], arry['v_k'][ik,:,:,ispin], arry['degen'][ispin][ik])           
    if prop == 'cond':
        '''compute charge conductivity tensor'''
        cpol,ipol = tensor
        for ispin in range(nspin):
            for ik in range(nk):
                oper_matrix1[ik,:,:,ispin],oper_matrix2[ik,:,:,ispin] = perturb_split(arry['dHksp'][ik,cpol,:,:,ispin], arry['dHksp'][ik,ipol,:,:,ispin], arry['v_k'][ik,:,:,ispin], arry['degen'][ispin][ik]) 
                
     
                
                 
    oper_matrix1 = oper_matrix1[:,:,arry['selected_bands'],:]
    oper_matrix1 = oper_matrix1[:,arry['selected_bands'],:,:]
    oper_matrix2 = oper_matrix2[:,:,arry['selected_bands'],:]
    oper_matrix2 = oper_matrix2[:,arry['selected_bands'],:,:]
    '''compute numerator now'''
    numerator = np.zeros((nk,len(arry['selected_bands']),len(arry['selected_bands']),nspin), dtype=complex)
    for ispin in range(nspin):
        for ik in range(nk):
            numerator[ik,:,:,ispin] = (oper_matrix1[ik,:,:,ispin]*oper_matrix2[ik,:,:,ispin].T) 
            
    oper_matrix1=oper_matrix2=None
                 
    for igam in range(denominator.shape[0]):
        prop_aux = np.zeros((nk,len(ene), nspin), dtype=float)
        for ispin in range(nspin):
            for ie in range(len(ene)):
                aux = np.real((numerator[:,:,:,ispin]*(gamma[igam]**2))/(denominator[igam,ie,:,:,:,ispin])) 
                aux = np.sum(aux,axis = 2)
                '''apply smearing function over bands now'''
                # smear = intgaussian(arry['E_k'][:,arry['selected_bands'],0], ene[ie], arry['deltakp'][:,arry['selected_bands'],0])
                # shc_aux[:,ie] = np.sum(aux*smear, axis=1)
                '''Marco- comment above two lines and uncomment one line below to not apply smearing'''
                prop_aux[:,ie,ispin] = np.sum(aux, axis=1)
                aux = None
                
        '''gather, fix unit and save'''
        if prop != 'cond':
            prop_aux = prop_aux[:,:,0]
        prop_aux_full = gather_full(prop_aux, attr['npool'])  
        if rank == 0:
            if prop == 'shc':
                prop_aux_full = np.sum(prop_aux_full, axis=0)*(1/prop_aux_full.shape[0]) *(attr['cgs_conv']/np.pi)*(-1)
            if prop == 'ree':
                prop_aux_full = np.sum(prop_aux_full, axis=0)*(1/prop_aux_full.shape[0]) *(bohr_to_cm/np.pi)*(-1)
            if prop == 'cond':
                prop_aux_full = np.sum(prop_aux_full, axis=0)*(1/prop_aux_full.shape[0]) *(attr['cgs_conv']/np.pi)
                      
        xzy = ['x','y','z'] 
        if prop == 'shc':
            fname = f'{igam}_SHC_eqn1_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat' 
            data_controller.write_file_row_col(fname, ene, prop_aux_full)
            prop_aux_full = None
            prop_aux = None
        if prop == 'ree':
            fname = f'{igam}_REE_eqn1_{xzy[spol]}{xzy[ipol]}.dat' 
            data_controller.write_file_row_col(fname, ene, prop_aux_full)
            prop_aux_full = None
            prop_aux = None 
        if prop == 'cond':
            for ispin in range(nspin):
                fname = f'{igam}_Cond_eqn1_{xzy[cpol]}{xzy[ipol]}_ispin{ispin}.dat' 
                data_controller.write_file_row_col(fname, ene, prop_aux_full)
                prop_aux_full = None
                prop_aux = None
        
        
def spin_current(data_controller = None,tensor= None):
    arry,attr = data_controller.data_dicts()
    nk,nbnd,nspin = arry['E_k'].shape
    spol,jspol,ipol = tensor
    Sj = arry['Sj'][spol]
    snktot,_,nawf,nawf,nspin = arry['dHksp'].shape
    jdHksp = np.empty((snktot,nawf,nawf,nspin), dtype=complex)
    for ispin in range(nspin):
        for ik in range(snktot):
            jdHksp[ik,:,:,ispin] = 0.5*(np.dot(Sj,arry['dHksp'][ik,jspol,:,:,ispin])+np.dot(arry['dHksp'][ik,jspol,:,:,ispin],Sj))
    return jdHksp
                


