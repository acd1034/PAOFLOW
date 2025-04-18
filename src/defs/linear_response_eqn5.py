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

bohr_to_cm = 5.29177249e-9

def do_chi2_simple(data_controller):  
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
           
    for properties in arry['prop']:
        if properties == 'shc':
            my_tensor = arry['s_tensor']
        if properties == 'ree' or properties == 'cond':
            my_tensor = arry['ree_tensor']            
        for tensor in my_tensor:
            if rank == 0:
                start_time = time.time()
            calc_chi2(data_controller = data_controller,tensor=tensor,prop=properties)
            if rank == 0:
                end_time = time.time()
                total_time = (end_time-start_time)/60
                if properties == 'shc':
                    print(f'EVEN SHC [{tensor[0]}{tensor[1]}{tensor[2]}] completed in {total_time:.4f} mins.')
                if properties == 'ree':
                    print(f'ODD REE [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')
                if properties == 'cond':
                    print(f'Conductivity [{tensor[0]}{tensor[1]}] completed in {total_time:.4f} mins.')
  
def calc_chi2(data_controller = None,tensor = None, prop=None):
    arry,attr = data_controller.data_dicts()
    nk,nbnd,nspin = arry['E_k'].shape
    attr['emaxH'] = np.amin(np.array([attr['shift'],attr['emaxH']]))
    ene = np.linspace(attr['eminH'], attr['emaxH'],attr['esize'])
    esize = attr['esize']
    
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
        
    Om_znkaux = np.zeros((nk,len(arry['selected_bands']), nspin), dtype=float)
    deltap = 0.05
    for ispin in range (nspin):
        for ik in range(nk):
            E_nm = (arry['E_k'][ik,arry['selected_bands'],ispin] - arry['E_k'][ik,arry['selected_bands'],ispin][:,None])**2+deltap**2
            E_nm[np.where(E_nm<1.e-4)] = np.inf
            Om_znkaux[ik,:,ispin] = -2.0*np.sum(np.imag(oper_matrix1[ik,:,:,ispin]*oper_matrix2[ik,:,:,ispin].T)/E_nm, axis=1)     
    oper_matrix1=oper_matrix2=None
    Om_zkaux = np.zeros((nk,esize, nspin), dtype=float) 
    
    for ispin in range(nspin):
        for i in range(esize):
            if attr['smearing'] == 'gauss':
                Om_zkaux[:,i,ispin] = np.sum((Om_znkaux[:,:,ispin]*intgaussian(arry['E_k'][:,arry['selected_bands'],ispin],ene[i],arry['deltakp'][:,arry['selected_bands'],ispin])), axis=1)
            elif attr['smearing'] == 'm-p':
                Om_zkaux[:,i,ispin] = np.sum(Om_znkaux[:,:,ispin]*intmetpax(arry['E_k'][:,arry['selected_bands'],ispin],ene[i],arry['deltakp'][:,arry['selected_bands'],ispin]), axis=1)
            else:
                Om_zkaux[:,i,ispin] = np.sum(Om_znkaux[:,:,ispin]*(0.5 * (-np.sign(arry['E_k'][:,arry['selected_bands'],ispin]-ene[i]) + 1)), axis=1)
       
    if prop != 'cond':
        Om_zkaux = Om_zkaux[:,:,0]  
    Om_zk = gather_full(Om_zkaux, attr['npool'])
    Om_zkaux = None
    if rank == 0:
        if prop == 'shc':
            Om_zk *= attr['cgs_conv']
            Om_zk = np.sum(Om_zk, axis=0)/float(attr['nkpnts'])     
        if prop == 'ree':
            Om_zk *= bohr_to_cm
            Om_zk = np.sum(Om_zk, axis=0)/float(attr['nkpnts'])     
        if prop == 'cond':
            Om_zk *= attr['cgs_conv']
            Om_zk = np.sum(Om_zk, axis=0)/float(attr['nkpnts'])     
    xzy = ['x','y','z']
    if prop == 'shc':
        fname = f'SHC_eqn5_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat' 
        data_controller.write_file_row_col(fname, ene, Om_zk)
        
    if prop == 'ree':
        fname = f'REE_eqn5_{xzy[spol]}{xzy[ipol]}.dat' 
        data_controller.write_file_row_col(fname, ene, Om_zk)
        
    if prop == 'cond':
        fname = f'AHE_eqn5_{xzy[cpol]}{xzy[ipol]}.dat' 
        data_controller.write_file_row_col(fname, ene, Om_zk)          
    ene=Om_zk=None
        
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
                


