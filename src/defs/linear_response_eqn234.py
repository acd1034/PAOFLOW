import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
from PAOFLOW.defs.perturb_split import perturb_split
from PAOFLOW.defs.constants import ELECTRONVOLT_SI,ANGSTROM_AU,H_OVER_TPI,LL
from PAOFLOW.defs.communication import gather_full
from PAOFLOW.defs.smearing import intgaussian, intmetpax, gaussian
import time
bohr_to_cm = 5.29177249e-9

def do_seperate_chi1s(data_controller):
    arry,attr = data_controller.data_dicts()
    if 'ree' in arry['prop'] or 'shc' in arry['prop']:
        if attr['dftSO'] == False:
            if rank == 0:
                print('Relativistic calculation with SO required')
                comm.Abort()
            comm.Barrier()  
    nk, nbnd, nspin = arry['E_k'].shape
        
    for properties in arry['prop']:          
        if properties == 'shc':
            for tensor in arry['s_tensor']:
                spol, jpol, ipol = tensor[0],tensor[1],tensor[2]       
                jdHksp = do_spin_current(data_controller = data_controller, tensor=tensor)  
                oper_matrix1 = np.empty((nk,nbnd,nbnd,nspin), dtype=complex)
                oper_matrix2 = np.empty((nk,nbnd,nbnd,nspin), dtype=complex)
                for ik in range(jdHksp.shape[0]):
                    for ispin in range(jdHksp.shape[3]):
                        oper_matrix1[ik,:,:,ispin],oper_matrix2[ik,:,:,ispin] = perturb_split(jdHksp[ik,:,:,ispin], arry['dHksp'][ik,ipol,:,:,ispin], arry['v_k'][ik,:,:,ispin], arry['degen'][ispin][ik])       
                jdHksp = None 
                for eqn_num in arry['eqn']:
                    if eqn_num == 4:
                        calc_chi2(data_controller = data_controller,tensor=tensor,prop=properties,oper_matrix1=oper_matrix1, oper_matrix2=oper_matrix2)
                    if eqn_num == 2:
                        fermi_surf(data_controller = data_controller,tensor=tensor,prop=properties,oper_matrix1=oper_matrix1, oper_matrix2=oper_matrix2)
                    if eqn_num == 3:
                        fermi_sea(data_controller = data_controller,tensor=tensor,prop=properties,oper_matrix1=oper_matrix1, oper_matrix2=oper_matrix2)
                oper_matrix1=oper_matrix2=None
                jdHksp=None
            
        if properties == 'ree':
            for tensor in arry['ree_tensor']:
                spol,ipol = tensor[0],tensor[1] 
                oper_matrix1 = np.empty((nk,nbnd,nbnd,nspin), dtype=complex)
                oper_matrix2 = np.empty((nk,nbnd,nbnd,nspin), dtype=complex)
                spol, ipol = tensor[0],tensor[1]         
                for ik in range(nk):
                    for ispin in range(nspin):
                        oper_matrix1[ik,:,:,ispin],oper_matrix2[ik,:,:,ispin] = perturb_split(arry['Sj'][spol,:,:], arry['dHksp'][ik,ipol,:,:,ispin], arry['v_k'][ik,:,:,ispin], arry['degen'][ispin][ik])  
                for eqn_num in arry['eqn']:
                    if eqn_num == 4:
                        calc_chi2(data_controller = data_controller,tensor=tensor,prop=properties,oper_matrix1=oper_matrix1, oper_matrix2=oper_matrix2)
                    if eqn_num == 2:
                        fermi_surf(data_controller = data_controller,tensor=tensor,prop=properties,oper_matrix1=oper_matrix1, oper_matrix2=oper_matrix2)
                    if eqn_num == 3:
                        fermi_sea(data_controller = data_controller,tensor=tensor,prop=properties,oper_matrix1=oper_matrix1, oper_matrix2=oper_matrix2)
                oper_matrix1=oper_matrix2=None
                        
        if properties == 'cond':
            for tensor in arry['ree_tensor']:
                cpol,ipol = tensor[0],tensor[1]
                oper_matrix1 = np.empty((nk,nbnd,nbnd,nspin), dtype=complex)
                oper_matrix2 = np.empty((nk,nbnd,nbnd,nspin), dtype=complex)
                cpol, ipol = tensor[0],tensor[1]         
                for ik in range(nk):
                    for ispin in range(nspin):
                        oper_matrix1[ik,:,:,ispin],oper_matrix2[ik,:,:,ispin] = perturb_split(arry['dHksp'][ik,cpol,:,:,ispin], arry['dHksp'][ik,ipol,:,:,ispin], arry['v_k'][ik,:,:,ispin], arry['degen'][ispin][ik])     
                for eqn_num in arry['eqn']:      
                    if eqn_num == 4:
                        calc_chi2(data_controller = data_controller,tensor=tensor,prop=properties,oper_matrix1=oper_matrix1, oper_matrix2=oper_matrix2)
                    if eqn_num == 2:
                        fermi_surf(data_controller = data_controller,tensor=tensor,prop=properties,oper_matrix1=oper_matrix1, oper_matrix2=oper_matrix2)
                    if eqn_num == 3:
                        fermi_sea(data_controller = data_controller,tensor=tensor,prop=properties,oper_matrix1=oper_matrix1, oper_matrix2=oper_matrix2)
                oper_matrix1=oper_matrix2=None
                            

            
def calc_chi2(data_controller = None,tensor=None,prop=None,oper_matrix1=None, oper_matrix2=None):
    '''EQUATION (4)'''
    arry, attr = data_controller.data_dicts()
    nk, nbnd, nspin = arry['E_k'].shape
    if attr['twoD']:
        av0,av1 = arry['a_vectors'][0,:],arry['a_vectors'][1,:]
        cgs_conv = 1./(np.linalg.norm(np.cross(av0,av1))*attr['alat']**2)
    else:
        cgs_conv = 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/(H_OVER_TPI*attr['omega'])  
        
    deltab = 0.001
    E_band = arry['E_k'][:,arry['selected_bands'],:]
    attr['emaxH'] = np.amin(np.array([attr['shift'],attr['emaxH']]))
    ene = np.linspace(attr['eminH'], attr['emaxH'],attr['esize'])   
    if prop == 'shc' or prop == 'ree':
        oper_matrix1 = oper_matrix1[:,:,:,0]
        oper_matrix1 = oper_matrix1[:,:,arry['selected_bands']]
        oper_matrix1 = oper_matrix1[:,arry['selected_bands'],:]
        
        oper_matrix2 = oper_matrix2[:,:,:,0]
        oper_matrix2 = oper_matrix2[:,:,arry['selected_bands']]
        oper_matrix2 = oper_matrix2[:,arry['selected_bands'],:]
        for igamma, gamma in enumerate(arry['gamma']):
            aux1 = np.zeros((nk,len(arry['selected_bands'])), dtype=float)
            for ik in range(nk):
                E_nm = (E_band[ik,:,0] - E_band[ik,:,0][:,None])**2
                aux = oper_matrix1[ik,:,:]*oper_matrix2[ik,:,:].T
                np.fill_diagonal(aux,0.0)
                aux = 2*np.imag(aux)*(gamma**2-E_nm) 
                aux /= ((E_nm+gamma**2)**2 +deltab**2)
                aux1[ik,:] = np.sum(aux, axis=1)
                aux = None 
            aux2 = np.zeros((nk,len(ene)), dtype=float)
            for ie in range(len(ene)):
                smear = intgaussian(E_band[:,:,0],ene[ie],arry['deltakp'][:,arry['selected_bands'],0])
                aux2[:,ie] = np.sum(aux1*smear, axis=1) 
            aux1 = None
            aux2_full = gather_full(aux2, attr['npool'])
            if rank == 0:
                aux2_full = np.sum(aux2_full, axis=0)/aux2_full.shape[0]
                if prop == 'shc':
                    aux2_full *= cgs_conv
                if prop == 'ree':
                    aux2_full *= bohr_to_cm
            xzy = ['x','y','z']
            if prop == 'shc':
                spol, jpol, ipol = tensor[0],tensor[1],tensor[2]
                fname = f'{igamma}_SHC_eqn4_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat' 
            if prop == 'ree':
                spol,ipol = tensor[0],tensor[1]
                fname = f'{igamma}_REE_eqn4_{xzy[spol]}{xzy[ipol]}.dat'      
            data_controller.write_file_row_col(fname, ene, aux2_full)
            aux2_full= None        
            
    if prop == 'cond':
        oper_matrix1 = oper_matrix1[:,:,arry['selected_bands'],:]
        oper_matrix1 = oper_matrix1[:,arry['selected_bands'],:,:]
        
        oper_matrix2 = oper_matrix2[:,:,arry['selected_bands'],:]
        oper_matrix2 = oper_matrix2[:,arry['selected_bands'],:,:]
        for ispin in range (nspin):
            for igamma, gamma in enumerate(arry['gamma']):
                aux1 = np.zeros((nk,len(arry['selected_bands'])), dtype=float)
                for ik in range(nk):
                    E_nm = (E_band[ik,:,ispin] - E_band[ik,:,ispin][:,None])**2
                    aux = oper_matrix1[ik,:,:,ispin]*oper_matrix2[ik,:,:,ispin].T
                    np.fill_diagonal(aux,0.0)
                    #NOTE: multiply eqn (5) with 2 to make it consistent with gamma-->0 defination of SHC-even
                    aux = -2*np.imag(aux)*(gamma**2-E_nm) ##minus sign for conductivity
                    aux /= ((E_nm+gamma**2)**2 +deltab**2)
                    aux1[ik,:] = np.sum(aux, axis=1)
                    aux = None    
                aux2 = np.zeros((nk,len(ene)), dtype=float)                
                for ie in range(len(ene)):
                    smear = intgaussian(E_band[:,:,ispin],ene[ie],arry['deltakp'][:,arry['selected_bands'],ispin])
                    aux2[:,ie] = np.sum(aux1*smear, axis=1) 
                aux1 = None
                aux2_full = gather_full(aux2, attr['npool'])
                if rank == 0:
                    aux2_full = np.sum(aux2_full, axis=0)*(cgs_conv/aux2_full.shape[0])
                xzy = ['x','y','z']
                cpol,ipol = tensor[0],tensor[1]
                fname = f'{igamma}_AHE_eqn4_{xzy[cpol]}{xzy[ipol]}_ispin{ispin}.dat'
                data_controller.write_file_row_col(fname, ene, aux2_full)
                aux2_full= None
    oper_matrix1=oper_matrix2=E_band=None 
                
def fermi_surf(data_controller = None,tensor=None,prop=None,oper_matrix1=None, oper_matrix2=None):
    '''Equation 2'''
    arry, attr = data_controller.data_dicts()
    if attr['twoD']:
        av0,av1 = arry['a_vectors'][0,:],arry['a_vectors'][1,:]
        cgs_conv = 1./(np.linalg.norm(np.cross(av0,av1))*attr['alat']**2)
    else:
        cgs_conv = 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/(H_OVER_TPI*attr['omega'])
        
    if prop == 'shc' or prop == 'ree':
        oper_matrix1 = np.diagonal(oper_matrix1[:,:,:,0], axis1=1, axis2=2)  # (snktot, nawf)
        oper_matrix1 = oper_matrix1[:,arry['selected_bands']]
        
        oper_matrix2 = np.diagonal(oper_matrix2[:,:,:,0], axis1=1, axis2=2)  # (snktot, nawf)
        oper_matrix2 = oper_matrix2[:,arry['selected_bands']]
        
        numerator = np.real(oper_matrix1 * oper_matrix2)    
        gamma = arry['gamma']
        attr['emaxH'] = np.amin(np.array([attr['shift'],attr['emaxH']]))
        ene = np.linspace(attr['eminH'], attr['emaxH'],attr['esize']) 
        for igamma,spec in enumerate(gamma):            
            E_bands = arry['E_k'][:,arry['selected_bands'],0]
            nk = E_bands.shape[0]
            aux_diag = np.zeros((nk,len(ene)), dtype=float)
            for ie in range(len(ene)):
                smear = gaussian(E_bands, ene[ie], arry['deltakp'][:,arry['selected_bands'],0])
                aux = numerator*smear
                aux = np.sum(aux, axis=1)
                aux_diag[:,ie] = aux
                aux = None
            aux_full = gather_full(aux_diag, attr['npool']) 
            if rank == 0:
                deltab = 0.0001 ###this prevents expression from blowing up when user input for Gamma = 0.0
                if prop == 'shc':
                    aux_full = np.sum(aux_full, axis=0)*(cgs_conv/aux_full.shape[0]) 
                if prop == 'ree':
                    aux_full = np.sum(aux_full, axis=0)*(bohr_to_cm/aux_full.shape[0])
                aux_full /= ((-2*spec)+deltab)
            xzy = ['x','y','z']
            if prop == 'shc': 
                spol, jpol, ipol = tensor[0],tensor[1],tensor[2]
                fname = f'{igamma}_SHC_eqn2_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat' 
            if prop == 'ree':
                spol,ipol = tensor[0],tensor[1]
                fname = f'{igamma}_REE_eqn2_{xzy[spol]}{xzy[ipol]}.dat'    
            data_controller.write_file_row_col(fname, ene, aux_full)
            aux_full = None
            E_bands=None
        numerator = None
        oper_matrix1=oper_matrix2=None
        
    if prop == 'cond':
        nk, nbnd, nspin = arry['E_k'].shape
        for ispin in range(nspin):
            oper_matrix1 = np.diagonal(oper_matrix1[:,:,:,ispin], axis1=1, axis2=2)  # (snktot, nawf)
            oper_matrix1 = oper_matrix1[:,arry['selected_bands']]
            
            oper_matrix2 = np.diagonal(oper_matrix2[:,:,:,ispin], axis1=1, axis2=2)  # (snktot, nawf)
            oper_matrix2 = oper_matrix2[:,arry['selected_bands']]
            
            numerator = np.real(oper_matrix1 * oper_matrix2)    
            gamma = arry['gamma']
            attr['emaxH'] = np.amin(np.array([attr['shift'],attr['emaxH']]))
            ene = np.linspace(attr['eminH'], attr['emaxH'],attr['esize']) 
            for igamma,spec in enumerate(gamma):            
                E_bands = arry['E_k'][:,arry['selected_bands'],ispin]
                nk = E_bands.shape[0]
                aux_diag = np.zeros((nk,len(ene)), dtype=float)
                for ie in range(len(ene)):
                    smear = gaussian(E_bands, ene[ie], arry['deltakp'][:,arry['selected_bands'],ispin])
                    aux = numerator*smear
                    aux = np.sum(aux, axis=1)
                    aux_diag[:,ie] = aux
                    aux = None
                aux_full = gather_full(aux_diag, attr['npool']) 
                if rank == 0:
                    deltab = 0.0001 ###this prevents expression from blowing up when user input for Gamma = 0.0
                    aux_full = np.sum(aux_full, axis=0)*(cgs_conv/aux_full.shape[0]) 
                    aux_full /= ((2*spec)+deltab) ##no minus sign for conductivity
                
                cpol,ipol = tensor[0],tensor[1]
                xzy = ['x','y','z']
                fname = f'{igamma}_Cond_eqn2_{xzy[cpol]}{xzy[ipol]}_ispin{ispin}.dat'    
                data_controller.write_file_row_col(fname, ene, aux_full)
                aux_full = None
                E_bands=None
            numerator = None
            oper_matrix1=oper_matrix2=None
    
def fermi_sea(data_controller = None,tensor=None,prop=None,oper_matrix1=None, oper_matrix2=None):
    '''EQUATION (3)'''
    arry, attr = data_controller.data_dicts()
    if attr['twoD']:
        av0,av1 = arry['a_vectors'][0,:],arry['a_vectors'][1,:]
        cgs_conv = 1./(np.linalg.norm(np.cross(av0,av1))*attr['alat']**2)
    else:
        cgs_conv = 1.0e8*ANGSTROM_AU*ELECTRONVOLT_SI**2/(H_OVER_TPI*attr['omega']) 
        
    deltab = 0.001
    
    if prop == 'shc' or prop == 'ree':
        oper_matrix1 = oper_matrix1[:,:,:,0]
        oper_matrix1 = oper_matrix1[:,:,arry['selected_bands']]
        oper_matrix1 = oper_matrix1[:,arry['selected_bands'],:]
        
        oper_matrix2 = oper_matrix2[:,:,:,0]
        oper_matrix2 = oper_matrix2[:,:,arry['selected_bands']]
        oper_matrix2 = oper_matrix2[:,arry['selected_bands'],:]
        
        nk = arry['E_k'].shape[0]
        E_band = arry['E_k'][:,arry['selected_bands'],:]
        attr['emaxH'] = np.amin(np.array([attr['shift'],attr['emaxH']]))
        ene = np.linspace(attr['eminH'], attr['emaxH'],attr['esize'])
        for igamma, gamma in enumerate(arry['gamma']):
            aux_odd = np.zeros((nk,len(arry['selected_bands'])), dtype=float)
            for ik in range(nk):
                E_nm = (E_band[ik,:,0] - E_band[ik,:,0][:,None]) ###NO SQUARE HERE
                aux = oper_matrix1[ik,:,:]*oper_matrix2[ik,:,:].T
                np.fill_diagonal(aux,0.0)
                aux = 2*np.real(aux)*gamma*E_nm
                aux /= ((E_nm**2+gamma**2)**2 +deltab**2)
                aux_odd[ik,:] = np.sum(aux, axis=1)
                aux = None 
            shc_aux = np.zeros((nk,len(ene)), dtype=float)
            
            for ie in range(len(ene)):
                smear = intgaussian(E_band[:,:,0],ene[ie],arry['deltakp'][:,arry['selected_bands'],0])
                shc_aux[:,ie] = (-1)*np.sum(aux_odd*smear, axis=1) ##HACK: confirm that it is negative
            aux_odd = None
            shc_aux_full = gather_full(shc_aux, attr['npool'])
            if rank == 0:
                shc_aux_full = np.sum(shc_aux_full, axis=0)/shc_aux_full.shape[0]
                if prop == 'shc':
                    shc_aux_full *= cgs_conv
                if prop == 'ree':
                    shc_aux_full *= bohr_to_cm
                    
            xzy = ['x','y','z']
            if prop == 'shc': 
                spol, jpol, ipol = tensor[0],tensor[1],tensor[2]
                fname = f'{igamma}_SHC_eqn3_{xzy[spol]}_{xzy[jpol]}{xzy[ipol]}.dat' 
            if prop == 'ree':
                spol,ipol = tensor[0],tensor[1]
                fname = f'{igamma}_REE_eqn3_{xzy[spol]}{xzy[ipol]}.dat' 
            data_controller.write_file_row_col(fname, ene, shc_aux_full)
            shc_aux_full = None
        oper_matrix1=oper_matrix2=None
        E_band=None
            
    if prop == 'cond':
        nk, nbnd, nspin = arry['E_k'].shape
        for ispin in range(nspin):
            oper_matrix1 = oper_matrix1[:,:,:,ispin]
            oper_matrix1 = oper_matrix1[:,:,arry['selected_bands']]
            oper_matrix1 = oper_matrix1[:,arry['selected_bands'],:]
            
            oper_matrix2 = oper_matrix2[:,:,:,ispin]
            oper_matrix2 = oper_matrix2[:,:,arry['selected_bands']]
            oper_matrix2 = oper_matrix2[:,arry['selected_bands'],:]
            
            nk = arry['E_k'].shape[0]
            E_band = arry['E_k'][:,arry['selected_bands'],:]
            attr['emaxH'] = np.amin(np.array([attr['shift'],attr['emaxH']]))
            ene = np.linspace(attr['eminH'], attr['emaxH'],attr['esize'])
            for igamma, gamma in enumerate(arry['gamma']):
                aux_odd = np.zeros((nk,len(arry['selected_bands'])), dtype=float)
                for ik in range(nk):
                    E_nm = (E_band[ik,:,ispin] - E_band[ik,:,ispin][:,None]) ###NO SQUARE HERE
                    aux = oper_matrix1[ik,:,:]*oper_matrix2[ik,:,:].T
                    np.fill_diagonal(aux,0.0)
                    aux = -2*np.real(aux)*gamma*E_nm ##minus for conductivity
                    aux /= ((E_nm**2+gamma**2)**2 +deltab**2)
                    aux_odd[ik,:] = np.sum(aux, axis=1)
                    aux = None 
                shc_aux = np.zeros((nk,len(ene)), dtype=float)
                
                for ie in range(len(ene)):
                    smear = intgaussian(E_band[:,:,0],ene[ie],arry['deltakp'][:,arry['selected_bands'],ispin])
                    shc_aux[:,ie] = np.sum(aux_odd*smear, axis=1) 
                aux_odd = None
                shc_aux_full = gather_full(shc_aux, attr['npool'])
                if rank == 0:
                    shc_aux_full = np.sum(shc_aux_full, axis=0)/shc_aux_full.shape[0]
                    shc_aux_full *= cgs_conv
                        
                xzy = ['x','y','z']
                cpol,ipol = tensor[0],tensor[1]
                fname = f'{igamma}_Cond_eqn3_{xzy[cpol]}{xzy[ipol]}.dat' 
                data_controller.write_file_row_col(fname, ene, shc_aux_full)
                shc_aux_full = None
            oper_matrix1=oper_matrix2=None
            E_band=None
                
def do_spin_current (data_controller = None, tensor=None):
    spol, jpol, ipol = tensor[0],tensor[1],tensor[2]            
    arry,attr = data_controller.data_dicts()
    Sj = arry['Sj'][spol]
    snktot,_,nawf,nawf,nspin = arry['dHksp'].shape
    jdHksp = np.empty((snktot,nawf,nawf,nspin), dtype=complex)
    for ispin in range(nspin):
        for ik in range(snktot):
            jdHksp[ik,:,:,ispin] = 0.5*(np.dot(Sj,arry['dHksp'][ik,jpol,:,:,ispin])+np.dot(arry['dHksp'][ik,jpol,:,:,ispin],Sj))
    return jdHksp







