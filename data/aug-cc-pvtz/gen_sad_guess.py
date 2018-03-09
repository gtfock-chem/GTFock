import sys
import numpy as np
from pyscf import gto, scf

basisname = 'augccpvtz'

mol = gto.M(atom='''C  0 0 0''', basis=basisname, cart=True, charge=0);
guess = scf.hf.init_guess_by_atom(mol)
np.savetxt('C.dat', guess.reshape(guess.size,1))

mol = gto.M(atom='''H  0 0 0''', basis=basisname, cart=True, charge=-1);
guess = scf.hf.init_guess_by_atom(mol)
np.savetxt('H.dat', guess.reshape(guess.size,1))

mol = gto.M(atom='''N  0 0 0''', basis=basisname, cart=True, charge=-1);
guess = scf.hf.init_guess_by_atom(mol)
np.savetxt('N.dat', guess.reshape(guess.size,1))

mol = gto.M(atom='''O  0 0 0''', basis=basisname, cart=True, charge=0);
guess = scf.hf.init_guess_by_atom(mol)
np.savetxt('O.dat', guess.reshape(guess.size,1))

mol = gto.M(atom='''P  0 0 0''', basis=basisname, cart=True, charge=-1);
guess = scf.hf.init_guess_by_atom(mol)
np.savetxt('P.dat', guess.reshape(guess.size,1))

mol = gto.M(atom='''S  0 0 0''', basis=basisname, cart=True, charge=0);
guess = scf.hf.init_guess_by_atom(mol)
np.savetxt('S.dat', guess.reshape(guess.size,1))

#      charge
# C  6
# H  1   -1 from add e
# N  7   -1
# O  8
# P 15   -1
# S 16
