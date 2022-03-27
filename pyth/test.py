
import nno

def CF(q, u):
   return 1;

class mpc_var:
   PH_length = 4;
   time_step = 0.025;
   appx_zero = 1e-6;
   cost_func = CF;

print(nno.mpc_root(mpc_var,1,[1,1],1))
