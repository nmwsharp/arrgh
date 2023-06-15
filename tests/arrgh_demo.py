import os
import sys
import os.path as path

# Path to where the library lives
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from arrgh import arrgh

if __name__ == "__main__":

    ## test it!

    # plain python vlaues
    noneval = None
    intval1 = 7
    intval2 = -3
    floatval0 = 42.0
    floatval1 = 5.5 * 1e-12
    floatval2 = 7.7232412351231231234 * 1e44

    # numpy values
    import numpy as np
    npval1 = np.arange(100)
    npval2 = np.arange(10000)
    npval3 = np.arange(10000).astype(np.uint64)
    npval4 = np.arange(10000).astype(np.float32).reshape(100,10,10)
    npval5 = np.arange(10000)[-1]

    # torch values 
    torchval1 = None
    torchval2 = None
    torchval3 = None
    torchval4 = None
    try:
        import torch
        torchval1 = torch.randn((1000,12,3))
        torchval2 = torch.randn((1000,12,3)).cuda()
        torchval3 = torch.arange(1000)
        torchval4 = torch.arange(1000)[0]
    except ModuleNotFoundError:
        pass
    
    # jax values 
    jaxval1 = None
    jaxval2 = None
    jaxval3 = None
    jaxval4 = None
    try:
        import jax
        import jax.numpy as jnp
        jaxval1 = jnp.linspace(0,1,10000)
        jaxval2 = jnp.linspace(0,1,10000).reshape(100,10,10)
        jaxval3 = jnp.arange(1000)
        jaxval4 = jnp.arange(1000)[0]
    except ModuleNotFoundError:
        pass
        
       
    print("\nbasic test")
    arrgh(noneval, \
             intval1, intval2, \
             floatval0, floatval1, floatval2, \
             npval1, npval2, npval3, npval4, npval4[0,:,2:], npval5, \
             torchval1, torchval2, torchval3, torchval4, \
             jaxval1, jaxval2, jaxval3, jaxval4, \
    )


    # test keyword name printrin
    print("\nkeyword test")
    arrgh(intval1, intval2, 2.*npval1, times2=2.323124123*npval1, times2_again=2.*npval1)
    
    print("\nwidth test")
    arrgh(intval1, intval2, 2.*npval1, times2=2.323124123*npval1, times2_again=2.*npval1, arrgh_float_width=10)
    
    print("\npassing invalid data test")
    arrgh(intval1, npval1, 'bad val', arrgh)
