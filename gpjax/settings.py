class Settings():
    jitter = 1e-10
    nat_grad_jitter = 0 #set to 1 to enable jitter in natural gradients

    enforce_psd=False


    #Useful if default behaviour is not wanted. Will throw errors if any defaults are needed.
    strict_mode=False

    verbose=True

    #Use monte-carlo sampling for the expected log likelihood and predictive distributions
    force_black_box=False

    seed=42
