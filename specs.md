# Specification

## DGP
```
p_dgp = GP([   
    PCA(
        [GP(X), GP(), GP(X, Y, likelihood=Poisson())], #if no X provided just use f?
        output_dim=2
    ),
    GP(X, Y) #if want to provide Y at different layers
])

m = GPModel(X, Y, likelihoods, prior=p_dgp, inference=<dsvi, ds-cvi, etc>)
```

```
FullyConnected(
	PCA([GP(), GP(), GP(), GP()]),
	[GP(), GP()]
)
```

Somehow need to support all combinations and topologies

## Moment Matching

```
t = FullyConnected(
	PCA([GP(), GP(), GP(), GP()]),
	[GP()]
) #probably has to be a single output

GP(X, Y, kernel=kernel.MomentMatchDGP(t))
```



## Standard GP

```
m = GP(X, Y, likelihood=Gauss(), infernce=<variational, batch, cvi>)
```


## MultiTask

```
p = LMC([GP(), GP(), GP()])
p = GPRN(f=[GP(), GP(), GP()], W=[GP(), GP(), GP()])

m = GPModel(X, Y, likelihoods=[Gauss, Poisson], prior=p, infernce=<variational, batch, cvi>)
```

## MultiTask + flow

```
p = Zip(
	LMC([GP(), GP(), GP()]),
	[Exp, Indentity]
)
```

## TGP

```
p = Flow(GP())
m = GPModel(X, Y, likelihoods=[], prior=p, inference=<vi>)
```

Inference could be defined like

```
TGP(CVI|VI|BATCH) 
```

which strips the final transformation and places into the likelihood?

Or just assume that it is a TGP and use closed forms if available

## TGP multioutput

```
p = Zip(
	[gp_1, gp_2]
	[flow_1, flow_2]
)
```

## MultiTask DGP

```
ConstrainedLMC(
	FullyConected(
		[GP, GP, GP], 
		[GP, GP]
	)
)
```

# Initial Parameters

## Explicit

```
approx_posterior = GaussianApproximatePosterior(m, S)
inf = VI(approx_posterior)
m = GP(X, Y, likelihood=Gauss(), infernce=<variational, batch, cvi>)
```


## Easier

```
#and then just pass through to VI
m = GP(X, Y, likelihood=Gauss(), infernce=<variational, batch, cvi>, m_init=m, S_init=S)
```

# Initializer

```
K = SM()
k = initialise(K, X=X, Y=Y)
```

```
flow = SAL()
k = initialise(flow, X=X, Y=Y, **kwargs) #from data
k = initialise(flow, **kwargs) #identity

k = initialise(flow, X=X, Y=Y, type='from_data', **kwargs) #from data
k = initialise(flow, type='to_identity', **kwargs) #identity
```

## Data

No data formatting to be done formatting due to jit, we can check if it is the correct format though

```
X, Y, indexes = TemporalPermutation(X, Y) 
X, Y = ReversePermutation(X, Y, indexes) 
```

```
X, Y, indexes = SpatioTemporalPermutation(X, Y) 
X, Y = ReversePermutation(X, Y, indexes) 
```

## Get Citations

```
m = GPModel(...)
cite(m) -> prints all citations required
```

or does this just invite lots of issues and arguing about who should get cited? -.-

