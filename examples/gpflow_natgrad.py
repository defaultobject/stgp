
def gpflow_model(q_mf):
    import gpflow
    from gpflow.models import VGP, GPR, SGPR, SVGP
    from gpflow.optimizers import NaturalGradient

    inducing_variable = Z

    data = (X, Y)

    svgp = SVGP(
        kernel=gpflow.kernels.RBF(lengthscales=0.2),
        likelihood=gpflow.likelihoods.Gaussian(variance=0.1),
        inducing_variable=inducing_variable,
        q_mu = q_mf[0][0][:, None],
        q_sqrt = np.linalg.cholesky(q_mf[0][1])[None, ...],
        whiten=False
    )

    print(-svgp.elbo(data).numpy())


    natgrad_opt = NaturalGradient(gamma=1.0)
    variational_params = [(svgp.q_mu, svgp.q_sqrt)]
    svgp_natgrad_loss = svgp.training_loss_closure(data)
    natgrad_opt.minimize(svgp_natgrad_loss, var_list=variational_params)

    print(-svgp.elbo(data).numpy())

    pred_mu, pred_var = svgp.predict_f(XS)

    plt.plot(XS, pred_mu)
    plt.scatter(X, Y)
    plt.scatter(Z, np.zeros_like(Z), color='grey')
    plt.show()

